//
//  NPULayerNorm.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPULayerNorm.hpp"
#include "NPUBackend.hpp"
#include "../custom/LayerNormCustomOp.hpp"
#include "core/FileLoader.hpp"
#include <cmath>
#include <sstream>

// On this DDK, hiai::op::LayerNorm silently zeros the gamma input — output
// always equals beta regardless of gamma values (verified empirically with
// gamma=0 vs gamma=real producing the same NPU output). Decompose LN into
// primitive ops (ReduceMean + Tile + Sub + Mul + Add + Rsqrt) instead.
// Set this to 0 only if you need to A/B against the broken HiAI LayerNorm op.
#ifndef MNN_HIAI_LN_USE_PRIMITIVES
#define MNN_HIAI_LN_USE_PRIMITIVES 0
#endif

// Preferred Kirin9030 custom LayerNorm path. This keeps MNN's current
// flatten-to-[M, normSize, 1, 1] convention and feeds gamma/beta/epsilon with
// the same semantics as the original hiai::op::LayerNorm bridge, while adding
// norm_size for the custom kernel's row-wise reduction.
#ifndef MNN_HIAI_LN_USE_CUSTOM
#define MNN_HIAI_LN_USE_CUSTOM 0
#endif

// Paddle-Lite Kirin NPU style path. When enabled (default), feeds gamma/beta
// to hiai::op::LayerNorm as 1D Const → hiai::op::Reshape → 4D activation
// tensor (NOT as a directly-baked 4D weight Const). Hypothesis: HiAI's
// weight pipeline (TransAndMergeWeights → NC1HWC0 layout bake) is what
// silently zeros gamma on this DDK; routing gamma through Reshape makes it
// a runtime activation tensor that takes a different code path.
//
// Reference: Paddle-Lite PR #8483 (TangYuan-Liu, 2022-02), still in master,
// NCHW/NHWC bug acknowledged but workaround works to ~1e-2 numerical
// tolerance on their hardware. See:
//   /data/dahu/mlsys/Paddle-Lite/lite/backends/nnadapter/nnadapter/src/
//       driver/huawei_kirin_npu/converter/layer_normalization.cc
//
// This macro takes precedence over MNN_HIAI_LN_USE_PRIMITIVES. Disable
// (define to 0) to fall back to decomposition or the original 4D-weight-
// const path (depending on MNN_HIAI_LN_USE_PRIMITIVES).
#ifndef MNN_HIAI_LN_USE_PADDLELITE
#define MNN_HIAI_LN_USE_PADDLELITE 0
#endif

// Debug stage selector for the decomposed path. Lets you expose any
// intermediate tensor as the LN output to bisect precision bugs:
//   0 = full LN (default, production)
//   1 = mean         (broadcast to [M, normSize, 1, 1] via Tile)
//   2 = centered = x - mean
//   3 = sq = centered^2
//   4 = var          (broadcast)
//   5 = var+eps      (broadcast)
//   6 = invStd       (broadcast)
//   7 = normalized = centered * invStd
//   8 = scaled = normalized * gamma
// Stages with reduced shape ([M,1,1,1]) get expanded back to [M,normSize,1,1]
// via hiai::op::Tile. We use Tile (HiAI 100.310.010.013) over BroadcastTo
// (100.500.010.010): older op = wider DDK support, and NPUTile.cpp shows
// it's already exercised in this codebase.
#ifndef MNN_HIAI_LN_DEBUG_STAGE
#define MNN_HIAI_LN_DEBUG_STAGE 0
#endif

using namespace std;

namespace MNN {

NPULayerNorm::NPULayerNorm(MNN::Backend *b, const MNN::Op *op, const std::vector<Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs) : NPUCommonExecution(b, op) {}

static bool tensorLooksAllZero(const std::vector<float>& data, float* minOut, float* maxOut) {
    if (data.empty()) {
        if (minOut) *minOut = 0.0f;
        if (maxOut) *maxOut = 0.0f;
        return true;
    }
    float mn = data[0];
    float mx = data[0];
    float absMax = std::fabs(data[0]);
    for (size_t i = 1; i < data.size(); ++i) {
        float v = data[i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        absMax = std::max(absMax, std::fabs(v));
    }
    if (minOut) *minOut = mn;
    if (maxOut) *maxOut = mx;
    return absMax < 1e-12f;
}

ErrorCode NPULayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);
    auto opName = mOp->name()->str();
    {
        std::ostringstream shapeOs;
        if (!inputs.empty() && inputs[0] != nullptr) {
            const auto& shp = inputs[0]->shape();
            for (size_t i = 0; i < shp.size(); ++i) {
                if (i) shapeOs << "x";
                shapeOs << shp[i];
            }
        } else {
            shapeOs << "?";
        }
        MNN_HIAI_LOGV2("[NPU_LN_BUILD_TAG=v3-bcast-2026-04-25] NPULayerNorm::onResize ENTER name=%s in_shape=%s",
                     opName.c_str(), shapeOs.str().c_str());
    }
    auto param = mOp->main_as_LayerNorm();
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto inputIndex = mOp->inputIndexes()->data()[0];
    auto iops = mNpuBackend->mGrapMap[inputIndex]; // x
    xOp = iops.back().first;

    constw = hiai::op::Const(opName + "_w_const");
    constb = hiai::op::Const(opName + "_b_const");

    auto shape = inputs[0]->shape();
    int32_t normSize = shape.empty() ? 0 : shape.back();
    vector<float> gammaData;
    vector<float> betaData;
    bool loadedGammaBeta = false;

    bool embeddedSuspiciousAllZero = false;
    // 1) Try embedded gamma/beta (already de-externalized by loader).
    if (param->gamma() != nullptr && param->beta() != nullptr &&
        param->gamma()->size() > 0 && param->gamma()->size() == param->beta()->size()) {
        int32_t count = static_cast<int32_t>(param->gamma()->size());
        gammaData.resize(count);
        betaData.resize(count);
        ::memcpy(gammaData.data(), param->gamma()->data(), count * sizeof(float));
        ::memcpy(betaData.data(), param->beta()->data(), count * sizeof(float));
        float gmin = 0.0f, gmax = 0.0f, bmin = 0.0f, bmax = 0.0f;
        bool gAllZero = tensorLooksAllZero(gammaData, &gmin, &gmax);
        bool bAllZero = tensorLooksAllZero(betaData, &bmin, &bmax);
        embeddedSuspiciousAllZero = gAllZero && bAllZero;
        loadedGammaBeta = !embeddedSuspiciousAllZero;
        MNN_HIAI_LOGV2("NPULayerNorm(%s): embedded gamma/beta count=%d g[min=%g max=%g] b[min=%g max=%g]%s",
                     opName.c_str(), count, gmin, gmax, bmin, bmax,
                     embeddedSuspiciousAllZero ? " (suspicious all-zero, will try external)" : "");
    }

    // 2) If embedded data is missing/suspicious, try loading from external file by offset.
    if (!loadedGammaBeta && param->external() != nullptr && param->external()->size() >= 3) {
        auto ext = param->external()->data();
        int64_t offset = ext[0];
        int64_t gammaBytes = ext[1];
        int64_t betaBytes = ext[2];
        if (gammaBytes > 0 && betaBytes > 0 && gammaBytes % sizeof(float) == 0 &&
            betaBytes % sizeof(float) == 0 && gammaBytes == betaBytes) {
            int32_t count = static_cast<int32_t>(gammaBytes / sizeof(float));
            gammaData.resize(count);
            betaData.resize(count);
            if (mOp->externalPath() != nullptr) {
                FileLoader loader(mOp->externalPath()->c_str());
                if (loader.valid()) {
                    bool ok = (loader.offset(offset) == 0);
                    ok = ok && loader.read(reinterpret_cast<char*>(gammaData.data()), gammaBytes);
                    ok = ok && loader.read(reinterpret_cast<char*>(betaData.data()), betaBytes);
                    loadedGammaBeta = ok;
                    if (!ok) {
                        MNN_HIAI_LOGV2("NPULayerNorm(%s): read external gamma/beta failed, file=%s off=%lld gb=%lld bb=%lld",
                                     opName.c_str(), mOp->externalPath()->c_str(),
                                     (long long)offset, (long long)gammaBytes, (long long)betaBytes);
                    } else {
                        float gmin = 0.0f, gmax = 0.0f, bmin = 0.0f, bmax = 0.0f;
                        bool gAllZero = tensorLooksAllZero(gammaData, &gmin, &gmax);
                        bool bAllZero = tensorLooksAllZero(betaData, &bmin, &bmax);
                        MNN_HIAI_LOGV2("NPULayerNorm(%s): loaded gamma/beta from external, count=%d g[min=%g max=%g] b[min=%g max=%g]%s",
                                     opName.c_str(), count, gmin, gmax, bmin, bmax,
                                     (gAllZero && bAllZero) ? " (WARNING: both all-zero)" : "");
                    }
                } else {
                    MNN_HIAI_LOGV2("NPULayerNorm(%s): externalPath invalid: %s",
                                 opName.c_str(), mOp->externalPath()->c_str());
                }
            } else {
                MNN_HIAI_LOGV2("NPULayerNorm(%s): gamma/beta absent and externalPath is null", opName.c_str());
            }
        }
    }

    if (!loadedGammaBeta && embeddedSuspiciousAllZero) {
        // Keep suspicious embedded data as a weaker fallback before identity.
        loadedGammaBeta = true;
        MNN_HIAI_LOGV2("NPULayerNorm(%s): external unavailable, fallback to embedded all-zero gamma/beta",
                     opName.c_str());
    }

    // 3) Last fallback: identity LN affine.
    if (!loadedGammaBeta) {
        if (normSize <= 0) {
            MNN_HIAI_LOGV2("NPULayerNorm(%s): invalid normSize=%d", opName.c_str(), normSize);
            return INPUT_DATA_ERROR;
        }
        gammaData.assign(normSize, 1.0f);
        betaData.assign(normSize, 0.0f);
        MNN_HIAI_LOGV2("NPULayerNorm(%s): fallback to identity gamma/beta, size=%d", opName.c_str(), normSize);
    }

    normSize = static_cast<int32_t>(gammaData.size());
    float eps = param->epsilon();

#if MNN_HIAI_LN_USE_CUSTOM
    // ===== Kirin9030 custom LayerNorm path =====
    vector<int64_t> gammaShape{1, static_cast<int64_t>(gammaData.size()), 1, 1};
    ge::TensorDesc gdesc(ge::Shape(gammaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
    gtensor->SetTensorDesc(gdesc);
    gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()), gammaData.size() * sizeof(float));
    constw.set_attr_value(gtensor);

    vector<int64_t> betaShape{1, static_cast<int64_t>(betaData.size()), 1, 1};
    ge::TensorDesc bdesc(ge::Shape(betaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
    btensor->SetTensorDesc(bdesc);
    btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()), betaData.size() * sizeof(float));
    constb.set_attr_value(btensor);

    int64_t totalElems = 1;
    for (auto d : shape) totalElems *= d;
    int64_t mLong = (normSize > 0) ? (totalElems / normSize) : 0;
    if (normSize <= 0 || mLong <= 0 || mLong * normSize != totalElems) {
        MNN_HIAI_LOGV2("NPULayerNorm(%s): cannot flatten shape for LayerNormCustom "
                     "(total=%lld normSize=%d)",
                     opName.c_str(), (long long)totalElems, normSize);
        return NOT_SUPPORT;
    }
    int32_t M = static_cast<int32_t>(mLong);

    mPreShapeConst = hiai::op::Const(opName + "_pre_shape");
    {
        std::vector<int32_t> preShape = {M, normSize, 1, 1};
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(preShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(preShape.data()),
                         preShape.size() * sizeof(int32_t));
        mPreShapeConst.set_attr_value(ptensor);
    }
    auto preReshape = std::make_shared<hiai::op::Reshape>(opName + "_pre_reshape");
    (*preReshape).set_input_x(*xOp.get()).set_input_shape(mPreShapeConst);

    auto layerNorm = std::make_shared<hiai::op::LayerNormCustom>(opName + "_custom_ln");
    (*layerNorm).set_input_x(*preReshape.get())
                .set_input_gamma(constw)
                .set_input_beta(constb)
                .set_attr_epsilon(eps)
                .set_attr_norm_size(normSize);

    mPostShapeConst = hiai::op::Const(opName + "_post_shape");
    {
        std::vector<int32_t> postShape(shape.begin(), shape.end());
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(postShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(postShape.data()),
                         postShape.size() * sizeof(int32_t));
        mPostShapeConst.set_attr_value(ptensor);
    }
    auto postReshape = std::make_shared<hiai::op::Reshape>(opName + "_post_reshape");
    (*postReshape).set_input_x(*layerNorm.get()).set_input_shape(mPostShapeConst);

    mNpuBackend->setOutputOps(mOp, {preReshape, layerNorm, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (LayerNormCustom) name=%s normSize=%d M=%d",
                 opName.c_str(), normSize, M);
    return NO_ERROR;
#else
#if MNN_HIAI_LN_USE_PADDLELITE
    // ===== Paddle-Lite Kirin NPU style path =====
    // Reference: paddle-lite/.../huawei_kirin_npu/converter/layer_normalization.cc
    //
    // Topology (5 ops):
    //   x [orig] --pre_reshape-> [M, normSize, 1, 1] -+
    //                                                  |--> hiai::op::LayerNorm
    //   gamma [normSize] --gamma_reshape-> [1, N, 1, 1] -+   (begin_norm_axis=1,
    //   beta  [normSize] --beta_reshape-> [1, N, 1, 1]  -+    epsilon=eps)
    //                                                       |
    //                                                       v
    //                                          --post_reshape-> [orig]
    //
    // Key difference vs the original 4D-const path: gamma/beta are 1D Consts
    // fed through a Reshape op (runtime activation tensor) instead of being
    // baked as 4D Const weights. This bypasses HiAI's TransAndMergeWeights /
    // NC1HWC0 weight pipeline that we suspect is silently zeroing gamma on
    // this DDK.
    //
    // Shape rule (paddle-lite layer_normalization.cc:23-32): for input
    // [1, S, H] with axis=-1=2, the (n, h, w) axis=2 case applies → 4D wrap
    // = (1*S, H, 1, 1) = [S, H, 1, 1] = [M, normSize, 1, 1].

    int32_t rank = static_cast<int32_t>(shape.size());
    if (rank <= 0 || normSize <= 0) {
        MNN_HIAI_LOGV2("NPULayerNorm(%s): invalid rank=%d normSize=%d",
                     opName.c_str(), rank, normSize);
        return INPUT_DATA_ERROR;
    }
    int64_t totalElems = 1;
    for (auto d : shape) totalElems *= d;
    int64_t mLong = totalElems / normSize;
    if (mLong <= 0 || mLong * normSize != totalElems) {
        MNN_HIAI_LOGV2("NPULayerNorm(%s): cannot flatten shape (total=%lld normSize=%d)",
                     opName.c_str(), (long long)totalElems, normSize);
        return NOT_SUPPORT;
    }
    int32_t M = static_cast<int32_t>(mLong);

    // pre-Reshape: input -> [M, normSize, 1, 1]
    mPreShapeConst = hiai::op::Const(opName + "_pre_shape");
    {
        std::vector<int32_t> preShape = {M, normSize, 1, 1};
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(preShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(preShape.data()),
                         preShape.size() * sizeof(int32_t));
        mPreShapeConst.set_attr_value(ptensor);
    }
    auto preReshape = std::make_shared<hiai::op::Reshape>(opName + "_pre_reshape");
    (*preReshape).set_input_x(*xOp.get()).set_input_shape(mPreShapeConst);

    // Affine shape Const = [1, normSize, 1, 1] (target shape for gamma/beta).
    mAffineShapeConst = hiai::op::Const(opName + "_affine_shape");
    {
        std::vector<int32_t> aShape = {1, normSize, 1, 1};
        ge::TensorDesc adesc(ge::Shape({static_cast<int64_t>(aShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr atensor = std::make_shared<ge::Tensor>();
        atensor->SetTensorDesc(adesc);
        atensor->SetData(reinterpret_cast<uint8_t*>(aShape.data()),
                         aShape.size() * sizeof(int32_t));
        mAffineShapeConst.set_attr_value(atensor);
    }

    // gamma 1D Const: shape [normSize]
    constw = hiai::op::Const(opName + "_w_const");
    {
        ge::TensorDesc gdesc(ge::Shape({static_cast<int64_t>(gammaData.size())}),
                             ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
        gtensor->SetTensorDesc(gdesc);
        gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()),
                         gammaData.size() * sizeof(float));
        constw.set_attr_value(gtensor);
    }
    // gamma Reshape: [normSize] -> [1, normSize, 1, 1]
    auto gammaReshape = std::make_shared<hiai::op::Reshape>(opName + "_gamma_reshape");
    (*gammaReshape).set_input_x(constw).set_input_shape(mAffineShapeConst);

    // beta 1D Const: shape [normSize]
    constb = hiai::op::Const(opName + "_b_const");
    {
        ge::TensorDesc bdesc(ge::Shape({static_cast<int64_t>(betaData.size())}),
                             ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
        btensor->SetTensorDesc(bdesc);
        btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()),
                         betaData.size() * sizeof(float));
        constb.set_attr_value(btensor);
    }
    // beta Reshape: [normSize] -> [1, normSize, 1, 1]
    auto betaReshape = std::make_shared<hiai::op::Reshape>(opName + "_beta_reshape");
    (*betaReshape).set_input_x(constb).set_input_shape(mAffineShapeConst);

    // hiai::op::LayerNorm
    auto layerNorm = std::make_shared<hiai::op::LayerNorm>(opName + "_ln");
    (*layerNorm).set_input_x(*preReshape.get())
                .set_input_gamma(*gammaReshape.get())
                .set_input_beta(*betaReshape.get())
                .set_attr_begin_norm_axis(1)
                .set_attr_begin_params_axis(1)
                .set_attr_epsilon(eps);

    // post-Reshape: LayerNorm output -> original input shape
    mPostShapeConst = hiai::op::Const(opName + "_post_shape");
    {
        std::vector<int32_t> postShape(shape.begin(), shape.end());
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(postShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(postShape.data()),
                         postShape.size() * sizeof(int32_t));
        mPostShapeConst.set_attr_value(ptensor);
    }
    auto postReshape = std::make_shared<hiai::op::Reshape>(opName + "_post_reshape");
    (*postReshape).set_input_x(*layerNorm.get()).set_input_shape(mPostShapeConst);

    mNpuBackend->setOutputOps(mOp,
        {preReshape, gammaReshape, betaReshape, layerNorm, postReshape},
        outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (paddlelite) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
    return NO_ERROR;
#else
#if MNN_HIAI_LN_USE_PRIMITIVES
    // ===== Decomposed LayerNorm (bypasses broken hiai::op::LayerNorm) =====
    // y = (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta
    //
    // Layout: canonical NCHW [M, normSize, 1, 1] (matches hiai::op::LayerNorm
    // and hiai::op::InstanceNorm conventions, see MindSpore Lite NPU delegate
    // instance_norm_npu.cc and MNN's existing NPULayerNorm hiai-op path).
    // Reduce on axis 1 (C = feature dim).
    //
    // Why explicit Tile on mean / invStd:
    //   NPU's elementwise Sub/Mul on [M,C,1,1] vs [M,1,1,1] (single-broadcast
    //   on the C dim with C=1024) fails ProduceTiling with code 1443627009
    //   ("Sub ln_out_centered compileInfo is null"). Materialising mean/invStd
    //   to the full [M, C, 1, 1] shape via hiai::op::Tile turns Sub/Mul
    //   into pure same-shape elementwise, which compiles cleanly.
    //
    // gamma/beta keep [1, normSize, 1, 1]: that is broadcast on the N dim
    // only, matching the InstanceNorm reference and the way the original
    // hiai::op::LayerNorm consumes them — proven supported on this DDK.
    int32_t rank = static_cast<int32_t>(shape.size());
    if (rank <= 0 || normSize <= 0) {
        MNN_HIAI_LOGV2("NPULayerNorm(%s): invalid rank=%d normSize=%d", opName.c_str(), rank, normSize);
        return INPUT_DATA_ERROR;
    }
    int64_t totalElems = 1;
    for (auto d : shape) totalElems *= d;
    int64_t mLong = (normSize > 0) ? (totalElems / normSize) : 0;
    if (mLong <= 0 || mLong * normSize != totalElems) {
        MNN_HIAI_LOGV2("NPULayerNorm(%s): cannot flatten shape for 4D wrap "
                     "(total=%lld normSize=%d)",
                     opName.c_str(), (long long)totalElems, normSize);
        return NOT_SUPPORT;
    }
    int32_t M = static_cast<int32_t>(mLong);

    // pre-Reshape: x -> [M, normSize, 1, 1]
    mPreShapeConst = hiai::op::Const(opName + "_pre_shape");
    {
        std::vector<int32_t> preShape = {M, normSize, 1, 1};
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(preShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(preShape.data()),
                         preShape.size() * sizeof(int32_t));
        mPreShapeConst.set_attr_value(ptensor);
    }
    auto preReshape = std::make_shared<hiai::op::Reshape>(opName + "_pre_reshape");
    (*preReshape).set_input_x(*xOp.get()).set_input_shape(mPreShapeConst);

    // Tile multiples Const = [1, normSize, 1, 1]. Used by hiai::op::Tile to
    // expand a reduced-shape op ([M,1,1,1]) to [M, normSize, 1, 1]. We use
    // Tile (older HiAI op, 100.310.010.013) rather than BroadcastTo
    // (100.500.010.010): wider DDK support and it's already used in MNN's
    // NPUTile.cpp. The multiples vector is itself a 1D rank-4 const.
    mTileMultConst = hiai::op::Const(opName + "_tile_mult");
    {
        std::vector<int32_t> mult = {1, normSize, 1, 1};
        ge::TensorDesc tdesc(ge::Shape({static_cast<int64_t>(mult.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ttensor = std::make_shared<ge::Tensor>();
        ttensor->SetTensorDesc(tdesc);
        ttensor->SetData(reinterpret_cast<uint8_t*>(mult.data()),
                         mult.size() * sizeof(int32_t));
        mTileMultConst.set_attr_value(ttensor);
    }

    // axes Const = [1] for ReduceMean (reduce C=normSize dim of [M,normSize,1,1]).
    // FORMAT_ND for axes matches NPUReduction.cpp's working pattern.
    mAxesConst = hiai::op::Const(opName + "_ln_axes");
    {
        std::vector<int32_t> axes = {1};
        ge::TensorDesc d(ge::Shape({static_cast<int64_t>(axes.size())}),
                         ge::FORMAT_ND, ge::DT_INT32);
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        t->SetTensorDesc(d);
        t->SetData(reinterpret_cast<uint8_t*>(axes.data()),
                   axes.size() * sizeof(int32_t));
        mAxesConst.set_attr_value(t);
    }

    // eps Const: scalar wrapped as [1,1,1,1] so Add(var, eps) is same-rank.
    // FORMAT must be NCHW, not ND — the NPU compiler converts weight Consts
    // to internal NC1HWC0 via TransAndMergeWeights; ND -> NC1HWC0 is
    // unimplemented on this DDK ("Trans 2 to 3 not support") and crashes
    // graph build. NCHW -> NC1HWC0 is supported.
    mEpsConst = hiai::op::Const(opName + "_ln_eps");
    {
        std::vector<int64_t> epsShape = {1, 1, 1, 1};
        ge::TensorDesc d(ge::Shape(epsShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr t = std::make_shared<ge::Tensor>();
        t->SetTensorDesc(d);
        t->SetData(reinterpret_cast<uint8_t*>(&eps), sizeof(float));
        mEpsConst.set_attr_value(t);
    }

    // gamma/beta Const: [1, normSize, 1, 1] — N-dim broadcast at consumption.
    std::vector<int64_t> affineShape = {1, static_cast<int64_t>(normSize), 1, 1};
    {
        ge::TensorDesc gdesc(ge::Shape(affineShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
        gtensor->SetTensorDesc(gdesc);
        gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()),
                         gammaData.size() * sizeof(float));
        constw.set_attr_value(gtensor);
    }
    {
        ge::TensorDesc bdesc(ge::Shape(affineShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
        ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
        btensor->SetTensorDesc(bdesc);
        btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()),
                         betaData.size() * sizeof(float));
        constb.set_attr_value(btensor);
    }

    // ---- Stage-conditional op construction ----
    // We ONLY construct ops actually used by the chosen stage's output path.
    // Previous code built all 11+ ops then included only a subset in
    // setOutputOps; the unused shared_ptrs went out of scope at function exit
    // while still holding pointers into the chained ops, which corrupted
    // HiAI's IR builder ("graph_builder.cpp BuildOutNode outOpImpl null"
    // crash). With per-stage construction every constructed op stays alive
    // through setOutputOps registration.
    //
    // We use hiai::op::Tile (NOT BroadcastTo) to expand reduced-shape
    // intermediates: Tile is older (100.310.010.013), more battle-tested,
    // and already proven by MNN's NPUTile.cpp.

    // mean = ReduceMean(x_pre, axes=[1], keep_dims=true) -> [M, 1, 1, 1]
    auto mean = std::make_shared<hiai::op::ReduceMean>(opName + "_mean");
    (*mean).set_input_x(*preReshape.get())
           .set_input_axes(mAxesConst)
           .set_attr_keep_dims(true);

    // mean_full = Tile(mean, mult=[1, normSize, 1, 1]) -> [M, normSize, 1, 1]
    auto meanFull = std::make_shared<hiai::op::Tile>(opName + "_mean_full");
    (*meanFull).set_input_x(*mean.get()).set_input_multiples(mTileMultConst);

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 2
    // centered = x_pre - mean_full   (same-shape elementwise, no broadcast)
    auto centered = std::make_shared<hiai::op::Sub>(opName + "_centered");
    (*centered).set_input_x1(*preReshape.get())
               .set_input_x2(*meanFull.get());
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 3
    // sq = centered * centered
    auto sq = std::make_shared<hiai::op::Mul>(opName + "_sq");
    (*sq).set_input_x1(*centered.get())
         .set_input_x2(*centered.get());
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 4
    // var = ReduceMean(sq, axes=[1], keep_dims=true) -> [M, 1, 1, 1]
    auto var = std::make_shared<hiai::op::ReduceMean>(opName + "_var");
    (*var).set_input_x(*sq.get())
          .set_input_axes(mAxesConst)
          .set_attr_keep_dims(true);
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 4
    // Stage 4 only: tile var to full shape so post-Reshape gets [M,N,1,1].
    auto varFull = std::make_shared<hiai::op::Tile>(opName + "_var_full");
    (*varFull).set_input_x(*var.get()).set_input_multiples(mTileMultConst);
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 5
    // var_eps = var + eps   (small tensors, single broadcast on N — supported)
    auto varEps = std::make_shared<hiai::op::Add>(opName + "_var_eps");
    (*varEps).set_input_x1(*var.get())
             .set_input_x2(mEpsConst);
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 5
    auto varEpsFull = std::make_shared<hiai::op::Tile>(opName + "_var_eps_full");
    (*varEpsFull).set_input_x(*varEps.get()).set_input_multiples(mTileMultConst);
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 6
    // invStd = rsqrt(var_eps) -> [M, 1, 1, 1]
    auto invStd = std::make_shared<hiai::op::Rsqrt>(opName + "_inv_std");
    (*invStd).set_input_x(*varEps.get());

    // invStd_full = Tile(invStd, mult=[1, normSize, 1, 1]) -> [M, normSize, 1, 1]
    auto invStdFull = std::make_shared<hiai::op::Tile>(opName + "_inv_std_full");
    (*invStdFull).set_input_x(*invStd.get()).set_input_multiples(mTileMultConst);
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 7
    // normalized = centered * invStd_full   (same-shape elementwise)
    auto normalized = std::make_shared<hiai::op::Mul>(opName + "_norm");
    (*normalized).set_input_x1(*centered.get())
                 .set_input_x2(*invStdFull.get());
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0 || MNN_HIAI_LN_DEBUG_STAGE >= 8
    // scaled = normalized * gamma   (gamma N-dim broadcast, out_H*out_W=1)
    auto scaled = std::make_shared<hiai::op::Mul>(opName + "_scaled");
    (*scaled).set_input_x1(*normalized.get())
             .set_input_x2(constw);
#endif

#if MNN_HIAI_LN_DEBUG_STAGE == 0
    // y = scaled + beta   (beta N-dim broadcast)
    auto y = std::make_shared<hiai::op::Add>(opName + "_y");
    (*y).set_input_x1(*scaled.get())
        .set_input_x2(constb);
#endif

    // post-Reshape: <chosen op> -> original input shape
    mPostShapeConst = hiai::op::Const(opName + "_post_shape");
    {
        std::vector<int32_t> postShape(shape.begin(), shape.end());
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(postShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(postShape.data()),
                         postShape.size() * sizeof(int32_t));
        mPostShapeConst.set_attr_value(ptensor);
    }
    auto postReshape = std::make_shared<hiai::op::Reshape>(opName + "_post_reshape");

#if MNN_HIAI_LN_DEBUG_STAGE == 0
    // Production: full LN output.
    (*postReshape).set_input_x(*y.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, var, varEps, invStd,
         invStdFull, normalized, scaled, y, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=0 production) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 1
    // Stage 1: mean broadcast back to full shape.
    (*postReshape).set_input_x(*meanFull.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=1 mean) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 2
    (*postReshape).set_input_x(*centered.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=2 centered) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 3
    (*postReshape).set_input_x(*sq.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=3 sq) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 4
    (*postReshape).set_input_x(*varFull.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, var, varFull, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=4 var) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 5
    (*postReshape).set_input_x(*varEpsFull.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, var, varEps, varEpsFull, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=5 var_eps) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 6
    (*postReshape).set_input_x(*invStdFull.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, var, varEps, invStd,
         invStdFull, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=6 invStd) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 7
    (*postReshape).set_input_x(*normalized.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, var, varEps, invStd,
         invStdFull, normalized, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=7 normalized) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#elif MNN_HIAI_LN_DEBUG_STAGE == 8
    (*postReshape).set_input_x(*scaled.get()).set_input_shape(mPostShapeConst);
    mNpuBackend->setOutputOps(mOp,
        {preReshape, mean, meanFull, centered, sq, var, varEps, invStd,
         invStdFull, normalized, scaled, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (decomposed-Tile, stage=8 scaled) "
                 "name=%s rank=%d normSize=%d M=%d eps=%g",
                 opName.c_str(), rank, normSize, M, eps);
#else
#  error "Unsupported MNN_HIAI_LN_DEBUG_STAGE value (use 0..8)"
#endif
    return NO_ERROR;
#else
    // ===== Original hiai::op::LayerNorm path (gamma silently zeroed on this DDK) =====
    shared_ptr<hiai::op::LayerNorm> layerNorm(new hiai::op::LayerNorm(opName));
    vector<int64_t> gammaShape{1, static_cast<int64_t>(gammaData.size()), 1, 1};
    ge::TensorDesc gdesc(ge::Shape(gammaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr gtensor = std::make_shared<ge::Tensor>();
    gtensor->SetTensorDesc(gdesc);
    gtensor->SetData(reinterpret_cast<uint8_t*>(gammaData.data()), gammaData.size() * sizeof(float));
    constw.set_attr_value(gtensor);

    vector<int64_t> betaShape{1, static_cast<int64_t>(betaData.size()), 1, 1};
    ge::TensorDesc bdesc(ge::Shape(betaShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr btensor = std::make_shared<ge::Tensor>();
    btensor->SetTensorDesc(bdesc);
    btensor->SetData(reinterpret_cast<uint8_t*>(betaData.data()), betaData.size() * sizeof(float));
    constb.set_attr_value(btensor);

    int64_t totalElems = 1;
    for (auto d : shape) totalElems *= d;
    int64_t mLong = (normSize > 0) ? (totalElems / normSize) : 0;
    if (normSize <= 0 || mLong <= 0 || mLong * normSize != totalElems) {
        MNN_HIAI_LOGV2("NPULayerNorm(%s): cannot flatten shape for HiAI begin_norm_axis=1 convention "
                     "(total=%lld normSize=%d)",
                     opName.c_str(), (long long)totalElems, normSize);
        return NOT_SUPPORT;
    }
    int32_t M = static_cast<int32_t>(mLong);

    mPreShapeConst = hiai::op::Const(opName + "_pre_shape");
    {
        std::vector<int32_t> preShape = {M, normSize, 1, 1};
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(preShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(preShape.data()),
                         preShape.size() * sizeof(int32_t));
        mPreShapeConst.set_attr_value(ptensor);
    }
    shared_ptr<hiai::op::Reshape> preReshape(new hiai::op::Reshape(opName + "_pre_reshape"));
    (*preReshape).set_input_x(*xOp.get()).set_input_shape(mPreShapeConst);

    (*layerNorm).set_input_x(*preReshape.get())
                .set_input_gamma(constw)
                .set_input_beta(constb)
                // .set_attr_begin_norm_axis(1)
                // .set_attr_begin_params_axis(1)
                .set_attr_epsilon(eps);

    mPostShapeConst = hiai::op::Const(opName + "_post_shape");
    {
        std::vector<int32_t> postShape(shape.begin(), shape.end());
        ge::TensorDesc pdesc(ge::Shape({static_cast<int64_t>(postShape.size())}),
                             ge::FORMAT_NCHW, ge::DT_INT32);
        ge::TensorPtr ptensor = std::make_shared<ge::Tensor>();
        ptensor->SetTensorDesc(pdesc);
        ptensor->SetData(reinterpret_cast<uint8_t*>(postShape.data()),
                         postShape.size() * sizeof(int32_t));
        mPostShapeConst.set_attr_value(ptensor);
    }
    shared_ptr<hiai::op::Reshape> postReshape(new hiai::op::Reshape(opName + "_post_reshape"));
    (*postReshape).set_input_x(*layerNorm.get()).set_input_shape(mPostShapeConst);

    mNpuBackend->setOutputOps(mOp, {preReshape, layerNorm, postReshape}, outputs);
    MNN_HIAI_LOGV2("NPULayerNorm::onResize EXIT (hiai::LayerNorm) name=%s normSize=%d M=%d",
                 opName.c_str(), normSize, M);
    return NO_ERROR;
#endif // MNN_HIAI_LN_USE_PRIMITIVES
#endif // MNN_HIAI_LN_USE_PADDLELITE
#endif // MNN_HIAI_LN_USE_CUSTOM
}

NPUCreatorRegister<TypedCreator<NPULayerNorm>> __LayerNorm_op(OpType_LayerNorm);

} // namespace MNN
