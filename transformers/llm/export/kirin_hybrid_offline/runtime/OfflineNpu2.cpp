#include "OfflineNpu2.hpp"
#include "llmconfig.hpp"
#include "rapidjson/document.h"
#include <MNN/expr/ExprCreator.hpp>
#include <chrono>
#include <cstring>
#include <algorithm>

namespace OfflineNpu2 {
using namespace MNN::Transformer;
using namespace MNN::Express;
namespace {
constexpr int H=2048;
class Hybrid final : public Llm {
public:
    Hybrid(std::shared_ptr<LlmConfig> config, std::unique_ptr<Vision> visual,
           std::string visionPath, std::shared_ptr<State> state, KirinFullNpu::Log log)
        : Llm(std::move(config)), visual(std::move(visual)), visionPath(std::move(visionPath)),
          state(std::move(state)), log(std::move(log)) {}
    bool load() override {
        // Base Llm loads only the language graph. Omni::load (CPU/chunk ViT) is
        // intentionally never entered. All MNN execution stays on CPU.
        if (!Llm::load()) { fail("MNN CPU language model load failed"); return false; }
        if (!visual->load(visionPath,state->error)) { fail(state->error); return false; }
        return true;
    }
    std::vector<int> tokenizer_encode(const std::string& input) override {
        state->error.clear(); features.clear(); deep.clear();
        if (!parseImages(input,prompt,state->error)) { fail(state->error); return {}; }
        auto ids=Llm::tokenizer_encode(prompt.text);
        if (ids.empty() || ids.size()>32768) { fail("Empty or oversized prompt"); return {}; }
        const int history=mConfig->reuse_kv()?mContext->all_seq_len:0;
        if (history==0) ropeEnds.clear();
        // Agent mode can erase the previous screenshot/answer back to its text
        // prefix. Restore that prefix's position, rather than the prior image's
        // compressed offset. Unknown imported KV histories are rejected.
        if (history>0 && !ropeEnds.count(history)) { fail("Missing M-RoPE history after KV edit"); return {}; }
        int base=history>0?ropeEnds[history]:0;
        if (!makePositions(ids,prompt,base,state->error)) { fail(state->error); return {}; }
        ropeEnds[history+ids.size()]=prompt.nextPosition;
        const auto start=std::chrono::steady_clock::now();
        for (const auto& image:prompt.images) {
            Features f;
            if (!visual->run(image,f,state->error)) { fail(state->error); return {}; }
            features.push_back(std::move(f));
        }
        if (!features.empty()) {
            mContext->vision_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
            mContext->pixels_mp+=features.size()*608.0f*256.0f/1e6f;
        }
        prepared=ids;
        return ids;
    }
    std::vector<int> tokenizer_encode(const MultimodalPrompt&) override {
        fail("Offline NPU currently requires <img> file paths, not in-memory media"); return {};
    }
    void response(const std::vector<int>& ids,std::ostream* os,const char* end,int limit) override {
        if (!state->error.empty()) return;
        // Public token-vector callers have not used the multimodal tokenizer.
        if (ids!=prepared) {
            features.clear(); prompt={}; deep.clear();
            if (!makePositions(ids,prompt,mConfig->reuse_kv()?mContext->all_seq_len:0,state->error)) { fail(state->error); return; }
            prepared=ids;
        }
        Llm::response(ids,os,end,limit);
    }
    VARP embedding(const std::vector<int>& ids) override {
        ExecutorScope scope(mExecutor);
        auto e=Llm::embedding(ids);
        if (mContext->gen_seq_len>0) { deep.clear(); return e; }
        if (e.get()==nullptr || !e->getInfo() || e->getInfo()->size!=int(ids.size()*H)) { fail("CPU embedding shape mismatch"); return nullptr; }
        if (ids!=prepared || !injectFeatures(prompt,features,ids.size(),e->writeMap<float>(),deep,state->error)) {
            fail(state->error.empty()?"Prefill tokens differ from prepared image spans":state->error); return nullptr;
        }
        return e;
    }
    VARP gen_position_ids(int count) override {
        ExecutorScope scope(mExecutor);
        auto positions=_Input({3,count},NCHW,halide_type_of<int>());
        auto* dst=positions->writeMap<int>();
        if (mContext->gen_seq_len>0) {
            int first=prompt.nextPosition-1+mContext->gen_seq_len;
            for (int axis=0;axis<3;++axis) for (int i=0;i<count;++i) dst[axis*count+i]=first+i;
            ropeEnds[mContext->all_seq_len+count]=first+count;
        } else {
            if (prompt.positions.size()!=size_t(3*count)) { fail("Prefill M-RoPE sequence mismatch"); return nullptr; }
            std::memcpy(dst,prompt.positions.data(),prompt.positions.size()*sizeof(int));
        }
        return positions;
    }
    std::vector<VARP> forwardRaw(VARP hidden,VARP mask,VARP pos,VARPS args={}) override {
        ExecutorScope scope(mExecutor);
        if (!state->error.empty() || hidden.get()==nullptr || !hidden->getInfo()) return {};
        const int count=hidden->getInfo()->size/H;
        if (count<=0 || !args.empty()) { fail("Unsupported CPU language input contract"); return {}; }
        auto ds=_Input({3,count,H},NCHW,halide_type_of<float>());
        auto* dst=ds->writeMap<float>(); std::fill(dst,dst+3*count*H,0.0f);
        if (mContext->gen_seq_len==0 && !deep.empty()) {
            const int total=static_cast<int>(prepared.size());
            const int offset=mChunkStart>=0?mChunkStart:0;
            const int valid=std::max(0,std::min(count,total-offset));
            if (valid>0) for (int layer=0;layer<3;++layer)
                std::memcpy(dst+size_t(layer)*count*H,deep.data()+(size_t(layer)*total+offset)*H,size_t(valid)*H*sizeof(float));
        }
        auto outputs=Llm::forwardRaw(hidden,mask,pos,{ds});
        if (outputs.empty() || mContext->status==LlmStatus::INTERNAL_ERROR) fail("MNN CPU language forward failed");
        return outputs;
    }
private:
    void fail(std::string error) {
        state->error=std::move(error);
        mContext->status=state->error=="cancelled"?LlmStatus::USER_CANCEL:LlmStatus::INTERNAL_ERROR;
        if(log) log("[OFFLINE_NPU2] level=error "+state->error);
    }
    std::unique_ptr<Vision> visual;
    std::string visionPath;
    std::shared_ptr<State> state;
    KirinFullNpu::Log log;
    Prompt prompt;
    std::vector<Features> features;
    std::vector<float> deep;
    std::vector<int> prepared;
    std::map<int,int> ropeEnds;
};
}
std::unique_ptr<Llm> create(const std::string& path,std::unique_ptr<KirinFullNpu::Backend> backend,
                          KirinFullNpu::Log log,KirinFullNpu::Cancel cancel,const std::shared_ptr<State>& state) {
    state->error.clear();
    // Parse marker before constructing LlmConfig; reject arbitrary/full-NPU packages.
    std::ifstream f(path); std::string content{std::istreambuf_iterator<char>(f),{}};
    rapidjson::Document d; d.Parse(content.c_str());
    if (d.HasParseError() || !d.IsObject() || !d.HasMember("offline_npu2") ||
        !d["offline_npu2"].IsObject() || !d["offline_npu2"].HasMember("vision_config") ||
        !d["offline_npu2"].HasMember("format") || !d["offline_npu2"]["format"].IsString() ||
        std::string(d["offline_npu2"]["format"].GetString())!="kirin_offline_npu2_v1" ||
        !d["offline_npu2"]["vision_config"].IsString() ||
        std::string(d["offline_npu2"]["vision_config"].GetString())!="vision/config.json") {
        state->error="Select a dedicated Offline NPU package (MNN CPU LLM + external-weight vision_full)"; return {};
    }
    try {
    auto config=std::make_shared<LlmConfig>(path);
    if(config->config_.value("model_type",std::string())!="qwen3_vl" ||
       config->config_.value("hidden_size",0)!=2048 || config->config_.value("layer_nums",0)!=28 ||
       !config->config_.value("has_deepstack",false) || !config->config_.value("is_mrope",false)) {
        state->error="Offline NPU requires the matching Mobi Qwen3-VL 2B CPU language model"; return {};
    }
    config->config_["backend_type"]="cpu";
    config->config_["prompt_cache"]=false;
    config->mllm_config_["backend_type"]="cpu";
    config->config_["mllm"]=config->mllm_config_;
    // Prevent accelerator block configs and speculative modules from taking over.
    config->config_["visual_split"]=false;
    config->config_["speculative_type"]="";
    auto visual=std::unique_ptr<Vision>(new Vision(std::move(backend),log,std::move(cancel)));
    return std::unique_ptr<Llm>(new Hybrid(config,std::move(visual),config->base_dir_+"vision/config.json",state,log));
    } catch (const std::exception& e) {
        state->error=std::string("Invalid Offline NPU MNN configuration: ")+e.what();
        return {};
    }
}
void prepareRequest(Llm* llm,const std::shared_ptr<State>& state) {
    if (!state || !llm) return;
    if (!state->error.empty() || llm->getContext()->status==LlmStatus::USER_CANCEL || llm->getContext()->status==LlmStatus::INTERNAL_ERROR) {
        llm->reset();
        const_cast<LlmContext*>(llm->getContext())->status=LlmStatus::NORMAL_FINISHED;
    }
    state->error.clear();
}
} // namespace OfflineNpu2
