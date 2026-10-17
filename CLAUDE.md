# MNN Project Instructions

MNN is a lightweight deep learning **inference engine** (not a training framework), targeting mobile and server platforms. Supports CNN / Transformer / LLM / Diffusion models. Code must prioritize **performance and binary size**.

## Restricted Access

> The following directories contain internal proprietary code. **Do NOT read, modify, or reference** any files within:
> - `schema/private/`
> - `source/internal/`

## Architecture Overview

MNN uses a **graph optimization + heterogeneous backend scheduling** architecture.

Two inference APIs are available:
- **Session API** (low-level): `Interpreter → createSession → runSession`, operates on Tensor directly
- **Module API** (high-level, recommended): `Module::load → onForward(VARP)`, Express-based dynamic graph. Used by LLM / Diffusion and most modern workloads

**Key abstractions** (see corresponding headers under `source/core/`):
- **Interpreter** / **Session**: model loading and inference session management
- **Backend** / **Execution**: hardware backend abstraction and per-op implementation (CPU/Metal/CUDA/OpenCL/Vulkan/...)
- **Tensor**: data container; internally uses NC4HW4 format (channels packed by 4 for SIMD)
- **Op / Schema**: FlatBuffers-defined operator descriptors (`schema/default/*.fbs`)

**Op registration pattern**: Schema definition → shape inference (`source/shape/`) → Geometry decomposition (optional) → Backend Execution implementation

## LLM Subsystem

MNN supports end-to-end LLM export and inference:

- **Python export** (`transformers/llm/export/`): HuggingFace model → MNN format. Core modules: `llmexport.py` entry point, `utils/model_mapper.py` (model field mapping), `utils/model.py` (unified LlmModel class), `utils/transformers.py` (Attention/Decoder/RoPE export)
- **C++ inference** (`transformers/llm/engine/`): `llm.cpp` (text inference), `omni.cpp` (multimodal: vision/audio), includes KVCache management and sampling strategies

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/MNN/` | Public C++ headers |
| `source/core/` | Inference core (Interpreter, Session, Pipeline, Backend) |
| `source/backend/` | Hardware backend implementations (cpu, arm82, metal, cuda, opencl, vulkan, ...) |
| `source/shape/` | Shape inference |
| `source/geometry/` | Geometry computation (op decomposition) |
| `express/` | Express API (high-level dynamic graph, VARP) |
| `schema/default/` | FlatBuffers schema (op definitions) |
| `tools/converter/` | Model converter (ONNX/TF/Caffe → MNN) |
| `transformers/llm/` | LLM export (Python) + inference engine (C++) |
| `transformers/diffusion/` | Diffusion model support |
| `pymnn/` | Python bindings |
| `test/` | Test cases |
| `skills/` | AI Agent Skills |

## Coding Style

- **C++**: Google Style variant, see `.clang-format`. 4-space indent, 120-char line width, attached braces. Class names `PascalCase`, functions `camelCase`, member variables `mCamelCase`. RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`). Default standard: C++11.
- **Python**: Standard Python conventions
- **Formatting**: `clang-format -i -style=file <file>`

## Build & Test

```bash
# Build C++ (with LLM)
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# Common CMake options: MNN_BUILD_TEST, MNN_BUILD_CONVERTER, MNN_METAL, MNN_OPENCL,
# MNN_VULKAN, MNN_CUDA, MNN_ARM82, MNN_BUILD_QUANTOOLS, MNN_SUPPORT_TRANSFORMER_FUSE
# Full list: see option() declarations at the top of CMakeLists.txt

# Unit tests
cd build && ./run_test.out

# LLM export
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL

# LLM test
cd build
./llm_demo /path/to/MODEL/config.json prompt.txt

# LLM benchmark
./llm_bench -m /path/to/MODEL/config.json
```

Test suite includes: unit tests (`run_test.out`), model tests, conversion tests (ONNX/TF/TFLite/Torch), quantization tests, LLM tests, PyMNN tests. See `test.sh` and `test/` directory for details.

## Skills

For the following tasks, **read the Skill entry file first** and execute step by step. Each step must pass its tests before proceeding.

**After completing any skill-driven task, run the Retrospective skill** to reflect on mistakes and update the skill with lessons learned.

| Skill | Entry File | Trigger |
|-------|-----------|---------|
| Support new LLM | `skills/support-new-llm/SKILL.md` | Add / adapt a new LLM model |
| Add new op | `skills/add-new-op/SKILL.md` | Add a new operator |
| ARM CPU optimization | `skills/arm-cpu-optimize/SKILL.md` | Optimize op performance on ARM CPU |
| Retrospective | `skills/retrospective/SKILL.md` | After any non-trivial task: reflect on mistakes, update relevant skills with lessons learned |


## 端侧鸿蒙 App 源码

本机 App 工程位于 `/home/ma-user/workspace/feh/mobiinfra-oh`，原生入口在
`entry/src/main/cpp/napi_init.cpp`，界面在 `entry/src/main/ets/`。
`entry/libs/arm64-v8a/libMNN.so` 来自本仓库的 Harmony 构建。修改本机源码后需要重新编译、安装 App，下载模型不会更新已安装的原生库。


## OMG/OMC 离线 NPU 图编译

### 主要方案：单张 ViT 离线图 + 外置权重 + CPU LLM

**现在“离线NPU”默认指本方案，原“离线NPU-2”按钮已改为“离线NPU”。**
先阅读 [单图视觉离线 NPU 实现与验证](transformers/llm/export/kirin_hybrid_offline/README.md)。

- ViT、最终 merger 和三路 DeepStack merger 合并为 **一张 `vision_full.omc`**，全部在 NPU 执行。
- 使用 **图与权重分离**：OMG `--target=omc --save_weights_as_external_data=true`，运行时通过 `HMS_HiAIExecutor_InitWeights` 加载外置 `SubGraph_0.weight`。保持一个视觉 executor，跨图片/请求复用。
- 视觉沿用 Full NPU 已冻结的 **W4A16/group64**，固定高 608、宽 256，每图 152 个语言图像 token。视觉 layers 5/11/17 的特征经各自 merger 后，注入语言 layers 0/1/2。
- LLM、文本 embedding、LM head、采样和 KV 全在 **MNN CPU**。当前语言权重沿用用户提供的 AutoRound/GPTQ **W8/group128**，head 为浮点；没有重新量化，不得宣称整包 W4 或完整复现源 AutoRound 的激活量化。
- 最新完整包：[双模式 v2](https://www.modelscope.cn/models/fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2)。它额外保留完整 MNN ViT，供在线 NPU/CPU 使用；离线模式不加载这些 MNN 视觉图。
- App 内部模式标识和模型协议继续使用 **`offline_npu2` / `kirin_offline_npu2_v1`**，不要因为界面改名而更改它们。旧持久化 `offline` 选择迁移到主方案；native 的旧 `offline` 分块分支仍保留为备选，不显示按钮。

复用已验证的单图视觉运行包和 MNN CPU 导出生成完整模型目录：

```bash
python3 transformers/llm/export/kirin_hybrid_offline/package_runtime.py \
  --cpu /path/to/mnn_gptq_export \
  --full-npu /path/to/validated_fusedvision_runtime \
  --output /path/to/new_offline_npu_package --with-mnn-vision
```

该打包步骤不重新量化、不执行 OMG；重建冻结视觉图的编译流程见下述 Full NPU 文档的第 4 节，需使用 `--fuse-vision` 与外置权重。不要用旧六段 `build_kirin_offline.py` 代替当前单图构建。
CPU 参考验证不能代替手机 OMC/HiAI 验证。用户已反馈本模式可正常运行，但回答质量、性能和长期稳定性仍需评估，不能将主方案切换描述为精度或速度问题已全部解决。

### 另行保留：全模型 NPU 实验路径（2026-10-17）

涉及 Qwen3-VL/mobi 的 **全图 NPU**、外置 W4 权重、FP16 KV、CPU 量化对照、
App 加载或性能排查时，先阅读
**[Kirin9030 全图 NPU 构建与验证](docs/inference/full-npu-kirin9030.md)**。
该文档包含单图视觉版本共 6 张 OMC（旧分图版为 13 张）的构建命令、图权重分离与 InitWeights 流程、量化问题及 CPU
校验方法。当前用户真机反馈精度阶段性对齐，**性能仍然非常非常慢，尚需继续优化**；
不能将编译或精度通过写成性能问题已解决。它使用 `full_npu` 标识，不是当前“离线NPU”按钮的执行路径。

## 备选：旧六段视觉离线图（保留复现流程）

**以下构建、量化和配置仅用于旧视觉分块方案。它不是默认离线方案，App 已移除对应按钮，但保留 native `offline` 和 `OfflineNpuChunkExecutor` 实现。新模型优先采用上面的单图外置权重方案。**

面向 **Kirin9030 (V311)**。目标产物：6 个视觉 chunk 的离线 NPU 图（`om/visual_blocks_npu_<i>.om`）。

> 历次实测数据、失败排查与结论见 **`docs/inference/visual-npu-notes.md`**。本文只讲怎么做。

### 核心规则（先读这条）

1. **`--target=omc` 才是离线图**。`--target=om` 生成的是设备侧在线编译的 HiAI IR 容器。
   本项目沿用 `.om` 扩展名（兼容 App 的 chunk 解析），须用 OMG 日志确认来自 `omc`。
2. **FP16（不加 `--compress_conf`）可作为浮点基线；ViT W4A16/group64 已获用户真机初测可用反馈**。
   W4A16 发布包使用配套 UINT4/LUT 权重与 `compress_conf`，详见下方及 notes §13。
   Kirin9030 上原 W8A8 `compress_conf` 的图文无关问题仍未修复，不能把 W4A16 的结果推广到 W8A8。
3. **Kirin9030 必须加载 AscendC 环境**。只设 `--platform` 不加载 AscendC，MatMul 会被拒并回退 CPU；
   即使产出文件也不算成功。
4. **FP16 必须直接从原始 HF 浮点权重导出**，不能先加载 DOPT `fake_quant_weight.pth` 再转 FP16
   （后者是伪量化/反量化后的权重，会改变数值）。
5. **`visual_blocks_om_deepstack_dup` 必须配对**，否则真机闪退（见下「配置深坑」）。

### W4A16 已发布配置（2026-10-08 更新）

完整模型包：[`fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_w4a16_g64`](https://www.modelscope.cn/models/fengerhu1/mnn_mobi_2B_visual6_npu_kirin9030_w4a16_g64)。用户在手机运行后反馈能区分不同图片，暂按图文功能可用记录。此次没有提供数值输出或运行日志，不能宣称 NPU 数值与 CPU 参考对齐，也不能宣称 W4A16 的量化精度优于 W8A8。

该版本仅将六张 ViT block 离线图改为 `Quant_act_weight_eco`、W4/group64、signed INT16 输入及逐通道 INT16 输出；LLM 和视觉前后处理保持原样。序列化必须使用 SDK 的 `quant_param_2=True`，并按配套 `fake_quant_weight.pth` 中的 UINT4/LUT 索引重新导出 ONNX；不能只将新参数文件配到旧浮点权重 ONNX。备选构建入口为 [`build_kirin_offline.py`](transformers/llm/export/plugin_quant_visual_matmul_route_v1/build_kirin_offline.py)，参数见其 `--help` 和本节历史流程。单段导出器保留旧序列化默认值，原生编码需要显式 `--quant_param_2`。发布包的隔离构建、电脑端误差与验证边界见 [`visual-npu-notes.md` §13](docs/inference/visual-npu-notes.md#13-vit-w4a16group64-真机初测可用2026-10-08)。

此为历史模型与旧入口的复现记录。现在仅保留 native `offline` 分支供备选调试，不再提供旧按钮；当前“离线NPU”按钮要求单图混合模型包，不能直接加载上述六段包。旧分支仍须保留 `visual_blocks_om_deepstack_dup=[2]`。当时每段使用 192 份真实路径 A 输入校准，没有运行完整 CUDA GPTQ/QAT 三段式权重优化。

### 环境准备

环境路径以仓库根目录的 `ENV.md` 为准。**不要把 `ENV.md` 中的令牌或密码**输出到日志、提交或命令行参数。

```bash
export DDK_PATH=/temp/huawei-sdk/DDK-tools-next-6.1.1.0

# conda 环境名是大写 CANN（小写 cann 会报 EnvironmentNameNotFound）
source /opt/conda/etc/profile.d/conda.sh && conda activate CANN
```

一次性初始化（新机器解压 DDK 后各做一次）：

```bash
# 1) omg 解压后可能缺执行位
chmod +x "$DDK_PATH"/tools/tools_omg/omg "$DDK_PATH"/tools/tools_omg/master/omg

# 2) AscendC 环境（注意在 bash -c 里 source：zsh 下 $0 检测会失效）
cd "$DDK_PATH/tools/tools_ascendc" && bash -c 'source install.sh'

# 3) 若 AscendC 的 python package 缺适配器，用 DDK 自带 wheel 离线装，勿从公网拉
python -m pip install --no-index --no-deps \
  --target "$DDK_PATH/tools/tools_ascendc/package/python" \
  "$DDK_PATH/tools/tools_ascendc/package/ascendc_adapter-0.1-py3-none-any.whl"
```

自检：`source set_ascendc_env.sh` 后能 `python -c 'import te_fusion'` 才算就绪。

### 步骤 0：编译 host 工具（x86）

`MNNConvert` 供 `llmexport.py` 把 ONNX 转成 MNN；`llm_demo` 供 host 侧验证与（路径 A 的）校准 dump。

```bash
mkdir -p /temp/work/build-host-converter && cd /temp/work/build-host-converter
cmake <repo_root> \
  -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON \
  -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON
make -j$(nproc) MNNConvert llm_demo
```

> 需要路径 A（引擎 dump 校准）时额外加 `-DMNN_VISUAL_CHUNK_INPUT_DUMP=ON`；
> 仅做 FP16 离线 OM 时不需要（FP16 路线不使用校准）。

### 步骤 1：导出 chunk ONNX

以下是原有 FP16 与 W8A8 导出入口。W4A16 发布包使用上述配套 LUT 重导流程，不能仅修改以下命令的位宽来复现：

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
export DDK_DOPT=$DDK_PATH/tools/tools_dopt/dopt_pytorch_py3
export PYTHONPATH=$DDK_DOPT:$(git rev-parse --show-toplevel)/transformers/llm/export:$PYTHONPATH

# A) FP16（推荐；从原始 HF 浮点权重导出，不需校准）
python visual_plugin_quant_matmul_route.py \
  --route_dir /temp/work/omc/chunk0 --chunk_index 0 --npu_chunks 6 \
  --sequence_length 608 --hidden_size 1024 --rotary_size 64 --fp16 \
  export-fp16

# B) W8A8（先 `all` 做 DOPT 量化 + 校准；Kirin9030 上产物数值有问题，见 notes）
python visual_plugin_quant_matmul_route.py \
  --route_dir /temp/work/omc/chunk0 --chunk_index 0 --npu_chunks 6 \
  --quant_strategy Quant_aigc_ptq --weight_bit 8 --weight_algo min_max \
  --act_bit 8 --input_algo min_max \
  --use_qwen3_style_rotary --input_dir <calib_npz_dir> --num_samples 16 --force_regen all
```

校准 npz 的生成见本文末「校准数据」一节。

### 步骤 2：OMG 编译

包装脚本：`transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh`

```bash
# FP16（推荐）
PLATFORM=kirin9030 \
TARGET_MODEL_TYPE=omc \
USE_COMPRESS_CONF=false \
LOAD_ASCENDC_ENV=auto \
OMG_TOOL=$DDK_PATH/tools/tools_omg/omg \
OMG_MASTER_DIR=$DDK_PATH/tools/tools_omg/master \
ASCENDC_ENV_SCRIPT=$DDK_PATH/tools/tools_ascendc/set_ascendc_env.sh \
bash transformers/llm/export/plugin_quant_visual_matmul_route_v1/run_visual_plugin_matmul_omc.sh \
  /temp/work/omc/chunk0 fp16
```

产物：`<route_dir>/omc_output/visual_plugin_matmul_quantized.omc`（FP16 约 **101.8 MB/chunk**）。

> `run_*.sh` 里的 `PLATFORM=kirin9020` 与 `SRC_IMG_DIR` 等路径均已过时（本机无 9020 插件、
> 旧 `/temp/fdh`、`/temp/csm` 不存在），**不要直接跑这些脚本**；按本节命令手动执行。

### 步骤 3：成功判据

日志须同时包含（只看文件存在不够）：

```text
partition type NPU:1, CPU:0
MemoryCalculateForGraph
SaveCompiledModelToFile SUCCESS
OMG generate offline model success
```

再确认：

1. 无 `CPU fallback`、分区非 `NPU:0, CPU:1`、MatMul 未全 unsupported；
2. FP16 路径的 `can't find key [shape]` 警告数应为 **0**；本次 W8A8 路径出现过 96 条，但其影响尚未确定，不能仅据此判定参数消费错误。

### 步骤 4：导出 MNN 模型并组装交付目录

```bash
cd transformers/llm/export
python llmexport.py \
  --path /temp/models/mobi0402_2B_halfimage_rl \
  --export mnn \
  --gptq_path /temp/models/autoround_export/mobi0402_2B_halfimage_rl-w8g128/ \
  --visual_gptq_path /temp/models/autoround_export/mobi0402_2B_halfimage_rl-w8g128/ \
  --quant_bit 8 --quant_block 128 --visual_quant_bit 8 --visual_quant_block 128 \
  --lm_quant_bit 16 --seperate_embed \
  --visual_split --visual_npu_chunks 6 \
  --visual_chunk_backends "npu,npu,npu,npu,npu,npu" \
  --mnnconvert <build>/MNNConvert \
  --dst_path /temp/work/mnn_export
```

组装运行时包（**只放运行必需文件**，不要带 `onnx/`、`*.mnn.json`、`export_args.json`、编译日志）：

```
config.json                     llm.mnn / llm.mnn.weight
llm_config.json                 visual_pre.mnn  / .weight
tokenizer.mtok                  visual_post.mnn / .weight
embeddings_bf16.bin             visual_blocks_npu_0..5.mnn / .weight
om/visual_blocks_npu_0..5.om    ← 6 个 OMC 产物，按此命名（App 约定）
```

`config.json` 关键字段：

```json
"visual_split": true,
"visual_blocks_backend_type": "hiai",
"visual_blocks_chunks": ["visual_blocks_npu_0.mnn", "...", "visual_blocks_npu_5.mnn"],
"visual_blocks_chunk_backends": ["npu","npu","npu","npu","npu","npu"],
"npu_model_dir": "om",
"visual_blocks_offline_om": ["om/visual_blocks_npu_0.om", "...", "om/visual_blocks_npu_5.om"],
"visual_blocks_om_deepstack_dup": [2]
```

发布前逐文件核对路径、大小与 SHA-256，并确认远端无孤立中间文件。

### 配置深坑：`visual_blocks_om_deepstack_dup`

**漏配会导致真机闪退。** 某 chunk 的 deepstack 层若**恰是该 chunk 的最后一层**，其 deepstack
输出与 `hidden_states` 是同一张量；ONNX 用 `Identity` 命名，而 OMG 会删掉这个冗余节点
（日志 `the node Identity dont have output tensor`），于是该 chunk 的 `.om` 只返回 1 个输出。
`visual_post` 因此少收一路输入 → `MNN_ASSERT` 在 release 下是空宏拦不住 → 越界崩溃。

本模型 `deepstack_visual_indexes=[5,11,17]`，6 chunk × 4 层 → **只有 chunk2 需要**：

| chunk | blocks | deepstack | 合并？ |
|---|---|---|---|
| 0 | [0..3] | — | |
| 1 | [4..7] | 5 | 否 |
| **2** | **[8..11]** | **11 = 末层** | **★ 需要 dup** |
| 3 | [12..15] | — | |
| 4 | [16..19] | 17 | 否 |
| 5 | [20..23] | — | |

**判定方法**：遍历各 chunk ONNX 的 `graph.output`，若 deepstack 输出来自 `Identity`
且其输入 == `hidden_states` 输出名，则该 chunk 需要 dup。注意取值只与 chunk 划分和
deepstack 索引有关，**与校准来源无关**。

> host 上把 `visual_blocks_chunk_backends` 全置 `cpu` 时走 MNN module 路径，会保留 chunk2 的
> 2 个输出，因此 **host 测不出这个问题**，只有真机 OM 路径才会暴露。

### 其他易踩点

- **`llm_config.json` 必须随模型一起拷贝**。缺它会报 `tensor [deepstack_embeds] is input but not found`
  / `Create module error`，看起来像 deepstack 不匹配，实际只是缺文件。
- **不要用 symlink 建测试目录后 `json.dump`**：会穿透改写源文件；测试 config 要用真实文件。
- **`visual_blocks_om_paths` 引擎不读**（只存在于文档）。OM 路径由 app 侧 `setNpuChunkExecutor`
  注入，app 按 `visual_blocks_offline_om` / `npu_model_dir` 解析
  （见 `mobiinfra-oh/entry/src/main/cpp/napi_init.cpp`）。
- **host 验证时**把 `visual_blocks_chunk_backends` 全置 `cpu` 才能跑通视觉前向（host 无 hiai NPU）。
- **只重编 OM、未改引擎代码时，不需要重新编译/替换 `libMNN.so`**；引擎有改动时才重编，
  并在复制到 App 前后校验哈希。
- 输入 shape 是编译期固定的。若改变图片预处理得到的 token 数或 hidden size，
  **必须按新 shape 重编**，不能复用旧 OMC。

### 引擎侧防御（已实现）

针对上面两个真机问题加的守卫，`omni.cpp` / `PipelineModule.cpp`：

- 调 `visual_post` 前校验 `blocksOut.size() == postInfo->inputNames.size()`，不匹配则
  `MNN_ERROR` 并返回空；
- `outputs` 为空时提前返回，避免 `outputs[0]` 越界；
- `PipelineModule::onForward` 在 `mInputSize != inputs.size()` 时报错返回，并给 submodule 的
  stack 索引加范围检查（原先只有 `MNN_ASSERT`，release 下是空操作）。

**这些守卫不修复模型问题，只把「崩溃」变成「明确报错」。** 注意 `PipelineModule.cpp` 静态链入
`libMNN.so`（App 侧无独立 Express 动态库），所以守卫需重编 `libMNN.so` 才生效。

## 校准数据

两条路径，产出的 npz **格式完全一致**，可互换。

| | 路径 A：引擎 dump | 路径 B：纯 Python 前向（推荐） |
|---|---|---|
| 依赖 | `llm_demo`（需 `MNN_VISUAL_CHUNK_INPUT_DUMP=ON`）+ **6chunk MNN 模型目录** | 仅 HF fp 模型 |
| 工具 | `select_images.py` → `llm_demo` → `bin_to_chunk_npz.py` | `generate_npz_calib.py` |
| 现状 | 6chunk 模型目录是外部依赖，**已不存在** | 换机器后最省事 |

### 路径 B（推荐）

```bash
cd transformers/llm/export/plugin_quant_visual_matmul_route_v1
python3 generate_npz_calib.py \
  --num_samples 16 \
  --hw 600,270 \
  --output_dir /temp/work/calib_npz
```

- `--model_path` 默认 `$MOBI_HF_MODEL` 或 `/temp/models/mobi0402_2B_halfimage_rl`（外部依赖）。
- 产物：`6 chunk × N` 个 fp16 npz，`seq_len=608`。
  `--num_samples 16` → 96 个 ≈144 MB；`256` → 1536 个 ≈3.1 GB。
- 每样本 shape：`hidden_states_in (1,608,1024)` / `rotary_pos_emb (2,608,1,64)` / `attention_mask (1,608,608)`。

**必须传 `--hw 600,270`**：导出端只用 `samples[0]` 固定 shape 且无 `dynamic_axes`，
而 `calib_images/` 的 496 张图自然尺寸不一，不加 `--hw` 会得到混合 `seq_len` 而失败。
脚本会在开始前只读图片头预检，尺寸不一致直接报错。

`--hw 600,270` 的语义与 prompt 的 `<hw>600,270</hw>` 等价（引擎 `qwen2VisionProcess` 先覆盖再
`smartResize`）：

```
600,270 -> smart_resize(factor=32) -> 608x256 -> grid 38x16 -> seq_len = 608
```

### 修改校准前向后必须做数值交叉验证

> ⚠️ **"格式一致" ≠ "数值等价"**。历史上只对比 shape/dtype 就认定路径 A/B 等价，结果漏掉了
> `visual_pre.mnn` 的 `patch_embed + pos_embeds`（`generate_npz_calib.py` 曾只算 `patch_embed`），
> 使 chunk0 输入被低估约 5.7 倍（absmax 5.75 vs 真值 32.62），且**不报错**、OMC 照常编译成功。

验证锚点（同图，hw=600,270）：

| 指标 | 期望值 |
|---|---|
| chunk0 `hidden_states_in` absmax | ≈ **32.6**（bug 版≈5.7） |
| chunk1 `hidden_states_in` absmax 均值 | ≈ **17.3**（对齐设备端日志 `hidden absmax=17.4068`） |

切换路径或改任一路径的前向后，务必比对这两项，不要只比 shape。

### 临时工作目录

所有中间产物放 `/temp/work/`（见 `ENV.md`），**不要写进仓库**：

```
/temp/work/build-host-converter/   # MNNConvert + llm_demo 的 x86 构建
/temp/work/calib_npz/              # 校准 npz
/temp/work/omc/chunk<i>/           # 各 chunk 的 ONNX + OMC 产物
/temp/work/mnn_export/             # llmexport.py 输出的 MNN 模型
```

`calib_dump_raw/`、`calib_npz/` 已在 `plugin_quant_visual_matmul_route_v1/.gitignore` 中忽略。

### 路径敏感配置（换目录后需重新生成）

`run_real_calib_256.sh`（路径 A）生成的两个存档文件含绝对路径：`config_dump_608.json` 的
`visual_chunk_input_dump_dir`、`image_prompt.txt` 每行的图片路径。仓库内副本的作用只是记录
复现参数（`--num 256 --hw 600,270 --seed 42`），实际流程会自行生成到 `/temp/work/`。
换目录后按脚本头部参数重新生成即可，**不要手改这两个文件**。

其中 `visual_chunk_input_dump_dir` 与 prompt 图片路径按**进程 CWD** 解析（`omni.cpp`），
而 `llm_model` / `visual_pre_model` 等按 **config 文件所在目录** 解析（`llmconfig.hpp`）——
解析基准不同，所以这两处必须写绝对路径。

##环境配置
查看 @ENV.md 文件

当前主要离线NPU方案（单图外置权重 ViT + MNN CPU LLM；内部标识仍为 `offline_npu2`）的实现、打包及主机精度验证见 @transformers/llm/export/kirin_hybrid_offline/README.md 。不能把 CPU 参考验证描述为 OMC 真机验证。
