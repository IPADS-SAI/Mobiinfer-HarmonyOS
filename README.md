# Mobiinfer — 麒麟 NPU 多模态模型推理

本仓库支持在 HarmonyOS / Kirin9030 上运行 Mobi Qwen3-VL。当前主要离线方案为：**ViT 使用一张预编译 OMC，图与权重分离；LLM 使用 MNN CPU 推理。**

## Demo

[![demo](doc/demo.gif)](doc/demo.mp4)

点击上图查看[原始视频](doc/demo.mp4)。App 公共工程见 [mobiinfer-oh](https://github.com/doulujiyao12/mobiinfer-oh)，本机修改目录为 `/home/ma-user/workspace/feh/mobiinfra-oh`。

## 1. 当前离线NPU方案

```text
图片 → CPU 解码、缩放和 patch 排列
     → 单张 vision_full.omc + 外置 SubGraph_0.weight（NPU）
     → 图像 embedding + 三路 DeepStack
     → MNN LLM、LM head、KV 和采样（CPU）
```

| 部分 | 执行方式 |
| --- | --- |
| ViT 前段、全部 24 层 blocks、最终 merger、三路 DeepStack merger | 合并为一张 `vision_full.omc`，由 HiAI NNRT 执行 |
| 视觉权重 | 冻结的 W4A16/group64，单独保存在外置权重文件中 |
| 文本 embedding、28 层 LLM、LM head、采样 | MNN CPU |
| 语言 KV Cache | MNN CPU 原生缓存，不向 NPU 上传语言 KV |

图与权重分离使用 OMG `--save_weights_as_external_data=true`；App 构建 executor 后，通过 `HMS_HiAIExecutor_InitWeights` 初始化外置权重。视觉图约 5.27 MB，外置视觉权重为 251,344,384 bytes。视觉 executor 在后续图片和请求间复用，CPU decode 不加载 NPU 语言图。

视觉输入固定为高 **608**、宽 **256**，patch16、temporal2、merge2，每图产生 **152** 个语言图像 token。视觉 blocks **5/11/17** 的特征分别经各自 merger 投影后，注入语言 layers **0/1/2**。不能把视觉取值层号当作语言注入层号。

当前 CPU 语言权重沿用用户提供的 AutoRound/GPTQ **W8/group128**，LM head 保留浮点权重。当前模型没有重新量化，不是整包 W4。GPTQ 整数权重和 scale 的正确导入，也不代表源 AutoRound 激活量化过程已被完整复现。

## 2. 下载模型与 App 模式

当前完整模型：[mobi_2B_offline_npu2_kirin9030_online_v2](https://www.modelscope.cn/models/fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2)，约 **3.54 GB**。它同时保留完整 MNN ViT，支持以下模式：

| App 按钮 | 视觉 | LLM |
| --- | --- | --- |
| **离线NPU**（主要离线方案） | 单张 OMC + 外置 W4 权重 | MNN W8 CPU |
| 在线 NPU | 完整 MNN ViT；六个 blocks chunk 交给 HiAI 在线编译，pre/post 在 CPU | 同一份 MNN W8 CPU |
| 全 CPU | 完整 MNN ViT 强制 CPU | 同一份 MNN W8 CPU |
| 全图 NPU（实验） | 专用全图模型的离线视觉图 | 专用模型的 NPU 语言图 |

下载整个仓库，保留子目录，加载根目录 `config.json`。内部 `vision/config.json` 仅用于视觉协议，不能作为整个模型入口。

```python
from modelscope import snapshot_download
model_dir = snapshot_download('fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2')
print(model_dir + '/config.json')
```

App 的“离线NPU”由原“离线NPU-2”改名而来，**内部标识继续使用 `offline_npu2`，模型协议继续使用 `kirin_offline_npu2_v1`**，已有单图模型包无需修改。旧分块离线按钮已移除，旧持久化的 `offline` 选择会迁移到主方案；旧 native 分块执行逻辑仍保留。

需要重新编译包含这些修改的 App。只下载模型不会更新已安装 App 的 `libentry.so`；旧版模式检查可能将完整双模式包拒绝为“requires Offline NPU-2”。本次源码调整不直接安装或修改手机 App。

## 3. 准备工具与 MNN CPU 权重

### 主机与设备工具

- 设备运行库使用匹配的 HiAI / CANN Kit；其头文件和 ARM64 库位于 `source/backend/hiai/3rdParty/`。厂商下载入口见[华为 DDK 页面](https://developer.huawei.com/consumer/cn/doc/hiai-Library/ddk-download-0000001053590180)。
- 离线编译使用主机 **DDK-tools-next-6.1.1.0** 的 DOPT、OMG、Kirin9030 平台插件和 AscendC。它与手机端运行库是不同的 SDK 包。
- Harmony App 使用 DevEco Studio 或匹配的 HarmonyOS Command Line Tools 构建，`HARMONY_HOME` 指向 OpenHarmony SDK 根目录。
- 量化工具：[mobi-autoround](https://github.com/doulujiyao12/mobi-autoround)，支持真实图片校准和 GPTQ 格式导出。已有 GPTQ 权重时可以直接复用，不必重新量化。

主机 MNN 工具示例：

```bash
cmake -S . -B /path/to/build-host   -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TOOLS=ON   -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON   -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON   -DMOBIINFER_MNN=OFF
cmake --build /path/to/build-host --target MNNConvert llm_demo -j8
```

### 导入已有 GPTQ 权重

以下命令生成 CPU 语言资产，并保留在线 NPU/CPU 所需的完整 MNN 视觉资产。命令中的六个 MNN chunks 用于在线路径，不是主要离线方案的六张 OMC。

```bash
python transformers/llm/export/llmexport.py   --path /path/to/original_hf_model --export mnn   --gptq_path /path/to/supplied_gptq_model   --visual_gptq_path /path/to/supplied_gptq_model   --quant_bit 8 --quant_block 128   --visual_quant_bit 8 --visual_quant_block 128   --lm_quant_bit 16 --seperate_embed   --visual_split --visual_npu_chunks 6   --visual_chunk_backends npu,npu,npu,npu,npu,npu   --mnnconvert /path/to/build-host/MNNConvert   --dst_path /path/to/mnn_gptq_export
```

位宽和 group size 必须与源 GPTQ 文件相符。上面的 W8/group128 对应当前用户提供的模型；不能只改参数就把已有 W8 权重变成 W4。

## 4. 编译单张 ViT 图与组装模型包

### 单图编译

完整构建和量化背景见 [Kirin9030 全图 NPU 构建与验证](docs/inference/full-npu-kirin9030.md)，主方案复用其中已冻结、已合并的视觉组件，不加载 NPU 语言组件。

已有匹配的冻结 W4 / stablemath 构建时，可使用以下入口合并、导出并编译视觉部分：

```bash
python transformers/llm/export/kirin_llm_offline/external_weights.py   --source-build /path/to/validated_frozen_w4_build   --output /path/to/new_fusedvision_build   --model /path/to/original_hf_model   --ddk /path/to/DDK-tools-next-6.1.1.0   --layers-per-chunk 28 --kv-dtype fp16   --activation-range-multiplier 1 --output-range-multiplier 4   --components vision --fuse-vision --steps prepare,export,compile
```

此命令依赖已保存的校准 checkpoint、量化参数及构建 manifest，不是从原始模型开始的一键重新量化。构建工具在 OMG 阶段使用：

```text
--target=omc
--save_weights_as_external_data=true
--compress_conf=<与该图索引权重配套的参数>
```

必须校验 NPU 分区、编译日志、图与外置权重的配对哈希，并分别核对最终图像 embedding 和三路 DeepStack。外置权重仍会占用设备内存；不能根据 OMC 文件很小就认为没有权重初始化或内存开销。完整视觉合并、CPU 对照及运行包生成步骤见上述构建文档和[实现说明](transformers/llm/export/kirin_hybrid_offline/README.md)。

### 从已验证产物组装主方案模型包

以下是当前已验证的打包入口，输入是 MNN 导出目录和已经验收的单图 Full NPU **运行包**，而非未打包的构建目录：

```bash
python transformers/llm/export/kirin_hybrid_offline/package_runtime.py   --cpu /path/to/mnn_gptq_export   --full-npu /path/to/validated_fusedvision_runtime   --output /path/to/new_offline_npu_package --with-mnn-vision
```

`--with-mnn-vision` 保留在线 NPU/CPU 所需的完整 MNN ViT；不加时生成只支持主要离线路径的精简包。此步骤复制并核验资产，不重新量化，也不执行 OMG。

```text
config.json / llm_config.json
llm.mnn / llm.mnn.weight
embeddings_bf16.bin / tokenizer.mtok
vision/config.json
vision/om/vision_full.omc
vision/weights/<group>/SubGraph_0.weight
vision/vision_constants.bin
visual_pre.mnn / visual_pre.mnn.weight             # 可选完整 MNN ViT
visual_blocks_npu_0..5.mnn / 对应 .mnn.weight      # 可选完整 MNN ViT
visual_post.mnn / visual_post.mnn.weight           # 可选完整 MNN ViT
manifest.json / release_manifest.json / SHA256SUMS # 发布包清单
```

资产来源、两套视觉格式的区别、DeepStack 协议及打包边界见[单图视觉离线 NPU 文档](transformers/llm/export/kirin_hybrid_offline/README.md)。

## 5. App 构建与验证范围

需要重建 MNN 库时，使用仓库提供的 Harmony 脚本：

```bash
export HARMONY_HOME=/path/to/openharmony-sdk
cd project/harmony
mkdir -p build
cd build
../build_64.sh
```

将生成的 `libMNN.so` 同步到 App 的 `entry/libs/arm64-v8a/`；按钮、模式选择和 NNRT 加载桥接代码位于 App 的 `entry/src/main/`，也必须参与 App 重新编译。仅修改按钮或模式路由时不需要重新量化模型、重编 OMC 或重建 MNN 库。

当前已完成的主机检查：

- 196 个语言矩阵及 144 个 MNN 视觉矩阵的整数编码、scale 与源 GPTQ 文件逐字节一致。
- 完整 MNN 路径的单图/双图重打包对照中，prefill logits 最大绝对误差为 0，前三个生成 token 一致。
- 单图离线视觉的 CPU 参考与桥接检查覆盖图像 token、M-RoPE、三路 DeepStack、decode、错误重试及 Agent prefix KV 擦除；prefill logits 与对应 CPU 对照一致。

这些验证不在 CPU 上执行 OMC，也不代替真机在线编译或端到端精度测试。用户已反馈主方案可以正常运行，但回答质量、速度、内存和长期稳定性仍需继续评估；本次文档与按钮调整不代表性能或精度问题已解决。

## 6. 备选路径

### 旧六段视觉离线图

原方案把 ViT blocks 编译为六段离线图，视觉 pre/post 和 LLM 由 MNN 执行。现在保留为备选，不再显示旧“离线 OM”按钮。

- 构建入口：[build_kirin_offline.py](transformers/llm/export/plugin_quant_visual_matmul_route_v1/build_kirin_offline.py)。
- 历史配置、校准、编译与排查：[CLAUDE.md 的旧方案章节](CLAUDE.md#备选旧六段视觉离线图保留复现流程)、[视觉 NPU 记录](docs/inference/visual-npu-notes.md)。
- native `offline` 分支及 `OfflineNpuChunkExecutor` 保留；当前“离线NPU”按钮使用单图外置权重方案。

### 全模型 NPU 实验

全图 NPU 将 embedding、ViT、LLM 和 head 都交给 NPU，使用独立模型包和 `full_npu` 模式，不能与当前 CPU LLM 的主方案混用。单图视觉版本共六张物理 OMC，历史分图版为十三张。

此前取得精度阶段性成功，但 **性能仍然非常非常慢，需要后续优化**。构建、图权重分离、量化问题和 CPU 检验方法见[全图 NPU 文档](docs/inference/full-npu-kirin9030.md)。

## 致谢

感谢 [alibaba/MNN](https://github.com/alibaba/MNN) 提供的基础能力与工程实现。
