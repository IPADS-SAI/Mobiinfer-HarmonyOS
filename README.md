# Mobiinfer-HarmonyOS

**基于鸿蒙系统与麒麟 NPU 的多模态大模型推理框架。**

[![Platform](https://img.shields.io/badge/Platform-HarmonyOS-blue)](https://developer.huawei.com/consumer/cn/)
[![NPU](https://img.shields.io/badge/NPU-Kirin-orange)](#推理方案)
[![Model](https://img.shields.io/badge/Model-Qwen3--VL-green)](#模型下载)
[![App](https://img.shields.io/badge/App-ClawMate-blue)](#app-体验)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE.txt)

[项目介绍](#项目介绍) · [推理方案](#推理方案) · [App 体验](#app-体验) · [模型下载](#模型下载) · [环境配置](#环境配置) · [编译与集成](#编译与集成) · [构建模型](#构建模型) · [文档](#文档)

## 项目介绍

Mobiinfer-HarmonyOS 基于 [MNN](https://github.com/alibaba/MNN) 构建，面向 HarmonyOS 设备提供多模态大模型的端侧推理能力，支持文本对话、图像理解和图文交互。

项目通过 HiAI / CANN Kit 调用麒麟 NPU，提供**在线 NPU 编译**和**离线 NPU 编译**两种部署方式。目前主要采用 **ViT 在 NPU、LLM 在 CPU** 的混合推理架构，并提供模型导出、量化、离线图编译及 HarmonyOS App 集成工具。

- **两种 NPU 编译方式**：在线版本在设备上编译视觉图；离线版本加载为目标芯片预编译的视觉图。
- **单图视觉推理**：离线版本将完整 ViT、最终 merger 和三路 DeepStack merger 合并为一张图。
- **图与权重分离**：离线视觉图与 W4A16 权重独立存储，通过 HiAI NNRT 加载并复用视觉 executor。
- **CPU 语言推理**：基于 MNN 执行 LLM、文本 embedding、LM head、KV Cache 和采样，支持导入 GPTQ 格式权重。
- **完整应用示例**：配套 ClawMate App，可切换在线、离线 NPU 模式并体验本地多模态对话。

### 演示

[![ClawMate 多模态推理演示](doc/demo.gif)](doc/demo.mp4)

点击上图查看[演示视频](doc/demo.mp4)。

## 推理方案

“在线”和“离线”指 **NPU 图的编译时机**。两种方式均在设备端执行模型推理。

| 对比项 | 在线 NPU 编译版本 | 离线 NPU 编译版本 |
| --- | --- | --- |
| 编译时机 | 设备运行时由 HiAI 编译 | 开发主机上提前编译 |
| 模型与设备的关系 | 分发 MNN 模型，无需为每种芯片预编译 | 分发与目标芯片及运行环境匹配的离线图 |
| 视觉执行方式 | ViT blocks 分为六段，在 NPU 执行；视觉 pre/post 在 CPU | 完整 ViT、最终 merger 和三路 DeepStack merger 在一张 NPU 图内执行 |
| 视觉模型文件 | MNN 图及配套权重 | `vision_full.omc` + 外置 `SubGraph_0.weight` |
| 当前发布包的视觉量化 | blocks 使用 W8/group128；pre/post 保留浮点计算 | W4A16/group64 |
| 语言模型 | MNN CPU，主干权重 W8/group128，LM head 保留浮点 | 与在线版本共用同一份 MNN CPU 语言模型 |

### 在线 NPU 编译版本（设备无关）

在线版本以 MNN 格式发布模型，运行时通过本项目的 HiAI 后端为设备编译视觉 blocks，并使用编译缓存减少后续开销。

```text
图片 → CPU 视觉 pre → NPU 视觉 blocks → CPU 视觉 post / DeepStack → CPU LLM → 输出
```

这里的“设备无关”指**模型包无需绑定特定芯片提前编译**。运行设备仍需支持相应的 HarmonyOS、HiAI 接口和算子；首次编译耗时及兼容性取决于设备环境。

### 离线 NPU 编译版本（设备相关）

离线版本使用 DDK-tools 和目标芯片插件，在主机上将视觉模型编译为 **一张 OMC 图**。当前发布模型面向 **Kirin9030**，采用图与权重分离的加载方式：

```text
vision_full.omc + 外置 W4A16 权重
                    ↓
图片 → CPU 图像预处理 → NPU 完整 ViT / merger / DeepStack → CPU LLM → 输出
```

编译时使用 `--target=omc --save_weights_as_external_data=true`；运行时通过 `HMS_HiAIExecutor_InitWeights` 初始化外置权重。视觉 executor 可跨图片和请求复用，语言 KV Cache 由 MNN CPU 管理。

离线图需要匹配目标芯片、编译工具链与设备运行库。更换目标平台时，应重新编译并验证；图与权重文件必须配套使用。完整流程见[单图视觉离线 NPU 文档](transformers/llm/export/kirin_hybrid_offline/README.md)。

## App 体验

### 直接下载

在**华为应用市场**搜索 **ClawMate**，下载后即可体验，无需自行编译应用。

1. 打开 **我的 → 模型与智能体**。
2. 选择 **在线 NPU** 或 **离线 NPU**，选择并下载对应模型。
3. 返回聊天界面，选择 **本地模型**，输入问题或添加图片开始对话。

离线模式请选择与手机芯片匹配的模型。下方完整模型包同时包含在线和离线路径需要的资产。

### 从源码构建

App 源码位于 [mobiinfra-oh 的 coevomind-app 分支](https://github.com/doulujiyao12/mobiinfra-oh/tree/coevomind-app)：

```bash
git clone --branch coevomind-app --single-branch \
  https://github.com/doulujiyao12/mobiinfra-oh.git
```

使用 DevEco Studio 打开工程，按下方[编译与集成](#编译与集成)准备原生库后，编译并安装应用。

## 模型下载

**Mobi 模型采用 Qwen3-VL 架构，由视觉编码器 ViT 和语言模型 LLM 组成。** 当前提供 Mobi 2B 的原始权重、GPTQ 格式量化权重和端侧运行包：

| 模型 | ModelScope | 用途 |
| --- | --- | --- |
| Mobi 2B 原始模型 | [YYDH21/mobi0402_2B_halfimage_rl](https://www.modelscope.cn/models/YYDH21/mobi0402_2B_halfimage_rl) | Hugging Face 格式原始权重，用于导出、量化及精度对照 |
| Mobi 2B GPTQ 模型 | [YYDH21/mobi0402_2B_halfimage_rl-w8g128](https://www.modelscope.cn/models/YYDH21/mobi0402_2B_halfimage_rl-w8g128) | AutoRound 量化、GPTQ 格式保存的 W8/group128 权重，用于 MNN 导入 |
| Mobi 2B NPU 完整运行包 | [fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2](https://www.modelscope.cn/models/fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2) | 包含完整 MNN ViT、单图离线 ViT 及 CPU LLM，支持在线 NPU 和 Kirin9030 离线 NPU，约 3.54 GB |

运行包中，离线 ViT 使用 W4A16，CPU LLM 沿用上述 GPTQ 模型的 W8 主干权重。各部分采用不同精度，具体布局见[模型包说明](transformers/llm/export/kirin_hybrid_offline/README.md)。

也可通过 ModelScope SDK 下载完整运行包：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2"
)
print(f"{model_dir}/config.json")
```

请保留完整目录结构，使用**根目录的 `config.json`** 作为模型入口。原始模型和 GPTQ 源模型需要先转换为端侧运行格式。

## 环境配置

### 需要准备的工具

**编译推理库与 App、编译离线模型是两个独立流程。** 使用已发布的模型包时，只需准备前两项；自行生成 OMC 时再准备 DDK-tools 和芯片插件。

| 工具包 | 用途 | 获取方式 |
| --- | --- | --- |
| **DevEco Studio 配套 HarmonyOS SDK** | `openharmony/native/` 提供交叉编译工具链与 NNRT；`hms/native/` 提供 App 使用的 CANN Kit 系统接口 | [华为开发者网站](https://developer.huawei.com/consumer/cn/)，通过 SDK Manager 安装 |
| **CANN Kit 独立包** | `ddk/ai_ddk_lib/include/` 和 `lib64/` 提供 MNN 后端所需的 HiAI 头文件及 ARM64 库 | [CANN Kit / HiAI 下载页](https://developer.huawei.com/consumer/cn/doc/hiai-Library/ddk-download-0000001053590180) |
| **DDK-tools**（可选） | OMG、DOPT、AscendC：用于离线模型量化与编译 | [环境准备指引](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations) |
| **kirin9030-plugin**（可选） | 提供 Kirin9030 平台配置、算子及编译支持 | 在同一[环境准备指引](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations)中查找 |

下面的脚本在 **Linux 主机**上执行，HarmonyOS 交叉编译工具链也应使用 Linux 版本；App 可在 Windows / macOS 的 DevEco Studio 中构建。

### 只配置一个文件

编辑 **[project/harmony/sdk_config.sh](project/harmony/sdk_config.sh)**，填写本机的工具包和 App 工程路径。配置项含义见文件内注释，主要需要确认：

| 配置内容 | 指向哪里 |
| --- | --- |
| 工具存放目录 | 工具解压的统一目录，默认 `~/mobi-sdk` |
| 压缩包目录 | 已下载的 ZIP 所在目录，默认 `~/mobi-sdk/downloads` |
| HarmonyOS SDK 路径 | SDK Manager 安装的 **`openharmony` 目录**，对应配置文件中的 `HARMONY_HOME`；其下应存在 `native/build/cmake/ohos.toolchain.cmake` |
| App 工程路径 | 上文克隆的 `mobiinfra-oh` / `coevomind-app` 工程，默认与本仓库同级 |

默认识别的压缩包名称为：

- `CANN-Kit-next-6.0.1.0.zip`
- `DDK-tools-next-6.1.1.0.zip`（仅自行编译离线模型时需要）
- `kirin9030-plugin-next-6.1.1.0.zip`（仅自行编译离线模型时需要）

将下载的文件放入压缩包目录即可。文件名或安装位置不同时，在配置文件中修改对应路径。DDK 安装目录请勿包含空格，这是厂商安装脚本的要求。

脚本自动定位本仓库并复用已有的完整 SDK；切换 SDK 版本时应选择新的解压目录。其他高级配置可按脚本注释调整，也可交由 Agent 完成。

SDK 工具包需自行下载。找不到可用的本地安装或压缩包时，脚本会明确报错，不会自动下载 SDK。

### 各工具最后放在哪里

```text
~/mobi-sdk/
├── downloads/                          # 手动下载的 ZIP
├── harmony-sdk/default/                # 示例 SDK Manager 安装位置
│   ├── openharmony/                    # HARMONY_HOME 指向这里
│   │   └── native/build/cmake/ohos.toolchain.cmake
│   └── hms/native/sysroot/usr/include/CANNKit/
├── CANN-Kit-next-6.0.1.0/               # 脚本解压的 CANN Kit
│   └── ddk/ai_ddk_lib/
│       ├── include/                    # MNN 使用的 HiAI 头文件
│       └── lib64/                      # 手机端 ARM64 库
├── DDK-tools-next-6.1.1.0/              # 脚本解压的 DDK-tools
│   └── tools/
│       ├── tools_omg/
│       ├── tools_dopt/
│       ├── tools_ascendc/
│       └── platform/kirin9030/         # 脚本自动安装芯片插件到这里
└── kirin9030-plugin-next-6.1.1.0/       # 插件解压目录
```

HarmonyOS SDK 的实际安装位置以 SDK Manager 为准。例如，Windows 上 SDK 位于 `D:/DevEco/sdk/default/` 时，对应的 `HARMONY_HOME` 为 `D:/DevEco/sdk/default/openharmony`。在 Linux 构建推理库时，请配置本机可用的 Linux SDK 路径。当前构建需要 API 20 的 SDK。

## 编译与集成

> **Tips**：下载好上述工具包和模型后，后续的编译、模型权重导出等步骤可以交由 Agent 完成。手动配置较为繁琐，可让 Agent 根据本机环境配置并执行下述脚本。

### 一键准备推理库与 App 依赖

完成配置、下载 CANN Kit，并克隆上文的 App 工程后，在本仓库根目录执行：

```bash
bash project/harmony/setup.sh app
```

它依次完成：**解压 CANN Kit → 复制 HiAI 头文件和 ARM64 库 → 编译 `libMNN.so` → 同步 App 所需的库与公共头文件**。这条流程使用已发布的模型，不需要 DDK-tools 和芯片插件。

完成后，使用 DevEco Studio 打开配置好的 App 工程，编译并安装 HAP。App 的 CMake 会生成并打包 **`libentry.so`**，无需额外手动复制。

### 分步运行

各步骤也可以独立执行，便于复用已有 SDK 或已编译的推理库：

| 脚本 | 自动完成的工作 | 何时执行 |
| --- | --- | --- |
| [01_prepare_hiai.sh](project/harmony/01_prepare_hiai.sh) | 解压 CANN Kit；将头文件复制到 `source/backend/hiai/3rdParty/include/`，将 ARM64 库复制到 `3rdParty/arm64-v8a/` | 首次准备或切换 CANN Kit 后 |
| [03_build_mnn.sh](project/harmony/03_build_mnn.sh) | 检查工具链与 HiAI 依赖；编译 ARM64 `libMNN.so` | 准备依赖后，或修改推理引擎后 |
| [04_sync_app.sh](project/harmony/04_sync_app.sh) | 将 `libMNN.so`、HiAI 库复制到 App 的 `entry/libs/arm64-v8a/`，同步 MNN / LLM 公共头文件 | 推理库编译完成后 |

```bash
bash project/harmony/01_prepare_hiai.sh
bash project/harmony/03_build_mnn.sh
bash project/harmony/04_sync_app.sh
```

推理库默认生成在 `project/harmony/build_oh/libMNN.so`。已有编译好的 ARM64 库时，在配置文件中指定其构建目录，直接运行同步脚本即可。

如果推理库在 Linux 编译、App 在 Windows 编译，可先同步到 Linux 上的 App 源码副本，再把更新后的工程传到 Windows 构建。

App 编译还会从 DevEco 配套 SDK 解析系统接口：

| 系统接口 | 来源 |
| --- | --- |
| `neural_network_runtime/neural_network_core.h`、`libneural_network_core.so` | HarmonyOS SDK 的 `openharmony/native/` |
| `CANNKit/hiai_helper.h`、`CANNKit/hiai_options.h`、`libhiai_foundation.so` | HarmonyOS SDK 的 `hms/native/` |

这里的 `CANNKit/` 系统头文件用于 App 桥接；独立 CANN Kit 包中的 `ddk/ai_ddk_lib/include/` 用于 MNN HiAI 后端。系统库使用 SDK 提供的链接库，手机系统提供运行实现。

### 一键安装离线模型编译工具（可选）

需要自行生成 `vision_full.omc` 和外置权重时，下载 DDK-tools 与芯片插件，在 Linux 主机上配置好工具包路径后执行：

```bash
bash project/harmony/setup.sh offline-tools
```

脚本会解压 DDK 和芯片插件，完成工具安装与初始化。重复执行时会检查并复用已有环境。

如果暂时只需要解压和放置工具，可以执行：

```bash
bash project/harmony/02_prepare_offline_tools.sh --extract-only
```

工具安装完成后，按下面的“构建模型”流程生成模型包。量化与编译原理可参阅[全图 NPU 构建与验证](docs/inference/full-npu-kirin9030.md#4-如何导出并编译当前结构)。

## 构建模型

这一流程从用户指定的**原始模型、GPTQ 权重和校准数据**生成 App 可加载的完整模型目录。它与前面的 `setup.sh app`（编译 C++ 推理库）是两个入口。

当前脚本适配 **Mobi Qwen3-VL 2B**，GPTQ 输入使用上文提供的 **W8/group128** 模型。

```text
原始 HF 模型 + GPTQ 权重 ──→ MNN CPU LLM + 完整 MNN 视觉资产
原始 HF 模型 + 校准图片 ────→ ViT W4A16 校准 → 单张 OMC + 外置权重
                                      ↓
                       CPU 参考检验 + GPTQ 导入核验
                                      ↓
                         同时支持在线 / 离线 NPU 的模型包
```

CPU LLM 和在线视觉导入现有 GPTQ 权重；离线视觉从原始浮点模型重新进行 **W4/group64 GPTQ 权重优化及 A16 校准**。新的视觉校准只量化 ViT 和各 merger，CPU 语言权重仍取自提供的 GPTQ 文件。

### 1. 准备模型构建环境

在 Linux 主机上完成上述工具安装，再准备模型导出环境。**重新进行 GPTQ 量化校准需要支持 CUDA 的 GPU**；复用已有的冻结校准结果时，无需重新执行 GPU 校准。

具体环境配置可交由 Agent 根据本机情况完成，相关设置及说明保留在配置脚本中。

### 2. 设置模型与校准数据

**未指定本地模型目录时，脚本默认从 ModelScope 下载对应的 Mobi 模型：**

- 原始模型：[YYDH21/mobi0402_2B_halfimage_rl](https://www.modelscope.cn/models/YYDH21/mobi0402_2B_halfimage_rl)。
- GPTQ 权重：[YYDH21/mobi0402_2B_halfimage_rl-w8g128](https://www.modelscope.cn/models/YYDH21/mobi0402_2B_halfimage_rl-w8g128)。

默认下载到 `~/mobi-models/`，已有完整本地文件时直接复用。下载中断后，重新运行脚本会继续下载。

编辑 **[project/harmony/model_config.sh](project/harmony/model_config.sh)**，提供校准图片和独立验证图片；已有本地模型时，也可在其中指定模型目录。手动指定的本地目录无效时会报错，不会自动替换成默认模型。工具包路径在前面的 `sdk_config.sh` 中配置。

| 输入 | 内容 |
| --- | --- |
| 原始模型 | 默认自动下载，也可指定已有的完整模型目录 |
| GPTQ 权重 | 默认自动下载，也可指定与原始模型对应的本地 GPTQ 模型目录 |
| 校准图片 | 用于 NPU 视觉量化的图片目录，支持 JPG、PNG、WebP |
| 验证图片 | 一张独立图片，用于检查导出精度，不应出现在校准集中 |

校准图片应覆盖模型的实际使用场景。已有 GPTQ 权重直接导入，所提供的校准图片用于重新构建离线 NPU 的 W4 视觉模型。

### 3. 执行模型构建脚本

先检查输入并预览实际命令：

```bash
bash project/harmony/build_model.sh --dry-run
```

然后执行完整流程：

```bash
bash project/harmony/build_model.sh
```

脚本依次完成 **MNN 模型导出 → 视觉 W4A16 量化校准 → CPU 精度检查 → 单图 OMC 与外置权重编译 → 完整模型打包**，同时保留在线 NPU 所需的视觉模型。

重复执行时会校验并复用已完成的步骤，日志保存在工作目录的 `logs/` 下。更换模型、校准数据或构建设置时，请在配置文件中选择新的工作目录。

### 4. 使用生成的模型

默认结果位于 **`~/mobi-model-build/mobi2b/package/`**，也可在配置文件中调整输出位置。目录同时包含：

```text
package/
├── config.json / llm_config.json
├── llm.mnn / llm.mnn.weight                 # GPTQ 导入的 CPU LLM
├── tokenizer.mtok / embeddings_bf16.bin
├── visual_pre.mnn / visual_blocks_npu_*.mnn / visual_post.mnn
├── 对应的 MNN 外置 .weight 文件              # 在线 NPU / CPU 视觉
├── vision/
│   ├── config.json / vision_constants.bin
│   ├── om/vision_full.omc                    # 离线 NPU 单图
│   └── weights/.../SubGraph_0.weight
├── validation/                              # GPTQ 导入审计与 CPU 视觉报告
├── build_provenance.json                     # 输入、校准数据与参数的来源记录
└── release_manifest.json / SHA256SUMS
```

保留整个目录，在配套 App 中加载根 `config.json`。在线 NPU 使用完整 MNN 视觉资产，离线 NPU 使用单张 OMC 与外置权重，两者共用 CPU LLM。

构建时的 CPU 精度检查用于发现导出和量化问题；模型的回答质量和端侧性能仍需在目标 Kirin9030 设备上验证。

## 文档

| 文档 | 内容 |
| --- | --- |
| [单图视觉离线 NPU](transformers/llm/export/kirin_hybrid_offline/README.md) | 当前主要离线方案、DeepStack 对接、模型打包与验证 |
| [全图 NPU 构建与验证](docs/inference/full-npu-kirin9030.md) | 图与权重分离、W4 量化、CPU 精度检验及全模型 NPU 实验 |
| [视觉 NPU 技术记录](docs/inference/visual-npu-notes.md) | 视觉分块方案、在线编译及历史问题排查 |
| [MNN LLM](transformers/README.md) | MNN 大模型导出与推理工具 |

当前主要方案为 **NPU ViT + CPU LLM**，精度、性能及兼容性仍在持续验证与优化。另行保留的**全模型 NPU 实验方案**已取得精度阶段性进展，但推理速度仍然很慢，需要后续优化，详见对应技术文档。

## 许可证

本项目使用 [Apache License 2.0](LICENSE.txt)。模型和厂商 SDK 请遵循各自的许可条款。

## 致谢

- [MNN](https://github.com/alibaba/MNN)：提供端侧推理引擎、模型转换及多模态 LLM 基础能力。
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)：提供多模态模型架构与开源生态。
- 华为 HarmonyOS、HiAI / CANN Kit：提供麒麟 NPU 的开发工具与运行接口。
