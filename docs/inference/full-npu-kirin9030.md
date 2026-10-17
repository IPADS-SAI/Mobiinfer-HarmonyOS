# Kirin9030 全图 NPU：构建、图权重分离、量化与验证

更新日期：2026-10-17。适用模型：Qwen3-VL 架构的 `mobi0402_2B_halfimage_rl`，目标设备 Kirin9030；不代表其他芯片或任意 Qwen 模型已适配。

**当前取得的是精度方面的阶段性成功：用户在最新真机运行后确认“精度是对齐了”。目前只是精度还可以看，性能仍然非常非常慢，性能问题尚未解决，后续必须继续优化。** 本次反馈没有附新的完整数值报告或性能统计，不能据此补写逐层误差、吞吐率或完整 benchmark 通过结论。

新模型：[fengerhu1/mobi_2B_full_npu_kirin9030_w4a16_g64_kv16_fusedvision_v1](https://www.modelscope.cn/models/fengerhu1/mobi_2B_full_npu_kirin9030_w4a16_g64_kv16_fusedvision_v1)。24 个文件、2,782,486,376 字节已校验远端大小和 SHA256；旧模型仓库保留。

新增单图视觉候选 `fusedvision_v1`：视觉部分由 8 张合成 `vision_full` 一张，整个模型由 13 张变为 **6 张 OMC**。它沿用 `stablemath_v1` 的五张语言图、冻结 W4/group64 GPTQ 和 FP16 KV，仍采用外置权重。包格式为 `kirin_full_npu_runtime_v3`，需要本轮更新的 App 源码。

三张未参与校准的图片上，合并前后 CPU ONNX 的图像 embedding 和三路 DeepStack 输出全部逐元素相同（relative L2 和最大绝对误差均为 0）。104 个视觉 W4 矩阵及对应编码、浮点前段、其余五图的身份均核对；单张视觉 OMC 已通过纯 NPU 编译。视觉第 5、11、17 层 **block 后** 的特征分别经过各自 merger，按顺序注入语言第 0、1、2 层。不能把视觉层号用作语言注入层号。

本候选未执行真机 OMC；以上证明合并保留 CPU 参考结果，不证明端侧精度、峰值内存或加速效果。原有解码阶段反复 `InitWeights` 的性能问题仍需优化。精度阶段性成功的用户反馈对应此前的 `stablemath_v1`：

- 模型：[fengerhu1/mobi_2B_full_npu_kirin9030_w4a16_g64_kv16_stablemath_v1](https://www.modelscope.cn/models/fengerhu1/mobi_2B_full_npu_kirin9030_w4a16_g64_kv16_stablemath_v1)。发布时 37 个文件、2,785,113,131 字节，已校验远端大小与 SHA256。
- 导出与运行代码：[kirin_llm_offline](../../transformers/llm/export/kirin_llm_offline/README.md)。模型采用 `kirin_full_npu_runtime_v2` 包格式，需配套 App 的 **全图 NPU** 模式。
- App 是相邻的 `mobiinfra-oh` 仓库；本机位置 `/home/ma-user/workspace/feh/mobiinfra-oh`。同步的 portable runtime 位于 `entry/src/main/cpp/full_npu/`。
- 最新反馈补充了真机精度阶段性可用的结论；下文已有审计文件保存的是反馈前的结果，保留其中的失败项和 `device_tested=false`，不改写历史证据。

## 1. 全图 NPU 实际执行什么

这里“全图”指神经网络部分的 embedding、视觉网络、语言网络和 LM head 都执行离线 NPU 图。CPU 仍负责分词、聊天模板、图像解码/缩放、位置编码和 mask 的构造、DeepStack 数据组织、KV 写回及采样。没有将自回归循环整体放进单个 OMC。

单图视觉候选共 6 张物理 OMC：

| 组件 | 图数 | 作用 |
| --- | ---: | --- |
| `embedding_prefill` / `embedding_decode` | 2 | token ID 转 2048 维词嵌入 |
| `vision_full` | 1 | patch、位置、全部 24 层视觉 blocks、最终 merger 与三路 DeepStack merger |
| `llm_prefill_0` / `llm_decode_0` | 2 | 每张覆盖全部 28 层语言主干 |
| `lm_head` | 1 | 最终 norm 和词表投影 |

纯文本执行 `embedding → LLM → head`；带图请求先执行视觉路径，再把视觉特征按协议送入语言路径。prefill 每块 128 tokens，decode 每次 1 token；总上下文上限 2048 tokens，包含模板、图片和生成内容。图片固定为高 608、宽 256，视觉 patch tokens 为 608，合并后占 152 个语言 tokens。改变 shape 或容量必须重导并重编图。

`stablemath_v1` 的视觉部分为 `vision_pre + vision_blocks_0..5 + vision_post`，共八图；本轮将其全部合并。此前的七段语言图和八片 embedding 方案也不是当前结构。不要混用不同包的配置、权重或 OMC。

## 2. 图与权重已经分离

**当前执行的是 OMC 离线图，图和权重已经分离。** 关键不是扩展名，而是实际的 OMG 编译开关和设备初始化流程：

```text
ONNX + 配对的量化参数
  → OMG --target=omc --save_weights_as_external_data=true
  → external.omc + SubGraph_0.weight
  → 运行包中的 om/<graph>.omc + weights/<group>/SubGraph_0.weight
```

构建侧由 [external_weights.py](../../transformers/llm/export/kirin_llm_offline/external_weights.py) 实现，打包侧由 [package_runtime.py](../../transformers/llm/export/kirin_llm_offline/package_runtime.py) 建立每张图的外置权重目录映射。模型运行包的结构为：

```text
config.json                         # 图列表、端口、外置权重目录与运行参数
tokenizer.mtok
vision_constants.bin
om/embedding_decode.omc
om/llm_decode_0.omc
om/vision_full.omc
om/...                              # 共 6 张；旧 stablemath_v1 为 13 张
weights/<group>/SubGraph_0.weight    # 单图视觉候选为 5 组唯一外置目录
validation_summary.json
release_manifest.json
SHA256SUMS
```

这里只对文件名与 SHA256 完全相同的权重文件做磁盘去重；embedding 的两个阶段可以使用同一份磁盘权重。**磁盘去重、模型常驻、跨 executor 的设备权重共享是三件事；目前没有证明或实现通用的设备权重共享。** 外置权重仍可能在每次 executor 初始化时重新读取或搬运。

App 的 [NnrtBackend.cpp](../../transformers/llm/export/kirin_llm_offline/runtime/NnrtBackend.cpp) 按以下顺序加载：

1. 检查图与外置权重文件，执行 HiAI 兼容性检查，选择 `HIAI_F`，关闭 CPU fallback。
2. `OH_NNCompilation_Build`，随后 `OH_NNExecutor_Construct`。
3. `HMS_HiAIExecutor_InitWeights(executor, weight_directory)`，传入权重所在目录。
4. 查询实际 tensor 的名称、dtype、shape 和大小，按名称绑定，再允许执行。

加载方式参考了 MindSpore Lite 的图权重分离流程，没有移植其整套 Qwen 管线或 Q4_0 自定义算子。当前仍采用本项目已能编译的华为 SDK W4 编码。

曾遇到首图报 `InitWeights; device API unavailable`：只用 `dlsym(RTLD_DEFAULT, ...)` 查找会漏掉局部加载库中的符号。现先 `dlopen("libhiai_foundation.so", RTLD_LAZY | RTLD_LOCAL)`，从库句柄解析，再尝试全局兼容路径，并保留库和 compilation 的生命周期。显式查找仍失败时应报错，不能跳过权重初始化。旧日志里的“加载成功”也不能替代实际 `exec_ok` 和数值核验。

## 3. 当前量化方案与遇到的问题

### 3.1 W4A16 的准确含义

| 对象 | 当前方案 |
| --- | --- |
| 语言 attention/MLP、视觉 Linear/merger | SDK `Quant_act_weight_eco`，W4/group64，GPTQ 优化 |
| LM head 矩阵 | SDK `Quant_lm_head`，4 bit，独立处理与 embedding 的共享关系 |
| 激活 | SDK signed INT16 输入与逐通道 INT16 输出量化；A16 不表示所有中间值都是 FP16 |
| embedding、patch Conv3D、学习位置表等 | 保留浮点来源；编译权重类型 FP16 |
| norm、attention、residual 等 | 保持相应浮点计算，未全部变为 4 bit |
| KV Cache | FP16，当前不使用 INT8 KV |

301 个压缩矩阵核对了 SDK 的 UINT4/LUT 编码。GPTQ 使用真实输入的 Hessian 做权重误差补偿，保持 SDK 的连续 group64 分组，比较每个矩阵的重建误差，保留优于 RTN 的结果。这不是完整复现官方 KD/QAT 训练流程。

本次冻结权重来自 64 组校准输入：32 张图片、32 条文本输入（文本来自 8 个循环 prompt），每组追加 4 步 decode；质量检查图片与校准图片分离。checkpoint SHA256 为 `4ebd75fcb73613f9a7e9a3835fe9efb05899cb18e24b65b7c89ca6802043325e`。重新校准不保证产生同一哈希；复现当前版本应优先复用已冻结、经校验的构建。

### 3.2 编码、激活范围与浮点稳定性要分别检查

| 遇到的问题 | 排查和处理 | 结论边界 |
| --- | --- | --- |
| RTN 候选回答循环、重复 | 改用 GPTQ，并保留逐矩阵重建损失及真实生成回答 | 矩阵重建改善不等于整模型质量通过 |
| UINT4/LUT 索引与浮点权重混淆 | 使用 SDK `quant_param_2=True`；重导配套索引 ONNX，核对 LUT+scale 还原权重与 MatMul 名称 | 不能只把新参数文件配到旧 ONNX，也不能把索引当浮点权重送给 ORT |
| 完整激活量化比“仅 W4 权重”明显变差 | 冻结 W4 权重，分别比较两条参考；当前输入范围系数为 1，输出范围额外扩大 4 倍 | 4 倍是本模型的诊断选择，不是通用最优参数；扩大范围会改变取整误差 |
| CPU 参考正常，但设备 hidden/logits 幅度严重偏离 | 核对输入、图和权重哈希，检查 FP16 中间值；当前使用稳定 RMSNorm、预缩放 QK、显式稳定 Softmax | 已知风险得到处理；最新用户反馈精度可用，仍不能倒推出此前唯一根因 |
| INT8 KV 增加协议、scale 和保护窗口复杂度 | 当前采用 FP16 KV，保留独立 KV 对照检查 | FP16 KV 不等于与 FP32 KV 逐值相同，旧长上下文失败仍保留 |

`stablemath_v1` 的 [static_kv.py](../../transformers/llm/export/kirin_llm_offline/static_kv.py) 具体处理如下：

- RMSNorm 在平方前按每行 `s=max(1, max(abs(x)))` 缩放，令 `u=x/s`，计算 `u / sqrt(mean(u²) + eps/s/s) * weight`。避免先对大残差平方，也避免直接计算 `s²`；不依赖编译器保留 FP32 中间值。
- QK 点积前分别把 Q、K 乘以 `head_dim**(-0.25)`，两者乘积等价于原来的 `1/sqrt(head_dim)` 缩放，降低未缩放 MatMul 中间结果溢出的风险。
- Softmax 显式减去每行最大值。

冻结的 W4 checkpoint、权重 scale 没有因此重新校准，但**浮点等价不等于有限精度或量化取整逐值等价**。新旧 CPU 首 token logits 的最大 relative L2 为 0.0929。新候选 Torch 与 ONNX 的逐图最大 relative L2 为 0.01555；这两项的比较对象不同，均不能解释成“相对原始浮点模型的量化误差只有多少”。

## 4. 如何导出并编译当前结构

以下命令从本仓库根目录执行，先替换输入路径。模型、DDK、冻结校准构建和 tokenizer 是外部依赖；下载运行包不会得到完整校准 checkpoint 和 CPU 参考 ONNX。大文件放独立工作目录，不放入源码仓库，也不要覆盖已发布构建。

### 4.1 环境和输入

```bash
KIRIN_ROOT="$PWD"
KIRIN_TOOLS="$KIRIN_ROOT/transformers/llm/export/kirin_llm_offline"
KIRIN_MODEL=/temp/models/mobi0402_2B_halfimage_rl
KIRIN_DDK=/temp/huawei-sdk/DDK-tools-next-6.1.1.0
KIRIN_CPU_PY=/home/ma-user/.conda/envs/CANN/bin/python
KIRIN_QUANT_PY=/path/to/cuda_enabled_python
KIRIN_FROZEN_W4=/path/to/frozen_gptq_build
KIRIN_TOKENIZER=/path/to/tokenizer.mtok
KIRIN_CALIB_IMAGES=/path/to/calibration_images
KIRIN_HELDOUT_IMAGE=/path/to/heldout_image.jpg
KIRIN_WORK=/tmp/kirin_full_npu_rebuild
KIRIN_BUILD="$KIRIN_WORK/build"
KIRIN_PACKAGE="$KIRIN_WORK/package"
```

CPU 环境需要项目所用版本的 Torch、Transformers、ONNX、ONNX Runtime、Pillow，以及 DDK 的 DOPT 模块；GPTQ 校准环境另需 CUDA Torch。现有机器使用 DDK `DDK-tools-next-6.1.1.0`。Python 依赖和 DDK 应与原构建兼容，不能只换一个 serializer 或模型库版本后沿用旧审计。

Kirin9030 的 MatMul 编译需要 AscendC 环境。解压后的工具先确认可执行位，按发行包说明运行一次 `tools/tools_ascendc/install.sh`；脚本使用 bash。`external_weights.py` 编译时会自动用 bash 加载 `set_ascendc_env.sh`。可先自检：

```bash
bash -c 'source "$1" && "$2" -c "import te_fusion"' kirin-check \
  "$KIRIN_DDK/tools/tools_ascendc/set_ascendc_env.sh" "$KIRIN_CPU_PY"
```

曾因沙箱禁止编译器的本地进程/Unix socket 通信而出现失败。先确认工具执行环境和权限，再判断平台算子是否支持；不要把这类错误直接当作算子不支持。

### 4.2 有冻结 W4 构建时：直接重建外置图

```bash
"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/external_weights.py" \
  --source-build "$KIRIN_FROZEN_W4" \
  --output "$KIRIN_BUILD" --model "$KIRIN_MODEL" --ddk "$KIRIN_DDK" \
  --layers-per-chunk 28 --kv-dtype fp16 \
  --activation-range-multiplier 1 --output-range-multiplier 4 \
  --fuse-vision \
  --components llm,head,embedding,vision --steps prepare,export,compile
```

该命令读取冻结校准状态，建立新目录，导出当前稳定数学版本并编译全模型。不会重新 GPTQ。`prepare` 校验校准状态及可复用编码文件的哈希；参数 JSON 和可能改写的编码对采用独立副本。若只复用不可变大文件，不要在目标目录原地修改它们。

`stablemath_v1` 实际是在旧外置权重构建上，仅重导 `--components llm,head`，其余十张已核验图直接复用。这样做要求视觉与 embedding 的导出逻辑未改变；若修改了这些组件，应将其加入重建列表。`manifest.json` 的 `static_math` 是构建身份的一部分，不能把新图放回旧 profile 后跳过检查。

导出和编译可拆开运行：首次使用 `--steps prepare,export`，完成后以**同一组参数、同一目录**执行 `--steps compile`。改变 shape、量化范围或构建身份时创建新目录。

若已有本机 `stablemath_v1` 完整构建，只合并视觉部分即可，其余五图逐文件复用：

```bash
"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/external_weights.py" \
  --source-build /tmp/kirin_full_vlm_external_w4_kv16_stablemath_v1 \
  --output /tmp/kirin_full_vlm_external_w4_kv16_fusedvision_v1 \
  --model "$KIRIN_MODEL" --ddk "$KIRIN_DDK" \
  --layers-per-chunk 28 --kv-dtype fp16 \
  --activation-range-multiplier 1 --output-range-multiplier 4 \
  --components vision --fuse-vision --steps prepare,export

"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/validate_fused_vision.py" \
  --source-build /tmp/kirin_full_vlm_external_w4_kv16_stablemath_v1 \
  --build /tmp/kirin_full_vlm_external_w4_kv16_fusedvision_v1 \
  --model "$KIRIN_MODEL" --image "$KIRIN_HELDOUT_IMAGE"
```

验证脚本逐路比较图像特征和 DeepStack，核对冻结编码和未选中组件，默认 relative L2 门槛为 0.01、cosine 不低于 0.9999；本轮三张图片实际误差均为 0。通过后以相同导出参数执行 `--steps compile`，再打包并运行 `verify_app_runtime.py`。`--fuse-vision` 进入 manifest 身份，不能在原分图构建目录直接切换。

### 4.3 没有冻结构建时：先校准

```bash
KIRIN_FROZEN_W4="$KIRIN_WORK/new_frozen_w4"
"$KIRIN_QUANT_PY" -B "$KIRIN_TOOLS/experiment.py" \
  --model "$KIRIN_MODEL" --output "$KIRIN_FROZEN_W4" --ddk "$KIRIN_DDK" \
  --precision w4a16 --weight-method gptq --group-size 64 --head-bit 4 \
  --quant-device cuda:0 --gptq-damping 0.01 --gptq-block-size 128 \
  --context 2048 --prefill-tokens 128 --layers-per-chunk 4 \
  --vision-tokens 608 --vision-layers-per-chunk 4 \
  --calibration-images "$KIRIN_CALIB_IMAGES" --calibration-samples 64 \
  --calibration-decode-steps 4 --image "$KIRIN_HELDOUT_IMAGE" --steps calibrate
```

校准完成后再执行 4.2。这里先校准完整模型，再按 28 层重新组织静态图；量化校准与图分块是两个步骤。自定义文本可传 `--calibration-texts`，格式为 `[{"text": "..."}]`。`--image` 指定的评估图会从新校准集排除。重新校准产生的是新量化候选，应重新检查质量，不能直接继承当前模型的精度反馈。

### 4.4 实际 OMG 参数和成功判据

外置编译器使用以下组合，输入 shape 按各图 manifest 生成：

```text
--framework=5
--target=omc
--platform=kirin9030
--input_shape=<每个输入的固定形状>
--save_weights_as_external_data=true
--weight_data_type=FP16
--compress_conf=<该图配对的 quant_params_file>  # 仅量化图
```

`--weight_data_type=FP16` 不会取消配套 `compress_conf` 的 W4 编码；embedding 等浮点图不使用压缩参数。编译前核对 ONNX、外部 initializer 数据与参数文件哈希，编译后记录 OMC 和所有 `.weight` 的哈希。

每图日志应有 `partition type NPU:1, CPU:0`、`SaveCompiledModelToFile SUCCESS` 和 `OMG generate offline model success`，并通过脚本的完整日志检查。仅出现最终 success 或文件存在不算纯 NPU 编译成功。当前 6 张图（旧分图版 13 张）都要有匹配产物和纯 NPU 记录。

## 5. 如何在 CPU 上检查量化结果

CPU 检查用于定位问题和阻止错误产物流入手机，**CPU 不执行 Kirin OMC**。需要分别保留以下参考：

| 比较 | 能定位的问题 |
| --- | --- |
| 原始 HF 浮点模型 vs 仅 W4 权重、浮点激活 | 权重量化自身的损失 |
| 仅 W4 权重 vs 完整 SDK 激活量化参考 | 激活范围、截断与取整的额外影响 |
| 同一量化 HF 参考 vs 静态 KV/分块 Torch | 结构转换、位置、mask、DeepStack、缓存协议 |
| 静态 Torch vs 反量化 CPU ONNX | ONNX 导出及算子重写 |
| App portable C++ 调度 vs 独立 Python 输入/缓存参考 | 分词、模板、数据布局、KV 增量更新和调度 |
| 真机输出 vs 相同保存输入的 CPU 参考 | 编译器、设备精度和运行时边界；此项必须另做设备验证 |

### 5.1 使用反量化 reference，不直接执行索引图

SDK 的 UINT4/LUT ONNX 是编译输入；其 FLOAT32 initializer 可能只是索引容器。CPU ORT 应读取每图的 `reference/model.onnx`，其中包含还原后的权重和激活量化操作。[ONNXGraphs](../../transformers/llm/export/kirin_llm_offline/experiment.py) 已按 manifest 自动选用参考图。

编码审计核对 `quant_params_file`、`fake_quant_weight.pth`、LUT/scale、每个 MatMul initializer 和算子路径。不能仅通过文件名、位宽字段或模型体积判断量化正确。GPTQ 处理 LM head 时要解除与 embedding 的共享，防止浮点 embedding 被连带改写。

### 5.2 权重与激活的独立审计

```bash
"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/audit_quantization.py" \
  --source-build "$KIRIN_BUILD" --model "$KIRIN_MODEL" --ddk "$KIRIN_DDK" \
  --output "$KIRIN_WORK/cpu_audit" --image "$KIRIN_HELDOUT_IMAGE" \
  --activation-range-multiplier 1 --output-range-multiplier 4 \
  --generation-tokens 32 --threads 8
```

该入口比较原始浮点、仅权重量化、完整激活量化的 logits、teacher-forced NLL 和真实回答，并单独检查 FP16/FP32 KV。图片专项可加 `--image-only`；只比较 logits 可再加 `--logits-only`。评估图片不能用于校准。审计目录必须独立，输出中的门槛失败应保留；执行完脚本不等于所有质量检查通过。

完整结构检查入口是 `experiment.py --validation-backend onnx --steps validate,quality,kv_validate`，需要同时传入与构建一致的 `--precision w4a16 --weight-method gptq --group-size 64 --head-bit 4 --layers-per-chunk 28 --kv-dtype fp16 --output-range-multiplier 4`、源模型、构建目录及 DDK。其他形状/GPTQ 参数也须匹配 manifest。`kv_validate` 应检查跨块 prefill、短/长上下文及各层 K/V，不能只检查最后 logits。

常用指标是 `relative L2 = ||candidate-reference||₂ / ||reference||₂`、cosine、max absolute error、NaN/Inf 数量和 token/NLL。高 cosine 可能掩盖幅度错误；只看 min/max 或“回答看起来相关”也不足以定位误差。量化前后不要求自由生成逐 token 相同，但结构转换检查应固定相同输入或 teacher-forced 前缀，避免把采样分歧继续放大。

FP16/仅权重的严格结构检查保留原有标准；完整 W4A16 或压缩 KV 的参考检查通常使用 relative L2 0.05。这里的比较对象是同一量化参考，**不是承诺相对原始浮点模型误差低于 5%**。不得为了报告通过而临时放宽阈值。

### 5.3 App C++ 的 CPU 回放

先打包，再构建 portable host driver。`KIRIN_HOST_MNN` 必须是已有的 **x86_64 主机** `libMNN.so`，不是手机 arm64 库。

```bash
"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/package_runtime.py" \
  --build "$KIRIN_BUILD" --model "$KIRIN_MODEL" --tokenizer "$KIRIN_TOKENIZER" \
  --output "$KIRIN_PACKAGE"

KIRIN_HOST_MNN=/path/to/x86_64/libMNN.so
KIRIN_HOST_BUILD="$KIRIN_WORK/host"
cmake -S "$KIRIN_TOOLS/runtime" -B "$KIRIN_HOST_BUILD" \
  -DMNN_HOST_LIBRARY="$KIRIN_HOST_MNN" -DKIRIN_RUNTIME_PACKAGE="$KIRIN_PACKAGE"
cmake --build "$KIRIN_HOST_BUILD" -j8
ctest --test-dir "$KIRIN_HOST_BUILD" --output-on-failure

"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/verify_app_runtime.py" \
  --driver "$KIRIN_HOST_BUILD/host_driver" --package "$KIRIN_PACKAGE" \
  --build "$KIRIN_BUILD" --model "$KIRIN_MODEL" --output "$KIRIN_WORK/cpp_replay" \
  --system-prompt 'You are a helpful assistant.' \
  --text-prompt '你是谁？' --text-prompt '8乘以7等于多少？只回答数字。' \
  --text-prompt '把“桌子上有一本蓝色的书。”翻译成英文。' \
  --image "$KIRIN_HELDOUT_IMAGE" --decode-steps 16 --threads 8
```

回放使用实际 portable C++ 调度器，通过 CPU ONNX 代替 NPU 后端，独立检查模板、token ID、M-RoPE、attention mask、DeepStack 和全部 28 层 FP16 KV。`report.json` 含包配置哈希与回答。它能发现 App 输入与缓存错误，不能验证 NNRT 内部搬运、OMC 数值或端侧速度。

回放通过后，将报告附入另一个空的发布目录，保持回放时的图、权重、tokenizer 和运行配置一致：

```bash
KIRIN_RELEASE="$KIRIN_WORK/release"
"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/package_runtime.py" \
  --build "$KIRIN_BUILD" --model "$KIRIN_MODEL" --tokenizer "$KIRIN_TOKENIZER" \
  --app-validation "$KIRIN_WORK/cpp_replay/report.json" --output "$KIRIN_RELEASE"
```

核对报告的 `package_config_sha256` 与最终 `config.json` 哈希相同，再在发布目录执行 `sha256sum -c SHA256SUMS`。所有报告应在封存清单前附入；清单封存后不再原地增删文件。上传后还要逐文件核对远端大小与 SHA256。

另有 Python/SDK 回归测试：

```bash
KIRIN_TEST_DDK="$KIRIN_DDK" PYTHONDONTWRITEBYTECODE=1 \
  "$KIRIN_CPU_PY" -B -m unittest discover -s "$KIRIN_TOOLS" -p 'test_*.py' -v
```

当前保存证据为 35 项 Python/SDK 测试、九组 C++ 测试通过，四个 CPU 回放共 2928 项独立张量检查通过。三条文本回放分别得到相关自我介绍、`56` 和正确的蓝色书本翻译；生成长度有限，不当作完整质量评测。

旧 FP16 KV 长上下文检查在约 1802 个 prompt tokens 时，逐层 KV relative L2 最大约 0.2759，超过 0.05，仍是失败记录。旧 W4 质量检查也有事实、OCR 与细节偏差。最新用户反馈是精度阶段性可用，不能据此把这些旧检查改成通过或宣布全量精度验收完成。

## 6. 如何在 App 中运行

运行已发布模型可跳过导出与编译，但需要匹配的 App 源码：

```bash
"$KIRIN_CPU_PY" -B "$KIRIN_TOOLS/sync_app_runtime.py" --app /path/to/mobiinfra-oh
```

该入口同步 `full_npu/` 及其依赖并生成 `sources.json`；检查新增头文件（例如 `KvTransfer.hpp`）与清单一起交付。不要只拷贝 `NnrtBackend.cpp`。此次 runtime 变更需要在 DevEco 中重新构建并安装 App；仅更新本地源码不会升级手机。此路径的改动不要求替换 `libMNN.so`。

在 App 设置中填写当前 ModelScope 仓库，下载到独立目录，选择模型和 **全图 NPU**。不要把仅 ViT 离线图的“离线 NPU 图”模式当作此模式。自建包应包含 tokenizer、视觉常量、13 张图、所有外置权重及一致的 config/哈希清单。

启用 Debug 时，加载和请求日志应能看到：

```text
resolve_HMS_HiAIExecutor_InitWeights ... result=success
OH_NNExecutor_Construct ... result=success
HMS_HiAIExecutor_InitWeights ret=OH_NN_SUCCESS(0)
graph_ready cpu_fallback=disabled
exec_ok graph=...
run_timing input_copy_ms=... run_sync_ms=... kv_copied_bytes=...
```

同时检查 `cache_hit`、`resident_evict`、`cache_backoff`、`cache_policy_restore`、实际生效预算以及采样 token ID。`graph_ready` 是累计加载事件，可能因重载超过 13，不是常驻图数或执行正确性的证明。本次文档工作没有连接手机。

## 7. KV 和性能现状：仍然非常慢

**当前精度阶段性可用，性能仍然非常非常慢；尚未达到性能优化完成或可流畅交互的结论。** 以下机制是已经实现的优化方向，不能据此声称当前真机已达到目标速度。

FP16 KV 按 `[1,8,2048,128]` 保存，每层 K/V 各一份，28 层共 **224 MiB**。图返回新增 `key_delta/value_delta`，CPU 将有效 token 转为 FP16 并追加到宿主缓存；不是每步重新计算全部历史。NNRT 输入有另一份缓冲：首次或失去同步时完整写入，后续可只写新行，单 token 全模型为 **112 KiB**。这不含设备内部读取、attention 及驱动搬运，也不是 OMC 内部原地维护 KV。

模型包仍携带原来的四图/2048 MiB 配置。2026-10-17 按用户要求，本地 App 源码
在全图 NPU 加载入口显式传入 16 图、4096 MiB 预算，覆盖模型包中的这两项；KV
配置继续从模型读取。**需重新编译并安装 App 后生效，手机现有配置未修改。**
配套运行时提供可选的 `ResidencyBudget` 加载参数；没有传入覆盖参数的 CPU 工具
和其他调用方仍按模型配置执行。当前 App 的有效初始预算为：

```json
{
  "npu_max_resident_graphs": 16,
  "npu_resident_model_mib": 4096,
  "npu_persistent_kv_mib": 256,
  "npu_incremental_kv": true
}
```

加载日志会先记录 `level=cache_budget_override`，随后是
`cache_config max_graphs=16 model_budget_bytes=4294967296`。这些是调度预算，不是设备实际内存测量。理想热身工作集是 embedding、整段 decoder、head 三个 executor；资源紧张时逐图淘汰，优先保留 decoder。只剩两个名额时 embedding/head 仍可能交替重载；一个名额时更明显。KV 池尽量独立保留，格式查询失败则禁止跨 executor 共享，但同一 executor 仍可增量同步。新请求和 prefill→decode 切换恢复预算，避免一次压力永久锁死后续请求。此次预算调整未取消资源失败回退或阶段释放，不代表解决了设备初始化失败。

**修复前的旧版日志**中，“你是谁？”生成 25 tokens 的 decode 为 59.17 秒（约 0.42 token/s），图加载/切换为 54.52 秒，占约 92%，缓存命中为零，25 步累计完整写入约 5.47 GiB KV。该数据来自修复前，**不是当前版本的最新测速**；最新反馈只确认仍很慢，没有提供新吞吐数据。

后续应先从同一请求的完整日志分解并排序瓶颈：

1. **图/权重重载**：分别统计 Build、Construct、InitWeights、淘汰和缓存命中；检查当前实际常驻数量。图已分离不表示 InitWeights 免费。
2. **KV 写入**：确认热身后是 112 KiB 增量还是 224 MiB 全量，并关联 executor 重建、缓存 epoch 和格式查询。宿主少拷贝不保证设备端零拷贝。
3. **真正 RunSync**：分离视觉、prefill、decode 和 head 时间。固定容量 attention、KV 读取、量化内核、图间边界仍可能有显著开销。
4. **其他宿主开销**：分别测量图像处理、模板/位置/mask、输出取回、Debug 的全张量统计与日志。比较相同输入的冷启动、热启动和多 token 稳态，报告首 token 延迟及 decode token/s。

确认瓶颈后再选择常驻策略、权重初始化复用、输入缓冲复用、上下文档位/attention 或内核优化。扩大预算可能再次触发设备分配失败；未确认 SDK 支持前，不能直接承诺跨 executor 权重共享、任意动态 KV 或图内自回归可用。后续性能改动应保留当前精度样例，重新做 CPU 对照与真机检查。

## 8. 源码与证据索引

| 内容 | 入口 |
| --- | --- |
| 构建、W4 和历史实验详细记录 | [kirin_llm_offline/README.md](../../transformers/llm/export/kirin_llm_offline/README.md) |
| W4/GPTQ 与 SDK 编码 | [w4a16.py](../../transformers/llm/export/kirin_llm_offline/w4a16.py)、[gptq.py](../../transformers/llm/export/kirin_llm_offline/gptq.py) |
| 数学、静态 KV 协议 | [static_kv.py](../../transformers/llm/export/kirin_llm_offline/static_kv.py) |
| 图权重分离编译 | [external_weights.py](../../transformers/llm/export/kirin_llm_offline/external_weights.py) |
| 设备加载、常驻与增量传输 | [NnrtBackend.cpp](../../transformers/llm/export/kirin_llm_offline/runtime/NnrtBackend.cpp)、[KvTransfer.hpp](../../transformers/llm/export/kirin_llm_offline/runtime/KvTransfer.hpp) |
| 最新修复及发布记录（用户反馈前） | [results_stablemath_cache_kirin9030.json](../../transformers/llm/export/kirin_llm_offline/results_stablemath_cache_kirin9030.json) |
| 最新编译和 CPU 回放 | [artifact_audit.json](../../transformers/llm/export/kirin_llm_offline/evidence_stablemath_cache/artifact_audit.json)、[cpp_host_validation.json](../../transformers/llm/export/kirin_llm_offline/evidence_stablemath_cache/cpp_host_validation.json)、[cpu_text_canary.json](../../transformers/llm/export/kirin_llm_offline/evidence_stablemath_cache/cpu_text_canary.json) |
| 旧独立量化及 KV 检查 | [quantization_audit.json](../../transformers/llm/export/kirin_llm_offline/evidence_external_w4_kv16/quantization_audit.json)、[kv_validation_torch.json](../../transformers/llm/export/kirin_llm_offline/evidence_external_w4_kv16/kv_validation_torch.json) |

后续维护本文时，分别记录用户反馈、CPU 量化参考、编译检查、真机数值与真机性能；每项注明版本和输入，不用其中一项替代另一项。当前优先事项是**在保持已恢复精度的前提下，继续解决非常慢的性能问题**。
