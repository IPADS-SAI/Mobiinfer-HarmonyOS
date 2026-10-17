# 离线NPU：单图 ViT + CPU LLM

这是当前主要离线方案，App 入口为 **离线NPU**（原名“离线NPU-2”）。为兼容已发布模型包，内部模式和包协议仍使用 `offline_npu2` / `kirin_offline_npu2_v1`。旧 `offline`（视觉分块 OM）仅保留后台实现，按钮已移除；`full_npu` 保留独立实验入口。只修改本机 App 源码，不连接手机。

模型已发布到 [ModelScope 离线NPU](https://www.modelscope.cn/models/fengerhu1/mobi_2B_offline_npu2_kirin9030_vit_w4a16_g64_llm_cpu_v1)。17 个发布文件，共 3,004,653,895 bytes，已逐项核验远端文件大小和 SHA256；发布记录见 `evidence_publication/modelscope_publication.json`。下载完整仓库，在新版 App 选择“离线NPU”并加载根目录 `config.json`。

## 双模式完整包

已发布到 [ModelScope 双模式 v2](https://www.modelscope.cn/models/fengerhu1/mobi_2B_offline_npu2_kirin9030_online_v2)：37 个发布文件，共 3,536,832,614 bytes，已核验远端大小和 SHA256。发布记录见 `evidence_publication/modelscope_dual_publication.json`。

`--with-mnn-vision` 额外保留完整的 MNN `visual_pre`、六个 blocks chunk、`visual_post` 及各自的 `.mnn.weight`，共 16 个视觉文件。它们来自用户提供的 AutoRound/GPTQ W8/group128 导出；pre/post、主 merger 和三个 DeepStack merger 保留原浮点权重。离线视觉仍是 Full NPU 冻结的 W4A16/group64 单图。两种视觉实现各自保留，不将厂商 LUT 文件当作 MNN 权重。

根 `config.json` 同时保留 MNN 视觉配置和 `offline_npu2` marker。更新后的本机 App 根据所选按钮执行：

| App 模式 | 视觉 | LLM |
| --- | --- | --- |
| 离线NPU | `vision_full.omc` + 外置 W4 权重 | 原 MNN W8 CPU |
| 在线NPU | 完整 MNN 视觉，pre/post 在 CPU，六个 blocks 交给 HiAI 在线编译 | 同一份 MNN W8 CPU |
| CPU | 完整 MNN 视觉强制 CPU | 同一份 MNN W8 CPU |

本次不重新量化、不修改激活量化规则，也不宣称两种视觉量化结果逐值相同。新包用 `offline_npu2_mnn_visual=true` 标明包含完整 MNN 视觉；App 仅对该完整包放行在线/CPU 模式，旧离线专用包仍会拒绝。桥接器在离线NPU模式下仍强制关闭 MNN 视觉。另提供不含离线 marker 的 `config_online.json`，便于独立验证普通 MNN 路径；App 常规入口仍使用根配置，需要重编本机 App 源码，不操作手机。

```sh
python3 transformers/llm/export/kirin_hybrid_offline/package_runtime.py \
  --cpu /temp/models/mnn_mobi_2B_w8a8_visual_npu_kirin9030_pathA \
  --full-npu /tmp/kirin_full_npu_external_w4_kv16_fusedvision_v1_release \
  --output /tmp/kirin_offline_npu2_dual_release --with-mnn-vision
```

双模式包有 24 个模型资产，合计 3,536,795,186 bytes，另有配置、说明和验证报告。CPU 语言、MNN 视觉、离线视觉均从对应源包复制并逐文件重新计算 SHA256；原发布包保留。

`audit_gptq_weights.py` 独立解包源 GPTQ 数据，对照 MNN 的 196 个语言矩阵及 144 个视觉矩阵（融合 QKV 在 MNN 中分为三个矩阵），逐项验证整数编码、group128 scale 和源零点约束。`verify_mnn_repack.py` 使用 `mnn_host_driver` 在主机执行真实 MNN 视觉和 CPU LLM，对照重打包前后的单图/双图 prefill logits 和生成 token；该测试不执行 HiAI，也不代表与 Full NPU 精度相同。`verify_host.py` 则继续检查离线视觉参考与 CPU 桥接的单图/双图、DeepStack、M-RoPE 和重试流程。

本次 340 个矩阵的整数编码与 scale 全部逐字节一致。完整 MNN 路径的单图/双图分别为 184/355 tokens，重打包前后的 prefill logits 最大绝对误差均为 0，前三个生成 token 完全一致。离线桥接的单图/双图为 167/321 tokens，独立参考的 prefill logits 最大绝对误差也均为 0。两套路径沿用各自的预处理尺寸，不能据此宣称 MNN W8 与离线 W4 的视觉数值相同。报告见 `validation/dual_*.json`。

## 执行边界

| 部分 | 执行方式 |
| --- | --- |
| 图像解码、RGB resize、patch 排列 | CPU，沿用 Full NPU 的预处理实现 |
| ViT、最终 merger、三路 DeepStack merger | HiAI NNRT，单张 `vision_full.omc` |
| 视觉权重 | 原样复用冻结的 W4A16 / GPTQ group64 外置权重 |
| 文本 embedding、LLM prefill/decode、head、采样 | MNN CPU |
| 语言 KV | MNN 原生 CPU KV，不使用 Full NPU 的固定容量 KV 输入协议 |

视觉图约 **5.27 MB**，外置视觉权重 **251,344,384 bytes**。通过现有 NNRT 后端的 `HMS_HiAIExecutor_InitWeights` 加载权重；不会因为图很小就声称权重无需设备内存。新模式只保留这一个视觉 executor，CPU 解码不触碰 NNRT，不加载 NPU embedding/decoder/head 图。视觉图在后续图片和请求之间保留，直到模型卸载。

CPU 语言部分加载 `llm.mnn` 和其外置 `llm.mnn.weight`，不使用 NPU 语言图。当前本地验证包来自现有 `mnn_mobi_2B_w8a8_visual_npu_kirin9030_pathA`，其 CPU 权重原样保留，**没有转换成 CPU W4，也不宣称整个包都是 W4**。本次没有重新量化任何模型。KV 的精度/布局由 MNN CPU 运行时决定，没有强制 INT8。

## 接口和精度要点

仅支持匹配的 Mobi Qwen3-VL 2B CPU 模型：28 层、hidden 2048、有 M-RoPE 和三路 DeepStack。当前视觉固定网格为 608×256，patch16、temporal2、merge2，每图产生 152 个图像 token；不从旧包的动态尺寸或 420 输入配置推导 NPU 网格。

- `vision_full` 接收 patches、position_indices/weights、视觉 RoPE cos/sin 和 attention_mask，输出图像 embedding 和投影后的三路 DeepStack。
- 视觉 blocks **5/11/17** 的 DeepStack 依次注入语言 layers **0/1/2**，不是语言 layers 5/11/17。prefill 中只在图像 token 的位置注入，文本位置和 decode 均填零。
- CPU LLM 接收 `[3, sequence, 2048]` 的 DeepStack。图像 embedding 只替换图像 token 的文本 lookup 结果。
- M-RoPE 使用合并后的 19×8 网格；后续 decode 从压缩后的位置继续。Agent 擦除到固定 prefix 的 KV 后，恢复该 prefix 对应的位置，避免上一张截图的位置偏移污染下一轮。
- 接受 `<img>文件路径</img>`，兼容 `<hw>` 在路径前/后的旧 App 标签；图片按已编译网格处理。视频、音频和内存图片 API 不在此版本范围内。
- 关闭当前混合模式的文本 prompt cache，避免它二次分词或切分图像 token 跨度。普通多轮聊天完整重新 prefill；Agent 固定 prefix 仍可复用 CPU KV。
- 错误直接返回，不降级到旧的 CPU/分块视觉路径。取消之后、视觉失败后均可重新请求。

## 生成独立模型包

在仓库根目录运行；两个输入目录不变，输出必须是新目录：

```sh
python3 transformers/llm/export/kirin_hybrid_offline/package_runtime.py \
  --cpu /temp/models/mnn_mobi_2B_w8a8_visual_npu_kirin9030_pathA \
  --full-npu /tmp/kirin_full_npu_external_w4_kv16_fusedvision_v1_release \
  --output /tmp/kirin_offline_npu2_release
```

本次已生成上述输出目录，8 个模型资产合计 **3,004,631,370 bytes**，另有 JSON 配置/manifest。每个资产记录 SHA256，复制后重新校验；CPU 与 Full NPU 源包的 tokenizer 也必须相同。它包含：

```text
config.json                      # 普通 MNN 配置 + offline_npu2 marker
llm_config.json
llm.mnn / llm.mnn.weight
embeddings_bf16.bin
tokenizer.mtok
vision/config.json               # 单独的视觉协议，不调用 FullNpuRuntime.load
vision/vision_constants.bin
vision/om/vision_full.omc
vision/weights/.../SubGraph_0.weight
manifest.json
```

不能直接选择原 Full NPU 包或旧视觉分块包的 `config.json`。App 需选择这个独立混合包的根配置。此脚本只在本机生成目录，不上传 ModelScope，不下载/安装 App。

## App 集成

本机源码目录 `/home/ma-user/workspace/feh/mobiinfra-oh` 已增加：

- 设置按钮、模式持久化和切换、native 类型声明。
- 独立 `offline_npu2` native 加载分支；加载、Chat、Generate、Agent、Profile 均有接入。
- `entry/src/main/cpp/offline_npu2/`：桥接源码与私有配置头文件，CMake 编入 `entry`。
- Profile 根据混合包 marker 选择新模式，并保留 CPU 线程预设。

App 通过现有 `Llm` 公共/受保护接口实现独立派生类，直接加载 CPU 语言模型，没有进入原 `Omni` 的视觉分块加载。原 `OfflineNpuChunkExecutor`、`full_npu` 源码及现有 `libMNN.so` 哈希不变。私有实现使用隐藏符号，避免覆盖 libMNN 内部实现。

后续同步桥接源码：

```sh
python3 transformers/llm/export/kirin_hybrid_offline/sync_app.py \
  --app /home/ma-user/workspace/feh/mobiinfra-oh --write
# 不带 --write 仅验证副本和 manifest。
```

App 需要在 DevEco Studio 重新构建。当前服务器没有完整 HarmonyOS native/ArkTS SDK，因此**未生成或验证 HAP**；没有执行 hdc、没有连接或更改手机。

## 主机验证及范围

```sh
cmake -S transformers/llm/export/kirin_hybrid_offline/runtime \
  -B /tmp/kirin_offline_npu2_host \
  -DMNN_HOST_LIBRARY=/temp/work/build-host-converter/libMNN.so \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/kirin_offline_npu2_host -j4
ctest --test-dir /tmp/kirin_offline_npu2_host --output-on-failure
/tmp/kirin_offline_npu2_host/test_protocol \
  /tmp/kirin_offline_npu2_release/vision/config.json \
  /tmp/kirin_offline_npu2_verify/source.png
```

`verify_host.py` 用真实桥接 C++、真实 MNN CPU 语言模型和 CPU ONNX 视觉参考验证。独立 HF 参考检查图像 token 与 M-RoPE，Python 独立构造 embedding/DeepStack，再直接调用未改动的 MNN CPU 模块作为 logits 对照。先执行仅 prefill 保存 logits，再生成 token，避免误把下一步 decode logits 当作 prefill。

```sh
/home/ma-user/.conda/envs/CANN/bin/python \
  transformers/llm/export/kirin_hybrid_offline/verify_host.py \
  --driver /tmp/kirin_offline_npu2_host/host_driver \
  --package /tmp/kirin_offline_npu2_release \
  --build /tmp/kirin_full_vlm_external_w4_kv16_fusedvision_v1 \
  --model /temp/models/mobi0402_2B_halfimage_rl \
  --image /tmp/kirin_offline_npu2_verify/source.png \
  --output /tmp/kirin_offline_npu2_verify
```

已验证的单图输入 167 tokens，其中 152 个图像 token：视觉位置常量和 RoPE 一致；图像 patch 相比 Pillow 最大误差一个 RGB 灰度级；CPU embedding 完全一致；prefill logits 最大绝对误差 **0**，首 token 一致。连续 decode、文本后续请求、失败后重试、Agent prefix KV 擦除后再次推理通过。资源生命周期替身测试确认仅 probe 一次、重复图片不释放 executor，NNRT persistent KV 为零。App 副本 C++ 通过本机语法检查，原 Full NPU 的 9 项主机回归测试通过。

双图用同一张无损图片重复，合计 321 tokens / 304 个图像 token，独立 HF 位置编码和 CPU prefill logits 同样精确对齐；跨请求八次视觉调用仅 probe 一次。报告保存在本目录 `validation/host_single.json`、`validation/host_multi.json`。复现双图可加 `--copies 2`；`--vision-reference` 可复用已验证单图的视觉输出，复用前逐项核对 build manifest 哈希、六个输入的全部字节和视觉输出哈希。

逐像素预处理测试使用 PNG。JPEG 的 Pillow/stb 解码存在原有差异，本次样例的 patch 最大差异约 0.02353；没有为其放宽 PNG 验收阈值，也没有修改既有 Full NPU 预处理。主机验证使用视觉参考输出，不运行 OMC；它证明桥接和 CPU 调度数值对齐，不能证明手机上的 OMC 数值、端到端回答质量或性能已验收。

目标是去掉此前 decode 反复初始化 NPU embedding/decoder/head 权重的耗时。CPU 推理速度、内存压力、持续运行温度和真机加速比例仍待后续实测，当前不承诺 token/s。
