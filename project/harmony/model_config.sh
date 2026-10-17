#!/usr/bin/env bash
# 本文件中的 MOBI_* 是构建脚本使用的环境变量，不是 C/C++ 宏或厂商 SDK 参数。
# 默认 Mobi 模型会在本地缺失时从 ModelScope 下载；也可填写已有模型目录。
# 输入文件的统一存放目录；下面的具体路径也可以分别修改。
export MOBI_MODEL_INPUT_ROOT="${MOBI_MODEL_INPUT_ROOT:-$HOME/mobi-models}"
# 原始 Hugging Face 模型目录。
export MOBI_HF_MODEL="${MOBI_HF_MODEL:-$MOBI_MODEL_INPUT_ROOT/mobi0402_2B_halfimage_rl}"
# 与原始模型配套的 GPTQ 模型目录；当前支持 W8/group128。
export MOBI_GPTQ_MODEL="${MOBI_GPTQ_MODEL:-$MOBI_MODEL_INPUT_ROOT/mobi0402_2B_halfimage_rl-w8g128}"
# 默认模型目录对应的 ModelScope 仓库。手动指定其他本地目录时，不自动替换模型。
# 自定义远程模型时可显式设置对应的 MODEL_ID；显式设为空则仅使用本地文件。
if [[ "$MOBI_HF_MODEL" == "$MOBI_MODEL_INPUT_ROOT/mobi0402_2B_halfimage_rl" ]]; then
    export MOBI_HF_MODEL_ID="${MOBI_HF_MODEL_ID-YYDH21/mobi0402_2B_halfimage_rl}"
else
    export MOBI_HF_MODEL_ID="${MOBI_HF_MODEL_ID-}"
fi
if [[ "$MOBI_GPTQ_MODEL" == "$MOBI_MODEL_INPUT_ROOT/mobi0402_2B_halfimage_rl-w8g128" ]]; then
    export MOBI_GPTQ_MODEL_ID="${MOBI_GPTQ_MODEL_ID-YYDH21/mobi0402_2B_halfimage_rl-w8g128}"
else
    export MOBI_GPTQ_MODEL_ID="${MOBI_GPTQ_MODEL_ID-}"
fi
# 下载版本；默认首次拉取 master，此后复用完整本地文件，不自动更新已下载权重。
export MOBI_HF_MODEL_REVISION="${MOBI_HF_MODEL_REVISION:-master}"
export MOBI_GPTQ_MODEL_REVISION="${MOBI_GPTQ_MODEL_REVISION:-master}"
# 用于 NPU 视觉量化校准的图片目录。
export MOBI_CALIB_IMAGES="${MOBI_CALIB_IMAGES:-$MOBI_MODEL_INPUT_ROOT/calibration/images}"
# 独立验证图片，不能与校准图片重复。
export MOBI_VALIDATION_IMAGE="${MOBI_VALIDATION_IMAGE:-$MOBI_MODEL_INPUT_ROOT/validation/example.png}"
# 可选文本校准数据，JSON 格式为 [{"text": "提示词"}]；留空使用内置提示词。
export MOBI_CALIB_TEXTS="${MOBI_CALIB_TEXTS:-}"

# 工作目录：存放中间产物和日志，必须与输入模型、校准数据和 SDK 目录分开。
export MOBI_MODEL_WORK="${MOBI_MODEL_WORK:-$HOME/mobi-model-build/mobi2b}"
# 主机版 MNN 转换工具的构建目录。
export MOBI_HOST_BUILD="${MOBI_HOST_BUILD:-$MOBI_MODEL_WORK/host_tools}"
# MNNConvert 转换工具的可执行文件；默认由脚本编译，也可指定现有文件。
export MOBI_MNNCONVERT="${MOBI_MNNCONVERT:-$MOBI_HOST_BUILD/MNNConvert}"
# MNN 模型导出的中间目录。
export MOBI_MNN_EXPORT="${MOBI_MNN_EXPORT:-$MOBI_MODEL_WORK/mnn}"
# 离线视觉图的导出、验证和编译目录。
export MOBI_VISION_BUILD="${MOBI_VISION_BUILD:-$MOBI_MODEL_WORK/vision_build}"
# 最终交付模型包目录，App 使用该目录中的 config.json。
export MOBI_MODEL_PACKAGE="${MOBI_MODEL_PACKAGE:-$MOBI_MODEL_WORK/package}"

# 以下是执行环境和量化参数，可由 Agent 根据本机环境配置。
# CPU 导出、验证、离线编译和打包使用的 Python 可执行文件。
# 需安装 export/requirements.txt 所需依赖，以及 ONNX Runtime；DOPT 从 DDK 加载。
export MOBI_MODEL_PYTHON="${MOBI_MODEL_PYTHON:-$MOBI_PYTHON}"
# 新 GPTQ 校准使用的 Python，需安装支持 CUDA 的 Torch 及相应导出依赖。
# 可以与上面的 Python 相同；这是构建主机环境，不是手机端环境。
export MOBI_QUANT_PYTHON="${MOBI_QUANT_PYTHON:-$MOBI_MODEL_PYTHON}"
# 校准使用的 GPU，默认第一张 CUDA 设备。
export MOBI_QUANT_DEVICE="${MOBI_QUANT_DEVICE:-cuda:0}"
# NPU 视觉权重的优化方法，默认 GPTQ；也支持显式选择 RTN。
export MOBI_VISION_WEIGHT_METHOD="${MOBI_VISION_WEIGHT_METHOD:-gptq}"
# 校准样本总数；默认 64，交替使用 32 个图片样本和 32 个文本样本。
export MOBI_CALIB_SAMPLES="${MOBI_CALIB_SAMPLES:-64}"
# 每个校准样本执行的 decode 步数。
export MOBI_CALIB_DECODE_STEPS="${MOBI_CALIB_DECODE_STEPS:-4}"
# CPU 模型处理线程数。
export MOBI_MODEL_THREADS="${MOBI_MODEL_THREADS:-8}"
# OMG 编译临时目录；路径仅可包含英文字母、数字、下划线和 /。
export MOBI_COMPILER_WORK_DIR="${MOBI_COMPILER_WORK_DIR:-/tmp}"

# 可选的已有产物复用，默认留空，脚本从指定输入重新构建。
# 冻结 W4 构建目录：须与原始模型同源，并含 manifest、量化配置和 calibrated.pth。
# 复用它可以跳过新的 GPU 校准；已发布的 App 运行包不包含完整校准状态。
export MOBI_FROZEN_W4="${MOBI_FROZEN_W4:-}"
# 完整 MNN 导出目录，须包含审计用的 .mnn.json；不能使用删掉审计文件的运行包。
export MOBI_REUSE_MNN="${MOBI_REUSE_MNN:-}"
