#!/usr/bin/env bash
# 本文件配置构建脚本使用的环境变量，可直接修改默认路径，也可通过同名环境变量覆盖。
# 本仓库路径由脚本自动定位，无需手动填写。
export MOBI_ROOT="${MOBI_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
# 工具包统一解压目录。
export MOBI_SDK_ROOT="${MOBI_SDK_ROOT:-$HOME/mobi-sdk}"
# 已下载的 SDK 压缩包目录。
export MOBI_DOWNLOAD_DIR="${MOBI_DOWNLOAD_DIR:-$MOBI_SDK_ROOT/downloads}"

# SDK Manager 安装的 openharmony 目录，其下应有 native/。
export HARMONY_HOME="${HARMONY_HOME:-$MOBI_SDK_ROOT/harmony-sdk/default/openharmony}"

# CANN Kit 解压目录和 ZIP 路径，用于准备 HiAI 头文件及手机端 ARM64 库。
export CANN_KIT_ROOT="${CANN_KIT_ROOT:-$MOBI_SDK_ROOT/CANN-Kit-next-6.0.1.0}"
export CANN_KIT_ZIP="${CANN_KIT_ZIP:-$MOBI_DOWNLOAD_DIR/CANN-Kit-next-6.0.1.0.zip}"

# DDK 解压目录和 ZIP 路径；仅自行编译离线模型时需要。
# 解压目录不能含空白字符，厂商 AscendC 安装脚本有此限制。
export KIRIN_DDK="${KIRIN_DDK:-$MOBI_SDK_ROOT/DDK-tools-next-6.1.1.0}"
export KIRIN_DDK_ZIP="${KIRIN_DDK_ZIP:-$MOBI_DOWNLOAD_DIR/DDK-tools-next-6.1.1.0.zip}"
# Kirin9030 芯片插件的解压目录和 ZIP 路径。
export KIRIN_PLUGIN_ROOT="${KIRIN_PLUGIN_ROOT:-$MOBI_SDK_ROOT/kirin9030-plugin-next-6.1.1.0}"
export KIRIN_PLUGIN_ZIP="${KIRIN_PLUGIN_ZIP:-$MOBI_DOWNLOAD_DIR/kirin9030-plugin-next-6.1.1.0.zip}"

# 本机的 mobiinfra-oh App 源码目录，使用 coevomind-app 分支。
export MOBI_APP_ROOT="${MOBI_APP_ROOT:-$(dirname "$MOBI_ROOT")/mobiinfra-oh}"
# HarmonyOS ARM64 推理库的构建目录，生成 libMNN.so。
export MOBI_BUILD_DIR="${MOBI_BUILD_DIR:-$MOBI_ROOT/project/harmony/build_oh}"
# C++ 并行编译任务数。
export MOBI_BUILD_JOBS="${MOBI_BUILD_JOBS:-8}"

# 执行配置脚本及初始化 DDK / AscendC 的 Python 可执行文件。
# 自行编译模型时，通常选择与 model_config.sh 中 CPU 导出相同的 Python 环境。
export MOBI_PYTHON="${MOBI_PYTHON:-python3}"
