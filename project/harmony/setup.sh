#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "${1:---help}" in
    app)
        shift
        source "$script_dir/sdk_common.sh"
        mobi_sdk check-app
        bash "$script_dir/01_prepare_hiai.sh"
        bash "$script_dir/03_build_mnn.sh" "$@"
        bash "$script_dir/04_sync_app.sh"
        ;;
    offline-tools)
        shift
        bash "$script_dir/02_prepare_offline_tools.sh" "$@"
        ;;
    model)
        shift
        bash "$script_dir/build_model.sh" "$@"
        ;;
    -h|--help)
        cat <<'USAGE'
Configure project/harmony/sdk_config.sh first. Requires Bash and Python 3.8+.

  bash project/harmony/setup.sh app [CMake options...]
      Unpack CANN Kit, stage HiAI dependencies, build libMNN.so, sync App libraries/headers.
      Finish building the App/HAP in DevEco Studio. DDK is not needed for this flow.

  bash project/harmony/setup.sh offline-tools [--extract-only]
      Unpack DDK and Kirin9030 plugin, initialize AscendC and verify te_fusion.
      --extract-only skips the vendor Python installer; model compilation is not included.

  bash project/harmony/setup.sh model [--dry-run] [--stages tools,mnn,calibrate,vision,verify,compile,package]
      Download missing default Mobi models from ModelScope, export MNN, build fused W4 ViT, package both NPU modes.
      Configure calibration/validation images in model_config.sh; local model paths can override downloads.

Each numbered script can also be run separately; existing SDKs are reused.
Set MOBI_SDK_CONFIG to use a separate configuration file instead of sdk_config.sh.
USAGE
        ;;
    *)
        printf 'Unknown command: %s (use --help)\n' "$1" >&2
        exit 2
        ;;
esac
