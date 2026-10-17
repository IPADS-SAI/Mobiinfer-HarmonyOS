#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/sdk_common.sh"
model_config="${MOBI_MODEL_CONFIG:-$MOBI_SCRIPT_DIR/model_config.sh}"
if [[ ! -f "$model_config" ]]; then
    printf 'Missing model configuration: %s\n' "$model_config" >&2
    exit 1
fi
source "$model_config"
"$MOBI_PYTHON" -B "$MOBI_SCRIPT_DIR/model_pipeline.py" "$@"
