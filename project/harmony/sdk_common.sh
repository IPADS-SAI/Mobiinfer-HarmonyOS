#!/usr/bin/env bash
set -euo pipefail

MOBI_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOBI_ROOT="${MOBI_ROOT:-$(cd -- "$MOBI_SCRIPT_DIR/../.." && pwd)}"
MOBI_SDK_CONFIG="${MOBI_SDK_CONFIG:-$MOBI_SCRIPT_DIR/sdk_config.sh}"
if [[ ! -f "$MOBI_SDK_CONFIG" ]]; then
    printf 'Missing SDK configuration: %s\n' "$MOBI_SDK_CONFIG" >&2
    exit 1
fi
source "$MOBI_SDK_CONFIG"
if ! command -v "$MOBI_PYTHON" >/dev/null 2>&1; then
    printf 'Python not found: %s (set MOBI_PYTHON in sdk_config.sh)\n' "$MOBI_PYTHON" >&2
    exit 1
fi

mobi_sdk() {
    "$MOBI_PYTHON" -B "$MOBI_SCRIPT_DIR/sdk_setup.py" "$@"
}
