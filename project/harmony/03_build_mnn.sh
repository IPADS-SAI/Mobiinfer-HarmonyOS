#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/sdk_common.sh"
mobi_sdk check-build
mkdir -p -- "$MOBI_BUILD_DIR"
cd -- "$MOBI_BUILD_DIR"
bash "$MOBI_SCRIPT_DIR/build_64.sh" "$@"
