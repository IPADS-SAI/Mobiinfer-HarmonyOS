#!/usr/bin/env bash
# Install DDK + Kirin9030 plugin, then initialize AscendC in MOBI_PYTHON's environment.
source "$(dirname -- "${BASH_SOURCE[0]}")/sdk_common.sh"
mobi_sdk prepare-offline "$@"
