#!/usr/bin/env bash
# Unpack CANN Kit and stage its ARM64 HiAI headers/libraries in the MNN source tree.
source "$(dirname -- "${BASH_SOURCE[0]}")/sdk_common.sh"
mobi_sdk prepare-hiai "$@"
