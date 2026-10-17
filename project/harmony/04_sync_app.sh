#!/usr/bin/env bash
# Copy libMNN, ARM64 HiAI libraries, and matching public headers into the App checkout.
source "$(dirname -- "${BASH_SOURCE[0]}")/sdk_common.sh"
mobi_sdk sync-app "$@"
