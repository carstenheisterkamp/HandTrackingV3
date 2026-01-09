#!/bin/bash
# Pre-build script: Kill old service before rebuilding
# Add this to CLion: Settings > Build > CMake > Build options: -D PRE_BUILD_SCRIPT=...
# Or run manually before each rebuild

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/kill_old_service.sh"

