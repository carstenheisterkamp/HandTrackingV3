#!/bin/bash
# Quick switcher for Lite vs Full models
# Usage: ./switch_models.sh [lite|full]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_CPP="$PROJECT_ROOT/src/main.cpp"

if [ $# -eq 0 ]; then
    echo "Usage: $0 [lite|full]"
    echo ""
    echo "Current model:"
    grep "const bool USE_FULL_MODELS" "$MAIN_CPP" || echo "  Not found in main.cpp"
    exit 1
fi

MODE="$1"

case "$MODE" in
    lite)
        echo "Switching to LITE models (fast, default)..."
        sed -i.bak 's/const bool USE_FULL_MODELS = true/const bool USE_FULL_MODELS = false/' "$MAIN_CPP"
        echo "✓ Set USE_FULL_MODELS = false"
        ;;
    full)
        echo "Switching to FULL models (high accuracy)..."
        sed -i.bak 's/const bool USE_FULL_MODELS = false/const bool USE_FULL_MODELS = true/' "$MAIN_CPP"
        echo "✓ Set USE_FULL_MODELS = true"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        echo "Use: lite or full"
        exit 1
        ;;
esac

rm -f "$MAIN_CPP.bak"

echo ""
echo "Next steps:"
echo "  1. Rebuild: cd cmake-build-debug-remote-host && ninja"
echo "  2. Restart: sudo systemctl restart hand-tracking"
echo "  3. Check log: journalctl -u hand-tracking -f"
echo ""
echo "See docs/MODEL_TESTING.md for detailed testing guide"

