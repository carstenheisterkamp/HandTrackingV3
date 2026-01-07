#!/bin/bash
# Jetson Orin Nano Performance Diagnostics (Fixed V2)

echo "=========================================="
echo "  Jetson Orin Nano System Diagnostics"
echo "=========================================="
echo ""

# 1. Power Mode
echo "=== 1. Power Mode (nvpmodel) ==="
if command -v nvpmodel &> /dev/null; then
    # Full output for debug
    RAW_OUTPUT=$(nvpmodel -q)
    # Extract Mode Name (e.g. "15W" or "MAXN")
    MODE_NAME=$(echo "$RAW_OUTPUT" | grep "Power Mode:" | awk -F': ' '{print $2}' | tr -d '\n')
    # Extract Mode ID (usually the line after Power Mode)
    MODE_ID=$(echo "$RAW_OUTPUT" | grep -v "Power Mode" | head -n 1 | tr -d ' ' | tr -d '\n')

    echo "  Raw Output: $(echo $RAW_OUTPUT | xargs)"
    echo "  Detected Label:     '$MODE_NAME'"

    # Validation Logic: ID must be 0 OR Label must contain 15W/MAXN
    IS_MAX_POWER=false
    if [ "$MODE_ID" == "0" ]; then
        IS_MAX_POWER=true
    elif [[ "$MODE_NAME" == *"15W"* ]]; then
        IS_MAX_POWER=true
    elif [[ "$MODE_NAME" == *"MAXN"* ]]; then
        IS_MAX_POWER=true
    fi

    if [ "$IS_MAX_POWER" = true ]; then
        echo "✅ SUCCESS: System is in Maximum Performance Mode ($MODE_NAME)."
    else
        echo "❌ WARNING: System might not be in Max Performance. Expected Mode 0 or 15W/MAXN."
    fi
else
    echo "ERROR: nvpmodel not found!"
    IS_MAX_POWER=false
fi
echo ""

# 2. Clocks
echo "=== 2. Clocks (jetson_clocks) ==="
if command -v jetson_clocks &> /dev/null; then
    # Check GPU Min/Max freq match
    CLOCKS_INFO=$(sudo jetson_clocks --show)
    GPU_LINE=$(echo "$CLOCKS_INFO" | grep "GPU")
    echo "$GPU_LINE"

    GPU_MIN=$(echo "$GPU_LINE" | grep -o 'MinFreq=[0-9]*' | cut -d= -f2)
    GPU_MAX=$(echo "$GPU_LINE" | grep -o 'MaxFreq=[0-9]*' | cut -d= -f2)

    if [ "$GPU_MIN" == "$GPU_MAX" ]; then
         echo "✅ SUCCESS: GPU Clocks are LOCKED at Max ($GPU_MAX Hz)."
         CLOCKS_LOCKED=true
    else
         echo "❌ WARNING: GPU Clocks conform to dynamic scaling (Min!=Max). Run 'sudo jetson_clocks'."
         CLOCKS_LOCKED=false
    fi
else
    echo "ERROR: jetson_clocks not found!"
    CLOCKS_LOCKED=false
fi
echo ""

echo "=== 3. Summary ==="
if [ "$IS_MAX_POWER" = true ] && [ "$CLOCKS_LOCKED" = true ]; then
    echo "✅✅ YOUR SYSTEM IS PERFECTLY CONFIGURED (100% Performance)."
else
    echo "⚠️  CHECK CONFIGURATION (See warnings above)."
fi
echo "=========================================="

