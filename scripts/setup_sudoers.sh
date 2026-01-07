#!/bin/bash
# Enable Password-Less Performance Tuning
# This script configures sudoers to allow running performance commands without password

set -e

SCRIPT_PATH="/home/nvidia/dev/HandTrackingV3/scripts/jetson_max_performance.sh"
SUDOERS_FILE="/etc/sudoers.d/jetson-performance"

echo "Configuring password-less sudo for performance commands..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Warning: Script not found at $SCRIPT_PATH (will be created later)"
fi

# Create sudoers configuration
# Allows user 'nvidia' to run performance commands without password
cat > "$SUDOERS_FILE" << 'EOF'
# Jetson Performance Tuning - No Password Required
nvidia ALL=(ALL) NOPASSWD: /bin/bash /home/nvidia/dev/HandTrackingV3/scripts/jetson_max_performance.sh
nvidia ALL=(ALL) NOPASSWD: /usr/bin/bash /home/nvidia/dev/HandTrackingV3/scripts/jetson_max_performance.sh
nvidia ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel
nvidia ALL=(ALL) NOPASSWD: /usr/bin/jetson_clocks
nvidia ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
nvidia ALL=(ALL) NOPASSWD: /usr/bin/tee /sys/devices/system/cpu/cpu*/cpuidle/state*/disable
EOF

chmod 0440 "$SUDOERS_FILE"

# Validate sudoers file
visudo -c -f "$SUDOERS_FILE"

echo "Success! Password-less sudo configured for:"
echo "  - $SCRIPT_PATH"
echo "  - nvpmodel"
echo "  - jetson_clocks"
echo "  - CPU governor/idle control"

