#!/usr/bin/env bash
# Bootstrap a GPU-enabled Ubuntu VM for NN model experiments.
#
# Usage:
#   bash experiments/cloud/bootstrap-gpu.sh          # first run: installs drivers, may reboot
#   bash experiments/cloud/bootstrap-gpu.sh          # after reboot: finishes setup
#
# If nvidia-smi works, skips driver installation. Otherwise installs
# nvidia-driver-550, reboots (required for the kernel module to load),
# and asks you to re-run the script after reconnecting.

set -euo pipefail

if ! nvidia-smi >/dev/null 2>&1; then
    echo "[bootstrap-gpu] nvidia-smi not working — installing drivers"
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        nvidia-driver-550

    echo
    echo "============================================="
    echo "  Drivers installed. Rebooting in 5 seconds."
    echo "  After reboot, SSH back in and re-run:"
    echo "    cd threshold-opt"
    echo "    bash experiments/cloud/bootstrap-gpu.sh"
    echo "============================================="
    sleep 5
    sudo reboot
    # Script ends here on first run
fi

echo "[bootstrap-gpu] GPU visible:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Regular bootstrap: uv, system build deps, uv sync
bash experiments/cloud/bootstrap.sh
export PATH="$HOME/.local/bin:$PATH"

# Verify torch sees the GPU
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after bootstrap'
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "[bootstrap-gpu] done. GPU ready."
