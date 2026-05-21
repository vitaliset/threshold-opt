#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu VM to run experiments from this repo.
# Usage: bash experiments/cloud/bootstrap.sh
#
# Assumes you've already cloned the repo and are running this from its root.
# Installs uv, system build deps, and syncs the locked Python environment.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure ~/.local/bin is on PATH in future shells (uv installer may not
# update ~/.bashrc on all Ubuntu images).
if ! grep -q 'local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "[bootstrap] added ~/.local/bin to ~/.bashrc"
fi

if command -v apt-get >/dev/null 2>&1; then
    echo "[bootstrap] installing system packages"
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends \
        build-essential git tmux rsync ca-certificates
fi

echo "[bootstrap] uv sync --frozen"
uv sync --frozen

echo "[bootstrap] done. Reload your shell, then launch a study:"
echo "  source ~/.bashrc"
echo "  make cloud-run STUDY=binary N_SEEDS=10"
