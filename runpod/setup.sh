#!/usr/bin/env bash
# One-time environment setup on a RunPod H100 pod (no Docker available).
# Everything lives under the persistent volume so a stopped pod keeps it.
#
# Usage (on the pod):
#   bash runpod/setup.sh
set -euo pipefail

WORKDIR="${RUNPOD_WORKDIR:-/workspace/WeatherNext2-typhoon-Krovanh}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.uv-python}"

# TODO: verify the pod template ships an NVIDIA driver compatible with the
#       CUDA 12 wheels pulled in by jax[cuda12] (nvidia-smi).

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$WORKDIR"
uv python install 3.12
uv sync --group gpu

uv run python -c "import jax; print('jax backend:', jax.default_backend(), jax.devices())"
