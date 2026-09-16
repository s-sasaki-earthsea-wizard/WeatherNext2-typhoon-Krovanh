#!/usr/bin/env bash
# One-time environment setup on a RunPod H100 pod (no Docker available).
# Everything lives under the persistent volume so a stopped pod keeps it.
#
# Create the pod with:
#   - an H100 80 GB (PCIe is enough; see docs/requirements.md decision 9)
#   - a 30 GB network volume mounted at /workspace (sizing: docs/design.md)
#   - SSH over an exposed TCP port, not only the ssh.runpod.io proxy, because
#     the proxy carries no rsync and runpod/sync.sh needs it
#
# Usage (on the pod):
#   bash runpod/setup.sh
set -euo pipefail

WORKDIR="${RUNPOD_WORKDIR:-/workspace/WeatherNext2-typhoon-Krovanh}"
# Keep the cache on the same filesystem as the venv: uv hardlinks wheels across
# them rather than copying, and the CUDA wheels are most of the footprint.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.uv-python}"

echo "== GPU and driver"
nvidia-smi || { echo "no nvidia-smi: wrong pod template?" >&2; exit 1; }

if ! command -v uv >/dev/null 2>&1; then
  echo "== Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# rsync is what sync.sh uses and is not in every RunPod image.
command -v rsync >/dev/null 2>&1 || {
  echo "== Installing rsync"
  apt-get update -qq && apt-get install -y -qq rsync
}

cd "$WORKDIR"
uv python install 3.12
uv sync --group gpu
uv cache prune

echo "== JAX backend"
uv run python -c "import jax; print('backend:', jax.default_backend(), jax.devices())"

echo
echo "Next:"
echo "  make smoke-gpu    # jax sees the GPU"
echo "  make smoke-mini   # the whole pipeline on the 1 deg model, on the GPU"
echo "                    # attention path (triblockdiag_mha), for a few dollars"
echo "                    # less than finding out during the real run"
