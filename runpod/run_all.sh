#!/usr/bin/env bash
# End-to-end run on the pod for one case: inference -> tracker.
#
# Usage (on the pod):
#   bash runpod/run_all.sh init-2026-08-31T18
set -euo pipefail

CASE="${1:?case id required}"
CONFIG="${CONFIG:-configs/krovanh.yaml}"

# TODO: log to outputs/<case>/run.log with timestamps; record nvidia-smi and
#       wall-clock per member so the cost of extra cases can be estimated.
uv run python scripts/run_inference.py --config "$CONFIG" --case "$CASE"
uv run python scripts/run_tracker.py   --config "$CONFIG" --case "$CASE"
