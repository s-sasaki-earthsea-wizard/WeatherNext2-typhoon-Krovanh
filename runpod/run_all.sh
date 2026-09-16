#!/usr/bin/env bash
# End-to-end run on the pod for one case.
#
# Runs a single member first and stops, so the wall-clock is known before the
# remaining members are committed to. Re-run with MEMBERS set to continue.
#
# Usage (on the pod):
#   bash runpod/run_all.sh init-2026-08-31T18            # one member, measure
#   MEMBERS=8 bash runpod/run_all.sh init-2026-08-31T18  # the rest
set -euo pipefail

CASE="${1:?case id required}"
CONFIG="${CONFIG:-configs/krovanh.yaml}"
MEMBERS="${MEMBERS:-1}"
START="${MEMBER_START:-0}"
LOG="outputs/${CASE}/run.log"

mkdir -p "outputs/${CASE}"
{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) case=${CASE} members=${MEMBERS} start=${START}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee -a "$LOG"

# run_inference tracks each member as it is produced, because tracking needs the
# global cyclone fields and those are never written. Timings land in the log.
uv run python scripts/run_inference.py \
  --config "$CONFIG" --case "$CASE" \
  --members "$MEMBERS" --member-start "$START" 2>&1 | tee -a "$LOG"

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) done" | tee -a "$LOG"
echo
echo "Pull the results from the Mac:  make runpod-pull-outputs CASE=${CASE}"
