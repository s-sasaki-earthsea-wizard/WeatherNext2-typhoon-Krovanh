#!/usr/bin/env bash
# Move data between the Mac and the pod over ssh (alias from ~/.ssh/config).
#
# Usage (on the Mac):
#   bash runpod/sync.sh push <case>   # upload prepared inputs
#   bash runpod/sync.sh pull <case>   # download tracks + cropped fields
set -euo pipefail

MODE="${1:?push|pull}"
CASE="${2:?case id required}"
HOST="${RUNPOD_HOST:-runpod}"
REMOTE="${RUNPOD_WORKDIR:-/workspace/WeatherNext2-typhoon-Krovanh}"

# TODO: RunPod's ssh proxy (ssh.runpod.io) does not support scp/rsync;
#       use the pod's direct TCP ssh port instead, or fall back to
#       tar-over-ssh. Decide once the pod is up.
case "$MODE" in
  push) echo "TODO: upload data/interim/${CASE}/ to ${HOST}:${REMOTE}/data/interim/${CASE}/" ;;
  pull) echo "TODO: download ${HOST}:${REMOTE}/outputs/${CASE}/ to outputs/${CASE}/" ;;
  *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
