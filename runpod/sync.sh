#!/usr/bin/env bash
# Move data between the Mac and the pod.
#
# RunPod's ssh.runpod.io proxy carries interactive sessions but not rsync, so
# create the pod with ssh over an exposed TCP port and point the `runpod` alias
# in ~/.ssh/config at that host and port. Then plain rsync works.
#
# Usage (on the Mac):
#   bash runpod/sync.sh push <case>   # upload prepared inputs
#   bash runpod/sync.sh pull <case>   # download tracks + regional crop
set -euo pipefail

MODE="${1:?push|pull}"
CASE="${2:?case id required}"
HOST="${RUNPOD_HOST:-runpod}"
REMOTE="${RUNPOD_WORKDIR:-/workspace/WeatherNext2-typhoon-Krovanh}"
RESULTS_ROOT="${RESULTS_ROOT:-/Volumes/EW-NAS-Atoll/WeatherNext2-typhoon-Krovanh}"

# --partial --append-verify so an interrupted multi-GB transfer resumes instead
# of restarting; -z is deliberately absent because NetCDF and zarr are already
# compressed and the pod is billed by the hour. --no-owner --no-group because
# neither end can chown: the pod's volume and the NAS are both network mounts
# that refuse it, and the uids mean different things on the two machines.
RSYNC_OPTS=(--archive --no-owner --no-group
            --human-readable --progress --partial --append-verify)

# macOS ships openrsync, which does not know --append-verify and answers with a
# usage dump rather than an error message. Homebrew's rsync does know it.
#
# The help text is captured rather than piped into grep: under `pipefail` a
# `grep -q` that matches early closes the pipe, rsync dies of SIGPIPE, and the
# pipeline reports failure even though the option is supported. openrsync exits
# non-zero on --help, hence the `|| true`.
check_local_rsync() {
  local help_text
  help_text="$(rsync --help 2>&1 || true)"
  case "$help_text" in
    *--append-verify*) return 0 ;;
  esac
  echo "error: the local rsync does not support --append-verify." >&2
  echo "  macOS ships openrsync; install the GNU one:  brew install rsync" >&2
  return 1
}

check_rsync_over_ssh() {
  if ssh -o BatchMode=yes "$HOST" 'command -v rsync' >/dev/null 2>&1; then
    return 0
  fi
  echo "error: cannot reach rsync on '$HOST'." >&2
  echo "  - is the pod up, and does ~/.ssh/config point 'Host $HOST' at the" >&2
  echo "    pod's exposed TCP port rather than ssh.runpod.io?" >&2
  echo "  - if rsync is missing on the pod: ssh $HOST 'apt-get install -y rsync'" >&2
  return 1
}

case "$MODE" in
  push)
    LOCAL="data/interim/${CASE}/"
    [ -d "$LOCAL" ] || { echo "error: no such directory: $LOCAL" >&2; exit 1; }
    check_local_rsync
    check_rsync_over_ssh
    ssh "$HOST" "mkdir -p '${REMOTE}/data/interim/${CASE}'"
    rsync "${RSYNC_OPTS[@]}" "$LOCAL" "${HOST}:${REMOTE}/data/interim/${CASE}/"
    ;;
  pull)
    DEST="${RESULTS_ROOT}/outputs/${CASE}/"
    # RESULTS_ROOT itself does not exist before the first pull, so testing it
    # is wrong; what has to hold is that the NAS is mounted. Walk up to the
    # nearest existing ancestor and require it to be writable: with the share
    # unmounted that walk stops at /Volumes, which is root-owned, while a
    # RESULTS_ROOT pointed somewhere local stops at a directory the user owns.
    ancestor="$RESULTS_ROOT"
    while [ ! -d "$ancestor" ]; do ancestor="$(dirname "$ancestor")"; done
    if [ ! -w "$ancestor" ]; then
      echo "error: results root not available: $RESULTS_ROOT" >&2
      echo "  stopped at '$ancestor', which is not writable." >&2
      echo "  mount the NAS, or set RESULTS_ROOT to somewhere local." >&2
      exit 1
    fi
    check_local_rsync
    check_rsync_over_ssh
    mkdir -p "$DEST"
    rsync "${RSYNC_OPTS[@]}" "${HOST}:${REMOTE}/outputs/${CASE}/" "$DEST"

    # Point the working copy at the pulled results so the analysis targets read
    # them in place. outputs/* is git-ignored, so the link is never committed.
    LINK="outputs/${CASE}"
    if [ -L "$LINK" ] || [ ! -e "$LINK" ]; then
      ln -sfn "${DEST%/}" "$LINK"
      echo "pulled to ${DEST} (linked as ${LINK})"
    else
      echo "pulled to ${DEST}"
      echo "note: ${LINK} exists and is not a symlink; left alone." >&2
    fi
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 1
    ;;
esac
