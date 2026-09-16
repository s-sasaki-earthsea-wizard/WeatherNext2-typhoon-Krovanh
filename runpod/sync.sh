#!/usr/bin/env bash
# Move data between the Mac and the pod.
#
# RunPod's ssh.runpod.io proxy carries interactive sessions but not rsync, so
# create the pod with ssh over an exposed TCP port and point the `runpod` alias
# in ~/.ssh/config at that host and port. Then plain rsync works.
#
# The pull runs in two legs. A single rsync from the pod to the NAS holds the
# pod open for the whole of the slow leg, and by then the pod is doing no
# compute while still being billed. The first leg lands the case on the Mac's
# SSD, after which the pod can be stopped; the second leg copies it to
# RESULTS_ROOT with the pod already off. The legs are separately invocable so a
# second leg that fails -- NAS unmounted, Tailscale down -- can be retried
# without renting a pod again.
#
# Usage (on the Mac):
#   bash runpod/sync.sh push <case>          # upload prepared inputs
#   bash runpod/sync.sh pull <case>          # both pull legs, in order
#   bash runpod/sync.sh pull-stage <case>    # pod -> staging/<case>/
#   bash runpod/sync.sh pull-publish <case>  # staging/<case>/ -> $RESULTS_ROOT
set -euo pipefail

MODE="${1:?push|pull|pull-stage|pull-publish}"
CASE="${2:?case id required}"
HOST="${RUNPOD_HOST:-runpod}"
REMOTE="${RUNPOD_WORKDIR:-/workspace/WeatherNext2-typhoon-Krovanh}"
RESULTS_ROOT="${RESULTS_ROOT:-/Volumes/EW-NAS-Atoll/Projects/personal-dev/WeatherNext2-typhoon-Krovanh}"

# Staging sits in the working copy (git-ignored) rather than under /tmp: a
# RESULTS_ROOT that is itself local is then on the same filesystem, so the
# second leg never leaves it, and macOS's periodic /tmp cleanup cannot delete a
# staged case that is waiting for its second leg to be retried.
STAGE="staging/${CASE}/"

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

# Gates the NAS leg only. The pod leg must run even with the share unmounted:
# that leg is the one costing money, and its output is what makes the retry
# cheap.
check_results_root() {
  # RESULTS_ROOT itself does not exist before the first pull, so testing it is
  # wrong; what has to hold is that the NAS is mounted. Walk up to the nearest
  # existing ancestor and require it to be writable: with the share unmounted
  # that walk stops at /Volumes, which is root-owned, while a RESULTS_ROOT
  # pointed somewhere local stops at a directory the user owns.
  local ancestor="$RESULTS_ROOT"
  while [ ! -d "$ancestor" ]; do ancestor="$(dirname "$ancestor")"; done
  if [ -w "$ancestor" ]; then
    return 0
  fi
  echo "error: results root not available: $RESULTS_ROOT" >&2
  echo "  stopped at '$ancestor', which is not writable." >&2
  echo "  mount the NAS, or set RESULTS_ROOT to somewhere local." >&2
  echo "  anything already staged is kept; finish it with:" >&2
  echo "    bash runpod/sync.sh pull-publish ${CASE}" >&2
  return 1
}

# Leg 1: pod -> local SSD. Nothing here touches the NAS.
pull_stage() {
  check_local_rsync
  check_rsync_over_ssh
  mkdir -p "$STAGE"
  rsync "${RSYNC_OPTS[@]}" "${HOST}:${REMOTE}/outputs/${CASE}/" "$STAGE"
  echo "staged to ${STAGE}; the pod is no longer needed and can be stopped."
}

# Leg 2: local SSD -> RESULTS_ROOT. Needs no pod, so it is safe to retry.
pull_publish() {
  local dest="${RESULTS_ROOT}/outputs/${CASE}/"
  if [ ! -d "$STAGE" ]; then
    echo "error: nothing staged at ${STAGE}" >&2
    echo "  run this first:  bash runpod/sync.sh pull-stage ${CASE}" >&2
    return 1
  fi
  check_local_rsync
  check_results_root
  mkdir -p "$dest"
  # rsync rather than mv: it resumes, --append-verify checks what it skips, and
  # it merges sensibly into a destination that already holds an earlier pull.
  rsync "${RSYNC_OPTS[@]}" "$STAGE" "$dest"

  # Only now is the staged copy redundant. Deleting it with rsync
  # --remove-source-files instead would delete file by file as the transfer
  # runs, and a leg that dies halfway would leave a half-staged case that
  # cannot simply be retried.
  rm -rf "${STAGE%/}"

  # Point the working copy at the published results so the analysis targets read
  # them in place. The link is at RESULTS_ROOT, never at the staging copy, which
  # is about to disappear. outputs/* is git-ignored, so it is never committed.
  local link="outputs/${CASE}"
  if [ -L "$link" ] || [ ! -e "$link" ]; then
    ln -sfn "${dest%/}" "$link"
    echo "pulled to ${dest} (linked as ${link})"
  else
    echo "pulled to ${dest}"
    echo "note: ${link} exists and is not a symlink; left alone." >&2
  fi
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
    # Back to back, but still two independent legs: if the second one fails the
    # staged copy survives and pull-publish finishes the job later.
    pull_stage
    pull_publish
    ;;
  pull-stage)
    pull_stage
    ;;
  pull-publish)
    pull_publish
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 1
    ;;
esac
