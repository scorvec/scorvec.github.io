#!/bin/bash
# Shared by the frames publishers/fetchers: decide whether the object store is
# in play and make boto3 available. Sourced, not executed.
#
# On a runner the FRAMES_S3_* / AWS_* variables come from repository secrets
# via the workflow env block. On the laptop launchd carries no environment, so
# they are read from ~/.config/scorvec/frames_store.env (chmod 600) if present.
#
#   frames_store_ready   -> 0 when the store is configured, else 1
#   FRAMES_PY            -> python with boto3 (installed on demand on a runner)
FRAMES_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ENVF="${FRAMES_STORE_ENV:-$HOME/.config/scorvec/frames_store.env}"
if [ -z "${FRAMES_S3_BUCKET:-}" ] && [ -f "$_ENVF" ]; then
  set -a; . "$_ENVF"; set +a
fi
: "${FRAMES_PY:=$(command -v python3)}"
[ -x /opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python ] \
  && FRAMES_PY=/opt/homebrew/Caskroom/miniconda/base/envs/mjo/bin/python
frames_store_ready() {
  [ -n "${FRAMES_S3_BUCKET:-}" ] && [ -n "${FRAMES_S3_ENDPOINT:-}" ] \
    && [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ] || return 1
  "$FRAMES_PY" -c 'import boto3' 2>/dev/null || "$FRAMES_PY" -m pip install -q boto3 >/dev/null 2>&1 \
    || { echo "  frames store configured but boto3 unavailable"; return 1; }
  return 0
}
