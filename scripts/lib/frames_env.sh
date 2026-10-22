#!/bin/bash
# Shared by the frames publishers/fetchers: decide where this job's frames go and make boto3 available. Sourced, not
# executed.
#
# On a runner the FRAMES_S3_* variables come from repository secrets via the workflow env block, and FRAMES_BRANCH
# from the repository variable of that name. Elsewhere they are read from the env file named by FRAMES_STORE_ENV
# (chmod 600) if present.
#
#   frames_store_ready   -> 0 when the store is configured and boto3 imports, else 1
#   FRAMES_MODE          -> where frames go, from FRAMES_BRANCH and the store:
#       branch   the store is not configured, or FRAMES_BRANCH names another data branch (verify-data): git only
#       dual     the store is configured and FRAMES_BRANCH is unset or "frames": publish to the store, then to the
#                branch; seed from the branch, which stays the complete copy until it is retired
#       store    FRAMES_BRANCH=off: the store alone
#       broken   FRAMES_BRANCH=off but the store is not usable: nothing can be published (and no branch named "off"
#                may ever be created in its place)
#   FRAMES_PY            -> python with boto3 (installed on demand on a runner)
FRAMES_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ENVF="${FRAMES_STORE_ENV:-}"
if [ -z "${FRAMES_S3_BUCKET:-}" ] && [ -n "$_ENVF" ] && [ -f "$_ENVF" ]; then
  set -a; . "$_ENVF"; set +a
fi
: "${FRAMES_PY:=$(command -v python3)}"
frames_store_ready() {
  [ -n "${FRAMES_S3_BUCKET:-}" ] && [ -n "${FRAMES_S3_ENDPOINT:-}" ] \
    && [ -n "${FRAMES_S3_ACCESS_KEY_ID:-}" ] && [ -n "${FRAMES_S3_SECRET_ACCESS_KEY:-}" ] || return 1
  "$FRAMES_PY" -c 'import boto3' 2>/dev/null && return 0
  # a runner's system python refuses a plain install (PEP 668, externally managed); setup-python's takes it
  "$FRAMES_PY" -m pip install -q boto3 >/dev/null 2>&1 \
    || "$FRAMES_PY" -m pip install -q --break-system-packages boto3 >/dev/null 2>&1 \
    || { echo "::warning::the frames store is configured but boto3 could not be installed" >&2; return 1; }
}
case "${FRAMES_BRANCH:-frames}" in
  off)    if frames_store_ready; then FRAMES_MODE=store; else FRAMES_MODE=broken; fi ;;
  frames) if frames_store_ready; then FRAMES_MODE=dual; else FRAMES_MODE=branch; fi ;;
  *)      FRAMES_MODE=branch ;;
esac
