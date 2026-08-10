#!/usr/bin/env bash
# Check out upstream DINOv3 at the pinned commit, ready to `pip install`.
#
# The pinned commit is on a feature branch, not main, so this fetches all refs rather
# than doing a default clone.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/dinov3.pin"

dest="${1:-${here}/dinov3}"

if [[ -d "${dest}/.git" ]]; then
  echo "already present: ${dest}"
else
  git init -q "${dest}"
  git -C "${dest}" remote add origin "${DINOV3_REPO_URL}"
fi

# All branches, not just main — the pinned commit is on ${DINOV3_BRANCH}.
git -C "${dest}" fetch --quiet origin '+refs/heads/*:refs/remotes/origin/*' --tags

if ! git -C "${dest}" cat-file -e "${DINOV3_COMMIT}^{commit}" 2>/dev/null; then
  echo "ERROR: pinned commit ${DINOV3_COMMIT} is not reachable in ${DINOV3_REPO_URL}." >&2
  echo "       It is on branch '${DINOV3_BRANCH}'." >&2
  exit 1
fi

git -C "${dest}" checkout --quiet --detach "${DINOV3_COMMIT}"
echo "DINOv3 at ${DINOV3_COMMIT} (branch ${DINOV3_BRANCH}) -> ${dest}"
echo "install it with: pip install ${dest}"
