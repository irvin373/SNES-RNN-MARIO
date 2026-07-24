#!/usr/bin/env bash
# Build (cached) the project's Linux dev image and run a command inside it,
# with the repo and the SMW ROM directory mounted, and the ROM imported.
#
# Usage: scripts/docker-run.sh <command...>
# Example: scripts/docker-run.sh pytest tests/test_env_smoke.py -v
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROM_DIR="${MARIO_RNN_ROM_DIR:-$HOME/roms/smw}"

# Skip the rebuild if an image is already tagged: `docker build` has been observed to
# hang indefinitely on this machine after heavy Docker Desktop use, even with nothing to
# rebuild. Delete the image (`docker rmi mario-rnn-dev`) to force a rebuild when needed.
if ! docker image inspect mario-rnn-dev >/dev/null 2>&1; then
  docker build --platform linux/amd64 -q -t mario-rnn-dev -f "$REPO_ROOT/Dockerfile" "$REPO_ROOT" >/dev/null
fi

docker run --rm --platform linux/amd64 \
  -v "$REPO_ROOT:/workspace" \
  -v "$ROM_DIR:/roms:ro" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  mario-rnn-dev \
  bash -c 'python -m stable_retro.import /roms >/dev/null 2>&1; exec "$@"' _ "$@"
