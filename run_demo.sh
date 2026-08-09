#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="kubricdockerhub/kubruntudev"
DOCKER_USER="$(id -u):$(id -g)"
DOCKER_MOUNT="${ROOT_DIR}:/workspace"

usage() {
  cat <<'EOF'
Usage:
  ./run_demo.sh hello      # Render hello world still outputs
  ./run_demo.sh sim        # Run simulation and export output/simulator.mp4
  ./run_demo.sh all        # Run hello then simulation

Notes:
- Requires Docker daemon running.
- For sim mode, this script installs imageio-ffmpeg inside the container at runtime.
EOF
}

run_hello() {
  echo "[kubric-demo] Running hello world demo..."
  docker run --rm --interactive \
    --user "${DOCKER_USER}" \
    --volume "${DOCKER_MOUNT}" \
    "${DOCKER_IMAGE}" \
    python3 examples/helloworld.py

  echo "[kubric-demo] Hello outputs:"
  ls -lh "${ROOT_DIR}/output/helloworld"* 2>/dev/null || true
}

run_sim() {
  echo "[kubric-demo] Running simulation demo and exporting MP4..."
  docker run --rm --interactive \
    --user "${DOCKER_USER}" \
    --volume "${DOCKER_MOUNT}" \
    "${DOCKER_IMAGE}" \
    sh -lc 'python3 -m pip install --quiet --disable-pip-version-check imageio-ffmpeg && python3 examples/simulator.py'

  echo "[kubric-demo] Simulation outputs:"
  ls -lh "${ROOT_DIR}/output/simulator.mp4" "${ROOT_DIR}/output/simulator.blend" 2>/dev/null || true
}

MODE="${1:-sim}"

case "${MODE}" in
  hello)
    run_hello
    ;;
  sim)
    run_sim
    ;;
  all)
    run_hello
    run_sim
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
