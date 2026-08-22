#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE="kubricdockerhub/kubruntudev"
DOCKER_USER="$(id -u):$(id -g)"
DOCKER_MOUNT="${ROOT_DIR}:/workspace"
INTERVENTION_OUTPUT="${ROOT_DIR}/output/demo_collision_intervention"

usage() {
  cat <<'EOF'
Usage:
  ./run_demo.sh hello                       # Render hello world still outputs
  ./run_demo.sh sim                         # Export output/simulator.mp4
  ./run_demo.sh intervention                # Build the three-panel MP4
  ./run_demo.sh intervention-physics-only   # Write replay data only
  ./run_demo.sh all                         # Run hello then simulation

Notes:
- Hello, sim, and full intervention modes require a running Docker daemon.
- For sim mode, this script installs imageio-ffmpeg inside the container at runtime.
- Intervention modes require the conda `thesis` environment. Full intervention
  mode also requires Docker and kubricdockerhub/kubruntudev; it fails instead
  of silently skipping Blender. See docs/trajectory_interventions.md.
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

resolve_thesis_python() {
  local python_bin=""

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "${conda_base}" ] && [ -x "${conda_base}/envs/thesis/bin/python" ]; then
      python_bin="${conda_base}/envs/thesis/bin/python"
    fi
  fi

  if [ -z "${python_bin}" ]; then
    local candidate
    for candidate in \
      "${HOME}/miniconda3/envs/thesis/bin/python" \
      "${HOME}/anaconda3/envs/thesis/bin/python" \
      "${HOME}/miniforge3/envs/thesis/bin/python" \
      "${ROOT_DIR}/thesis/bin/python"; do
      if [ -x "${candidate}" ]; then
        python_bin="${candidate}"
        break
      fi
    done
  fi

  if [ -z "${python_bin}" ]; then
    echo "[kubric-demo] ERROR: conda environment 'thesis' was not found." >&2
    echo "[kubric-demo] Install it as documented in docs/trajectory_interventions.md." >&2
    return 1
  fi

  printf '%s\n' "${python_bin}"
}

run_intervention_physics() {
  local python_bin="$1"
  echo "[kubric-demo] Running trajectory intervention collision demo with: ${python_bin}"
  (
    cd "${ROOT_DIR}"
    "${python_bin}" -m scripts.demo_collision_intervention \
      --output "${INTERVENTION_OUTPUT}"
  )
}

require_blender_renderer() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[kubric-demo] ERROR: Docker command not found; Blender rendering is required." >&2
    return 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[kubric-demo] ERROR: Docker daemon is unavailable; Blender rendering is required." >&2
    return 1
  fi
  if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    echo "[kubric-demo] ERROR: required Docker image ${DOCKER_IMAGE} is unavailable; it supplies Blender." >&2
    return 1
  fi
}

render_blender_branches() {
  local render_command
  require_blender_renderer

  echo "[kubric-demo] Replaying all three branches through Blender in ${DOCKER_IMAGE}..."
  render_command='set -eu; cd /workspace; if ! python3 -c "import imageio_ffmpeg" >/dev/null 2>&1; then python3 -m pip install --quiet --disable-pip-version-check --no-cache-dir --target /tmp/kubric-demo-imageio imageio-ffmpeg; export PYTHONPATH="/tmp/kubric-demo-imageio${PYTHONPATH:+:$PYTHONPATH}"; fi; python3 /workspace/scripts/render_demo_branches_blender.py --states-dir /workspace/output/demo_collision_intervention --branches normal trajectory_changed target_removed'
  if ! docker run --rm --interactive \
    --user "${DOCKER_USER}" \
    --volume "${DOCKER_MOUNT}" \
    "${DOCKER_IMAGE}" \
    sh -lc "${render_command}"; then
    echo "[kubric-demo] ERROR: Blender renderer failed or is unavailable in ${DOCKER_IMAGE}." >&2
    return 1
  fi
}

compose_intervention_video() {
  local python_bin="$1"
  echo "[kubric-demo] Composing the synchronized three-panel video..."
  (
    cd "${ROOT_DIR}"
    "${python_bin}" -m scripts.compose_intervention_demo \
      --states-dir "${INTERVENTION_OUTPUT}" \
      --output "${INTERVENTION_OUTPUT}/trajectory_intervention_demo.mp4"
  )
}

show_outputs() {
  local heading="$1"
  shift
  local path

  echo "[kubric-demo] ${heading}:"
  for path in "$@"; do
    if [ ! -f "${path}" ]; then
      echo "[kubric-demo] ERROR: expected output is missing: ${path}" >&2
      return 1
    fi
    ls -lh -- "${path}"
  done
}

show_intervention_physics_outputs() {
  show_outputs "Physics replay outputs" \
    "${INTERVENTION_OUTPUT}/normal_states.npy" \
    "${INTERVENTION_OUTPUT}/normal_presence.npy" \
    "${INTERVENTION_OUTPUT}/trajectory_changed_states.npy" \
    "${INTERVENTION_OUTPUT}/trajectory_changed_presence.npy" \
    "${INTERVENTION_OUTPUT}/target_removed_states.npy" \
    "${INTERVENTION_OUTPUT}/target_removed_presence.npy" \
    "${INTERVENTION_OUTPUT}/contacts.json" \
    "${INTERVENTION_OUTPUT}/summary.json"
}

show_intervention_outputs() {
  show_intervention_physics_outputs
  show_outputs "Rendered video outputs" \
    "${INTERVENTION_OUTPUT}/normal_blender.mp4" \
    "${INTERVENTION_OUTPUT}/trajectory_changed_blender.mp4" \
    "${INTERVENTION_OUTPUT}/target_removed_blender.mp4" \
    "${INTERVENTION_OUTPUT}/trajectory_intervention_demo.mp4"
}

run_intervention_physics_only() {
  local python_bin
  python_bin="$(resolve_thesis_python)"
  run_intervention_physics "${python_bin}"
  show_intervention_physics_outputs
}

run_intervention() {
  local python_bin
  python_bin="$(resolve_thesis_python)"
  run_intervention_physics "${python_bin}"

  render_blender_branches
  compose_intervention_video "${python_bin}"
  show_intervention_outputs
}

MODE="${1:-sim}"

case "${MODE}" in
  hello)
    run_hello
    ;;
  sim)
    run_sim
    ;;
  intervention)
    run_intervention
    ;;
  intervention-physics-only)
    run_intervention_physics_only
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
