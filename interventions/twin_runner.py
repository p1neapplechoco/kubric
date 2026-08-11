"""Deterministic factual/counterfactual execution from backend-neutral schemas.

``ObjectConfig.size`` maps directly to Kubric's ``scale``.  Kubric primitives use
unit half-extents, so a cube configured with size ``s`` has physical width ``2*s``.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

import kubric as kb

try:
  import fcntl
except ImportError:  # pragma: no cover - pair publication targets POSIX workers.
  fcntl = None

from interventions.graph_extraction import extract_ground_truth
from interventions.kinematic_simulator import KinematicDragSimulator
from interventions.logging import (
    SimulationLog,
    read_simulation_log,
    write_simulation_log,
)
from interventions.schema import (
    SCHEMA_VERSION,
    CameraConfig,
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    ObjectConfig,
    SceneConfig,
    to_jsonable,
)
from interventions.tagging import derive_tags
from interventions.trajectory import (
    max_position_deviation,
    perturb_path,
    validate_path,
)


PathLike = Union[str, os.PathLike[str]]
_INITIAL_TOLERANCE = 1e-9
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_PERTURB_ATTEMPTS = 64
_PAIR_MANIFEST = "manifest.json"
_PAIR_GENERATIONS = "generations"
_EXTRACTION_THRESHOLDS = {
    "force_threshold": 0.0,
    "min_episode_impulse": 0.0,
    "force_tolerance": 1e-6,
    "position_epsilon": 1e-3,
    "velocity_epsilon": 1e-3,
    "quaternion_epsilon": 1e-3,
}


def _identifier(value: Any, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if not value.strip():
    raise ValueError("{} must not be empty".format(name))
  return value


def _rng_seed(value: Any) -> int:
  if isinstance(value, bool) or not isinstance(value, numbers.Integral):
    raise TypeError("rng_seed must be an integer")
  result = int(value)
  if result < 0:
    raise ValueError("rng_seed must be nonnegative")
  return result


def _duration_steps(scene_config: SceneConfig) -> int:
  frame_start, frame_end = scene_config.frame_range
  duration = (frame_end - frame_start) * scene_config.step_rate
  steps = duration // scene_config.frame_rate
  if steps < 2:
    raise ValueError("scene duration must contain at least two Bullet steps")
  return steps


def _integer_window(intervention: Intervention, steps: int) -> Tuple[int, int]:
  values = intervention.time_window
  if not all(float(value).is_integer() for value in values):
    raise ValueError("intervention time_window values must be integer-valued steps")
  start, end = (int(value) for value in values)
  if not 0 <= start < end <= steps:
    raise ValueError(
        "intervention time_window must satisfy 0 <= start < end <= {}".format(
            steps
        )
    )
  return start, end


def _target_config(
    scene_config: SceneConfig,
    target_id: str,
    intervention: Intervention,
) -> ObjectConfig:
  if not isinstance(scene_config, SceneConfig):
    raise TypeError("scene_config must be a SceneConfig")
  if not isinstance(intervention, Intervention):
    raise TypeError("intervention must be an Intervention")
  target_id = _identifier(target_id, "target_id")
  if target_id != intervention.target_id:
    raise ValueError("target_id must match intervention.target_id")
  by_id = {item.object_id: item for item in scene_config.objects}
  if target_id not in by_id:
    raise ValueError("target_id {!r} does not exist in SceneConfig".format(target_id))
  target = by_id[target_id]
  if not target.static:
    raise ValueError("intervention target must have static=True")
  if any(value != 0.0 for value in target.angular_velocity):
    raise ValueError("intervention target angular velocity must be zero")
  return target


def _validate_mapping(scene_config: SceneConfig) -> None:
  """Rejects all unsupported schema-to-Kubric mappings before opening clients."""
  _validate_float32(scene_config.gravity, "scene gravity")
  if scene_config.camera is not None:
    _validate_float32(scene_config.camera.position, "camera position")
    _validate_float32(scene_config.camera.look_at, "camera look_at")
  for item in scene_config.objects:
    if item.shape == "sphere" and not (
        item.size[0] == item.size[1] == item.size[2]
    ):
      raise ValueError(
          "sphere {!r} requires a uniform size".format(item.object_id)
      )
    if item.friction > 1.0:
      raise ValueError(
          "friction for {!r} lies outside the Kubric domain [0, 1]".format(
              item.object_id
          )
      )
    _validate_float32(item.position, "{} position".format(item.object_id))
    _validate_float32(item.size, "{} size".format(item.object_id))
    _validate_float32(
        item.quaternion, "{} quaternion".format(item.object_id)
    )
    _validate_float32(
        item.linear_velocity, "{} linear_velocity".format(item.object_id)
    )
    _validate_float32(
        item.angular_velocity, "{} angular_velocity".format(item.object_id)
    )
    logical_id = item.metadata.get("logical_id")
    if logical_id is not None and logical_id != item.object_id:
      raise ValueError(
          "metadata logical_id for {!r} conflicts with object_id".format(
              item.object_id
          )
      )


def _validate_float32(values: Any, name: str) -> None:
  array = np.asarray(values, dtype=np.float64)
  if not np.isfinite(array).all() or np.any(np.abs(array) > _FLOAT32_MAX):
    raise ValueError("{} must be representable as finite float32 values".format(name))


def _rotation_matrix(quaternion: Iterable[float]) -> np.ndarray:
  w, x, y, z = (float(value) for value in quaternion)
  return np.asarray(
      (
          (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
          (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
          (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
      ),
      dtype=np.float64,
  )


def _object_aabb(item: ObjectConfig) -> np.ndarray:
  center = np.asarray(item.position, dtype=np.float64)
  if item.shape == "sphere":
    extent = np.full(3, item.size[0], dtype=np.float64)
  else:
    extent = np.abs(_rotation_matrix(item.quaternion)) @ np.asarray(
        item.size, dtype=np.float64
    )
  return np.stack((center - extent, center + extent))


def _static_aabbs(
    scene_config: SceneConfig, target_id: str
) -> Tuple[np.ndarray, ...]:
  return tuple(
      _object_aabb(item)
      for item in scene_config.objects
      if item.static and item.object_id != target_id
  )


def _target_extents(target: ObjectConfig, path: np.ndarray) -> np.ndarray:
  if target.shape == "sphere":
    return np.full((len(path), 3), target.size[0], dtype=np.float64)
  size = np.asarray(target.size, dtype=np.float64)
  return np.stack(
      [np.abs(_rotation_matrix(quaternion)) @ size for quaternion in path[:, 3:7]]
  )


def _geometry_tolerance(*values: np.ndarray) -> float:
  scale = max(
      1.0,
      *(float(np.max(np.abs(value))) for value in values if value.size),
  )
  return 16.0 * float(np.finfo(np.float64).eps) * scale


def _segment_intersects_open_aabb(
    start: np.ndarray,
    end: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
  """Whether a segment enters an AABB interior; boundary tangency is allowed."""
  entry = 0.0
  exit_ = 1.0
  for axis in range(3):
    start_value = float(start[axis])
    delta = float(end[axis]) - start_value
    lower = float(minimum[axis])
    upper = float(maximum[axis])
    if delta == 0.0:
      if not lower < start_value < upper:
        return False
      continue
    first = (lower - start_value) / delta
    second = (upper - start_value) / delta
    entry = max(entry, min(first, second))
    exit_ = min(exit_, max(first, second))
    if entry >= exit_:
      return False
  return entry < exit_ and exit_ > 0.0 and entry < 1.0


def _validate_target_sweep(
    path: np.ndarray,
    target: ObjectConfig,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    static_aabbs: Tuple[np.ndarray, ...],
) -> None:
  positions = path[:, :3]
  extents = _target_extents(target, path)
  lower = np.asarray(bounds[0], dtype=np.float64)
  upper = np.asarray(bounds[1], dtype=np.float64)
  bounds_tolerance = _geometry_tolerance(positions, extents, lower, upper)
  if np.any(positions - extents < lower - bounds_tolerance) or np.any(
      positions + extents > upper + bounds_tolerance
  ):
    raise ValueError("target swept volume falls outside scene bounds")
  for obstacle in static_aabbs:
    for index, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
      extent = np.maximum(extents[index], extents[index + 1])
      minimum = obstacle[0] - extent
      maximum = obstacle[1] + extent
      tolerance = _geometry_tolerance(start, end, minimum, maximum)
      if _segment_intersects_open_aabb(
          start,
          end,
          minimum + tolerance,
          maximum - tolerance,
      ):
        raise ValueError("target swept volume intersects a static obstacle AABB")


def _validate_path_float32(path: np.ndarray, step_rate: int) -> None:
  _validate_float32(path, "commanded path")
  velocities = np.zeros((len(path), 3), dtype=np.float64)
  with np.errstate(over="ignore", invalid="ignore"):
    velocities[1:] = (path[1:, :3] - path[:-1, :3]) * float(step_rate)
  _validate_float32(velocities, "commanded path velocity")


def _default_path(target: ObjectConfig, steps: int, step_rate: int) -> np.ndarray:
  offsets = np.arange(steps, dtype=np.float64)[:, None]
  positions = np.asarray(target.position, dtype=np.float64)[None, :] + (
      offsets * np.asarray(target.linear_velocity, dtype=np.float64)[None, :]
      / float(step_rate)
  )
  quaternions = np.tile(np.asarray(target.quaternion, dtype=np.float64), (steps, 1))
  return np.column_stack((positions, quaternions))


def _explicit_path(path: Any, target: ObjectConfig, steps: int) -> np.ndarray:
  try:
    untyped = np.asarray(path)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("factual_path must be a numeric array") from error
  if untyped.shape != (steps, 7):
    raise ValueError(
        "factual_path shape must match scene duration ({}, 7)".format(steps)
    )
  validate_path(path)
  result = np.array(untyped, dtype=np.float64, copy=True)
  if not np.allclose(
      result[0, :3], target.position, rtol=0.0, atol=_INITIAL_TOLERANCE
  ):
    raise ValueError("factual_path initial position must match target config")
  initial_quaternion = result[0, 3:7]
  configured_quaternion = np.asarray(target.quaternion, dtype=np.float64)
  if not (
      np.allclose(
          initial_quaternion,
          configured_quaternion,
          rtol=0.0,
          atol=_INITIAL_TOLERANCE,
      )
      or np.allclose(
          initial_quaternion,
          -configured_quaternion,
          rtol=0.0,
          atol=_INITIAL_TOLERANCE,
      )
  ):
    raise ValueError("factual_path initial quaternion must match target config")
  return result


def _prepare_paths(
    scene_config: SceneConfig,
    target: ObjectConfig,
    intervention: Intervention,
    rng_seed: int,
    factual_path: Any,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
  steps = _duration_steps(scene_config)
  window = _integer_window(intervention, steps)
  if intervention.magnitude > 0.0 and window[1] - window[0] < 3:
    raise ValueError("a nonzero intervention window requires at least three samples")
  aabbs = _static_aabbs(scene_config, target.object_id)
  if factual_path is None:
    factual = _default_path(target, steps, scene_config.step_rate)
  else:
    factual = _explicit_path(factual_path, target, steps)
  validate_path(factual)
  _validate_path_float32(factual, scene_config.step_rate)
  _validate_target_sweep(factual, target, scene_config.scene_bounds, aabbs)

  start, end = window
  counterfactual = np.array(factual, dtype=np.float64, copy=True)
  if intervention.magnitude > 0.0:
    rng = np.random.default_rng(
        np.random.SeedSequence([scene_config.seed, rng_seed])
    )
    for _ in range(_PERTURB_ATTEMPTS):
      try:
        candidate = perturb_path(
            factual[start:end],
            intervention.recipe,
            intervention.magnitude,
            rng,
            bounds=scene_config.scene_bounds,
            static_aabbs=aabbs,
            clearance=0.0,
            max_attempts=1,
        )
      except ValueError:
        continue
      trial = np.array(factual, dtype=np.float64, copy=True)
      trial[start:end] = candidate
      try:
        _validate_path_float32(trial, scene_config.step_rate)
        _validate_target_sweep(
            trial, target, scene_config.scene_bounds, aabbs
        )
      except ValueError:
        continue
      counterfactual = trial
      break
    else:
      raise ValueError(
          "unable to sample a valid target-volume perturbation within {} attempts".format(
              _PERTURB_ATTEMPTS
          )
      )
  validate_path(counterfactual)
  _validate_path_float32(counterfactual, scene_config.step_rate)
  _validate_target_sweep(
      counterfactual, target, scene_config.scene_bounds, aabbs
  )
  if not np.array_equal(counterfactual[:start], factual[:start]) or not np.array_equal(
      counterfactual[end:], factual[end:]
  ):
    raise RuntimeError("trajectory perturbation escaped the intervention window")
  return factual, counterfactual, window


def _build_scene(
    scene_config: SceneConfig,
) -> Tuple[kb.Scene, Dict[str, kb.PhysicalObject]]:
  frame_start, frame_end = scene_config.frame_range
  scene = kb.Scene(
      frame_start=frame_start,
      frame_end=frame_end - 1,
      frame_rate=scene_config.frame_rate,
      step_rate=scene_config.step_rate,
      gravity=scene_config.gravity,
  )
  assets: Dict[str, kb.PhysicalObject] = {}
  for item in scene_config.objects:
    metadata = dict(item.metadata)
    metadata["logical_id"] = item.object_id
    constructor = kb.Cube if item.shape == "cube" else kb.Sphere
    asset = constructor(
        name=item.object_id,
        scale=item.size,
        mass=item.mass,
        friction=item.friction,
        restitution=item.restitution,
        position=item.position,
        quaternion=item.quaternion,
        velocity=item.linear_velocity,
        angular_velocity=item.angular_velocity,
        static=item.static,
        metadata=metadata,
    )
    scene += asset
    assets[item.object_id] = asset
  if scene_config.camera is not None:
    camera = kb.PerspectiveCamera(
        position=scene_config.camera.position,
        look_at=scene_config.camera.look_at,
        focal_length=scene_config.camera.focal_length,
    )
    scene.camera = camera
  return scene, assets


def _configure_physics(
    simulator: KinematicDragSimulator,
    scene_config: SceneConfig,
    assets: Mapping[str, kb.PhysicalObject],
) -> None:
  for item in scene_config.objects:
    asset = assets[item.object_id]
    body = asset.linked_objects[simulator]
    simulator.bullet_client.changeDynamics(
        body,
        -1,
        mass=0.0 if item.static else item.mass,
        lateralFriction=item.friction,
        restitution=item.restitution,
    )
    simulator.bullet_client.resetBaseVelocity(
        body,
        linearVelocity=item.linear_velocity,
        angularVelocity=item.angular_velocity,
    )


def _run_branch(
    scene_config: SceneConfig,
    target_id: str,
    path: np.ndarray,
    push_mass: float,
    branch: str,
) -> SimulationLog:
  scene, assets = _build_scene(scene_config)
  with tempfile.TemporaryDirectory(prefix="kubric-twin-") as scratch:
    with KinematicDragSimulator(scene, scratch_dir=Path(scratch)) as simulator:
      _configure_physics(simulator, scene_config, assets)
      return simulator.run_with_intervention(
          assets[target_id],
          path,
          push_mass=push_mass,
          branch=branch,
          start_step=0,
          write_keyframes=False,
      )


def _schema_digest(value: Any) -> str:
  return hashlib.sha256(_canonical_json(value)).hexdigest()


def _with_pair_provenance(
    log: SimulationLog,
    scene_config: SceneConfig,
    intervention: Intervention,
    rng_seed: Optional[int] = None,
) -> SimulationLog:
  metadata = dict(log.metadata)
  metadata.update(
      scene_config_sha256=_schema_digest(scene_config),
      intervention_sha256=_schema_digest(intervention),
  )
  if rng_seed is not None:
    metadata["rng_seed"] = rng_seed
  return SimulationLog(
      branch=log.branch,
      object_ids=log.object_ids,
      steps=log.steps,
      states=log.states,
      contacts=log.contacts,
      step_rate=log.step_rate,
      commanded_path=log.commanded_path,
      metadata=metadata,
      schema_version=log.schema_version,
  )


def _prefix_contacts(log: SimulationLog, start: int):
  return tuple(record for record in log.contacts if record.step < start)


def _enforce_twin_consistency(
    factual: SimulationLog,
    counterfactual: SimulationLog,
    target_id: str,
    intervention_start: int,
) -> None:
  if factual.object_ids != counterfactual.object_ids:
    raise RuntimeError("twin object identifiers differ before comparison")
  if not np.array_equal(
      factual.states[:intervention_start],
      counterfactual.states[:intervention_start],
  ):
    raise RuntimeError("twin states differ before intervention start")
  if _prefix_contacts(factual, intervention_start) != _prefix_contacts(
      counterfactual, intervention_start
  ):
    raise RuntimeError("twin contacts differ before intervention start")
  non_target_indices = tuple(
      index
      for index, object_id in enumerate(factual.object_ids)
      if object_id != target_id
  )
  if factual.states.shape[0] and non_target_indices and not np.array_equal(
      factual.states[0, non_target_indices],
      counterfactual.states[0, non_target_indices],
  ):
    raise RuntimeError("non-target twin state differs at the first logged step")


def generate_paired_instance(
    scene_config: SceneConfig,
    target_id: str,
    intervention: Intervention,
    rng_seed: int,
    *,
    factual_path: Any = None,
) -> Tuple[SimulationLog, SimulationLog]:
  """Runs deterministic factual and counterfactual worlds in fresh clients."""
  seed = _rng_seed(rng_seed)
  target = _target_config(scene_config, target_id, intervention)
  _validate_mapping(scene_config)
  factual_path_value, counterfactual_path, window = _prepare_paths(
      scene_config, target, intervention, seed, factual_path
  )

  factual = _run_branch(
      scene_config,
      target_id,
      factual_path_value,
      intervention.push_mass,
      "factual",
  )
  counterfactual = _run_branch(
      scene_config,
      target_id,
      counterfactual_path,
      intervention.push_mass,
      "counterfactual",
  )
  factual = _with_pair_provenance(factual, scene_config, intervention)
  counterfactual = _with_pair_provenance(
      counterfactual, scene_config, intervention, seed
  )
  _enforce_twin_consistency(factual, counterfactual, target_id, window[0])
  return factual, counterfactual


def _environment_ids(scene_config: SceneConfig, target_id: str) -> Tuple[str, ...]:
  return tuple(
      item.object_id
      for item in scene_config.objects
      if item.static and item.object_id != target_id
  )


def _validate_log_metadata(
    log: SimulationLog,
    scene_config: SceneConfig,
    intervention: Intervention,
    *,
    rng_seed: Optional[int],
) -> None:
  expected = {
      "target_id": intervention.target_id,
      "push_mass": intervention.push_mass,
      "dt": 1.0 / float(scene_config.step_rate),
      "kinematic_emulation": True,
      "scene_config_sha256": _schema_digest(scene_config),
      "intervention_sha256": _schema_digest(intervention),
  }
  for key, value in expected.items():
    if log.metadata.get(key) != value:
      raise ValueError("log metadata {!r} does not match pair provenance".format(key))
  if log.branch == "counterfactual":
    recorded_seed = log.metadata.get("rng_seed")
    if (
        isinstance(recorded_seed, bool)
        or not isinstance(recorded_seed, numbers.Integral)
        or int(recorded_seed) < 0
    ):
      raise ValueError("counterfactual metadata rng_seed is invalid")
    if rng_seed is not None and int(recorded_seed) != rng_seed:
      raise ValueError("counterfactual rng_seed does not match pair provenance")
  elif "rng_seed" in log.metadata:
    raise ValueError("factual metadata must remain independent of rng_seed")


def _validate_pair_logs(
    scene_config: SceneConfig,
    intervention: Intervention,
    factual_log: SimulationLog,
    counterfactual_log: SimulationLog,
    *,
    rng_seed: Optional[int] = None,
) -> Tuple[ObjectConfig, Tuple[int, int]]:
  if not isinstance(factual_log, SimulationLog) or not isinstance(
      counterfactual_log, SimulationLog
  ):
    raise TypeError("factual_log and counterfactual_log must be SimulationLog values")
  target = _target_config(scene_config, intervention.target_id, intervention)
  steps = _duration_steps(scene_config)
  window = _integer_window(intervention, steps)
  if intervention.magnitude > 0.0 and window[1] - window[0] < 3:
    raise ValueError("a nonzero intervention window requires at least three samples")
  if factual_log.branch != "factual":
    raise ValueError("factual_log branch must be 'factual'")
  if counterfactual_log.branch != "counterfactual":
    raise ValueError("counterfactual_log branch must be 'counterfactual'")
  expected_ids = tuple(sorted(item.object_id for item in scene_config.objects))
  if factual_log.object_ids != expected_ids or counterfactual_log.object_ids != expected_ids:
    raise ValueError("log object_ids/order does not match SceneConfig")
  expected_steps = tuple(range(steps))
  if factual_log.steps != expected_steps or counterfactual_log.steps != expected_steps:
    raise ValueError("log steps do not match SceneConfig duration")
  if (
      factual_log.step_rate != scene_config.step_rate
      or counterfactual_log.step_rate != scene_config.step_rate
  ):
    raise ValueError("log step_rate does not match SceneConfig")
  _validate_log_metadata(
      factual_log, scene_config, intervention, rng_seed=None
  )
  _validate_log_metadata(
      counterfactual_log,
      scene_config,
      intervention,
      rng_seed=rng_seed,
  )
  if factual_log.commanded_path is None or counterfactual_log.commanded_path is None:
    raise ValueError("paired logs require commanded_path values")
  factual_path = _explicit_path(factual_log.commanded_path, target, steps)
  counterfactual_path = np.array(
      counterfactual_log.commanded_path, dtype=np.float64, copy=True
  )
  if counterfactual_path.shape != (steps, 7):
    raise ValueError("counterfactual commanded_path must have shape [T, 7]")
  validate_path(counterfactual_path)
  aabbs = _static_aabbs(scene_config, target.object_id)
  for path in (factual_path, counterfactual_path):
    _validate_path_float32(path, scene_config.step_rate)
    _validate_target_sweep(path, target, scene_config.scene_bounds, aabbs)
  start, end = window
  if not np.array_equal(counterfactual_path[:start], factual_path[:start]) or not np.array_equal(
      counterfactual_path[end:], factual_path[end:]
  ):
    raise ValueError("counterfactual commanded_path differs outside time_window")
  if not np.array_equal(counterfactual_path[start], factual_path[start]) or not np.array_equal(
      counterfactual_path[end - 1], factual_path[end - 1]
  ):
    raise ValueError("counterfactual commanded_path does not preserve window anchors")
  deviation = max_position_deviation(factual_path, counterfactual_path)
  if deviation > intervention.magnitude + 1e-12:
    raise ValueError("counterfactual commanded_path exceeds intervention magnitude")
  if intervention.magnitude == 0.0 and not np.array_equal(
      factual_path, counterfactual_path
  ):
    raise ValueError("zero-magnitude commanded_path values must be identical")
  recorded_seed = int(counterfactual_log.metadata["rng_seed"])
  _, expected_counterfactual, _ = _prepare_paths(
      scene_config,
      target,
      intervention,
      recorded_seed,
      factual_path,
  )
  if not np.array_equal(counterfactual_path, expected_counterfactual):
    raise ValueError(
        "counterfactual commanded_path does not match recipe and rng_seed provenance"
    )
  return target, window


def extract_pair_ground_truth(
    scene_config: SceneConfig,
    intervention: Intervention,
    factual_log: SimulationLog,
    counterfactual_log: SimulationLog,
    **thresholds: Any,
) -> GroundTruth:
  """Extracts causal labels while excluding static environment propagation."""
  if not isinstance(intervention, Intervention):
    raise TypeError("intervention must be an Intervention")
  target, window = _validate_pair_logs(
      scene_config, intervention, factual_log, counterfactual_log
  )
  start = window[0]
  reserved = {"target_id", "intervention_start", "exclude_nodes"}
  conflicts = reserved.intersection(thresholds)
  if conflicts:
    raise TypeError(
        "thresholds cannot override {}".format(", ".join(sorted(conflicts)))
    )
  physically_identical = (
      factual_log.object_ids == counterfactual_log.object_ids
      and factual_log.steps == counterfactual_log.steps
      and factual_log.step_rate == counterfactual_log.step_rate
      and np.array_equal(factual_log.states, counterfactual_log.states)
      and factual_log.contacts == counterfactual_log.contacts
  )
  truth = extract_ground_truth(
      factual_log,
      counterfactual_log,
      target.object_id,
      intervention_start=start,
      exclude_nodes=_environment_ids(scene_config, target.object_id),
      **thresholds,
  )
  if physically_identical:
    return GroundTruth(graph_delta=GraphEdgeDelta())
  return truth


def _canonical_json(value: Any) -> bytes:
  return (
      json.dumps(
          to_jsonable(value),
          sort_keys=True,
          separators=(",", ":"),
          ensure_ascii=False,
          allow_nan=False,
      )
      + "\n"
  ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
  with path.open("xb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())


def _path_exists(path: Path) -> bool:
  return os.path.lexists(path)


def _write_temp(directory: Path, prefix: str, payload: bytes) -> Path:
  with tempfile.NamedTemporaryFile(
      mode="wb", prefix=prefix, dir=str(directory), delete=False
  ) as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    return Path(stream.name)


def _fsync_directory(directory: Path) -> None:
  flags = os.O_RDONLY
  if hasattr(os, "O_DIRECTORY"):
    flags |= os.O_DIRECTORY
  descriptor = os.open(str(directory), flags)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


@contextmanager
def _pair_publisher_lock(target: Path):
  """Serializes pair-pointer updates by locking the artifact parent inode."""
  if fcntl is None:
    raise RuntimeError("paired artifact publication requires advisory locking")
  flags = os.O_RDONLY
  if hasattr(os, "O_DIRECTORY"):
    flags |= os.O_DIRECTORY
  descriptor = os.open(str(target.parent), flags)
  try:
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    yield
  finally:
    try:
      fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
      os.close(descriptor)


def _generation_file_records(directory: Path) -> Mapping[str, Mapping[str, Any]]:
  records = {}
  for path in sorted(directory.rglob("*")):
    if path.is_symlink():
      raise ValueError("paired artifact generations cannot contain symbolic links")
    if not path.is_file() or path.name == ".publish.lock":
      continue
    relative = path.relative_to(directory).as_posix()
    payload = path.read_bytes()
    records[relative] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
  if not records:
    raise ValueError("paired artifact generation has no payload files")
  return records


def _pair_manifest(directory: Path) -> Mapping[str, Any]:
  bare_records = _generation_file_records(directory)
  generation = hashlib.sha256(_canonical_json(bare_records)).hexdigest()
  records = {
      relative: {
          **record,
          "path": "{}/{}/{}".format(
              _PAIR_GENERATIONS, generation, relative
          ),
      }
      for relative, record in bare_records.items()
  }
  return {
      "files": records,
      "generation": generation,
      "schema_version": SCHEMA_VERSION,
  }


def _validate_pair_manifest(target: Path) -> Mapping[str, Any]:
  try:
    payload = json.loads((target / _PAIR_MANIFEST).read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError("paired artifact has an invalid or missing manifest.json") from error
  if not isinstance(payload, dict) or set(payload) != {
      "files",
      "generation",
      "schema_version",
  }:
    raise ValueError("paired artifact manifest has unexpected fields")
  generation = payload["generation"]
  files = payload["files"]
  if (
      payload["schema_version"] != SCHEMA_VERSION
      or not isinstance(generation, str)
      or len(generation) != 64
      or any(character not in "0123456789abcdef" for character in generation)
      or not isinstance(files, dict)
      or not files
  ):
    raise ValueError("paired artifact manifest is malformed")
  bare_records = {}
  for relative, record in files.items():
    parts = Path(relative).parts if isinstance(relative, str) else ()
    expected_path = "{}/{}/{}".format(
        _PAIR_GENERATIONS, generation, relative
    )
    if (
        not parts
        or Path(relative).is_absolute()
        or ".." in parts
        or not isinstance(record, dict)
        or set(record) != {"path", "sha256", "size"}
        or record.get("path") != expected_path
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in record["sha256"]
        )
        or isinstance(record.get("size"), bool)
        or not isinstance(record.get("size"), numbers.Integral)
        or record["size"] < 0
    ):
      raise ValueError("paired artifact manifest file record is malformed")
    try:
      content = (target / expected_path).read_bytes()
    except OSError as error:
      raise ValueError("paired artifact generation is incomplete") from error
    if (
        len(content) != int(record["size"])
        or hashlib.sha256(content).hexdigest() != record["sha256"]
    ):
      raise ValueError("paired artifact generation integrity mismatch")
    bare_records[relative] = {
        "sha256": record["sha256"],
        "size": int(record["size"]),
    }
  if hashlib.sha256(_canonical_json(bare_records)).hexdigest() != generation:
    raise ValueError("paired artifact generation hash mismatch")
  generation_directory = target / _PAIR_GENERATIONS / generation
  actual_files = set(_generation_file_records(generation_directory))
  if actual_files != set(files):
    raise ValueError("paired artifact generation has unexpected files")
  return payload


def _verify_equal_generation(existing: Path, staged: Path) -> None:
  existing_records = _generation_file_records(existing)
  staged_records = _generation_file_records(staged)
  if existing_records != staged_records:
    raise ValueError("paired artifact generation hash collision")
  for relative in staged_records:
    if (existing / relative).read_bytes() != (staged / relative).read_bytes():
      raise ValueError("paired artifact generation hash collision")


def _publish_pair_generation(
    staged_generation: Path,
    target: Path,
    manifest: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
  manifest_payload = _canonical_json(manifest)
  with _pair_publisher_lock(target):
    existing = _path_exists(target)
    if existing and (target.is_symlink() or not target.is_dir()):
      raise ValueError("paired artifact target must be a real directory")
    if existing and not overwrite:
      raise FileExistsError("paired artifact already exists: {}".format(target))

    generation = manifest["generation"]
    if not existing:
      root_staging = Path(
          tempfile.mkdtemp(
              prefix=".{}.root-".format(target.name), dir=target.parent
          )
      )
      try:
        generations = root_staging / _PAIR_GENERATIONS
        generations.mkdir()
        final_generation = generations / generation
        os.rename(staged_generation, final_generation)
        _fsync_directory(final_generation)
        _fsync_directory(generations)
        _write_bytes(root_staging / _PAIR_MANIFEST, manifest_payload)
        _fsync_directory(root_staging)
        os.rename(root_staging, target)
        _fsync_directory(target)
        _fsync_directory(target.parent)
      finally:
        if root_staging.exists():
          shutil.rmtree(root_staging, ignore_errors=True)
      return

    _validate_pair_manifest(target)
    generations = target / _PAIR_GENERATIONS
    final_generation = generations / generation
    if final_generation.exists():
      _verify_equal_generation(final_generation, staged_generation)
      shutil.rmtree(staged_generation)
    else:
      os.rename(staged_generation, final_generation)
    _fsync_directory(final_generation)
    _fsync_directory(generations)
    temporary_manifest = _write_temp(target, ".manifest-", manifest_payload)
    try:
      os.replace(temporary_manifest, target / _PAIR_MANIFEST)
      temporary_manifest = None
      _fsync_directory(target)
    finally:
      if temporary_manifest is not None:
        try:
          temporary_manifest.unlink()
        except FileNotFoundError:
          pass


def _normalized_threshold_inputs(thresholds: Mapping[str, Any]) -> Dict[str, Any]:
  unknown = set(thresholds) - set(_EXTRACTION_THRESHOLDS)
  if unknown:
    raise TypeError(
        "unknown extraction thresholds: {}".format(", ".join(sorted(unknown)))
    )
  result = dict(_EXTRACTION_THRESHOLDS)
  result.update(thresholds)
  return result


def _read_json_object(path: Path, name: str) -> Dict[str, Any]:
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError("{} is missing or invalid JSON".format(name)) from error
  if not isinstance(payload, dict):
    raise ValueError("{} must contain a JSON object".format(name))
  return payload


def _scene_from_payload(payload: Any) -> SceneConfig:
  if not isinstance(payload, Mapping):
    raise ValueError("pair scene_config provenance is malformed")
  values = dict(payload)
  try:
    object_payloads = values.pop("objects")
    camera_payload = values.pop("camera")
    objects = tuple(ObjectConfig(**dict(item)) for item in object_payloads)
    camera = (
        None
        if camera_payload is None
        else CameraConfig(**dict(camera_payload))
    )
    return SceneConfig(objects=objects, camera=camera, **values)
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError("pair scene_config provenance is malformed") from error


def _intervention_from_payload(payload: Any) -> Intervention:
  if not isinstance(payload, Mapping):
    raise ValueError("pair intervention provenance is malformed")
  try:
    return Intervention(**dict(payload))
  except (TypeError, ValueError) as error:
    raise ValueError("pair intervention provenance is malformed") from error


def _ground_truth_from_payload(payload: Mapping[str, Any]) -> GroundTruth:
  try:
    values = dict(payload)
    delta_payload = dict(values.pop("graph_delta"))
    delta = GraphEdgeDelta(**delta_payload)
    return GroundTruth(graph_delta=delta, **values)
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError("ground_truth.json is malformed") from error


def read_paired_artifact(
    directory: PathLike,
) -> Tuple[SimulationLog, SimulationLog, GroundTruth, Mapping[str, Any]]:
  """Reads and integrity-validates the generation selected by ``manifest.json``."""
  source = Path(directory)
  manifest = _validate_pair_manifest(source)
  generation = source / _PAIR_GENERATIONS / manifest["generation"]
  pair_payload = _read_json_object(generation / "pair.json", "pair.json")
  expected_pair_keys = {
      "schema_version",
      "target_id",
      "rng_seed",
      "scene_config",
      "intervention",
      "factual",
      "counterfactual",
      "tags",
      "extraction_thresholds",
  }
  if set(pair_payload) != expected_pair_keys:
    raise ValueError("pair.json has missing or unexpected fields")
  if (
      pair_payload["schema_version"] != SCHEMA_VERSION
      or pair_payload["factual"] != "factual"
      or pair_payload["counterfactual"] != "counterfactual"
  ):
    raise ValueError("pair.json branch or schema provenance is invalid")
  scene_config = _scene_from_payload(pair_payload["scene_config"])
  intervention = _intervention_from_payload(pair_payload["intervention"])
  seed = _rng_seed(pair_payload["rng_seed"])
  if pair_payload["target_id"] != intervention.target_id:
    raise ValueError("pair.json target_id does not match intervention")
  thresholds = pair_payload["extraction_thresholds"]
  if not isinstance(thresholds, Mapping) or set(thresholds) != set(
      _EXTRACTION_THRESHOLDS
  ):
    raise ValueError("pair.json extraction_thresholds are incomplete")

  factual = read_simulation_log(generation / "factual")
  counterfactual = read_simulation_log(generation / "counterfactual")
  _validate_pair_logs(
      scene_config,
      intervention,
      factual,
      counterfactual,
      rng_seed=seed,
  )
  truth_payload = _read_json_object(
      generation / "ground_truth.json", "ground_truth.json"
  )
  truth = _ground_truth_from_payload(truth_payload)
  expected_truth = extract_pair_ground_truth(
      scene_config,
      intervention,
      factual,
      counterfactual,
      **dict(thresholds),
  )
  if truth != expected_truth:
    raise ValueError("ground_truth.json does not match logs and thresholds")
  environment_ids = _environment_ids(scene_config, intervention.target_id)
  expected_tags = derive_tags(
      truth,
      target_id=intervention.target_id,
      environment_ids=environment_ids,
  )
  if pair_payload["tags"] != list(expected_tags):
    raise ValueError("pair.json tags do not match ground truth")
  return factual, counterfactual, truth, pair_payload


def write_paired_artifact(
    directory: PathLike,
    scene_config: SceneConfig,
    intervention: Intervention,
    rng_seed: int,
    factual_log: SimulationLog,
    counterfactual_log: SimulationLog,
    *,
    overwrite: bool = False,
    **thresholds: Any,
) -> GroundTruth:
  """Publishes an immutable pair generation through an atomic manifest pointer."""
  if not isinstance(overwrite, bool):
    raise TypeError("overwrite must be a bool")
  seed = _rng_seed(rng_seed)
  target = Path(directory)
  if _path_exists(target) and not overwrite:
    raise FileExistsError("paired artifact already exists: {}".format(target))
  if _path_exists(target) and (target.is_symlink() or not target.is_dir()):
    raise ValueError("overwrite supports existing real directories only")
  _validate_pair_logs(
      scene_config,
      intervention,
      factual_log,
      counterfactual_log,
      rng_seed=seed,
  )
  normalized_inputs = _normalized_threshold_inputs(thresholds)
  truth = extract_pair_ground_truth(
      scene_config,
      intervention,
      factual_log,
      counterfactual_log,
      **normalized_inputs,
  )
  normalized_thresholds = {
      key: float(value) for key, value in normalized_inputs.items()
  }
  environment_ids = _environment_ids(scene_config, intervention.target_id)
  tags = derive_tags(
      truth,
      target_id=intervention.target_id,
      environment_ids=environment_ids,
  )
  pair_payload = {
      "schema_version": SCHEMA_VERSION,
      "target_id": intervention.target_id,
      "rng_seed": seed,
      "scene_config": scene_config,
      "intervention": intervention,
      "factual": "factual",
      "counterfactual": "counterfactual",
      "tags": tags,
      "extraction_thresholds": normalized_thresholds,
  }

  target.parent.mkdir(parents=True, exist_ok=True)
  staged_generation = Path(
      tempfile.mkdtemp(
          prefix=".{}.generation-".format(target.name), dir=target.parent
      )
  )
  try:
    write_simulation_log(factual_log, staged_generation / "factual")
    write_simulation_log(
        counterfactual_log, staged_generation / "counterfactual"
    )
    _write_bytes(
        staged_generation / "ground_truth.json", _canonical_json(truth)
    )
    _write_bytes(staged_generation / "pair.json", _canonical_json(pair_payload))
    _fsync_directory(staged_generation)
    manifest = _pair_manifest(staged_generation)
    _publish_pair_generation(
        staged_generation, target, manifest, overwrite=overwrite
    )
  finally:
    if staged_generation.exists():
      shutil.rmtree(staged_generation, ignore_errors=True)
  return truth


__all__ = [
    "extract_pair_ground_truth",
    "generate_paired_instance",
    "read_paired_artifact",
    "write_paired_artifact",
]
