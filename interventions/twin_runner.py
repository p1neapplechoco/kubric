"""Deterministic factual/counterfactual execution from backend-neutral schemas.

``ObjectConfig.size`` maps directly to Kubric's ``scale``.  Kubric primitives use
unit half-extents, so a cube configured with size ``s`` has physical width ``2*s``.
"""

from __future__ import annotations

import json
import numbers
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

import kubric as kb

from interventions.graph_extraction import extract_ground_truth
from interventions.kinematic_simulator import KinematicDragSimulator
from interventions.logging import SimulationLog, write_simulation_log
from interventions.schema import (
    SCHEMA_VERSION,
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    ObjectConfig,
    SceneConfig,
    to_jsonable,
)
from interventions.tagging import derive_tags
from interventions.trajectory import perturb_path, validate_path


PathLike = Union[str, os.PathLike[str]]
_INITIAL_TOLERANCE = 1e-9


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
  for item in scene_config.objects:
    if item.shape == "sphere" and not (
        item.size[0] == item.size[1] == item.size[2]
    ):
      raise ValueError(
          "sphere {!r} requires a uniform size".format(item.object_id)
      )
    logical_id = item.metadata.get("logical_id")
    if logical_id is not None and logical_id != item.object_id:
      raise ValueError(
          "metadata logical_id for {!r} conflicts with object_id".format(
              item.object_id
          )
      )


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
  aabbs = _static_aabbs(scene_config, target.object_id)
  if factual_path is None:
    factual = _default_path(target, steps, scene_config.step_rate)
  else:
    factual = _explicit_path(factual_path, target, steps)
  validate_path(
      factual,
      bounds=scene_config.scene_bounds,
      static_aabbs=aabbs,
      clearance=0.0,
  )

  start, end = window
  counterfactual = np.array(factual, dtype=np.float64, copy=True)
  if end - start >= 2:
    rng = np.random.default_rng(
        np.random.SeedSequence([scene_config.seed, rng_seed])
    )
    counterfactual[start:end] = perturb_path(
        factual[start:end],
        intervention.recipe,
        intervention.magnitude,
        rng,
        bounds=scene_config.scene_bounds,
        static_aabbs=aabbs,
        clearance=0.0,
    )
  validate_path(
      counterfactual,
      bounds=scene_config.scene_bounds,
      static_aabbs=aabbs,
      clearance=0.0,
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
  _enforce_twin_consistency(factual, counterfactual, target_id, window[0])
  return factual, counterfactual


def _environment_ids(scene_config: SceneConfig, target_id: str) -> Tuple[str, ...]:
  return tuple(
      item.object_id
      for item in scene_config.objects
      if item.static and item.object_id != target_id
  )


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
  target = _target_config(scene_config, intervention.target_id, intervention)
  start, _ = _integer_window(intervention, _duration_steps(scene_config))
  if not isinstance(factual_log, SimulationLog) or not isinstance(
      counterfactual_log, SimulationLog
  ):
    raise TypeError("factual_log and counterfactual_log must be SimulationLog values")
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


def _publish_directory(staging: Path, target: Path, overwrite: bool) -> None:
  existing = _path_exists(target)
  if existing and not overwrite:
    raise FileExistsError("paired artifact already exists: {}".format(target))
  if existing and (target.is_symlink() or not target.is_dir()):
    raise ValueError("overwrite supports existing real directories only")

  backup: Optional[Path] = None
  if existing:
    backup = target.parent / ".{}.backup-{}".format(target.name, uuid.uuid4().hex)
    os.replace(target, backup)
  try:
    os.replace(staging, target)
  except BaseException:
    if backup is not None and _path_exists(backup) and not _path_exists(target):
      os.replace(backup, target)
    raise
  if backup is not None:
    shutil.rmtree(backup, ignore_errors=True)


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
  """Atomically publishes nested logs plus canonical labels and provenance.

  With ``overwrite=True`` an existing real directory is first moved to a sibling
  backup and restored if publication fails. Files and symbolic links are rejected.
  """
  if not isinstance(overwrite, bool):
    raise TypeError("overwrite must be a bool")
  seed = _rng_seed(rng_seed)
  target = Path(directory)
  if _path_exists(target) and not overwrite:
    raise FileExistsError("paired artifact already exists: {}".format(target))
  if _path_exists(target) and (target.is_symlink() or not target.is_dir()):
    raise ValueError("overwrite supports existing real directories only")
  truth = extract_pair_ground_truth(
      scene_config,
      intervention,
      factual_log,
      counterfactual_log,
      **thresholds,
  )
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
  }

  target.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(
      tempfile.mkdtemp(prefix=".{}.tmp-".format(target.name), dir=target.parent)
  )
  published = False
  try:
    write_simulation_log(factual_log, staging / "factual")
    write_simulation_log(counterfactual_log, staging / "counterfactual")
    _write_bytes(staging / "ground_truth.json", _canonical_json(truth))
    _write_bytes(staging / "pair.json", _canonical_json(pair_payload))
    _publish_directory(staging, target, overwrite)
    published = True
  finally:
    if not published and staging.exists():
      shutil.rmtree(staging, ignore_errors=True)
  return truth


__all__ = [
    "extract_pair_ground_truth",
    "generate_paired_instance",
    "write_paired_artifact",
]
