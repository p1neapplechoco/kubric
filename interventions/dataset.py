"""Deterministic sampling, quality control, and dataset publication.

The module keeps sampling and post-processing independent from simulator state.
Batch generation is deliberately single-worker: each paired candidate owns fresh
Bullet clients through :mod:`interventions.twin_runner`.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import shutil
import tempfile
import traceback
import uuid
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from interventions.logging import (
    ANGULAR_VELOCITY_SLICE,
    LINEAR_VELOCITY_SLICE,
    SimulationLog,
)
from interventions.schema import (
    GroundTruth,
    Intervention,
    ObjectConfig,
    SceneConfig,
    to_jsonable,
)
from interventions.trajectory import build_path
from interventions.twin_runner import (
    extract_pair_ground_truth,
    generate_paired_instance,
    write_paired_artifact,
)


PathLike = Union[str, os.PathLike[str]]
_EXPECTED_EFFECTS = frozenset(("non_null", "null"))
_HOP_BUCKETS = ("0", "1", "2", "3+")
_SPLITS = ("train", "val", "test")


def _identifier(value: Any, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if not value.strip():
    raise ValueError("{} must not be empty".format(name))
  return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, numbers.Integral):
    raise TypeError("{} must be an integer".format(name))
  result = int(value)
  if result < minimum:
    raise ValueError("{} must be at least {}".format(name, minimum))
  return result


def _finite(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise TypeError("{} must be a real number".format(name))
  result = float(value)
  if not math.isfinite(result):
    raise ValueError("{} must be finite".format(name))
  return result


def _freeze(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
  if isinstance(value, (tuple, list)):
    return tuple(_freeze(item) for item in value)
  if value is None or isinstance(value, (str, bool, numbers.Number)):
    return value
  raise TypeError("unsupported immutable value: {!r}".format(type(value).__name__))


def _canonical_bytes(value: Any) -> bytes:
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


def _readonly_path(value: Any) -> np.ndarray:
  try:
    array = np.asarray(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("factual_path must be a numeric array") from error
  if array.shape[1:] != (7,) or array.ndim != 2 or array.shape[0] < 2:
    raise ValueError("factual_path must have shape [T, 7] with T >= 2")
  if array.dtype.kind not in {"i", "u", "f"}:
    raise ValueError("factual_path must contain real numeric values")
  result = np.array(array, dtype=np.float64, order="C", copy=True)
  if not np.isfinite(result).all():
    raise ValueError("factual_path must be finite")
  norms = np.linalg.norm(result[:, 3:7], axis=1)
  if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-6):
    raise ValueError("factual_path quaternions must be unit-normalized")
  immutable = result.tobytes(order="C")
  return np.frombuffer(immutable, dtype=np.float64).reshape(result.shape)


@dataclass(frozen=True)
class InstanceSpec:
  """A fully sampled, immutable simulator input."""

  attempt_index: int
  instance_seed: int
  instance_id: str
  scene_config: SceneConfig
  target_id: str
  factual_path: np.ndarray
  intervention: Intervention
  expected_effect: str
  intervention_start_step: int

  def __post_init__(self) -> None:
    attempt = _integer(self.attempt_index, "attempt_index")
    seed = _integer(self.instance_seed, "instance_seed")
    instance_id = _identifier(self.instance_id, "instance_id")
    target_id = _identifier(self.target_id, "target_id")
    if not isinstance(self.scene_config, SceneConfig):
      raise TypeError("scene_config must be a SceneConfig")
    if not isinstance(self.intervention, Intervention):
      raise TypeError("intervention must be an Intervention")
    if target_id != self.intervention.target_id:
      raise ValueError("target_id must match intervention.target_id")
    if target_id not in {item.object_id for item in self.scene_config.objects}:
      raise ValueError("target_id is absent from scene_config")
    expected = _identifier(self.expected_effect, "expected_effect")
    if expected not in _EXPECTED_EFFECTS:
      raise ValueError("expected_effect must be 'non_null' or 'null'")
    start = _integer(self.intervention_start_step, "intervention_start_step")
    if float(self.intervention.time_window[0]) != float(start):
      raise ValueError("intervention_start_step must match intervention.time_window")
    object.__setattr__(self, "attempt_index", attempt)
    object.__setattr__(self, "instance_seed", seed)
    object.__setattr__(self, "instance_id", instance_id)
    object.__setattr__(self, "target_id", target_id)
    object.__setattr__(self, "factual_path", _readonly_path(self.factual_path))
    object.__setattr__(self, "expected_effect", expected)
    object.__setattr__(self, "intervention_start_step", start)

  def to_dict(self) -> Mapping[str, Any]:
    return {
        "attempt_index": self.attempt_index,
        "instance_seed": self.instance_seed,
        "instance_id": self.instance_id,
        "scene_config": self.scene_config.to_dict(),
        "target_id": self.target_id,
        "factual_path": to_jsonable(self.factual_path),
        "intervention": self.intervention.to_dict(),
        "expected_effect": self.expected_effect,
        "intervention_start_step": self.intervention_start_step,
    }


@dataclass(frozen=True)
class QCResult:
  """All quality-control findings for one paired rollout."""

  accepted: bool
  reason_codes: Tuple[str, ...] = ()
  metrics: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not isinstance(self.accepted, bool):
      raise TypeError("accepted must be a bool")
    if isinstance(self.reason_codes, (str, bytes)):
      raise TypeError("reason_codes must be an iterable of strings")
    reasons = tuple(sorted({_identifier(item, "reason code") for item in self.reason_codes}))
    if self.accepted == bool(reasons):
      raise ValueError("accepted must be true exactly when reason_codes is empty")
    if not isinstance(self.metrics, Mapping):
      raise TypeError("metrics must be a mapping")
    metrics = _freeze(to_jsonable(self.metrics))
    object.__setattr__(self, "reason_codes", reasons)
    object.__setattr__(self, "metrics", metrics)

  def to_dict(self) -> Mapping[str, Any]:
    return {
        "accepted": self.accepted,
        "reason_codes": list(self.reason_codes),
        "metrics": to_jsonable(self.metrics),
    }


@dataclass(frozen=True)
class CandidateSummary:
  """Small journal record used for balancing and grouped splitting."""

  instance_id: str
  attempt_index: int
  category: str
  hop_depth: int
  hop_bucket: str
  topology_signature: str
  artifact_path: str

  def __post_init__(self) -> None:
    instance_id = _identifier(self.instance_id, "instance_id")
    attempt = _integer(self.attempt_index, "attempt_index")
    category = _identifier(self.category, "category")
    depth = _integer(self.hop_depth, "hop_depth")
    bucket = _identifier(self.hop_bucket, "hop_bucket")
    expected_bucket = "3+" if depth >= 3 else str(depth)
    if bucket not in _HOP_BUCKETS or bucket != expected_bucket:
      raise ValueError("hop_bucket does not match hop_depth")
    signature = _identifier(self.topology_signature, "topology_signature")
    artifact = _identifier(self.artifact_path, "artifact_path")
    object.__setattr__(self, "instance_id", instance_id)
    object.__setattr__(self, "attempt_index", attempt)
    object.__setattr__(self, "category", category)
    object.__setattr__(self, "hop_depth", depth)
    object.__setattr__(self, "hop_bucket", bucket)
    object.__setattr__(self, "topology_signature", signature)
    object.__setattr__(self, "artifact_path", artifact)

  def to_dict(self) -> Mapping[str, Any]:
    return {
        "instance_id": self.instance_id,
        "attempt_index": self.attempt_index,
        "category": self.category,
        "hop_depth": self.hop_depth,
        "hop_bucket": self.hop_bucket,
        "topology_signature": self.topology_signature,
        "artifact_path": self.artifact_path,
    }


def load_ranges(path: PathLike) -> Mapping[str, Any]:
  """Loads a YAML range document into recursively immutable containers."""
  try:
    import yaml
  except ImportError as error:  # pragma: no cover - environment dependency.
    raise ImportError("loading dataset ranges requires PyYAML") from error
  source = Path(path)
  try:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
  except OSError as error:
    raise ValueError("cannot read range config: {}".format(source)) from error
  except yaml.YAMLError as error:
    raise ValueError("invalid YAML range config") from error
  if not isinstance(payload, Mapping):
    raise ValueError("range config must contain a YAML mapping")
  return _freeze(to_jsonable(payload))


def derive_seed(master_seed: int, index: int, domain: str) -> int:
  """Derives a reproducible unsigned 63-bit seed with SHA256 separation."""
  master = _integer(master_seed, "master_seed")
  attempt = _integer(index, "index")
  domain = _identifier(domain, "domain")
  material = "{}\0{}\0{}".format(master, attempt, domain).encode("utf-8")
  return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _section(ranges: Mapping[str, Any], name: str) -> Mapping[str, Any]:
  try:
    result = ranges[name]
  except (KeyError, TypeError) as error:
    raise ValueError("range config is missing {!r}".format(name)) from error
  if not isinstance(result, Mapping):
    raise ValueError("{} must be a mapping".format(name))
  return result


def _pair(section: Mapping[str, Any], key: str, *, integer: bool = False,
          minimum: Optional[float] = None) -> Tuple[float, float]:
  try:
    values = tuple(section[key])
  except (KeyError, TypeError) as error:
    raise ValueError("{} must be a two-value range".format(key)) from error
  if len(values) != 2:
    raise ValueError("{} must be a two-value range".format(key))
  if integer:
    low = _integer(values[0], "{}[0]".format(key), minimum=0)
    high = _integer(values[1], "{}[1]".format(key), minimum=0)
  else:
    low = _finite(values[0], "{}[0]".format(key))
    high = _finite(values[1], "{}[1]".format(key))
  if low > high:
    raise ValueError("{} range minimum exceeds maximum".format(key))
  if minimum is not None and low < minimum:
    raise ValueError("{} range values must be at least {}".format(key, minimum))
  return low, high


def _unit_pair(
    section: Mapping[str, Any], key: str, *, minimum: float = 0.0
) -> Tuple[float, float]:
  result = _pair(section, key, minimum=minimum)
  if result[1] > 1.0:
    raise ValueError("{} range values must not exceed 1".format(key))
  return result


def _sample_float(rng: np.random.Generator, pair: Tuple[float, float]) -> float:
  low, high = pair
  if low == high:
    return float(low)
  return float(rng.uniform(low, high))


def _sample_int(rng: np.random.Generator, pair: Tuple[float, float]) -> int:
  low, high = (int(value) for value in pair)
  if low == high:
    return low
  return int(rng.integers(low, high + 1))


def _choice(rng: np.random.Generator, values: Any, name: str) -> Any:
  if isinstance(values, (str, bytes)):
    raise ValueError("{} must be a nonempty sequence".format(name))
  try:
    options = tuple(values)
  except TypeError as error:
    raise ValueError("{} must be a nonempty sequence".format(name)) from error
  if not options:
    raise ValueError("{} must be a nonempty sequence".format(name))
  return options[int(rng.integers(0, len(options)))]


def _overlap(
    position_a: Sequence[float], extent_a: Sequence[float],
    position_b: Sequence[float], extent_b: Sequence[float],
) -> bool:
  return bool(np.all(
      np.abs(np.asarray(position_a) - np.asarray(position_b))
      < np.asarray(extent_a) + np.asarray(extent_b)
  ))


def _duration_steps(scene: Mapping[str, Any]) -> int:
  frame_range = tuple(scene.get("frame_range", (0, 1)))
  if len(frame_range) != 2:
    raise ValueError("frame_range must contain start and end")
  frame_start = _integer(frame_range[0], "frame_range[0]")
  frame_end = _integer(frame_range[1], "frame_range[1]")
  frame_rate = _integer(scene.get("frame_rate", 24), "frame_rate", minimum=1)
  step_rate = _integer(scene.get("step_rate", 240), "step_rate", minimum=1)
  if frame_end <= frame_start or step_rate % frame_rate:
    raise ValueError("scene frame/step rates produce an invalid duration")
  result = (frame_end - frame_start) * step_rate // frame_rate
  if result < 2:
    raise ValueError("scene duration must contain at least two steps")
  return result


def sample_instance_spec(
    ranges: Mapping[str, Any], master_seed: int, index: int
) -> InstanceSpec:
  """Samples one valid spec without touching NumPy's global RNG."""
  if not isinstance(ranges, Mapping):
    raise TypeError("ranges must be a mapping")
  master = _integer(master_seed, "master_seed")
  attempt = _integer(index, "index")
  data = to_jsonable(ranges)
  scene_ranges = _section(data, "scene")
  object_ranges = _section(data, "objects")
  target_ranges = _section(data, "target")
  trajectory_ranges = _section(data, "trajectory")
  intervention_ranges = _section(data, "intervention")
  rng = np.random.default_rng(derive_seed(master, attempt, "sampling"))

  bounds = np.asarray(scene_ranges.get("bounds"), dtype=np.float64)
  if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
    raise ValueError("scene bounds must have shape [2, 3] with increasing limits")
  steps = _duration_steps(scene_ranges)
  frame_range = tuple(scene_ranges.get("frame_range", (0, 1)))
  frame_rate = _integer(scene_ranges.get("frame_rate", 24), "frame_rate", minimum=1)
  step_rate = _integer(scene_ranges.get("step_rate", 240), "step_rate", minimum=1)
  gravity = tuple(scene_ranges.get("gravity", (0, 0, -9.81)))

  floor_size = tuple(_finite(value, "floor_size") for value in object_ranges.get(
      "floor_size", (4.0, 4.0, 0.25)
  ))
  if len(floor_size) != 3 or any(value <= 0 for value in floor_size):
    raise ValueError("floor_size must contain three positive values")
  floor = ObjectConfig(
      "floor", "cube", size=floor_size, mass=0.0, static=True,
      position=(0.0, 0.0, -floor_size[2]),
      metadata={"role": "environment", "qc_clip_exempt": True},
  )

  target_size_range = _pair(target_ranges, "size", minimum=1e-12)
  target_x_range = _pair(target_ranges, "x")
  target_y_range = _pair(target_ranges, "y")
  displacement_x_range = _pair(target_ranges, "displacement_x")
  displacement_y_range = _pair(target_ranges, "displacement_y")
  target_friction_range = _unit_pair(target_ranges, "friction")
  target_restitution_range = _unit_pair(target_ranges, "restitution")
  target_extent = target_size_range[1]
  possible_centers = (
      (
          min(target_x_range[0], target_x_range[0] + displacement_x_range[0]),
          max(target_x_range[1], target_x_range[1] + displacement_x_range[1]),
      ),
      (
          min(target_y_range[0], target_y_range[0] + displacement_y_range[0]),
          max(target_y_range[1], target_y_range[1] + displacement_y_range[1]),
      ),
  )
  if any(
      center_range[0] - target_extent < bounds[0, axis]
      or center_range[1] + target_extent > bounds[1, axis]
      for axis, center_range in enumerate(possible_centers)
  ) or 0.0 < bounds[0, 2] or 2.0 * target_extent > bounds[1, 2]:
    raise ValueError("target trajectory volume leaves scene bounds")

  target_size = _sample_float(rng, target_size_range)
  target_position = (
      _sample_float(rng, target_x_range),
      _sample_float(rng, target_y_range),
      target_size,
  )
  target_shape = target_ranges.get("shape", "cube")
  target = ObjectConfig(
      "target", target_shape, size=target_size, mass=0.0, static=True,
      position=target_position,
      friction=_sample_float(rng, target_friction_range),
      restitution=_sample_float(rng, target_restitution_range),
      metadata={"kinematic_emulation": True, "role": "target"},
  )
  if np.any(np.asarray(target_position) - target_size < bounds[0]) or np.any(
      np.asarray(target_position) + target_size > bounds[1]
  ):
    raise ValueError("sampled target does not fit within scene bounds")

  object_count = _sample_int(rng, _pair(object_ranges, "count", integer=True))
  object_size_range = _pair(object_ranges, "size", minimum=1e-12)
  mass_range = _pair(object_ranges, "mass", minimum=1e-12)
  friction_range = _unit_pair(object_ranges, "friction")
  restitution_range = _unit_pair(object_ranges, "restitution")
  x_range = _pair(object_ranges, "x")
  y_range = _pair(object_ranges, "y")
  placed = [(target.position, target.size)]
  dynamic_objects = []
  for object_index in range(object_count):
    shape = str(_choice(rng, object_ranges.get("shapes"), "objects.shapes"))
    size = _sample_float(rng, object_size_range)
    position = None
    for _ in range(512):
      candidate = (
          _sample_float(rng, x_range), _sample_float(rng, y_range), size
      )
      extent = (size, size, size)
      if np.any(np.asarray(candidate) - extent < bounds[0]) or np.any(
          np.asarray(candidate) + extent > bounds[1]
      ):
        continue
      if not any(_overlap(candidate, extent, other_position, other_extent)
                 for other_position, other_extent in placed):
        position = candidate
        break
    if position is None:
      raise ValueError("object ranges cannot produce a non-overlapping scene")
    item = ObjectConfig(
        "object_{}".format(object_index), shape, size=size,
        mass=_sample_float(rng, mass_range),
        friction=_sample_float(rng, friction_range),
        restitution=_sample_float(rng, restitution_range),
        position=position,
        metadata={"role": "dynamic"},
    )
    dynamic_objects.append(item)
    placed.append((item.position, item.size))

  scene_seed = derive_seed(master, attempt, "scene")
  scene_config = SceneConfig(
      objects=(floor, target) + tuple(dynamic_objects),
      seed=scene_seed,
      scene_bounds=(tuple(bounds[0]), tuple(bounds[1])),
      gravity=gravity,
      frame_range=frame_range,
      frame_rate=frame_rate,
      step_rate=step_rate,
  )

  displacement = np.asarray((
      _sample_float(rng, displacement_x_range),
      _sample_float(rng, displacement_y_range),
      0.0,
  ))
  waypoint_count = _sample_int(
      rng, _pair(trajectory_ranges, "waypoint_count", integer=True)
  )
  if waypoint_count < 2:
    raise ValueError("waypoint_count must be at least two")
  positions = np.linspace(
      np.asarray(target_position), np.asarray(target_position) + displacement,
      waypoint_count,
  )
  if np.any(positions - target_size < bounds[0]) or np.any(
      positions + target_size > bounds[1]
  ):
    raise ValueError("sampled target trajectory volume leaves scene bounds")
  waypoints = np.column_stack((
      positions,
      np.tile(np.asarray((1.0, 0.0, 0.0, 0.0)), (waypoint_count, 1)),
  ))
  factual_path = build_path(
      waypoints, steps, method=str(trajectory_ranges.get("method", "linear"))
  )

  expected = str(_choice(
      rng, intervention_ranges.get("expected_effects"),
      "intervention.expected_effects",
  ))
  if expected not in _EXPECTED_EFFECTS:
    raise ValueError("expected_effects entries must be 'non_null' or 'null'")
  start = _sample_int(
      rng, _pair(intervention_ranges, "start_step", integer=True)
  )
  duration = _sample_int(
      rng, _pair(intervention_ranges, "duration_steps", integer=True)
  )
  if duration < 1 or start >= steps or start + duration > steps:
    raise ValueError("intervention window must lie inside the sampled trajectory")
  recipe = str(_choice(
      rng, intervention_ranges.get("recipes"), "intervention.recipes"
  ))
  magnitude = _sample_float(
      rng, _pair(intervention_ranges, "magnitude", minimum=0.0)
  )
  push_mass = _sample_float(
      rng, _pair(intervention_ranges, "push_mass", minimum=1e-12)
  )
  intervention = Intervention(
      "target", recipe, magnitude, (start, start + duration),
      metadata={"expected_effect": expected}, push_mass=push_mass,
  )
  instance_seed = derive_seed(master, attempt, "instance")
  identity_payload = {
      "attempt_index": attempt,
      "instance_seed": instance_seed,
      "scene_config": scene_config,
      "factual_path": factual_path,
      "intervention": intervention,
      "expected_effect": expected,
  }
  instance_id = "instance_{}".format(
      hashlib.sha256(_canonical_bytes(identity_payload)).hexdigest()[:20]
  )
  return InstanceSpec(
      attempt, instance_seed, instance_id, scene_config, "target", factual_path,
      intervention, expected, start,
  )


def generate_candidate(
    spec: InstanceSpec,
) -> Tuple[SimulationLog, SimulationLog, GroundTruth]:
  """Runs both branches and derives their oracle ground truth."""
  if not isinstance(spec, InstanceSpec):
    raise TypeError("spec must be an InstanceSpec")
  factual, counterfactual = generate_paired_instance(
      spec.scene_config,
      spec.target_id,
      spec.intervention,
      spec.instance_seed,
      factual_path=spec.factual_path,
  )
  truth = extract_pair_ground_truth(
      spec.scene_config, spec.intervention, factual, counterfactual
  )
  return factual, counterfactual, truth


def _log_parts(log: Any) -> Tuple[Optional[Tuple[str, ...]], Optional[Tuple[int, ...]],
                                  Optional[np.ndarray], Optional[Tuple[Any, ...]],
                                  Optional[float]]:
  try:
    object_ids = tuple(log.object_ids)
    steps = tuple(int(step) for step in log.steps)
    states = np.asarray(log.states, dtype=np.float64)
    contacts = tuple(log.contacts)
    step_rate = float(log.step_rate)
  except (AttributeError, TypeError, ValueError, OverflowError):
    return None, None, None, None, None
  if (
      states.ndim != 3 or states.shape != (len(steps), len(object_ids), 13)
      or not math.isfinite(step_rate) or step_rate <= 0
  ):
    return object_ids, steps, None, contacts, step_rate
  return object_ids, steps, states, contacts, step_rate


def _max_speed(states: Optional[np.ndarray], component: slice) -> float:
  if states is None:
    return float("inf")
  vectors = states[:, :, component]
  if not np.isfinite(vectors).all():
    return float("inf")
  return float(np.max(np.linalg.norm(vectors, axis=2), initial=0.0))


def _prefix_contacts(contacts: Optional[Tuple[Any, ...]], start: int) -> Optional[Tuple[Any, ...]]:
  if contacts is None:
    return None
  try:
    return tuple(record for record in contacts if int(record.step) < start)
  except (AttributeError, TypeError, ValueError):
    return None


def _rotation_extent(size: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
  w, x, y, z = (float(value) for value in quaternion)
  rotation = np.asarray((
      (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
      (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
      (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
  ))
  return np.abs(rotation) @ np.asarray(size, dtype=np.float64)


def _object_extent(item: ObjectConfig, quaternion: Sequence[float]) -> np.ndarray:
  if item.shape == "sphere":
    return np.full(3, item.size[0], dtype=np.float64)
  return _rotation_extent(item.size, quaternion)


def evaluate_qc(
    spec: InstanceSpec,
    factual_log: Any,
    counterfactual_log: Any,
    ground_truth: Any,
    qc_config: Optional[Mapping[str, Any]] = None,
) -> QCResult:
  """Evaluates all applicable QC rules and returns stable sorted codes."""
  if not isinstance(spec, InstanceSpec):
    raise TypeError("spec must be an InstanceSpec")
  config = {} if qc_config is None else dict(qc_config)
  linear_ceiling = _finite(
      config.get("linear_velocity_ceiling", 100.0), "linear_velocity_ceiling"
  )
  angular_ceiling = _finite(
      config.get("angular_velocity_ceiling", 100.0), "angular_velocity_ceiling"
  )
  clip_epsilon = _finite(config.get("clip_epsilon", 1e-9), "clip_epsilon")
  if linear_ceiling < 0 or angular_ceiling < 0 or clip_epsilon < 0:
    raise ValueError("QC ceilings and clip_epsilon must be nonnegative")

  factual = _log_parts(factual_log)
  counterfactual = _log_parts(counterfactual_log)
  f_ids, f_steps, f_states, f_contacts, f_rate = factual
  c_ids, c_steps, c_states, c_contacts, c_rate = counterfactual
  reasons = set()
  aligned = (
      f_ids is not None and c_ids is not None
      and f_steps is not None and c_steps is not None
      and f_ids == c_ids and f_steps == c_steps and f_rate == c_rate
      and getattr(factual_log, "branch", None) == "factual"
      and getattr(counterfactual_log, "branch", None) == "counterfactual"
      and f_states is not None and c_states is not None
  )
  if not aligned:
    reasons.add("branch_misaligned")
  if (
      f_states is None or c_states is None
      or not np.isfinite(f_states).all() or not np.isfinite(c_states).all()
  ):
    reasons.add("nonfinite_state")

  f_linear = _max_speed(f_states, LINEAR_VELOCITY_SLICE)
  c_linear = _max_speed(c_states, LINEAR_VELOCITY_SLICE)
  f_angular = _max_speed(f_states, ANGULAR_VELOCITY_SLICE)
  c_angular = _max_speed(c_states, ANGULAR_VELOCITY_SLICE)
  max_linear = max(f_linear, c_linear)
  max_angular = max(f_angular, c_angular)
  if max_linear > linear_ceiling:
    reasons.add("linear_velocity_ceiling")
  if max_angular > angular_ceiling:
    reasons.add("angular_velocity_ceiling")

  if aligned:
    prefix_indices = tuple(
        index for index, step in enumerate(f_steps)
        if step < spec.intervention_start_step
    )
    state_equal = all(np.array_equal(f_states[index], c_states[index])
                      for index in prefix_indices)
    contact_equal = _prefix_contacts(
        f_contacts, spec.intervention_start_step
    ) == _prefix_contacts(c_contacts, spec.intervention_start_step)
    if not state_equal or not contact_equal:
      reasons.add("twin_prefix_mismatch")

  target_indices = []
  for ids, states in ((f_ids, f_states), (c_ids, c_states)):
    if ids is None or states is None or spec.target_id not in ids:
      continue
    target_indices.append((states, ids.index(spec.target_id)))
  bounds = np.asarray(spec.scene_config.scene_bounds, dtype=np.float64)
  if any(np.any(states[:, index, :3] < bounds[0]) or
         np.any(states[:, index, :3] > bounds[1])
         for states, index in target_indices):
    reasons.add("target_out_of_bounds")

  by_id = {item.object_id: item for item in spec.scene_config.objects}
  target_config = by_id[spec.target_id]
  static_geometry = tuple(
      item for item in spec.scene_config.objects
      if item.static and item.object_id != spec.target_id
      and not bool(item.metadata.get("qc_clip_exempt", False))
  )
  clipped = False
  for states, index in target_indices:
    for state in states:
      target_position = state[index, :3]
      target_extent = _object_extent(target_config, state[index, 3:7])
      for obstacle in static_geometry:
        obstacle_extent = _object_extent(obstacle, obstacle.quaternion)
        overlap_depth = (
            target_extent + obstacle_extent
            - np.abs(target_position - np.asarray(obstacle.position))
        )
        if np.all(overlap_depth > clip_epsilon):
          clipped = True
          break
      if clipped:
        break
    if clipped:
      break
  if clipped:
    reasons.add("target_static_clip")

  valid_truth = isinstance(ground_truth, GroundTruth)
  affected = bool(
      valid_truth and (ground_truth.hard_affected or ground_truth.soft_affected)
  )
  graph_changed = bool(
      valid_truth and (
          ground_truth.graph_delta.added
          or ground_truth.graph_delta.removed
          or ground_truth.graph_delta.changed
      )
  )
  if spec.expected_effect == "non_null" and not affected:
    reasons.add("empty_affected")
  if spec.expected_effect == "null" and (affected or graph_changed):
    reasons.add("expected_null_mismatch")
  if not valid_truth:
    reasons.add("branch_misaligned")

  metrics = {
      "max_linear_velocity": max_linear,
      "max_angular_velocity": max_angular,
      "affected_count": (
          len(ground_truth.hard_affected) + len(ground_truth.soft_affected)
          if valid_truth else 0
      ),
      "contact_delta_count": (
          len(ground_truth.graph_delta.added)
          + len(ground_truth.graph_delta.removed)
          + len(ground_truth.graph_delta.changed)
          if valid_truth else 0
      ),
  }
  # Metrics are persisted as strict JSON; malformed states use a large finite marker.
  metrics = {
      key: (float(np.finfo(np.float64).max) if value == float("inf") else value)
      for key, value in metrics.items()
  }
  return QCResult(not reasons, tuple(sorted(reasons)), metrics)


def primary_category(ground_truth: GroundTruth) -> str:
  """Returns the single primary contact/state category."""
  if not isinstance(ground_truth, GroundTruth):
    raise TypeError("ground_truth must be a GroundTruth")
  delta = ground_truth.graph_delta
  present = tuple(bool(bucket) for bucket in (delta.added, delta.removed, delta.changed))
  if sum(present) > 1:
    return "mixed_contact_delta"
  if delta.added:
    return "contact_added"
  if delta.removed:
    return "contact_removed"
  if delta.changed:
    return "contact_changed"
  if ground_truth.hard_affected or ground_truth.soft_affected:
    return "state_only"
  return "null_effect"


def propagation_hop_depth(ground_truth: GroundTruth) -> Tuple[int, str]:
  """Returns maximum propagation edges and its ``0/1/2/3+`` bucket."""
  if not isinstance(ground_truth, GroundTruth):
    raise TypeError("ground_truth must be a GroundTruth")
  depth = max(
      (max(0, len(path) - 1) for path in ground_truth.propagation_path.values()),
      default=0,
  )
  return depth, "3+" if depth >= 3 else str(depth)


def _sha_text(*parts: Any) -> str:
  material = "\0".join(str(part) for part in parts).encode("utf-8")
  return hashlib.sha256(material).hexdigest()


def topology_signature(
    scene_config: SceneConfig, factual_log: Any, target_id: Optional[str] = None
) -> str:
  """Computes an ID-invariant Weisfeiler-Lehman contact topology hash."""
  if not isinstance(scene_config, SceneConfig):
    raise TypeError("scene_config must be a SceneConfig")
  ids = tuple(item.object_id for item in scene_config.objects)
  if target_id is None:
    target_id = "target" if "target" in ids else next(
        (item.object_id for item in scene_config.objects
         if item.metadata.get("kinematic_emulation")), None
    )
  target_id = _identifier(target_id, "target_id")
  if target_id not in ids:
    raise ValueError("target_id is absent from scene_config")
  labels = {}
  for item in scene_config.objects:
    role = "target" if item.object_id == target_id else (
        "environment" if item.static else "dynamic"
    )
    labels[item.object_id] = "{}|{}|{}".format(role, item.shape, int(item.static))
  edges = set()
  try:
    contacts = tuple(factual_log.contacts)
  except (AttributeError, TypeError) as error:
    raise ValueError("factual_log contacts are malformed") from error
  for record in contacts:
    try:
      left, right = sorted((str(record.object_a), str(record.object_b)))
    except AttributeError as error:
      raise ValueError("factual_log contacts are malformed") from error
    if left == right:
      continue
    if left not in labels:
      labels[left] = "unknown|unknown|0"
    if right not in labels:
      labels[right] = "unknown|unknown|0"
    edges.add((left, right))
  neighbors = {node: set() for node in labels}
  for left, right in edges:
    neighbors[left].add(right)
    neighbors[right].add(left)
  colors = {node: _sha_text(label) for node, label in labels.items()}
  for _ in range(len(colors)):
    refined = {
        node: _sha_text(colors[node], *sorted(colors[item] for item in neighbors[node]))
        for node in colors
    }
    if refined == colors:
      break
    colors = refined
  canonical_edges = sorted(
      tuple(sorted((colors[left], colors[right]))) for left, right in edges
  )
  payload = {
      "nodes": sorted(colors.values()),
      "edges": canonical_edges,
  }
  return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def select_balanced(
    candidates: Iterable[CandidateSummary], capacity: int, *, seed: int = 0
) -> Tuple[CandidateSummary, ...]:
  """Round-robins category/hop strata; excess capacity returns every item."""
  count = _integer(capacity, "capacity")
  seed = _integer(seed, "seed")
  try:
    values = tuple(candidates)
  except TypeError as error:
    raise TypeError("candidates must be iterable") from error
  if not all(isinstance(item, CandidateSummary) for item in values):
    raise TypeError("candidates must contain CandidateSummary values")
  if len({item.instance_id for item in values}) != len(values):
    raise ValueError("candidate instance_id values must be unique")
  strata = defaultdict(list)
  for item in values:
    strata[(item.category, item.hop_bucket)].append(item)
  for key in strata:
    strata[key].sort(key=lambda item: (_sha_text(seed, item.instance_id), item.instance_id))
  order = sorted(strata, key=lambda key: (_sha_text(seed, *key), key))
  selected = []
  cursor = {key: 0 for key in order}
  target = min(count, len(values))
  while len(selected) < target:
    progressed = False
    for key in order:
      index = cursor[key]
      if index < len(strata[key]) and len(selected) < target:
        selected.append(strata[key][index])
        cursor[key] += 1
        progressed = True
    if not progressed:
      break
  return tuple(selected)


def assign_grouped_splits(
    candidates: Iterable[CandidateSummary],
    fractions: Optional[Mapping[str, float]] = None,
    *,
    seed: int = 0,
) -> Mapping[str, str]:
  """Greedily assigns whole topology groups near requested split fractions."""
  seed = _integer(seed, "seed")
  values = tuple(candidates)
  if not all(isinstance(item, CandidateSummary) for item in values):
    raise TypeError("candidates must contain CandidateSummary values")
  if len({item.instance_id for item in values}) != len(values):
    raise ValueError("candidate instance_id values must be unique")
  requested = {"train": 0.8, "val": 0.1, "test": 0.1}
  if fractions is not None:
    if set(fractions) != set(_SPLITS):
      raise ValueError("fractions must contain train, val, and test")
    requested = {name: _finite(fractions[name], "{} fraction".format(name))
                 for name in _SPLITS}
  if any(value < 0 for value in requested.values()) or not math.isclose(
      math.fsum(requested.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
  ):
    raise ValueError("split fractions must be nonnegative and sum to one")
  groups = defaultdict(list)
  for item in values:
    groups[item.topology_signature].append(item)
  group_order = sorted(
      groups,
      key=lambda signature: (
          -len(groups[signature]), _sha_text(seed, signature), signature
      ),
  )
  desired = {name: requested[name] * len(values) for name in _SPLITS}
  counts = {name: 0 for name in _SPLITS}
  result = {}
  split_tiebreak = sorted(_SPLITS, key=lambda name: _sha_text(seed, name))
  for signature in group_order:
    size = len(groups[signature])
    split = min(
        split_tiebreak,
        key=lambda name: (
            sum(
                (counts[other] + (size if other == name else 0) - desired[other]) ** 2
                for other in _SPLITS
            ),
            split_tiebreak.index(name),
        ),
    )
    counts[split] += size
    for item in groups[signature]:
      result[item.instance_id] = split
  return MappingProxyType({key: result[key] for key in sorted(result)})


def _write_once(path: Path, payload: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.parent / ".{}.tmp-{}".format(path.name, uuid.uuid4().hex)
  try:
    with temporary.open("xb") as stream:
      stream.write(payload)
      stream.flush()
      os.fsync(stream.fileno())
    try:
      os.link(temporary, path)
    except FileExistsError:
      raise FileExistsError("immutable journal record exists: {}".format(path))
  finally:
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


def _write_atomic(path: Path, payload: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      mode="wb", prefix=".{}-".format(path.name), dir=path.parent, delete=False
  ) as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    temporary = Path(stream.name)
  try:
    os.replace(temporary, path)
  finally:
    try:
      temporary.unlink()
    except FileNotFoundError:
      pass


def _yaml_snapshot(ranges: Mapping[str, Any]) -> bytes:
  try:
    import yaml
  except ImportError as error:  # pragma: no cover - environment dependency.
    raise ImportError("dataset generation requires PyYAML") from error
  return yaml.safe_dump(
      to_jsonable(ranges), sort_keys=True, allow_unicode=True
  ).encode("utf-8")


def _file_manifest(directory: Path) -> Mapping[str, Any]:
  files = {}
  for path in sorted(directory.rglob("*")):
    if path.is_file() and path.name != "instance_manifest.json":
      payload = path.read_bytes()
      files[str(path.relative_to(directory))] = {
          "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)
      }
  return files


def _publish_instance(
    root: Path,
    spec: InstanceSpec,
    factual: SimulationLog,
    counterfactual: SimulationLog,
) -> Path:
  instances = root / "instances"
  instances.mkdir(parents=True, exist_ok=True)
  destination = instances / spec.instance_id
  if destination.exists():
    raise FileExistsError("instance artifact exists: {}".format(destination))
  staging = instances / ".{}.tmp-{}".format(spec.instance_id, uuid.uuid4().hex)
  try:
    write_paired_artifact(
        staging,
        spec.scene_config,
        spec.intervention,
        spec.instance_seed,
        factual,
        counterfactual,
    )
    _write_once(staging / "spec.json", _canonical_bytes(spec.to_dict()))
    instance_manifest = {
        "instance_id": spec.instance_id,
        "files": _file_manifest(staging),
    }
    _write_once(
        staging / "instance_manifest.json", _canonical_bytes(instance_manifest)
    )
    os.rename(staging, destination)
  except BaseException:
    if staging.exists():
      shutil.rmtree(staging, ignore_errors=True)
    raise
  return destination


def _summary_from_dict(value: Mapping[str, Any]) -> CandidateSummary:
  return CandidateSummary(**dict(value))


def _attempt_records(root: Path) -> Tuple[Mapping[str, Any], ...]:
  records = []
  for path in sorted((root / "attempts").glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")):
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
      raise ValueError("corrupt attempt journal: {}".format(path)) from error
    if not isinstance(payload, Mapping) or payload.get("attempt_index") != int(path.stem):
      raise ValueError("corrupt attempt journal: {}".format(path))
    records.append(payload)
  return tuple(records)


def _manifest(
    root: Path,
    config_sha256: str,
    master_seed: int,
    num_instances: int,
    records: Sequence[Mapping[str, Any]],
    balance_seed: int,
    split_seed: int,
    fractions: Mapping[str, float],
) -> Mapping[str, Any]:
  candidates = tuple(
      _summary_from_dict(record["candidate"])
      for record in records if record.get("status") == "accepted"
  )
  selected = select_balanced(candidates, num_instances, seed=balance_seed)
  splits = assign_grouped_splits(selected, fractions, seed=split_seed)
  status = "complete" if len(selected) >= num_instances else "capacity_exhausted"
  return {
      "status": status,
      "config_sha256": config_sha256,
      "master_seed": master_seed,
      "num_instances": num_instances,
      "attempt_count": len(records),
      "attempts": [int(record["attempt_index"]) for record in records],
      "candidates": [item.to_dict() for item in candidates],
      "selected_ids": [item.instance_id for item in selected],
      "splits": to_jsonable(splits),
  }


def run_batch(
    ranges: Mapping[str, Any],
    output: PathLike,
    master_seed: int,
    num_instances: int,
    max_attempts: int,
    *,
    resume: bool = False,
    workers: int = 1,
) -> Mapping[str, Any]:
  """Runs an immutable, resumable single-worker generation journal."""
  if not isinstance(ranges, Mapping):
    raise TypeError("ranges must be a mapping")
  seed = _integer(master_seed, "master_seed")
  count = _integer(num_instances, "num_instances", minimum=1)
  maximum = _integer(max_attempts, "max_attempts", minimum=1)
  workers = _integer(workers, "workers", minimum=1)
  if workers != 1:
    raise ValueError("only workers=1 is supported")
  if not isinstance(resume, bool):
    raise TypeError("resume must be a bool")
  root = Path(output)
  snapshot = _yaml_snapshot(ranges)
  config_hash = hashlib.sha256(snapshot).hexdigest()
  run_payload = {
      "config_sha256": config_hash,
      "master_seed": seed,
      "num_instances": count,
  }
  run_path = root / "run.json"
  if root.exists() and not resume:
    raise FileExistsError("dataset output exists; pass resume=True")
  if not root.exists():
    if resume:
      raise ValueError("cannot resume a missing dataset output")
    root.mkdir(parents=True)
    (root / "attempts").mkdir()
    (root / "errors").mkdir()
    (root / "instances").mkdir()
    _write_once(root / "config.yaml", snapshot)
    _write_once(run_path, _canonical_bytes(run_payload))
  else:
    try:
      existing = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
      raise ValueError("resume metadata is missing or corrupt") from error
    if existing.get("config_sha256") != config_hash:
      raise ValueError("resume config mismatch")
    if existing.get("master_seed") != seed:
      raise ValueError("resume seed mismatch")
    if existing.get("num_instances") != count:
      raise ValueError("resume count mismatch")
    if hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest() != config_hash:
      raise ValueError("resume config snapshot mismatch")

  qc_config = dict(_section(to_jsonable(ranges), "qc"))
  balance_config = dict(to_jsonable(ranges).get("balance", {}))
  split_config = dict(to_jsonable(ranges).get("split", {}))
  balance_seed = _integer(balance_config.get("seed", seed), "balance seed")
  split_seed = _integer(split_config.get("seed", seed), "split seed")
  fractions = split_config.get(
      "fractions", {"train": 0.8, "val": 0.1, "test": 0.1}
  )

  records = _attempt_records(root)
  occupied = {int(record["attempt_index"]) for record in records}
  # Fill the deterministic attempt budget before balancing. Stopping as soon as
  # capacity is reached would make later strata unable to displace early surplus.
  while len(occupied) < maximum:
    missing = next((index for index in range(maximum) if index not in occupied), None)
    if missing is None:
      break
    attempt_path = root / "attempts" / "{:08d}.json".format(missing)
    try:
      spec = sample_instance_spec(ranges, seed, missing)
      factual, counterfactual, truth = generate_candidate(spec)
      qc = evaluate_qc(spec, factual, counterfactual, truth, qc_config)
      record: Dict[str, Any] = {
          "attempt_index": missing,
          "instance_id": spec.instance_id,
          "instance_seed": spec.instance_seed,
          "qc": qc.to_dict(),
      }
      if qc.accepted:
        artifact = _publish_instance(root, spec, factual, counterfactual)
        depth, bucket = propagation_hop_depth(truth)
        summary = CandidateSummary(
            spec.instance_id,
            missing,
            primary_category(truth),
            depth,
            bucket,
            topology_signature(spec.scene_config, factual, spec.target_id),
            str(artifact.relative_to(root)),
        )
        record.update(status="accepted", candidate=summary.to_dict())
      else:
        record.update(status="rejected", spec=spec.to_dict())
    except Exception as error:  # Candidate-local failures must not abort later attempts.
      error_record = {
          "attempt_index": missing,
          "error_type": type(error).__name__,
          "message": str(error),
          "traceback": traceback.format_exc(),
      }
      _write_once(
          root / "errors" / "{:08d}.json".format(missing),
          _canonical_bytes(error_record),
      )
      record = {**error_record, "status": "error"}
    _write_once(attempt_path, _canonical_bytes(record))
    occupied.add(missing)
    records = _attempt_records(root)
    current = _manifest(
        root, config_hash, seed, count, records, balance_seed, split_seed, fractions
    )
    _write_atomic(root / "manifest.json", _canonical_bytes(current))

  records = _attempt_records(root)
  result = _manifest(
      root, config_hash, seed, count, records, balance_seed, split_seed, fractions
  )
  _write_atomic(root / "manifest.json", _canonical_bytes(result))
  return _freeze(result)


__all__ = [
    "CandidateSummary",
    "InstanceSpec",
    "QCResult",
    "assign_grouped_splits",
    "derive_seed",
    "evaluate_qc",
    "generate_candidate",
    "load_ranges",
    "primary_category",
    "propagation_hop_depth",
    "run_batch",
    "sample_instance_spec",
    "select_balanced",
    "topology_signature",
]
