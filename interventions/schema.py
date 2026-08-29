"""Validated backend- and renderer-independent intervention schemas.

Purpose: define immutable experiment values and deterministic JSON conversion.
Public API: ObjectConfig, CameraConfig, SceneConfig, Intervention,
GraphEdgeDelta, GroundTruth, constants, to_jsonable(), shape_half_extents(),
half_extents(), and oriented_aabb().
Dependencies: Python's standard library only, so validation and JSON preparation
never import Kubric, a renderer, or a simulator backend.
Trust boundary: validation enforces backend-neutral shape, numeric, and JSON-safe
contracts; it does not prove physical feasibility, execution, or data origin.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.0"
SUPPORTED_SHAPES = frozenset(("cube", "sphere", "cylinder", "capsule"))
TARGET_SHAPES = frozenset(("cube", "sphere"))
_RADIAL_SHAPES = frozenset(("sphere", "cylinder", "capsule"))
INTERVENTION_RECIPES = frozenset(
    (
        "remove_collision",
        "create_collision",
        "retime",
        "break_contact",
        "maintain_contact",
    )
)


def _json_sort_key(value: Any) -> str:
  """Returns a stable ordering key for an already JSON-compatible value."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_mapping_keys(value: Mapping[Any, Any]) -> None:
  if not all(isinstance(key, str) for key in value):
    raise ValueError("mapping keys must be strings")


def _json_integer(value: numbers.Integral) -> int:
  result = int(value)
  try:
    json.dumps(result)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("integer is not JSON-encodable") from error
  return result


def to_jsonable(value: Any) -> Any:
  """Recursively converts schemas and common containers to JSON-safe values.

  Dataclasses become dictionaries, mappings retain deterministically sorted keys,
  tuples and arrays become lists, and sets become deterministically sorted lists.
  Unsupported values raise ``TypeError`` rather than being silently stringified.
  """
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
        item.name: to_jsonable(getattr(value, item.name))
        for item in dataclasses.fields(value)
    }
  if value is None or isinstance(value, (str, bool)):
    return value
  if isinstance(value, numbers.Integral):
    return _json_integer(value)
  if isinstance(value, numbers.Real):
    try:
      result = float(value)
    except (OverflowError, ValueError) as error:
      raise ValueError("numbers must be finite") from error
    if not math.isfinite(result):
      raise ValueError("non-finite numbers are not JSON-safe")
    return result
  if isinstance(value, Mapping):
    _validate_mapping_keys(value)
    return {key: to_jsonable(value[key]) for key in sorted(value)}
  if isinstance(value, (set, frozenset)):
    converted = [to_jsonable(item) for item in value]
    return sorted(converted, key=_json_sort_key)
  if isinstance(value, (tuple, list)):
    return [to_jsonable(item) for item in value]

  # NumPy scalars and arrays expose these methods. Keeping the check generic avoids
  # importing NumPy into the schema-only layer.
  if hasattr(value, "item"):
    try:
      scalar = value.item()
    except (TypeError, ValueError):
      scalar = value
    if scalar is not value:
      return to_jsonable(scalar)
  if hasattr(value, "tolist"):
    try:
      listed = value.tolist()
    except (TypeError, ValueError):
      listed = value
    if listed is not value:
      return to_jsonable(listed)
  raise TypeError("unsupported JSON value: {!r}".format(type(value).__name__))


def _freeze(value: Any) -> Any:
  """Copies JSON-like data into recursively immutable containers."""
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    raise TypeError("nested dataclass values are not supported")
  if value is None or isinstance(value, (str, bool, numbers.Number)):
    # Validate numeric leaves while preserving integer/float identity.
    to_jsonable(value)
    return value
  if isinstance(value, Mapping):
    _validate_mapping_keys(value)
    frozen = {}
    for key in sorted(value):
      frozen[key] = _freeze(value[key])
    return MappingProxyType(frozen)
  if isinstance(value, (set, frozenset)):
    frozen_items = [_freeze(item) for item in value]
    return tuple(sorted(frozen_items, key=lambda item: _json_sort_key(to_jsonable(item))))
  if isinstance(value, (tuple, list)):
    return tuple(_freeze(item) for item in value)
  if hasattr(value, "tolist"):
    try:
      return _freeze(value.tolist())
    except (TypeError, ValueError):
      pass
  raise TypeError("metadata contains unsupported value: {!r}".format(type(value).__name__))


def _real(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise TypeError("{} must be a real number".format(name))
  try:
    result = float(value)
  except (OverflowError, ValueError) as error:
    raise ValueError("{} must be finite".format(name)) from error
  if not math.isfinite(result):
    raise ValueError("{} must be finite".format(name))
  return result


def _integer(value: Any, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, numbers.Integral):
    raise TypeError("{} must be an integer".format(name))
  try:
    return _json_integer(value)
  except ValueError as error:
    raise ValueError("{} must be JSON-encodable".format(name)) from error


def _vector(value: Any, length: int, name: str) -> Tuple[float, ...]:
  if isinstance(value, (str, bytes)):
    raise TypeError("{} must be a numeric sequence".format(name))
  try:
    values = tuple(value)
  except TypeError as error:
    raise TypeError("{} must be a numeric sequence".format(name)) from error
  if len(values) != length:
    raise ValueError("{} must contain {} values".format(name, length))
  return tuple(_real(item, "{}[{}]".format(name, index))
               for index, item in enumerate(values))


def _nonempty_string(value: Any, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if not value.strip():
    raise ValueError("{} must not be empty".format(name))
  return value


def _metadata(value: Any) -> Mapping[str, Any]:
  if value is None:
    return MappingProxyType({})
  if not isinstance(value, Mapping):
    raise TypeError("metadata must be a mapping")
  return _freeze(value)


def _version(value: Any) -> str:
  if value != SCHEMA_VERSION:
    raise ValueError("schema_version must be {!r}".format(SCHEMA_VERSION))
  return SCHEMA_VERSION


class _SchemaMixin:
  """Shared serialization behavior for persisted schemas."""

  def to_dict(self) -> Mapping[str, Any]:
    """Returns an independent JSON-safe dictionary representation."""
    return to_jsonable(self)


@dataclass(frozen=True)
class ObjectConfig(_SchemaMixin):
  """Logical primitive object and its initial physical state.

  Quaternion components use WXYZ order. Non-zero quaternion inputs are normalized,
  while scalar sizes are expanded to an XYZ tuple.

  ``size`` is interpreted per shape, always as a local half-extent:
  cube ``(half_x, half_y, half_z)``; sphere ``(radius, radius, radius)``;
  cylinder ``(radius, radius, half_height)``; capsule
  ``(radius, radius, cylinder_half_height)``, whose total Z half-extent is
  ``cylinder_half_height + radius``.
  """

  object_id: str
  shape: str
  size: Any = 1.0
  mass: float = 1.0
  friction: float = 0.5
  restitution: float = 0.5
  position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
  quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
  angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
  static: bool = False
  metadata: Mapping[str, Any] = field(default_factory=dict)
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    object.__setattr__(self, "object_id", _nonempty_string(self.object_id, "object_id"))
    if not isinstance(self.shape, str):
      raise TypeError("shape must be a string")
    shape = self.shape.lower()
    if shape not in SUPPORTED_SHAPES:
      raise ValueError("unsupported shape: {!r}".format(self.shape))
    object.__setattr__(self, "shape", shape)

    if isinstance(self.size, numbers.Real) and not isinstance(self.size, bool):
      scalar = _real(self.size, "size")
      size = (scalar, scalar, scalar)
    else:
      size = _vector(self.size, 3, "size")
    if any(component <= 0.0 for component in size):
      raise ValueError("size components must be positive")
    if shape in _RADIAL_SHAPES and size[0] != size[1]:
      raise ValueError("{} requires equal X and Y radii".format(shape))
    if shape == "sphere" and size[1] != size[2]:
      raise ValueError("sphere requires equal radii on every axis")
    object.__setattr__(self, "size", size)

    mass = _real(self.mass, "mass")
    if mass < 0.0 or (mass == 0.0 and not self.static):
      raise ValueError("mass must be positive for dynamic objects")
    friction = _real(self.friction, "friction")
    if friction < 0.0:
      raise ValueError("friction must be nonnegative")
    restitution = _real(self.restitution, "restitution")
    if not 0.0 <= restitution <= 1.0:
      raise ValueError("restitution must lie in [0, 1]")
    if not isinstance(self.static, bool):
      raise TypeError("static must be a bool")

    quaternion = _vector(self.quaternion, 4, "quaternion")
    scale = max(abs(component) for component in quaternion)
    if scale == 0.0:
      raise ValueError("quaternion must be non-zero")
    scaled = tuple(component / scale for component in quaternion)
    norm = math.hypot(*scaled)
    quaternion = tuple(component / norm for component in scaled)
    unit_norm = math.hypot(*quaternion)
    if not math.isfinite(unit_norm) or abs(unit_norm - 1.0) > 1e-12:
      raise ValueError("quaternion normalization failed")

    object.__setattr__(self, "mass", mass)
    object.__setattr__(self, "friction", friction)
    object.__setattr__(self, "restitution", restitution)
    object.__setattr__(self, "position", _vector(self.position, 3, "position"))
    object.__setattr__(self, "quaternion", quaternion)
    object.__setattr__(
        self, "linear_velocity", _vector(self.linear_velocity, 3, "linear_velocity")
    )
    object.__setattr__(
        self, "angular_velocity", _vector(self.angular_velocity, 3, "angular_velocity")
    )
    object.__setattr__(self, "metadata", _metadata(self.metadata))
    object.__setattr__(self, "schema_version", _version(self.schema_version))

  @property
  def initial_position(self) -> Tuple[float, float, float]:
    """Alias emphasizing that ``position`` describes the initial state."""
    return self.position

  @property
  def initial_quaternion(self) -> Tuple[float, float, float, float]:
    """Alias emphasizing that ``quaternion`` describes the initial state."""
    return self.quaternion


def shape_half_extents(
    shape: str, size: Sequence[float]
) -> Tuple[float, float, float]:
  """Returns the local, unrotated half-extents implied by ``shape`` and ``size``."""
  if shape not in SUPPORTED_SHAPES:
    raise ValueError("unsupported shape: {!r}".format(shape))
  extents = _vector(size, 3, "size")
  if shape == "capsule":
    return (extents[0], extents[1], extents[2] + extents[0])
  return extents


def half_extents(config: ObjectConfig) -> Tuple[float, float, float]:
  """Returns the local, unrotated half-extents of ``config``."""
  if not isinstance(config, ObjectConfig):
    raise TypeError("config must be an ObjectConfig")
  return shape_half_extents(config.shape, config.size)


def _rotate(
    quaternion: Sequence[float], vector: Sequence[float]
) -> Tuple[float, float, float]:
  """Rotates ``vector`` by a unit WXYZ ``quaternion`` using plain arithmetic."""
  w, x, y, z = (float(component) for component in quaternion)
  vx, vy, vz = (float(component) for component in vector)
  return (
      vx * (1 - 2 * (y * y + z * z)) + vy * 2 * (x * y - z * w)
      + vz * 2 * (x * z + y * w),
      vx * 2 * (x * y + z * w) + vy * (1 - 2 * (x * x + z * z))
      + vz * 2 * (y * z - x * w),
      vx * 2 * (x * z - y * w) + vy * 2 * (y * z + x * w)
      + vz * (1 - 2 * (x * x + y * y)),
  )


def oriented_aabb(
    config: ObjectConfig,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
  """Returns the world-space bounds of ``config`` after applying its rotation.

  The box is a conservative bound for the radial shapes, whose rotated silhouette
  is smaller than the rotated box around them.
  """
  extents = half_extents(config)
  corners = [
      _rotate(config.quaternion, (sx * extents[0], sy * extents[1], sz * extents[2]))
      for sx in (-1.0, 1.0)
      for sy in (-1.0, 1.0)
      for sz in (-1.0, 1.0)
  ]
  lower = tuple(
      min(corner[axis] for corner in corners) + config.position[axis]
      for axis in range(3)
  )
  upper = tuple(
      max(corner[axis] for corner in corners) + config.position[axis]
      for axis in range(3)
  )
  return lower, upper


@dataclass(frozen=True)
class CameraConfig(_SchemaMixin):
  """Pinhole camera placement and focal length."""

  position: Tuple[float, float, float]
  look_at: Tuple[float, float, float]
  focal_length: float
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    position = _vector(self.position, 3, "position")
    look_at = _vector(self.look_at, 3, "look_at")
    if position == look_at:
      raise ValueError("camera position and look_at must differ")
    focal_length = _real(self.focal_length, "focal_length")
    if focal_length <= 0.0:
      raise ValueError("focal_length must be positive")
    object.__setattr__(self, "position", position)
    object.__setattr__(self, "look_at", look_at)
    object.__setattr__(self, "focal_length", focal_length)
    object.__setattr__(self, "schema_version", _version(self.schema_version))


@dataclass(frozen=True)
class SceneConfig(_SchemaMixin):
  """Complete backend-independent scene configuration."""

  objects: Tuple[ObjectConfig, ...]
  camera: Optional[CameraConfig] = None
  seed: int = 0
  scene_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
      (-10.0, -10.0, -10.0),
      (10.0, 10.0, 10.0),
  )
  gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
  frame_range: Tuple[int, int] = (0, 24)
  frame_rate: int = 24
  step_rate: int = 240
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if isinstance(self.objects, (str, bytes)):
      raise TypeError("objects must be an iterable of ObjectConfig")
    try:
      objects = tuple(self.objects)
    except TypeError as error:
      raise TypeError("objects must be an iterable of ObjectConfig") from error
    if not all(isinstance(item, ObjectConfig) for item in objects):
      raise TypeError("objects must contain only ObjectConfig values")
    identifiers = [item.object_id for item in objects]
    if len(set(identifiers)) != len(identifiers):
      raise ValueError("object_id values must be unique")
    if self.camera is not None and not isinstance(self.camera, CameraConfig):
      raise TypeError("camera must be CameraConfig or None")

    seed = _integer(self.seed, "seed")
    if seed < 0:
      raise ValueError("seed must be nonnegative")
    if len(self.scene_bounds) != 2:
      raise ValueError("scene_bounds must contain minimum and maximum XYZ vectors")
    bounds_min = _vector(self.scene_bounds[0], 3, "scene_bounds[0]")
    bounds_max = _vector(self.scene_bounds[1], 3, "scene_bounds[1]")
    if any(lower >= upper for lower, upper in zip(bounds_min, bounds_max)):
      raise ValueError("scene_bounds minimum must be below maximum on every axis")
    for item in objects:
      if item.shape not in SUPPORTED_SHAPES:
        raise ValueError("unsupported object shape: {!r}".format(item.shape))
      if any(position < lower or position > upper for position, lower, upper in
             zip(item.position, bounds_min, bounds_max)):
        raise ValueError(
            "initial position for {!r} is outside scene_bounds".format(item.object_id)
        )

    if len(self.frame_range) != 2:
      raise ValueError("frame_range must contain start and end")
    frame_start = _integer(self.frame_range[0], "frame_range[0]")
    frame_end = _integer(self.frame_range[1], "frame_range[1]")
    if frame_start < 0 or frame_end <= frame_start:
      raise ValueError("frame_range must be a nonnegative half-open interval")
    frame_rate = _integer(self.frame_rate, "frame_rate")
    step_rate = _integer(self.step_rate, "step_rate")
    if frame_rate <= 0 or step_rate <= 0:
      raise ValueError("frame_rate and step_rate must be positive")
    if step_rate % frame_rate:
      raise ValueError("step_rate must be divisible by frame_rate")

    object.__setattr__(self, "objects", objects)
    object.__setattr__(self, "seed", seed)
    object.__setattr__(self, "scene_bounds", (bounds_min, bounds_max))
    object.__setattr__(self, "gravity", _vector(self.gravity, 3, "gravity"))
    object.__setattr__(self, "frame_range", (frame_start, frame_end))
    object.__setattr__(self, "frame_rate", frame_rate)
    object.__setattr__(self, "step_rate", step_rate)
    object.__setattr__(self, "schema_version", _version(self.schema_version))


@dataclass(frozen=True)
class Intervention(_SchemaMixin):
  """A requested counterfactual edit over a half-open time interval."""

  target_id: str
  recipe: str
  magnitude: float
  time_window: Tuple[float, float]
  metadata: Mapping[str, Any] = field(default_factory=dict)
  schema_version: str = SCHEMA_VERSION
  push_mass: float = 1.0

  def __post_init__(self) -> None:
    target_id = _nonempty_string(self.target_id, "target_id")
    if not isinstance(self.recipe, str):
      raise TypeError("recipe must be a string")
    if self.recipe not in INTERVENTION_RECIPES:
      raise ValueError("unsupported recipe: {!r}".format(self.recipe))
    magnitude = _real(self.magnitude, "magnitude")
    if magnitude < 0.0:
      raise ValueError("magnitude must be nonnegative")
    time_window = _vector(self.time_window, 2, "time_window")
    if time_window[0] < 0.0 or time_window[1] <= time_window[0]:
      raise ValueError("time_window must be a nonnegative half-open interval")
    push_mass = _real(self.push_mass, "push_mass")
    if push_mass <= 0.0:
      raise ValueError("push_mass must be positive")
    object.__setattr__(self, "target_id", target_id)
    object.__setattr__(self, "magnitude", magnitude)
    object.__setattr__(self, "time_window", time_window)
    object.__setattr__(self, "push_mass", push_mass)
    object.__setattr__(self, "metadata", _metadata(self.metadata))
    object.__setattr__(self, "schema_version", _version(self.schema_version))


def _temporal_edge(record: Any, name: str, index: int) -> Mapping[str, Any]:
  if not isinstance(record, Mapping):
    raise TypeError("{}[{}] must be a temporal edge mapping".format(name, index))
  required = ("object_a", "object_b", "start_step", "end_step")
  missing = [key for key in required if key not in record]
  if missing:
    raise ValueError(
        "{}[{}] is missing required fields: {}".format(
            name, index, ", ".join(missing)
        )
    )
  object_a = _nonempty_string(record["object_a"], "object_a")
  object_b = _nonempty_string(record["object_b"], "object_b")
  if object_a == object_b:
    raise ValueError("temporal edge endpoints must be distinct")
  object_a, object_b = sorted((object_a, object_b))
  start_step = _integer(record["start_step"], "start_step")
  end_step = _integer(record["end_step"], "end_step")
  if start_step < 0 or end_step <= start_step:
    raise ValueError("temporal edge steps must satisfy 0 <= start_step < end_step")

  normalized = dict(record)
  normalized.update(
      object_a=object_a,
      object_b=object_b,
      start_step=start_step,
      end_step=end_step,
  )
  return _freeze(normalized)


def _records(value: Iterable[Any], name: str) -> Tuple[Any, ...]:
  if value is None:
    return ()
  if isinstance(value, Mapping):
    records = (value,)
  elif isinstance(value, (str, bytes)):
    raise TypeError("{} must be an iterable of records".format(name))
  else:
    try:
      records = tuple(value)
    except TypeError as error:
      raise TypeError("{} must be an iterable of records".format(name)) from error
  by_identity = {}
  serialized = {}
  for index, record in enumerate(records):
    frozen_record = _temporal_edge(record, name, index)
    identity = (
        frozen_record["object_a"],
        frozen_record["object_b"],
        frozen_record["start_step"],
        frozen_record["end_step"],
    )
    payload = _json_sort_key(to_jsonable(frozen_record))
    if identity in serialized and serialized[identity] != payload:
      raise ValueError(
          "{} contains conflicting payloads for edge identity {!r}".format(
              name, identity
          )
      )
    by_identity[identity] = frozen_record
    serialized[identity] = payload
  frozen = tuple(by_identity[identity] for identity in sorted(by_identity))
  # Force validation now so artifact writing cannot fail later.
  to_jsonable(frozen)
  return frozen


@dataclass(frozen=True)
class GraphEdgeDelta(_SchemaMixin):
  """Added, removed, and changed temporal edge records.

  An edge's canonical identity is ``(object_a, object_b, start_step, end_step)``
  with its object identifiers lexicographically ordered.
  """

  added: Tuple[Any, ...] = ()
  removed: Tuple[Any, ...] = ()
  changed: Tuple[Any, ...] = ()
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    added = _records(self.added, "added")
    removed = _records(self.removed, "removed")
    changed = _records(self.changed, "changed")
    identities_by_bucket = {
        "added": {
            (item["object_a"], item["object_b"], item["start_step"], item["end_step"])
            for item in added
        },
        "removed": {
            (item["object_a"], item["object_b"], item["start_step"], item["end_step"])
            for item in removed
        },
        "changed": {
            (item["object_a"], item["object_b"], item["start_step"], item["end_step"])
            for item in changed
        },
    }
    owners = {}
    for bucket, identities in identities_by_bucket.items():
      for identity in identities:
        if identity in owners:
          raise ValueError(
              "edge identity {!r} appears in multiple buckets: {} and {}".format(
                  identity, owners[identity], bucket
              )
          )
        owners[identity] = bucket
    object.__setattr__(self, "added", added)
    object.__setattr__(self, "removed", removed)
    object.__setattr__(self, "changed", changed)
    object.__setattr__(self, "schema_version", _version(self.schema_version))

  @property
  def added_edges(self) -> Tuple[Any, ...]:
    """Compatibility alias for ``added`` records."""
    return self.added

  @property
  def removed_edges(self) -> Tuple[Any, ...]:
    """Compatibility alias for ``removed`` records."""
    return self.removed

  @property
  def changed_edges(self) -> Tuple[Any, ...]:
    """Compatibility alias for ``changed`` records."""
    return self.changed


def _affected_ids(value: Iterable[str], name: str) -> Tuple[str, ...]:
  if isinstance(value, (str, bytes)):
    raise TypeError("{} must be an iterable of identifiers".format(name))
  try:
    identifiers = tuple(value)
  except TypeError as error:
    raise TypeError("{} must be an iterable of identifiers".format(name)) from error
  normalized = {_nonempty_string(item, name) for item in identifiers}
  return tuple(sorted(normalized))


def _propagation_paths(
    value: Mapping[str, Sequence[str]],
) -> Mapping[str, Tuple[str, ...]]:
  if not isinstance(value, Mapping):
    raise TypeError("propagation_path must be a mapping")
  _validate_mapping_keys(value)
  result = {}
  for key in sorted(value):
    normalized_key = _nonempty_string(key, "propagation_path key")
    path = value[key]
    if isinstance(path, (str, bytes)) or not isinstance(path, SequenceABC):
      raise ValueError("propagation paths must be an ordered sequence, not a string")
    normalized_path = tuple(
        _nonempty_string(item, "propagation path item") for item in path
    )
    result[normalized_key] = normalized_path
  return MappingProxyType(result)


@dataclass(frozen=True)
class GroundTruth(_SchemaMixin):
  """Expected graph changes, affected objects, and ordered propagation paths.

  Each ``propagation_path`` value must be an ordered non-string ``Sequence``;
  generators, sets, and NumPy arrays are rejected.
  """

  graph_delta: GraphEdgeDelta
  hard_affected: Tuple[str, ...] = ()
  soft_affected: Tuple[str, ...] = ()
  propagation_path: Mapping[str, Sequence[str]] = field(default_factory=dict)
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if not isinstance(self.graph_delta, GraphEdgeDelta):
      raise TypeError("graph_delta must be GraphEdgeDelta")
    hard = _affected_ids(self.hard_affected, "hard_affected")
    soft = _affected_ids(self.soft_affected, "soft_affected")
    if set(hard).intersection(soft):
      raise ValueError("hard_affected and soft_affected must be disjoint")
    object.__setattr__(self, "hard_affected", hard)
    object.__setattr__(self, "soft_affected", soft)
    object.__setattr__(
        self, "propagation_path", _propagation_paths(self.propagation_path)
    )
    object.__setattr__(self, "schema_version", _version(self.schema_version))
