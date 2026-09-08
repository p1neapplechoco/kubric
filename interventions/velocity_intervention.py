"""Free-body "velocity intervention" scenes: sampling, simulation, QC, ground truth.

Purpose:
  Builds table-top scenes of 4-6 fixed-mass rigid bodies in which exactly one
  body (the *subject*) is launched with an initial linear velocity and nothing
  else is actuated. Every scene is realized as three branches sharing one
  visual scene and one static camera: ``factual`` (sampled initial velocity),
  ``counterfactual`` (the factual velocity discarded and a *new* initial velocity
  applied at step 0) and ``subject_removed`` (the subject absent). The module
  samples the scene, runs each branch with the existing Kubric PyBullet view,
  classifies rolling versus sliding motion, applies QC and extracts ground truth
  with :mod:`interventions.graph_extraction`.

Public API:
  ``BRANCHES``, ``VelocityInstanceSpec``, ``QCReport``, ``GeneratedInstance``,
  ``load_ranges``, ``sample_instance``, ``simulate_scene``, ``simulate_branches``,
  ``motion_summary``, ``evaluate_qc``, ``extract_instance_ground_truth``,
  ``generate_instance``, ``write_instance``, ``read_instance_spec``.

Dependencies:
  NumPy, PyBullet (through :mod:`kubric.simulator.pybullet`), :mod:`kubric.core`,
  and the sibling modules :mod:`interventions.schema`,
  :mod:`interventions.logging`, :mod:`interventions.graph_extraction`,
  :mod:`interventions.appearance`, :mod:`interventions.appearance_sampling`,
  :mod:`interventions.materials` and :mod:`interventions.dataset`
  (``load_ranges`` / ``derive_seed``). Blender is never imported here.

Trust boundary:
  Range documents are validated on load and every sampled value is re-validated
  by the frozen schema dataclasses. Simulation output is only trusted after
  ``evaluate_qc`` passes; callers must treat ``QCReport.passed == False`` as a
  rejected instance. Written artifacts reuse the atomic simulation-log publisher
  and are safe to read back with ``read_simulation_log``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from interventions import appearance, appearance_sampling, materials
from interventions.dataset import derive_seed, load_ranges
from interventions.graph_extraction import (
    contact_log_to_temporal_graph,
    extract_ground_truth,
    graph_delta,
)
from interventions.logging import (
    ANGULAR_VELOCITY_SLICE,
    LINEAR_VELOCITY_SLICE,
    POSITION_SLICE,
    QUATERNION_SLICE,
    ContactLogger,
    SimulationLog,
    read_simulation_log,
    write_simulation_log,
)
from interventions.schema import (
    CameraConfig,
    GroundTruth,
    ObjectConfig,
    SceneConfig,
    shape_half_extents,
    to_jsonable,
)

PathLike = Union[str, "os.PathLike[str]"]

BRANCHES: Tuple[str, str, str] = ("factual", "counterfactual", "subject_removed")
ROLES: Tuple[str, ...] = ("subject", "interactive", "bystander", "floor")
SPEC_FILENAME = "instance.json"
SIM_LOG_DIRNAME = "sim_log"
_PLACEMENT_TRIES = 200
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Small validated helpers
# ---------------------------------------------------------------------------


def _pair(section: Mapping[str, Any], key: str) -> Tuple[float, float]:
  try:
    value = section[key]
  except KeyError as error:
    raise ValueError("missing range key: {!r}".format(key)) from error
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return (float(value), float(value))
  values = tuple(float(item) for item in value)
  if len(values) != 2 or values[0] > values[1]:
    raise ValueError("range {!r} must be [low, high] with low <= high".format(key))
  return values


def _uniform(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
  low, high = bounds
  if low == high:
    return float(low)
  return float(rng.uniform(low, high))


def _int_inclusive(rng: np.random.Generator, bounds: Tuple[float, float]) -> int:
  low, high = int(round(bounds[0])), int(round(bounds[1]))
  if low == high:
    return low
  return int(rng.integers(low, high + 1))


def _weighted_choice(rng: np.random.Generator, weights: Mapping[str, Any]) -> str:
  keys = tuple(sorted(weights))
  raw = np.asarray([float(weights[key]) for key in keys], dtype=np.float64)
  if raw.size == 0 or raw.sum() <= 0.0:
    raise ValueError("weights must be a non-empty mapping with positive mass")
  return str(keys[int(rng.choice(len(keys), p=raw / raw.sum()))])


def _yaw_quaternion(yaw: float) -> Tuple[float, float, float, float]:
  return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _radial_extent(shape: str, size: Sequence[float]) -> float:
  """Largest ground-plane extent regardless of yaw (used for corridor tests)."""
  if shape == "cube":
    return math.hypot(size[0], size[1])
  return float(size[0])


def _volume(shape: str, size: Sequence[float]) -> float:
  return materials.proxy_volume(shape, shape_half_extents(shape, size))


def _round_floats(value: Any, digits: int = 9) -> Any:
  if isinstance(value, float):
    return round(value, digits)
  if isinstance(value, Mapping):
    return {key: _round_floats(item, digits) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_round_floats(item, digits) for item in value]
  return value


def _sha256(payload: Any) -> str:
  # Floats are rounded so that a JSON round trip (and the quaternion
  # re-normalization it triggers) yields the same identifier.
  encoded = json.dumps(_round_floats(to_jsonable(payload)), sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Instance specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VelocityInstanceSpec:
  """One sampled scene plus everything needed to realize its three branches.

  ``scene`` is the factual scene: the subject already carries
  ``factual_velocity`` as its ``linear_velocity``. ``counterfactual_scene`` swaps
  in ``counterfactual_velocity`` and ``removed_scene`` drops the subject.
  ``physics`` maps object ids to the per-body PyBullet dynamics that are not part
  of :class:`ObjectConfig` (rolling/spinning friction, material family).
  """

  master_seed: int
  index: int
  attempt: int
  scene: SceneConfig
  subject_id: str
  floor_id: str
  factual_velocity: Tuple[float, float, float]
  counterfactual_velocity: Tuple[float, float, float]
  roles: Mapping[str, str]
  physics: Mapping[str, Mapping[str, Any]]
  visual: appearance.VisualSceneSpec
  metadata: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    ids = {item.object_id for item in self.scene.objects}
    if self.subject_id not in ids or self.floor_id not in ids:
      raise ValueError("subject_id and floor_id must name scene objects")
    if set(self.roles) != ids:
      raise ValueError("roles must cover exactly the scene objects")
    if any(role not in ROLES for role in self.roles.values()):
      raise ValueError("unknown role in roles")
    if set(self.physics) != ids:
      raise ValueError("physics must cover exactly the scene objects")
    subject = self.object(self.subject_id)
    if tuple(subject.linear_velocity) != tuple(self.factual_velocity):
      raise ValueError("scene subject velocity must equal factual_velocity")
    appearance.validate_scene_correspondence(self.visual, self.scene)

  @property
  def instance_id(self) -> str:
    """Stable identifier: seed, index and a short hash of the scene payload."""
    return "vi_{:d}_{:06d}_{}".format(
        self.master_seed, self.index, _sha256(self.scene.to_dict())[:8]
    )

  @property
  def object_ids(self) -> Tuple[str, ...]:
    """Object ids in scene order (this order defines segmentation ids)."""
    return tuple(item.object_id for item in self.scene.objects)

  def object(self, object_id: str) -> ObjectConfig:
    """Returns the factual :class:`ObjectConfig` for ``object_id``."""
    for item in self.scene.objects:
      if item.object_id == object_id:
        return item
    raise KeyError(object_id)

  @property
  def counterfactual_scene(self) -> SceneConfig:
    """Initial scene for counterfactual (matches factual at t=0; velocity intervenes mid-video)."""
    return self.scene

  @property
  def removed_scene(self) -> SceneConfig:
    """Initial scene for subject_removed (matches factual at t=0; subject removed mid-video)."""
    return self.scene

  def scene_for(self, branch: str) -> SceneConfig:
    """Returns the initial :class:`SceneConfig` realized by ``branch`` at t=0."""
    if branch in BRANCHES:
      return self.scene
    raise ValueError("unknown branch: {!r}".format(branch))

  def to_dict(self) -> Mapping[str, Any]:
    """Deterministic JSON-compatible payload (includes ``instance_id``)."""
    payload = dict(to_jsonable(self))
    payload["instance_id"] = self.instance_id
    payload["visual_scene_hash"] = appearance.visual_scene_hash(self.visual)
    return payload


def _spec_from_payload(payload: Mapping[str, Any]) -> VelocityInstanceSpec:
  scene_payload = payload["scene"]
  objects = tuple(ObjectConfig(**item) for item in scene_payload["objects"])
  camera = scene_payload.get("camera")
  scene = SceneConfig(
      objects=objects,
      camera=CameraConfig(**camera) if camera else None,
      seed=scene_payload["seed"],
      scene_bounds=scene_payload["scene_bounds"],
      gravity=scene_payload["gravity"],
      frame_range=scene_payload["frame_range"],
      frame_rate=scene_payload["frame_rate"],
      step_rate=scene_payload["step_rate"],
  )
  return VelocityInstanceSpec(
      master_seed=int(payload["master_seed"]),
      index=int(payload["index"]),
      attempt=int(payload["attempt"]),
      scene=scene,
      subject_id=payload["subject_id"],
      floor_id=payload["floor_id"],
      factual_velocity=tuple(payload["factual_velocity"]),
      counterfactual_velocity=tuple(payload["counterfactual_velocity"]),
      roles=dict(payload["roles"]),
      physics=dict(payload["physics"]),
      visual=appearance.visual_scene_from_payload(payload["visual"]),
      metadata=dict(payload.get("metadata", {})),
  )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass
class _Body:
  object_id: str
  role: str
  shape: str
  size: Tuple[float, float, float]
  position: Tuple[float, float, float]
  quaternion: Tuple[float, float, float, float]

  @property
  def radial(self) -> float:
    return _radial_extent(self.shape, self.size)


def _sample_size(
    rng: np.random.Generator, shape: str, size_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
) -> Tuple[float, float, float]:
  primary = _uniform(rng, size_range)
  if shape == "sphere":
    return (primary, primary, primary)
  height = primary * _uniform(rng, aspect_range)
  return (primary, primary, height)


def _rest_height(shape: str, size: Sequence[float]) -> float:
  # Half-extent along local Z for an upright body, plus a hair of clearance so the
  # first step does not start inside the floor.
  return shape_half_extents(shape, size)[2] + 1e-3


def _clear_of(body: _Body, others: Sequence[_Body], gap: float) -> bool:
  for other in others:
    distance = math.hypot(
        body.position[0] - other.position[0], body.position[1] - other.position[1]
    )
    if distance < body.radial + other.radial + gap:
      return False
  return True


def _inside(position: Sequence[float], radial: float, bounds) -> bool:
  (x_min, y_min, _), (x_max, y_max, _) = bounds
  return (x_min + radial < position[0] < x_max - radial
          and y_min + radial < position[1] < y_max - radial)


def _sample_layout(
    ranges: Mapping[str, Any], rng: np.random.Generator
) -> Tuple[List[_Body], Tuple[float, float, float], Tuple[float, float, float]]:
  """Samples subject, interactive and bystander bodies plus both velocities."""
  objects = ranges["objects"]
  subject_ranges = ranges["subject"]
  interactive_ranges = ranges["interactive"]
  bystander_ranges = ranges["bystanders"]
  counterfactual = ranges["counterfactual"]
  scene_bounds = ranges["scene"]["bounds"]
  floor_half = ranges["floor"]["half_extents"]
  floor_bounds = ((-floor_half[0], -floor_half[1], 0.0), (floor_half[0], floor_half[1], 0.0))
  gap = float(objects["min_gap"])

  total = _int_inclusive(rng, _pair(objects, "count"))
  interactive_count = min(
      _int_inclusive(rng, _pair(interactive_ranges, "count")), total - 1
  )
  bystander_count = total - 1 - interactive_count
  if bystander_count < 1 and int(ranges["qc"].get("min_untouched", 0)) >= 1:
    interactive_count -= 1
    bystander_count += 1

  # Subject
  subject_shape = _weighted_choice(rng, subject_ranges["shapes"])
  subject_size = _sample_size(
      rng, subject_shape, _pair(subject_ranges, "size"), _pair(objects, "aspect_ratio")
  )
  start = (
      _uniform(rng, _pair(subject_ranges, "start_x")),
      _uniform(rng, _pair(subject_ranges, "start_y")),
      _rest_height(subject_shape, subject_size),
  )
  heading = _uniform(rng, _pair(subject_ranges, "heading"))
  speed = _uniform(rng, _pair(subject_ranges, "speed"))
  direction = (math.cos(heading), math.sin(heading))
  normal = (-direction[1], direction[0])
  yaw = _uniform(rng, (0.0, _TWO_PI)) if subject_shape != "sphere" else 0.0
  subject = _Body(
      object_id=str(subject_ranges.get("object_id", "subject")),
      role="subject",
      shape=subject_shape,
      size=subject_size,
      position=start,
      quaternion=_yaw_quaternion(yaw),
  )
  bodies = [subject]

  # Interactive bodies inside the corridor, ordered along the heading.
  shape_weights = objects["shapes"]
  size_range = _pair(objects, "size")
  aspect_range = _pair(objects, "aspect_ratio")
  distance_range = _pair(interactive_ranges, "distance")
  lateral_range = _pair(interactive_ranges, "lateral")
  placed = 0
  for _ in range(_PLACEMENT_TRIES):
    if placed >= interactive_count:
      break
    shape = _weighted_choice(rng, shape_weights)
    size = _sample_size(rng, shape, size_range, aspect_range)
    radial = _radial_extent(shape, size)
    along = _uniform(rng, distance_range)
    lateral = _uniform(rng, lateral_range) * (subject.radial + radial)
    position = (
        start[0] + along * direction[0] + lateral * normal[0],
        start[1] + along * direction[1] + lateral * normal[1],
        _rest_height(shape, size),
    )
    candidate = _Body(
        object_id="obj_{:02d}".format(len(bodies)),
        role="interactive",
        shape=shape,
        size=size,
        position=position,
        quaternion=_yaw_quaternion(_uniform(rng, (0.0, _TWO_PI)) if shape != "sphere" else 0.0),
    )
    if not _inside(position, radial, floor_bounds) or not _inside(position, radial, scene_bounds):
      continue
    if not _clear_of(candidate, bodies, gap):
      continue
    bodies.append(candidate)
    placed += 1
  if placed < interactive_count:
    raise RuntimeError("could not place interactive bodies")

  # Bystanders outside the corridor.
  clearance = float(bystander_ranges["lateral_clearance"])
  x_range = _pair(bystander_ranges, "x")
  y_range = _pair(bystander_ranges, "y")
  placed = 0
  for _ in range(_PLACEMENT_TRIES):
    if placed >= bystander_count:
      break
    shape = _weighted_choice(rng, shape_weights)
    size = _sample_size(rng, shape, size_range, aspect_range)
    radial = _radial_extent(shape, size)
    position = (_uniform(rng, x_range), _uniform(rng, y_range), _rest_height(shape, size))
    rel = (position[0] - start[0], position[1] - start[1])
    along = rel[0] * direction[0] + rel[1] * direction[1]
    lateral = abs(rel[0] * normal[0] + rel[1] * normal[1])
    ahead = along > -subject.radial
    if ahead and lateral < subject.radial + radial + clearance:
      continue
    candidate = _Body(
        object_id="obj_{:02d}".format(len(bodies)),
        role="bystander",
        shape=shape,
        size=size,
        position=position,
        quaternion=_yaw_quaternion(_uniform(rng, (0.0, _TWO_PI)) if shape != "sphere" else 0.0),
    )
    if not _inside(position, radial, floor_bounds) or not _inside(position, radial, scene_bounds):
      continue
    if not _clear_of(candidate, bodies, gap):
      continue
    bodies.append(candidate)
    placed += 1
  if placed < bystander_count:
    raise RuntimeError("could not place bystander bodies")

  if bool(objects.get("require_sphere_and_box", False)):
    shapes = {body.shape for body in bodies}
    if "sphere" not in shapes or not shapes.intersection({"cube", "cylinder"}):
      raise RuntimeError("layout lacks both a rolling and a sliding body")

  factual_velocity = (speed * direction[0], speed * direction[1], 0.0)
  cf_speed = _uniform(rng, _pair(counterfactual, "speed"))
  delta = _uniform(rng, _pair(counterfactual, "heading_delta"))
  sign = 1.0 if rng.random() < 0.5 else -1.0
  cf_heading = heading + sign * delta
  counterfactual_velocity = (
      cf_speed * math.cos(cf_heading), cf_speed * math.sin(cf_heading), 0.0
  )
  return bodies, factual_velocity, counterfactual_velocity


def _project_normalized(
    camera_position: Sequence[float], look_at: Sequence[float], focal_length: float,
    sensor_width: float, point: Sequence[float],
) -> Optional[Tuple[float, float]]:
  """Pinhole projection to [0, 1]^2 image coordinates (square sensor, +Z up)."""
  eye = np.asarray(camera_position, dtype=np.float64)
  forward = np.asarray(look_at, dtype=np.float64) - eye
  forward /= np.linalg.norm(forward)
  up = np.array([0.0, 0.0, 1.0])
  right = np.cross(forward, up)
  if np.linalg.norm(right) < 1e-9:
    right = np.array([1.0, 0.0, 0.0])
  right /= np.linalg.norm(right)
  true_up = np.cross(right, forward)
  relative = np.asarray(point, dtype=np.float64) - eye
  depth = float(relative @ forward)
  if depth <= 1e-6:
    return None
  x = float(relative @ right) / depth * focal_length / sensor_width
  y = float(relative @ true_up) / depth * focal_length / sensor_width
  return (0.5 + x, 0.5 - y)


def _static_camera(
    ranges: Mapping[str, Any], rng: np.random.Generator, scene: SceneConfig,
    dynamic_ids: Sequence[str], subject_id: str, factual_velocity: Sequence[float],
) -> Tuple[appearance.CameraRenderSpec, CameraConfig]:
  """One camera pose per scene, framed so every body and the subject's path fit."""
  camera_ranges = ranges["appearance"]["camera"]
  bodies = {item.object_id: item for item in scene.objects if item.object_id in dynamic_ids}
  subject = bodies[subject_id]
  speed = math.hypot(factual_velocity[0], factual_velocity[1])
  direction = (factual_velocity[0] / speed, factual_velocity[1] / speed) if speed else (1.0, 0.0)
  travel = float(camera_ranges.get("expected_travel", 2.5))
  path_end = (subject.position[0] + travel * direction[0], subject.position[1] + travel * direction[1], 0.0)
  points = [item.position for item in bodies.values()] + [path_end]
  centroid = np.mean(np.asarray(points), axis=0)
  look_at = (
      float(centroid[0]), float(centroid[1]),
      _uniform(rng, _pair(camera_ranges, "look_at_z")) if "look_at_z" in camera_ranges else 0.15,
  )
  radius_range = _pair(camera_ranges, "radius")
  radius = _uniform(rng, radius_range)
  elevation = _uniform(rng, _pair(camera_ranges, "elevation"))
  azimuth = _uniform(rng, _pair(camera_ranges, "azimuth"))
  focal_length = _uniform(rng, _pair(camera_ranges, "focal_length"))
  sensor_width = float(camera_ranges.get("sensor_width", 36.0))
  margin = float(camera_ranges.get("frame_margin", 0.08))
  position = look_at
  for _ in range(8):
    position = (
        look_at[0] + radius * math.cos(elevation) * math.cos(azimuth),
        look_at[1] + radius * math.cos(elevation) * math.sin(azimuth),
        look_at[2] + radius * math.sin(elevation),
    )
    projected = [
        _project_normalized(position, look_at, focal_length, sensor_width, point)
        for point in points
    ]
    if all(p is not None and margin <= p[0] <= 1.0 - margin and margin <= p[1] <= 1.0 - margin
           for p in projected):
      break
    radius = min(radius * 1.15, radius_range[1] * 1.6)  # widen the shot, bounded
  frames = len(appearance.frame_steps_for(scene))
  spec = appearance.CameraRenderSpec(
      positions=(position,) * frames,
      look_ats=(look_at,) * frames,
      focal_length=focal_length,
      sensor_width=sensor_width,
      clipping_range=tuple(camera_ranges.get("clipping_range", (0.1, 200.0))),
  )
  return spec, CameraConfig(position=position, look_at=look_at, focal_length=focal_length)


def sample_instance(
    ranges: Mapping[str, Any], master_seed: int, index: int, attempt: int = 0
) -> VelocityInstanceSpec:
  """Samples one instance deterministically from ``(master_seed, index, attempt)``.

  Raises ``RuntimeError`` when the layout cannot be placed; callers resample with
  the next ``attempt``.
  """
  attempt_seed = derive_seed(int(master_seed), int(index), "attempt:{:d}".format(int(attempt)))
  rng = np.random.default_rng(derive_seed(attempt_seed, index, "layout"))
  physics_rng = np.random.default_rng(derive_seed(attempt_seed, index, "physics"))
  camera_rng = np.random.default_rng(derive_seed(attempt_seed, index, "static_camera"))
  floor_rng = np.random.default_rng(derive_seed(attempt_seed, index, "floor_material"))

  scene_ranges = ranges["scene"]
  floor_ranges = ranges["floor"]
  physics_ranges = ranges["physics"]
  mass = float(ranges["objects"]["mass"])

  bodies, factual_velocity, counterfactual_velocity = _sample_layout(ranges, rng)
  floor_id = str(floor_ranges.get("object_id", "floor"))
  floor_half = tuple(float(v) for v in floor_ranges["half_extents"])
  floor = ObjectConfig(
      object_id=floor_id,
      shape="cube",
      size=floor_half,
      mass=0.0,
      friction=float(floor_ranges["friction"]),
      restitution=float(floor_ranges["restitution"]),
      position=(0.0, 0.0, -floor_half[2]),
      static=True,
      metadata={"role": "floor"},
  )

  preliminary_objects = [floor] + [
      ObjectConfig(
          object_id=body.object_id, shape=body.shape, size=body.size, mass=mass,
          position=body.position, quaternion=body.quaternion,
          metadata={"role": body.role},
      )
      for body in bodies
  ]
  preliminary = SceneConfig(
      objects=tuple(preliminary_objects),
      seed=attempt_seed % (2 ** 31),
      scene_bounds=tuple(tuple(v) for v in scene_ranges["bounds"]),
      gravity=tuple(scene_ranges["gravity"]),
      frame_range=tuple(scene_ranges["frame_range"]),
      frame_rate=int(scene_ranges["frame_rate"]),
      step_rate=int(scene_ranges["step_rate"]),
  )

  # Appearance is sampled with the existing component; the camera is replaced by a
  # single pose held for the whole clip and the floor gets its own material family.
  visual = appearance_sampling.sample_visual_scene(ranges, preliminary, attempt_seed, index)
  validated = appearance_sampling.validate_appearance_ranges(ranges)
  floor_material_ranges = ranges["appearance"].get("floor_material")
  visual_objects = []
  for item in visual.objects:
    if item.object_id == floor_id and floor_material_ranges:
      family = _weighted_choice(
          floor_rng, dict(zip(floor_material_ranges["families"], floor_material_ranges["weights"]))
      )
      color = appearance_sampling.sample_color(validated, floor_rng)
      texture = appearance_sampling.sample_texture(validated, floor_rng, family)
      material = materials.sample_material(floor_rng, family, color, texture)
      item = dataclasses.replace(item, material=material)
    visual_objects.append(item)
  family_by_id = {item.object_id: item.material.family for item in visual_objects}

  # Physics coupled to the sampled material family; mass stays fixed.
  physics: Dict[str, Mapping[str, Any]] = {}
  final_objects = []
  for item in preliminary.objects:
    family = family_by_id[item.object_id]
    if item.object_id == floor_id:
      physics[item.object_id] = {
          "material_family": family, "rolling_friction": 0.0, "spinning_friction": 0.0,
          "linear_damping": 0.0, "angular_damping": 0.0,
      }
      final_objects.append(item)
      continue
    coupled = materials.coupled_physics(physics_rng, family, _volume(item.shape, item.size))
    physics[item.object_id] = {
        "material_family": family,
        "rolling_friction": _uniform(physics_rng, _pair(physics_ranges, "rolling_friction")),
        "spinning_friction": _uniform(physics_rng, _pair(physics_ranges, "spinning_friction")),
        "linear_damping": float(physics_ranges.get("linear_damping", 0.0)),
        "angular_damping": float(physics_ranges.get("angular_damping", 0.0)),
        "effective_density": coupled["effective_density"],
    }
    velocity = (0.0, 0.0, 0.0)
    angular = (0.0, 0.0, 0.0)
    if item.metadata["role"] == "subject":
      velocity = factual_velocity
      spin = _uniform(rng, _pair(ranges["subject"], "angular_velocity"))
      # Spin about the axis perpendicular to the heading (a forward roll).
      speed = math.hypot(velocity[0], velocity[1])
      if spin and speed > 0.0:
        angular = (-velocity[1] / speed * spin, velocity[0] / speed * spin, 0.0)
    final_objects.append(dataclasses.replace(
        item, friction=coupled["friction"], restitution=coupled["restitution"],
        linear_velocity=velocity, angular_velocity=angular,
    ))
  scene = dataclasses.replace(preliminary, objects=tuple(final_objects))
  dynamic_ids = [item.object_id for item in scene.objects if not item.static]
  subject_id = next(item.object_id for item in scene.objects if item.metadata["role"] == "subject")
  camera_spec, camera_config = _static_camera(
      ranges, camera_rng, scene, dynamic_ids, subject_id, factual_velocity
  )
  scene = dataclasses.replace(scene, camera=camera_config)
  visual = dataclasses.replace(visual, objects=tuple(visual_objects), camera=camera_spec)

  roles = {item.object_id: item.metadata["role"] for item in scene.objects}
  return VelocityInstanceSpec(
      master_seed=int(master_seed),
      index=int(index),
      attempt=int(attempt),
      scene=scene,
      subject_id=subject_id,
      floor_id=floor_id,
      factual_velocity=factual_velocity,
      counterfactual_velocity=counterfactual_velocity,
      roles=roles,
      physics=physics,
      visual=visual,
      metadata={
          "attempt_seed": attempt_seed,
          "object_count": len(dynamic_ids),
          "interactive_ids": [oid for oid, role in roles.items() if role == "interactive"],
          "bystander_ids": [oid for oid, role in roles.items() if role == "bystander"],
          "factual_speed": math.hypot(*factual_velocity[:2]),
          "counterfactual_speed": math.hypot(*counterfactual_velocity[:2]),
          "heading_change_rad": math.atan2(
              factual_velocity[0] * counterfactual_velocity[1]
              - factual_velocity[1] * counterfactual_velocity[0],
              factual_velocity[0] * counterfactual_velocity[0]
              + factual_velocity[1] * counterfactual_velocity[1],
          ),
      },
  )


# ---------------------------------------------------------------------------
# Simulation with the existing Kubric PyBullet view
# ---------------------------------------------------------------------------


def _kubric_asset(item: ObjectConfig):
  from kubric import core  # pylint: disable=import-outside-toplevel

  kinds = {"cube": core.Cube, "sphere": core.Sphere,
           "cylinder": core.Cylinder, "capsule": core.Capsule}
  asset = kinds[item.shape](
      scale=item.size,
      position=item.position,
      quaternion=item.quaternion,
      mass=item.mass if not item.static else 1.0,
      friction=item.friction,
      restitution=item.restitution,
      static=item.static,
  )
  asset.metadata["logical_id"] = item.object_id
  return asset


def simulate_scene(
    scene: SceneConfig, physics: Mapping[str, Mapping[str, Any]], branch: str,
    scratch_dir: Optional[PathLike] = None,
    subject_id: Optional[str] = None,
    intervention_step: Optional[int] = None,
    heading_change_rad: Optional[float] = None,
) -> SimulationLog:
  """Runs ``scene`` as free rigid bodies and returns a per-step :class:`SimulationLog`.

  For ``factual``, the subject runs with its initial velocity throughout.
  For ``counterfactual``, the subject's velocity heading is altered mid-video at ``intervention_step``.
  For ``subject_removed``, the subject is removed mid-video at ``intervention_step`` before collision.
  """
  from kubric import core  # pylint: disable=import-outside-toplevel
  from kubric.simulator.pybullet import PyBullet  # pylint: disable=import-outside-toplevel

  frame_start, frame_end = scene.frame_range
  steps_per_frame = scene.step_rate // scene.frame_rate
  total_steps = (frame_end - frame_start) * steps_per_frame
  owned_scratch = scratch_dir is None
  scratch = Path(tempfile.mkdtemp(prefix="velocity_sim_")) if owned_scratch else Path(scratch_dir)

  kscene = core.Scene(
      frame_start=frame_start, frame_end=frame_end - 1, frame_rate=scene.frame_rate,
      step_rate=scene.step_rate, resolution=(64, 64), gravity=scene.gravity,
  )
  simulator = PyBullet(kscene, scratch_dir=str(scratch))
  client = simulator._physics_client  # pylint: disable=protected-access
  try:
    client.setTimeStep(1.0 / float(scene.step_rate))
    assets = []
    body_ids: Dict[int, str] = {}
    subject_body = None
    subject_col = None
    for column, item in enumerate(scene.objects):
      asset = _kubric_asset(item)
      kscene.add(asset)
      body = int(asset.linked_objects[simulator])
      dynamics = physics[item.object_id]
      client.changeDynamics(
          body, -1,
          lateralFriction=item.friction,
          restitution=item.restitution,
          rollingFriction=float(dynamics.get("rolling_friction", 0.0)),
          spinningFriction=float(dynamics.get("spinning_friction", 0.0)),
          linearDamping=float(dynamics.get("linear_damping", 0.0)),
          angularDamping=float(dynamics.get("angular_damping", 0.0)),
      )
      if not item.static:
        client.resetBaseVelocity(
            body, linearVelocity=item.linear_velocity, angularVelocity=item.angular_velocity
        )
      assets.append((item.object_id, body))
      body_ids[body] = item.object_id
      if item.object_id == subject_id:
        subject_body = body
        subject_col = column

    logger = ContactLogger(body_ids, step_rate=float(scene.step_rate), force_epsilon=1e-6)
    states = np.zeros((total_steps + 1, len(assets), 13), dtype=np.float64)

    def snapshot(row: int) -> None:
      for column, (_, body) in enumerate(assets):
        position, xyzw = client.getBasePositionAndOrientation(body)
        velocity, angular = client.getBaseVelocity(body)
        states[row, column, POSITION_SLICE] = position
        states[row, column, QUATERNION_SLICE] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
        states[row, column, LINEAR_VELOCITY_SLICE] = velocity
        states[row, column, ANGULAR_VELOCITY_SLICE] = angular

    snapshot(0)
    for step in range(1, total_steps + 1):
      # Mid-video intervention at intervention_step
      if intervention_step is not None and step == intervention_step and subject_body is not None:
        if branch == "counterfactual":
          cur_vel, cur_ang = client.getBaseVelocity(subject_body)
          speed = math.hypot(cur_vel[0], cur_vel[1])
          if speed > 1e-4 and heading_change_rad is not None:
            heading = math.atan2(cur_vel[1], cur_vel[0])
            new_heading = heading + heading_change_rad
            new_vx = speed * math.cos(new_heading)
            new_vy = speed * math.sin(new_heading)
            spin = math.hypot(cur_ang[0], cur_ang[1])
            new_ang = (
                (-new_vy / speed * spin, new_vx / speed * spin, cur_ang[2])
                if spin > 1e-4 else cur_ang
            )
            client.resetBaseVelocity(
                subject_body, linearVelocity=(new_vx, new_vy, cur_vel[2]), angularVelocity=new_ang
            )
        elif branch == "subject_removed":
          client.resetBasePositionAndOrientation(subject_body, [0.0, 0.0, -1000.0], [0.0, 0.0, 0.0, 1.0])
          client.resetBaseVelocity(subject_body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

      client.stepSimulation()

      contacts = client.getContactPoints()
      if (
          branch == "subject_removed"
          and intervention_step is not None
          and step >= intervention_step
          and subject_body is not None
      ):
        client.resetBasePositionAndOrientation(subject_body, [0.0, 0.0, -1000.0], [0.0, 0.0, 0.0, 1.0])
        client.resetBaseVelocity(subject_body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        contacts = tuple(
            cp for cp in contacts
            if cp[1] != subject_body and cp[2] != subject_body
        )

      logger.log(step, tuple(contacts))
      snapshot(step)

      if (
          branch == "subject_removed"
          and intervention_step is not None
          and step >= intervention_step
          and subject_col is not None
      ):
        states[step, subject_col, POSITION_SLICE] = (0.0, 0.0, -1000.0)
        states[step, subject_col, QUATERNION_SLICE] = (1.0, 0.0, 0.0, 0.0)
        states[step, subject_col, LINEAR_VELOCITY_SLICE] = (0.0, 0.0, 0.0)
        states[step, subject_col, ANGULAR_VELOCITY_SLICE] = (0.0, 0.0, 0.0)
  finally:
    try:
      client.disconnect()
    except Exception:  # pragma: no cover - best effort cleanup
      pass
    client._client = -1  # pylint: disable=protected-access
    if owned_scratch:
      shutil.rmtree(scratch, ignore_errors=True)

  return SimulationLog(
      branch=branch,
      object_ids=tuple(object_id for object_id, _ in assets),
      steps=tuple(range(total_steps + 1)),
      states=states,
      contacts=logger.records,
      step_rate=float(scene.step_rate),
      metadata={
          "frame_range": list(scene.frame_range),
          "frame_rate": scene.frame_rate,
          "steps_per_frame": steps_per_frame,
          "simulator": "kubric.simulator.pybullet.PyBullet",
          "intervention_step": intervention_step if intervention_step is not None else 0,
          "intervention_frame": (intervention_step // steps_per_frame) if intervention_step is not None else 0,
      },
  )


def _simulate_colliding_counterfactual(
    spec: VelocityInstanceSpec,
    factual_log: SimulationLog,
    intervention_step: int,
    rng: np.random.Generator,
) -> SimulationLog:
  """Finds a candidate heading change at ``intervention_step`` that prioritizes causing a collision."""
  steps_per_frame = spec.scene.step_rate // spec.scene.frame_rate
  total_steps = (spec.scene.frame_range[1] - spec.scene.frame_range[0]) * steps_per_frame
  rem_time = (total_steps - intervention_step) / spec.scene.step_rate

  sub_col = factual_log.object_ids.index(spec.subject_id)
  sub_pos = factual_log.states[intervention_step, sub_col, POSITION_SLICE]
  sub_vel = factual_log.states[intervention_step, sub_col, LINEAR_VELOCITY_SLICE]
  sub_speed = math.hypot(sub_vel[0], sub_vel[1])
  cur_heading = math.atan2(sub_vel[1], sub_vel[0])

  f_touched = _touched_ids(factual_log, spec.floor_id).get(spec.subject_id, set())
  by_id = {item.object_id: item for item in spec.scene.objects}
  sub_item = by_id[spec.subject_id]
  r_sub = _radial_extent(sub_item.shape, sub_item.size)

  candidates = []
  for oid in spec.object_ids:
    if oid in (spec.subject_id, spec.floor_id):
      continue
    col = factual_log.object_ids.index(oid)
    pos = factual_log.states[intervention_step, col, POSITION_SLICE]
    item = by_id[oid]
    r_obj = _radial_extent(item.shape, item.size)
    r_contact = r_sub + r_obj

    dx = pos[0] - sub_pos[0]
    dy = pos[1] - sub_pos[1]
    dist = math.hypot(dx, dy)
    if dist <= r_contact or dist > sub_speed * rem_time * 0.95:
      continue
    angle = math.atan2(dy, dx)
    delta = (angle - cur_heading + math.pi) % (2 * math.pi) - math.pi
    if abs(delta) > math.pi / 2:  # must be in front
      continue
    if abs(delta) < 0.10:  # significant heading change
      continue
    span = math.asin(min(0.99, r_contact / dist))
    is_novel = oid not in f_touched
    candidates.append((is_novel, -abs(delta), delta, span, oid))

  candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

  # Try simulating candidate targets to guarantee a counterfactual collision
  for is_novel, _, delta, span, oid in candidates:
    jitter = float(rng.uniform(-0.35 * span, 0.35 * span))
    target_delta = delta + jitter
    log = simulate_scene(
        spec.scene, spec.physics, "counterfactual",
        subject_id=spec.subject_id, intervention_step=intervention_step,
        heading_change_rad=target_delta,
    )
    struck = _touched_ids(log, spec.floor_id).get(spec.subject_id, set())
    if struck:
      new_meta = dict(log.metadata)
      new_meta["counterfactual_target_id"] = oid
      new_meta["heading_change_rad"] = target_delta
      return dataclasses.replace(log, metadata=new_meta)

  # Fallback to metadata heading change if no candidate produced a collision
  fallback_delta = float(spec.metadata.get("heading_change_rad", 0.4))
  if abs(fallback_delta) < 0.1:
    fallback_delta = 0.4
  log = simulate_scene(
      spec.scene, spec.physics, "counterfactual",
      subject_id=spec.subject_id, intervention_step=intervention_step,
      heading_change_rad=fallback_delta,
  )
  new_meta = dict(log.metadata)
  new_meta["counterfactual_target_id"] = None
  new_meta["heading_change_rad"] = fallback_delta
  return dataclasses.replace(log, metadata=new_meta)


def simulate_branches(spec: VelocityInstanceSpec) -> Mapping[str, SimulationLog]:
  """Simulates ``factual``, ``counterfactual`` (mid-video velocity change) and ``subject_removed`` (mid-video removal)."""
  factual_log = simulate_scene(spec.scene, spec.physics, "factual", subject_id=spec.subject_id)

  steps_per_frame = spec.scene.step_rate // spec.scene.frame_rate
  total_steps = (spec.scene.frame_range[1] - spec.scene.frame_range[0]) * steps_per_frame
  collision_steps = [
      r.step for r in factual_log.contacts
      if (r.object_a == spec.subject_id or r.object_b == spec.subject_id)
      and r.object_a != spec.floor_id and r.object_b != spec.floor_id
  ]
  if collision_steps:
    first_collision = min(collision_steps)
    intervention_step = max(steps_per_frame, min(int(round(first_collision * 0.5)), first_collision - steps_per_frame))
  else:
    intervention_step = max(steps_per_frame, total_steps // 3)

  rng = np.random.default_rng(derive_seed(spec.master_seed, spec.index, "cf_target"))
  counterfactual_log = _simulate_colliding_counterfactual(
      spec, factual_log, intervention_step, rng
  )
  removed_log = simulate_scene(
      spec.scene, spec.physics, "subject_removed",
      subject_id=spec.subject_id, intervention_step=intervention_step,
  )
  return {
      "factual": factual_log,
      "counterfactual": counterfactual_log,
      "subject_removed": removed_log,
  }


# ---------------------------------------------------------------------------
# Motion classification and QC
# ---------------------------------------------------------------------------


def motion_summary(
    log: SimulationLog, scene: SceneConfig, *, slip_threshold: float = 0.15,
    roll_tolerance: float = 0.25, speed_floor: float = 0.05,
) -> Mapping[str, Mapping[str, Any]]:
  """Classifies each dynamic body as rolling, sliding, mixed or static.

  For a body resting on the floor with radius/half-height ``r``, the slip speed at
  the contact point is ``|v_xy + (omega x (0,0,-r))_xy|``. Frames with speed above
  ``speed_floor`` count as *rolling* when ``|omega_xy| r`` matches ``|v_xy|``
  within ``roll_tolerance`` and the slip is small, and as *sliding* when the slip
  exceeds ``slip_threshold``.
  """
  by_id = {item.object_id: item for item in scene.objects}
  summary: Dict[str, Mapping[str, Any]] = {}
  for column, object_id in enumerate(log.object_ids):
    item = by_id[object_id]
    if item.static:
      continue
    states = log.states[:, column, :]
    velocity = states[:, LINEAR_VELOCITY_SLICE]
    angular = states[:, ANGULAR_VELOCITY_SLICE]
    r = shape_half_extents(item.shape, item.size)[2]
    speed_xy = np.linalg.norm(velocity[:, :2], axis=1)
    # omega x (0,0,-r) = (-omega_y r, omega_x r, 0)
    slip = np.linalg.norm(
        velocity[:, :2] + np.stack([-angular[:, 1] * r, angular[:, 0] * r], axis=1), axis=1
    )
    omega_r = np.linalg.norm(angular[:, :2], axis=1) * r
    moving = speed_xy > speed_floor
    rolling = moving & (slip <= slip_threshold) & (
        np.abs(omega_r - speed_xy) <= roll_tolerance * np.maximum(speed_xy, 1e-9)
    )
    sliding = moving & (slip > slip_threshold)
    moving_count = int(moving.sum())
    rolling_fraction = float(rolling.sum() / moving_count) if moving_count else 0.0
    sliding_fraction = float(sliding.sum() / moving_count) if moving_count else 0.0
    if moving_count == 0:
      label = "static"
    elif rolling_fraction >= 0.6 and sliding_fraction < 0.25:
      label = "rolling"
    elif sliding_fraction >= 0.6 and rolling_fraction < 0.25:
      label = "sliding"
    elif rolling_fraction < 0.1 and sliding_fraction < 0.1:
      label = "drifting"  # moving but neither regime dominates (e.g. airborne)
    else:
      label = "mixed"
    travel = float(np.linalg.norm(
        states[-1, POSITION_SLICE][:2] - states[0, POSITION_SLICE][:2]
    ))
    summary[object_id] = {
        "label": label,
        "rolling_fraction": rolling_fraction,
        "sliding_fraction": sliding_fraction,
        "moving_steps": moving_count,
        "travel": travel,
        "max_speed": float(np.linalg.norm(velocity, axis=1).max()),
        "max_angular_speed": float(np.linalg.norm(angular, axis=1).max()),
        "final_speed": float(np.linalg.norm(velocity[-1])),
    }
  return summary


def _touched_ids(log: SimulationLog, floor_id: str) -> Dict[str, set]:
  """Maps every object id to the set of non-floor ids it contacted."""
  touched: Dict[str, set] = {object_id: set() for object_id in log.object_ids}
  for record in log.contacts:
    if floor_id in (record.object_a, record.object_b):
      continue
    touched[record.object_a].add(record.object_b)
    touched[record.object_b].add(record.object_a)
  return touched


@dataclass(frozen=True)
class QCReport:
  """Outcome of :func:`evaluate_qc` with human-readable rejection reasons."""

  passed: bool
  reasons: Tuple[str, ...]
  metrics: Mapping[str, Any]

  def to_dict(self) -> Mapping[str, Any]:
    """JSON-compatible payload."""
    return to_jsonable(self)


def evaluate_qc(
    spec: VelocityInstanceSpec, logs: Mapping[str, SimulationLog],
    ranges: Mapping[str, Any],
) -> QCReport:
  """Checks the physical and causal requirements of the dataset on all branches."""
  qc = ranges["qc"]
  reasons: List[str] = []
  metrics: Dict[str, Any] = {}
  factual = logs["factual"]
  counterfactual = logs["counterfactual"]
  removed = logs["subject_removed"]
  floor_id = spec.floor_id
  subject_id = spec.subject_id

  motion = {
      branch: motion_summary(
          logs[branch], spec.scene_for(branch),
          slip_threshold=float(qc.get("slip_threshold", 0.15)),
          roll_tolerance=float(qc.get("roll_tolerance", 0.25)),
      )
      for branch in BRANCHES
  }
  metrics["motion"] = motion

  touched = _touched_ids(factual, floor_id)
  struck = sorted(touched[subject_id])
  dynamic_ids = [oid for oid in factual.object_ids if oid != floor_id]
  untouched = sorted(oid for oid in dynamic_ids if oid != subject_id and not touched[oid])
  metrics["factual_struck"] = struck
  metrics["factual_untouched"] = untouched
  if len(struck) < int(qc.get("min_struck", 1)):
    reasons.append("subject struck {} bodies (< {})".format(len(struck), qc.get("min_struck", 1)))
  if len(untouched) < int(qc.get("min_untouched", 0)):
    reasons.append("only {} untouched bodies".format(len(untouched)))

  cf_touched = _touched_ids(counterfactual, floor_id)
  cf_struck = sorted(cf_touched[subject_id])
  metrics["counterfactual_struck"] = cf_struck
  min_cf_struck = int(qc.get("min_counterfactual_struck", 1))
  if len(cf_struck) < min_cf_struck:
    reasons.append("counterfactual subject struck {} bodies (< {})".format(len(cf_struck), min_cf_struck))

  factual_motion = motion["factual"]
  labels = {label for info in factual_motion.values() for label in (info["label"],)}
  has_rolling = any(info["rolling_fraction"] > 0.2 for info in factual_motion.values())
  has_sliding = any(info["sliding_fraction"] > 0.2 for info in factual_motion.values())
  metrics["factual_labels"] = sorted(labels)
  if bool(qc.get("require_rolling", True)) and not has_rolling:
    reasons.append("no rolling body in factual")
  if bool(qc.get("require_sliding", True)) and not has_sliding:
    reasons.append("no sliding body in factual")

  travel = factual_motion[subject_id]["travel"]
  metrics["subject_travel"] = travel
  if travel < float(qc.get("min_subject_travel", 0.0)):
    reasons.append("subject travelled {:.2f} m".format(travel))

  linear_ceiling = float(qc.get("linear_velocity_ceiling", math.inf))
  angular_ceiling = float(qc.get("angular_velocity_ceiling", math.inf))
  for branch in BRANCHES:
    log = logs[branch]
    max_linear = float(np.linalg.norm(log.states[:, :, LINEAR_VELOCITY_SLICE], axis=2).max())
    max_angular = float(np.linalg.norm(log.states[:, :, ANGULAR_VELOCITY_SLICE], axis=2).max())
    metrics["{}_max_linear".format(branch)] = max_linear
    metrics["{}_max_angular".format(branch)] = max_angular
    if max_linear > linear_ceiling:
      reasons.append("{}: linear speed {:.1f} exceeds ceiling".format(branch, max_linear))
    if max_angular > angular_ceiling:
      reasons.append("{}: angular speed {:.1f} exceeds ceiling".format(branch, max_angular))
    if bool(qc.get("keep_in_bounds", True)):
      lower, upper = spec.scene.scene_bounds
      positions = log.states[:, :, POSITION_SLICE]
      valid_pos = positions[positions[:, :, 2] > -500.0]
      if len(valid_pos) > 0 and (
          (valid_pos < np.asarray(lower)).any() or (valid_pos > np.asarray(upper)).any()
      ):
        reasons.append("{}: a body left scene bounds".format(branch))

  # The removed branch: subject is removed before collision; non-subject bodies must remain at rest.
  removed_motion = motion["subject_removed"]
  non_subject_travel = max(
      (info["travel"] for oid, info in removed_motion.items() if oid != subject_id),
      default=0.0,
  )
  metrics["removed_max_travel"] = non_subject_travel
  if non_subject_travel > 0.05:
    reasons.append("non-subject body moved in subject_removed branch ({:.3f} m)".format(non_subject_travel))

  removed_touched = _touched_ids(removed, floor_id)
  removed_struck = sorted(removed_touched[subject_id])
  if removed_struck:
    reasons.append("subject struck bodies before removal in subject_removed branch: {}".format(removed_struck))

  factual_graph = contact_log_to_temporal_graph(
      tuple(r for r in factual.contacts if floor_id not in (r.object_a, r.object_b)),
      factual.step_rate,
  )
  counterfactual_graph = contact_log_to_temporal_graph(
      tuple(r for r in counterfactual.contacts if floor_id not in (r.object_a, r.object_b)),
      counterfactual.step_rate,
  )
  factual_pairs = {(e.object_a, e.object_b) for e in factual_graph.edges}
  counterfactual_pairs = {(e.object_a, e.object_b) for e in counterfactual_graph.edges}
  metrics["factual_contact_pairs"] = sorted(factual_pairs)
  metrics["counterfactual_contact_pairs"] = sorted(counterfactual_pairs)
  delta = graph_delta(factual_graph, counterfactual_graph)
  metrics["graph_delta_counts"] = {
      "added": len(delta.added), "removed": len(delta.removed), "changed": len(delta.changed),
  }
  if bool(qc.get("require_graph_delta", True)) and not (
      delta.added or delta.removed or delta.changed
  ):
    reasons.append("counterfactual velocity did not change the contact graph")

  metrics["removed_object_ids"] = list(removed.object_ids)
  return QCReport(passed=not reasons, reasons=tuple(reasons), metrics=metrics)


def extract_instance_ground_truth(
    spec: VelocityInstanceSpec, logs: Mapping[str, SimulationLog]
) -> GroundTruth:
  """Graph delta, affected objects and propagation paths (floor excluded)."""
  intervention_step = int(logs["counterfactual"].metadata.get("intervention_step", 0))
  return extract_ground_truth(
      logs["factual"], logs["counterfactual"], spec.subject_id, intervention_step,
      exclude_nodes=(spec.floor_id,), force_threshold=1e-6,
  )


# ---------------------------------------------------------------------------
# End-to-end generation and artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedInstance:
  """A QC-accepted instance: spec, per-branch logs, QC report and ground truth."""

  spec: VelocityInstanceSpec
  logs: Mapping[str, SimulationLog]
  qc: QCReport
  ground_truth: GroundTruth
  rejected_attempts: Tuple[Mapping[str, Any], ...] = ()


def generate_instance(
    ranges: Mapping[str, Any], master_seed: int, index: int,
    max_attempts: Optional[int] = None,
) -> GeneratedInstance:
  """Samples and simulates until QC passes or ``max_attempts`` is exhausted."""
  limit = int(ranges["qc"].get("max_attempts", 16)) if max_attempts is None else int(max_attempts)
  rejected: List[Mapping[str, Any]] = []
  for attempt in range(limit):
    try:
      spec = sample_instance(ranges, master_seed, index, attempt)
    except RuntimeError as error:
      rejected.append({"attempt": attempt, "reasons": ["layout: {}".format(error)]})
      continue
    logs = simulate_branches(spec)
    report = evaluate_qc(spec, logs, ranges)
    if report.passed:
      truth = extract_instance_ground_truth(spec, logs)
      return GeneratedInstance(
          spec=spec, logs=logs, qc=report, ground_truth=truth,
          rejected_attempts=tuple(rejected),
      )
    rejected.append({"attempt": attempt, "reasons": list(report.reasons)})
  raise RuntimeError(
      "index {} failed QC after {} attempts: {}".format(index, limit, rejected[-1] if rejected else "")
  )


def write_instance(directory: PathLike, generated: GeneratedInstance, *, overwrite: bool = False) -> Path:
  """Writes ``instance.json``, ``qc.json``, ``ground_truth.json`` and branch logs.

  Layout::

      <directory>/instance.json
      <directory>/qc.json
      <directory>/ground_truth.json
      <directory>/<branch>/sim_log/   (atomic simulation-log artifact)
      <directory>/<branch>/graph.json (temporal contact graph without the floor)
  """
  target = Path(directory)
  target.mkdir(parents=True, exist_ok=True)
  spec = generated.spec
  for branch in BRANCHES:
    branch_dir = target / branch
    branch_dir.mkdir(exist_ok=True)
    if overwrite and (branch_dir / SIM_LOG_DIRNAME).exists():
      shutil.rmtree(branch_dir / SIM_LOG_DIRNAME)
    write_simulation_log(generated.logs[branch], branch_dir / SIM_LOG_DIRNAME, overwrite=overwrite)
    log = generated.logs[branch]
    graph = contact_log_to_temporal_graph(
        tuple(r for r in log.contacts if spec.floor_id not in (r.object_a, r.object_b)),
        log.step_rate,
    )
    floor_graph = contact_log_to_temporal_graph(log.contacts, log.step_rate)
    payload = {
        "branch": branch,
        "instance_id": spec.instance_id,
        "step_rate": log.step_rate,
        "steps_per_frame": spec.scene.step_rate // spec.scene.frame_rate,
        "nodes": list(log.object_ids),
        "roles": {oid: spec.roles[oid] for oid in log.object_ids},
        "edges": [
            {
                "object_a": e.object_a, "object_b": e.object_b,
                "start_step": e.start_step, "end_step": e.end_step,
                "start_frame": e.start_step / (spec.scene.step_rate // spec.scene.frame_rate),
                "end_frame": e.end_step / (spec.scene.step_rate // spec.scene.frame_rate),
                "total_impulse": e.total_impulse, "peak_force": e.peak_force,
            }
            for e in graph.edges
        ],
        "floor_edges": [
            {"object_a": e.object_a, "object_b": e.object_b,
             "start_step": e.start_step, "end_step": e.end_step}
            for e in floor_graph.edges
            if spec.floor_id in (e.object_a, e.object_b)
        ],
    }
    (branch_dir / "graph.json").write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
  (target / SPEC_FILENAME).write_text(
      json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
  )
  (target / "qc.json").write_text(
      json.dumps(to_jsonable({
          "report": generated.qc.to_dict(),
          "rejected_attempts": list(generated.rejected_attempts),
      }), indent=2, sort_keys=True),
      encoding="utf-8",
  )
  (target / "ground_truth.json").write_text(
      json.dumps(generated.ground_truth.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
  )
  return target


def read_instance_spec(directory: PathLike) -> VelocityInstanceSpec:
  """Reads ``instance.json`` back into a validated :class:`VelocityInstanceSpec`."""
  payload = json.loads((Path(directory) / SPEC_FILENAME).read_text(encoding="utf-8"))
  spec = _spec_from_payload(payload)
  if payload.get("instance_id") not in (None, spec.instance_id):
    raise ValueError("instance_id does not match the stored scene payload")
  return spec


def read_branch_log(directory: PathLike, branch: str) -> SimulationLog:
  """Reads the simulation log written for ``branch`` under ``directory``."""
  if branch not in BRANCHES:
    raise ValueError("unknown branch: {!r}".format(branch))
  return read_simulation_log(Path(directory) / branch / SIM_LOG_DIRNAME)


__all__ = [
    "BRANCHES",
    "ROLES",
    "GeneratedInstance",
    "QCReport",
    "VelocityInstanceSpec",
    "evaluate_qc",
    "extract_instance_ground_truth",
    "generate_instance",
    "load_ranges",
    "motion_summary",
    "read_branch_log",
    "read_instance_spec",
    "sample_instance",
    "simulate_branches",
    "simulate_scene",
    "write_instance",
]
