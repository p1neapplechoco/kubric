"""Velocity-intervention scenes: three branches from one initial-velocity push.

Purpose: sample and roll out scenes in which every body is a free dynamic body,
exactly one subject carries an initial linear velocity along a single direction,
and the counterfactual replaces that initial velocity rather than perturbing a
prescribed path.
Public API: BRANCHES, VELOCITY_TRUST_MODEL, VelocityInstanceSpec, load_ranges,
sample_instance, generate_triplet, pair_ground_truth, removal_ground_truth,
motion_modes, evaluate_qc, instance_summary.
Dependencies: NumPy, PyYAML, and the standard library at sampling time; Kubric
and PyBullet are imported lazily, only when a rollout actually runs.
Trust boundary: this module reports what it simulated. The `subject_removed`
branch is a genuine third world rather than a presentation trick, but it shares
no prefix with the other two, so nothing here claims twin-prefix equality.

Why this exists alongside ``interventions.dataset``: the shipped pipeline drives
its target kinematically along a ``factual_path`` and intervenes by perturbing
that path (``remove_collision``, ``retime``, ...). None of those recipes can
express "discard the current velocity and apply a new initial velocity", and the
shipped sampler derives mass from material density, which a fixed-mass dataset
must not do. Both pipelines write the same ``SimulationLog`` and feed the same
``graph_extraction`` code, so downstream readers are unaffected.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from interventions import appearance, appearance_sampling, materials
from interventions.graph_extraction import (
    TemporalGraph,
    contact_log_to_temporal_graph,
    graph_delta,
)
from interventions.logging import ContactLogger, SimulationLog
from interventions.schema import (
    CameraConfig,
    GraphEdgeDelta,
    GroundTruth,
    ObjectConfig,
    SceneConfig,
    derive_seed,
    to_jsonable,
)


VELOCITY_TRUST_MODEL = "velocity_intervention_v1"

#: Branch names, in publication order. ``factual`` and ``counterfactual`` share
#: a scene; ``subject_removed`` runs the same scene minus the subject.
BRANCHES: Tuple[str, ...] = ("factual", "counterfactual", "subject_removed")

SUBJECT_ID = "subject"
FLOOR_ID = "floor"

#: Seed domains. Each draws from its own stream so that adding a draw in one
#: domain cannot shift another domain's samples.
_DOMAINS = (
    "placement",
    "material",
    "velocity",
    "appearance",
    "texture",
    "camera",
    "lighting",
    "background",
    "render",
)

_ZERO3 = (0.0, 0.0, 0.0)
_IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)

# Objects rest this far above the floor so Bullet's first step resolves a
# separating contact rather than an interpenetrating one.
_REST_EPSILON = 1e-3

#: Camera search budget. A draw is accepted only if it frames every movable
#: body for every frame; on exhaustion the last direction is dollied back.
_CAMERA_ATTEMPTS = 512
_CAMERA_DOLLY_FACTOR = 1.12
_CAMERA_DOLLY_STEPS = 48


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def load_ranges(path: Any) -> Mapping[str, Any]:
  """Reads and validates a velocity-scene range file."""
  with open(path, "r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
  return validate_ranges(payload)


def validate_ranges(ranges: Mapping[str, Any]) -> Mapping[str, Any]:
  """Raises ``ValueError`` unless ``ranges`` can drive this pipeline."""
  if not isinstance(ranges, Mapping):
    raise TypeError("ranges must be a mapping")
  for section in ("scene", "objects", "subject", "counterfactual"):
    if section not in ranges:
      raise ValueError("ranges is missing the {!r} section".format(section))

  scene = ranges["scene"]
  frame_rate = int(scene["frame_rate"])
  step_rate = int(scene["step_rate"])
  if step_rate % frame_rate != 0:
    raise ValueError("scene.step_rate must be a multiple of scene.frame_rate")

  objects = ranges["objects"]
  low, high = _int_pair(objects, "count")
  if low < 2:
    raise ValueError("objects.count must allow at least two bodies")
  mass = objects.get("mass")
  if isinstance(mass, Sequence) and not isinstance(mass, (str, bytes)):
    raise ValueError(
        "objects.mass must be a scalar; this pipeline pins mass rather than "
        "sampling it"
    )
  if float(mass) <= 0.0:
    raise ValueError("objects.mass must be positive")
  shapes = tuple(objects.get("shapes", ()))
  unsupported = set(shapes) - {"cube", "sphere"}
  if unsupported:
    raise ValueError(
        "objects.shapes supports only cube and sphere here; got {}".format(
            sorted(unsupported)
        )
    )
  in_low, in_high = _int_pair(objects, "in_path_count")
  if in_low < 1:
    raise ValueError("objects.in_path_count must plant at least one body")
  if in_high > high - 1:
    raise ValueError(
        "objects.in_path_count upper bound leaves no room for the subject"
    )

  turn_low, turn_high = _pair(ranges["counterfactual"], "turn")
  if turn_low <= 0.0:
    raise ValueError("counterfactual.turn lower bound must be positive")
  if turn_high > math.pi:
    raise ValueError("counterfactual.turn upper bound must not exceed pi")

  if ranges.get("appearance", {}).get("enabled", False):
    appearance_sampling.validate_appearance_ranges(ranges)
  return ranges


def _pair(section: Mapping[str, Any], key: str) -> Tuple[float, float]:
  value = section[key]
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    return (float(value), float(value))
  low, high = value
  low = float(low)
  high = float(high)
  if low > high:
    raise ValueError("{} lower bound exceeds its upper bound".format(key))
  return (low, high)


def _int_pair(section: Mapping[str, Any], key: str) -> Tuple[int, int]:
  low, high = _pair(section, key)
  return (int(low), int(high))


def _uniform(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
  if bounds[0] == bounds[1]:
    return float(bounds[0])
  return float(rng.uniform(bounds[0], bounds[1]))


def _uniform_int(rng: np.random.Generator, bounds: Tuple[int, int]) -> int:
  return int(rng.integers(bounds[0], bounds[1] + 1))


def _weighted_choice(
    rng: np.random.Generator, weights: Mapping[str, float]
) -> str:
  keys = sorted(weights)
  values = np.array([float(weights[key]) for key in keys], dtype=np.float64)
  total = float(values.sum())
  if total <= 0.0:
    raise ValueError("weights must sum to a positive number")
  return keys[int(rng.choice(len(keys), p=values / total))]


# ---------------------------------------------------------------------------
# Instance specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VelocityInstanceSpec:
  """One sampled scene plus the two initial velocities that define its branches."""

  index: int
  instance_id: str
  master_seed: int
  scene_config: SceneConfig
  subject_id: str
  factual_velocity: Tuple[float, float, float]
  counterfactual_velocity: Tuple[float, float, float]
  factual_azimuth: float
  counterfactual_azimuth: float
  families: Mapping[str, str]
  roles: Mapping[str, str]
  visual_scene: Optional[appearance.VisualSceneSpec]
  rollout_steps: int

  @property
  def visual_scene_hash(self) -> Optional[str]:
    if self.visual_scene is None:
      return None
    return appearance.visual_scene_hash(self.visual_scene)

  def scene_without_subject(self) -> SceneConfig:
    """The same scene with the subject deleted, for the removal branch."""
    return replace(
        self.scene_config,
        objects=tuple(
            item
            for item in self.scene_config.objects
            if item.object_id != self.subject_id
        ),
    )


def instance_id_for(master_seed: int, index: int) -> str:
  """The instance id an index will produce, without sampling or simulating it.

  Public because a resumable batch driver needs it: the id depends on nothing
  but ``(master_seed, index)``, so a run can tell whether an index has already
  been published by looking at the filesystem, instead of paying a full
  three-branch rollout to rediscover a name it could have computed.
  """
  digest = hashlib.sha256(
      "velocity/{}/{}".format(int(master_seed), int(index)).encode("utf-8")
  ).hexdigest()
  return "instance_{}".format(digest[:20])


def _instance_id(master_seed: int, index: int) -> str:
  return instance_id_for(master_seed, index)


def _rngs(master_seed: int, index: int) -> Dict[str, np.random.Generator]:
  return {
      domain: np.random.default_rng(derive_seed(master_seed, index, domain))
      for domain in _DOMAINS
  }


def _half_extents(shape: str, size: Sequence[float]) -> Tuple[float, float, float]:
  if shape == "sphere":
    radius = float(size[0])
    return (radius, radius, radius)
  return tuple(float(component) for component in size)


def _circumradius(shape: str, size: Sequence[float]) -> float:
  extents = _half_extents(shape, size)
  if shape == "sphere":
    return extents[0]
  return math.sqrt(sum(component * component for component in extents))


def _yaw_quaternion(yaw: float) -> Tuple[float, float, float, float]:
  half = 0.5 * float(yaw)
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def sample_instance(
    ranges: Mapping[str, Any], master_seed: int, index: int
) -> VelocityInstanceSpec:
  """Samples one scene, its subject velocity, and its replacement velocity."""
  validate_ranges(ranges)
  rngs = _rngs(master_seed, index)
  scene_ranges = ranges["scene"]
  object_ranges = ranges["objects"]
  subject_ranges = ranges["subject"]
  cf_ranges = ranges["counterfactual"]

  frame_start, frame_end = (int(value) for value in scene_ranges["frame_range"])
  frame_rate = int(scene_ranges["frame_rate"])
  step_rate = int(scene_ranges["step_rate"])
  rollout_steps = (frame_end - frame_start) * step_rate // frame_rate

  fixed_mass = float(object_ranges["mass"])
  floor_size = tuple(float(value) for value in object_ranges["floor_size"])
  placement = rngs["placement"]
  material_rng = rngs["material"]
  velocity_rng = rngs["velocity"]

  # --- direction. One azimuth defines the whole scene: the subject's initial
  # velocity, where the struck bodies sit, and which bodies count as bystanders.
  azimuth = _uniform(velocity_rng, _pair(subject_ranges, "azimuth"))
  direction = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
  lateral = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])

  total_movable = _uniform_int(placement, _int_pair(object_ranges, "count"))
  in_path_count = min(
      _uniform_int(placement, _int_pair(object_ranges, "in_path_count")),
      total_movable - 1,
  )
  bystander_count = total_movable - 1 - in_path_count

  shape_weights = {
      key: float(value)
      for key, value in object_ranges.get(
          "shape_weights", {name: 1.0 for name in object_ranges["shapes"]}
      ).items()
  }
  require_both = bool(object_ranges.get("require_both_shapes", False))

  # --- shapes. Draw them up front so the "must show both sliding and rolling"
  # constraint is resolved by construction rather than by rejecting the scene.
  # Slot 0 is the subject and slot 1 is the nearest body in its path — the only
  # two the subject is guaranteed to set in motion. Making those two differ is
  # what actually puts a slide and a roll in the same clip; a sphere parked off
  # to the side would satisfy a looser constraint while never rolling.
  shapes = [_weighted_choice(placement, shape_weights) for _ in range(total_movable)]
  if require_both and shapes[1] == shapes[0]:
    shapes[1] = "cube" if shapes[0] == "sphere" else "sphere"

  subject_shape = shapes[0]
  subject_size_range = _pair(subject_ranges, "size")
  if subject_shape == "sphere":
    radius = _uniform(placement, subject_size_range)
    subject_size = (radius, radius, radius)
  else:
    subject_size = tuple(
        _uniform(placement, subject_size_range) for _ in range(3)
    )

  start_radius = _uniform(placement, _pair(subject_ranges, "start_radius"))
  subject_extents = _half_extents(subject_shape, subject_size)
  subject_position = tuple(
      (-start_radius * direction)[:2].tolist()
      + [subject_extents[2] + _REST_EPSILON]
  )

  placed: List[Dict[str, Any]] = [{
      "object_id": SUBJECT_ID,
      "shape": subject_shape,
      "size": subject_size,
      "position": subject_position,
      "role": "subject",
  }]

  size_range = _pair(object_ranges, "size")
  along_range = _pair(object_ranges, "along_path")
  lateral_range = _pair(object_ranges, "lateral_jitter")
  bystander_x = _pair(object_ranges, "bystander_x")
  bystander_y = _pair(object_ranges, "bystander_y")
  clearance = float(object_ranges["bystander_clearance"])
  margin = float(object_ranges.get("separation_margin", 0.0))
  attempts = int(object_ranges.get("placement_attempts", 512))

  def _size_for(shape: str) -> Tuple[float, float, float]:
    if shape == "sphere":
      radius = _uniform(placement, size_range)
      return (radius, radius, radius)
    return tuple(_uniform(placement, size_range) for _ in range(3))

  def _clear_of_others(position, shape, size) -> bool:
    radius = _circumradius(shape, size)
    for other in placed:
      other_radius = _circumradius(other["shape"], other["size"])
      gap = math.dist(position[:2], other["position"][:2])
      if gap < radius + other_radius + margin:
        return False
    return True

  origin = np.array(subject_position)

  # Distances along the corridor, stratified so that in-path slot i is the
  # (i+1)-th body the subject reaches and no two land on top of each other.
  # Independent uniform draws routinely produce three near-identical distances,
  # which the overlap check can only resolve sideways and often cannot.
  along_distances = []
  span = (along_range[1] - along_range[0]) / max(1, in_path_count)
  for slot in range(in_path_count):
    low = along_range[0] + slot * span
    along_distances.append(_uniform(placement, (low, low + span)))

  for slot in range(total_movable - 1):
    shape = shapes[slot + 1]
    in_path = slot < in_path_count
    chosen = None
    for attempt in range(attempts):
      size = _size_for(shape)
      extents = _half_extents(shape, size)
      if in_path:
        # Hold the stratified distance while there is still room to resolve an
        # overlap sideways; only widen into the neighbouring strata late, and
        # never past them, so the ordering the shape constraint relies on holds.
        if attempt < attempts // 2:
          along = along_distances[slot]
        else:
          low = along_range[0] + slot * span
          along = _uniform(placement, (low, low + span))
        offset = _uniform(placement, lateral_range)
        point = origin + along * direction + offset * lateral
      else:
        point = np.array([
            _uniform(placement, bystander_x),
            _uniform(placement, bystander_y),
            0.0,
        ])
        # Reject anything the subject could reach: distance to the forward ray.
        relative = point - origin
        along = float(np.dot(relative, direction))
        perpendicular = float(abs(np.dot(relative, lateral)))
        if along > -clearance and perpendicular < clearance:
          continue
      position = (
          float(point[0]),
          float(point[1]),
          extents[2] + _REST_EPSILON,
      )
      if abs(position[0]) > floor_size[0] - extents[0]:
        continue
      if abs(position[1]) > floor_size[1] - extents[1]:
        continue
      if not _clear_of_others(position, shape, size):
        continue
      chosen = {
          "object_id": "object_{}".format(slot),
          "shape": shape,
          "size": size,
          "position": position,
          "role": "in_path" if in_path else "bystander",
      }
      break
    if chosen is None:
      raise ValueError(
          "could not place object_{} after {} attempts".format(slot, attempts)
      )
    placed.append(chosen)

  # --- materials. The family drives appearance, friction, and restitution.
  # It does not drive mass: `coupled_physics` proposes one and we discard it.
  family_weights = _family_weights(ranges)
  families: Dict[str, str] = {}
  configs: List[ObjectConfig] = []

  configs.append(ObjectConfig(
      object_id=FLOOR_ID,
      shape="cube",
      size=floor_size,
      mass=0.0,
      friction=float(object_ranges["floor_friction"]),
      restitution=float(object_ranges["floor_restitution"]),
      position=(0.0, 0.0, -floor_size[2]),
      quaternion=_IDENTITY_QUATERNION,
      linear_velocity=_ZERO3,
      angular_velocity=_ZERO3,
      static=True,
      metadata={"role": "environment", "qc_clip_exempt": True},
  ))

  for item in placed:
    family = _weighted_choice(material_rng, family_weights)
    families[item["object_id"]] = family
    volume = materials.proxy_volume(item["shape"], _half_extents(
        item["shape"], item["size"]
    ))
    coupled = materials.coupled_physics(material_rng, family, volume)
    yaw = _uniform(placement, (0.0, 2.0 * math.pi))
    is_subject = item["object_id"] == SUBJECT_ID
    configs.append(ObjectConfig(
        object_id=item["object_id"],
        shape=item["shape"],
        size=item["size"],
        # Fixed, by construction. `coupled["mass"]` is recorded in metadata so
        # the discarded material-implied mass stays auditable.
        mass=fixed_mass,
        friction=float(coupled["friction"]),
        restitution=float(coupled["restitution"]),
        position=item["position"],
        quaternion=_IDENTITY_QUATERNION if item["shape"] == "sphere"
        else _yaw_quaternion(yaw),
        # Only the subject is given a velocity, and only a linear one.
        linear_velocity=_ZERO3,
        angular_velocity=_ZERO3,
        static=False,
        metadata={
            "role": item["role"],
            "material_family": family,
            "mass_mode": "fixed",
            "material_implied_mass": float(coupled["unclamped_mass"]),
            "effective_density": float(coupled["effective_density"]),
        },
    ))

  # --- velocities. Both branches are a single-direction initial linear
  # velocity. The counterfactual is not a perturbation of the factual one; it
  # replaces it outright, which is why it is sampled from its own range.
  speed = _uniform(velocity_rng, _pair(subject_ranges, "speed"))
  factual_velocity = tuple((speed * direction).tolist())

  turn = _uniform(velocity_rng, _pair(cf_ranges, "turn"))
  if velocity_rng.random() < 0.5:
    turn = -turn
  cf_azimuth = azimuth + turn
  cf_speed = _uniform(velocity_rng, _pair(cf_ranges, "speed"))
  min_delta = float(cf_ranges.get("min_speed_delta", 0.0))
  if min_delta > 0.0 and abs(cf_speed - speed) < min_delta:
    cf_speed = speed + math.copysign(min_delta, cf_speed - speed or 1.0)
  cf_direction = np.array([math.cos(cf_azimuth), math.sin(cf_azimuth), 0.0])
  counterfactual_velocity = tuple((cf_speed * cf_direction).tolist())

  camera_spec = None
  visual_scene = None
  scene_config = SceneConfig(
      objects=tuple(configs),
      camera=None,
      seed=int(derive_seed(master_seed, index, "scene") % (2 ** 31)),
      scene_bounds=tuple(
          tuple(float(value) for value in corner)
          for corner in scene_ranges["bounds"]
      ),
      gravity=tuple(float(value) for value in scene_ranges["gravity"]),
      frame_range=(frame_start, frame_end),
      frame_rate=frame_rate,
      step_rate=step_rate,
  )

  if ranges.get("appearance", {}).get("enabled", False):
    # Framed on the starting layout. ``refit_camera`` re-frames it on the
    # rollout once the rollout exists; nothing here knows where bodies end up.
    camera_spec = _sample_fixed_camera(
        ranges, rngs["camera"], scene_config, _framing_targets(scene_config)
    )
    visual_scene = _build_visual_scene(
        ranges, scene_config, camera_spec, families, rngs, master_seed, index
    )
    scene_config = replace(scene_config, camera=CameraConfig(
        position=camera_spec.positions[0],
        look_at=camera_spec.look_ats[0],
        focal_length=camera_spec.focal_length,
    ))
    appearance.validate_scene_correspondence(visual_scene, scene_config)

  roles = {
      item.object_id: str(item.metadata.get("role", "unknown"))
      for item in scene_config.objects
  }

  return VelocityInstanceSpec(
      index=int(index),
      instance_id=_instance_id(master_seed, index),
      master_seed=int(master_seed),
      scene_config=scene_config,
      subject_id=SUBJECT_ID,
      factual_velocity=factual_velocity,
      counterfactual_velocity=counterfactual_velocity,
      factual_azimuth=float(azimuth),
      counterfactual_azimuth=float(cf_azimuth),
      families=dict(families),
      roles=roles,
      visual_scene=visual_scene,
      rollout_steps=int(rollout_steps),
  )


def refit_camera(
    ranges: Mapping[str, Any],
    spec: VelocityInstanceSpec,
    logs: Mapping[str, SimulationLog],
) -> VelocityInstanceSpec:
  """Re-frames the fixed camera on the trajectories the rollout produced.

  ``sample_instance`` can only frame the starting layout, and bodies move. This
  re-runs the same sampler against every logged pose of every movable body in
  every branch, so a published clip shows each simulated body whole for its
  whole duration — which is what makes the mask and the 2-D tracks usable.

  The camera is drawn from a fresh generator over the same ``camera`` seed
  domain, so the sequence of candidate viewpoints is the one ``sample_instance``
  would have drawn; only the acceptance test has more information. Every other
  domain is untouched, so appearance, physics, and velocities are unchanged.
  """
  if spec.visual_scene is None:
    return spec

  frame_steps = appearance.frame_steps_for(spec.scene_config)
  targets = []
  for log in logs.values():
    sampled = np.stack([log.states[step, :, 0:3] for step in frame_steps], axis=0)
    # The removal branch has no subject column; ``_framing_targets`` indexes by
    # id, so it takes each branch's own object list.
    subset = replace(
        spec.scene_config,
        objects=tuple(
            item for item in spec.scene_config.objects
            if item.object_id in set(log.object_ids)
        ),
    )
    targets.append(_framing_targets(subset, sampled, log.object_ids))
  all_targets = np.concatenate(targets, axis=0)

  rng = np.random.default_rng(derive_seed(spec.master_seed, spec.index, "camera"))
  camera = _sample_fixed_camera(ranges, rng, spec.scene_config, all_targets)
  visual = replace(spec.visual_scene, camera=camera)
  scene_config = replace(spec.scene_config, camera=CameraConfig(
      position=camera.positions[0],
      look_at=camera.look_ats[0],
      focal_length=camera.focal_length,
  ))
  appearance.validate_scene_correspondence(visual, scene_config)
  return replace(spec, scene_config=scene_config, visual_scene=visual)


def _family_weights(ranges: Mapping[str, Any]) -> Mapping[str, float]:
  section = ranges.get("appearance", {}).get("materials", {})
  families = tuple(section.get("families", ()))
  if not families:
    return {name: 1.0 for name in sorted(materials.FAMILY_PRIORS)}
  weights = tuple(section.get("weights", ()) or [1.0] * len(families))
  return {
      family: float(weight) for family, weight in zip(families, weights)
  }


def _corner_targets(
    positions: np.ndarray, radii: Sequence[float]
) -> np.ndarray:
  """Bounding-box corners for every body at every supplied pose.

  ``positions`` is ``[T, N, 3]`` (or ``[N, 3]``), ``radii`` is one circumradius
  per body. Framing the corners rather than the centres is what keeps a body
  whole in the image instead of merely present in it.
  """
  points = np.asarray(positions, dtype=np.float64)
  if points.ndim == 2:
    points = points[None, :, :]
  radius = np.asarray(radii, dtype=np.float64).reshape(1, -1, 1, 1)
  offsets = np.array(
      [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
      dtype=np.float64,
  )
  corners = points[:, :, None, :] + offsets[None, None, :, :] * radius
  return corners.reshape(-1, 3)


def _frames_all(
    position: Sequence[float],
    look_at: Sequence[float],
    focal_length: float,
    sensor_width: float,
    aspect: float,
    targets: np.ndarray,
    margin: float,
) -> bool:
  """Whether every target point projects inside the image with a margin.

  ``appearance_sampling._camera_fits`` compares the *scene bounds* against a
  fixed half-angle proxy that ignores the sampled focal length and the output
  aspect ratio, so at the focal lengths this dataset uses it accepts cameras
  that cannot frame the action at all. This is the real test: the actual
  frustum of the actual lens against the points that actually have to be seen.
  """
  origin = np.asarray(position, dtype=np.float64)
  forward = np.asarray(look_at, dtype=np.float64) - origin
  norm = float(np.linalg.norm(forward))
  if norm <= 1e-9:
    return False
  forward = forward / norm

  right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
  if float(np.linalg.norm(right)) <= 1e-9:
    right = np.array([1.0, 0.0, 0.0])
  right = right / float(np.linalg.norm(right))
  up = np.cross(right, forward)

  relative = np.asarray(targets, dtype=np.float64) - origin
  depth = relative @ forward
  if float(depth.min(initial=np.inf)) <= 1e-3:
    return False

  # Blender fits ``sensor_width`` to the longer image axis; these are the
  # tangents of the horizontal and vertical half-angles.
  half_x = (float(sensor_width) / 2.0) / float(focal_length)
  half_y = half_x / float(aspect) if aspect >= 1.0 else half_x * float(aspect)
  keep = max(0.0, 1.0 - float(margin))

  horizontal = np.abs(relative @ right) / depth
  vertical = np.abs(relative @ up) / depth
  return bool(
      horizontal.max() <= half_x * keep and vertical.max() <= half_y * keep
  )


def _render_aspect(ranges: Mapping[str, Any]) -> float:
  resolution = ranges.get("render", {}).get("resolution", (512, 384))
  width, height = (float(value) for value in resolution)
  return width / height


def _sample_fixed_camera(
    ranges: Mapping[str, Any],
    rng: np.random.Generator,
    scene_config: SceneConfig,
    targets: np.ndarray,
) -> appearance.CameraRenderSpec:
  """Samples one viewpoint that frames ``targets`` and holds it for the clip.

  ``appearance_sampling.sample_camera`` resamples radius, elevation, and azimuth
  once per frame regardless of ``camera.motion``, which produces a jittering
  camera rather than the fixed one this dataset requires. A static camera is a
  repeated entry in ``CameraRenderSpec.positions``, so building it here keeps the
  result a plain ``CameraRenderSpec`` that hashes and validates like any other.

  Acceptance is the real frustum test in ``_frames_all``. If no draw frames the
  targets, the last direction is kept and the camera is dollied back along it
  until it does, which converges because angular extent falls off with distance.
  """
  camera_ranges = ranges["appearance"]["camera"]
  margin = float(camera_ranges.get("frame_margin", 0.08))
  radius_range = _pair(camera_ranges, "radius")
  elevation_range = _pair(camera_ranges, "elevation")
  azimuth_range = _pair(camera_ranges, "azimuth")
  focal_range = _pair(camera_ranges, "focal_length")
  sensor_width = float(camera_ranges.get("sensor_width", 36.0))
  aspect = _render_aspect(ranges)
  frame_count = len(appearance.frame_steps_for(scene_config))

  # Look at the middle of what has to be seen, not the middle of the (much
  # larger) configured scene bounds.
  look_at = tuple(
      float(value) for value in np.asarray(targets, dtype=np.float64).mean(axis=0)
  )

  def _position(radius, elevation, azimuth):
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)
    if z <= 0.0:
      z = abs(z) + 0.05
    return (look_at[0] + x, look_at[1] + y, look_at[2] + z)

  radius = elevation = azimuth = None
  focal_length = None
  for _ in range(_CAMERA_ATTEMPTS):
    radius = _uniform(rng, radius_range)
    elevation = _uniform(rng, elevation_range)
    azimuth = _uniform(rng, azimuth_range)
    focal_length = _uniform(rng, focal_range)
    candidate = _position(radius, elevation, azimuth)
    if _frames_all(
        candidate, look_at, focal_length, sensor_width, aspect, targets, margin
    ):
      position = candidate
      break
  else:
    # Keep the last sampled direction, open the lens as wide as the config
    # allows, and dolly back until the targets fit.
    focal_length = focal_range[0]
    position = None
    for _ in range(_CAMERA_DOLLY_STEPS):
      radius *= _CAMERA_DOLLY_FACTOR
      candidate = _position(radius, elevation, azimuth)
      if _frames_all(
          candidate, look_at, focal_length, sensor_width, aspect, targets, margin
      ):
        position = candidate
        break
    if position is None:
      raise ValueError(
          "no camera within {} draws frames the scene, even dollied back to "
          "{:.1f} m at {:.1f} mm".format(_CAMERA_ATTEMPTS, radius, focal_length)
      )

  return appearance.CameraRenderSpec(
      positions=tuple(position for _ in range(frame_count)),
      look_ats=tuple(look_at for _ in range(frame_count)),
      focal_length=focal_length,
      sensor_width=sensor_width,
      clipping_range=tuple(
          float(value) for value in camera_ranges.get("clipping_range", (0.1, 200.0))
      ),
  )


def _framing_targets(
    scene_config: SceneConfig,
    positions: Optional[np.ndarray] = None,
    object_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
  """The points the camera must contain: every movable body, whole.

  The floor is excluded on purpose. It is a 10 m backdrop that no reasonable
  lens frames from this distance, and cropping it costs nothing; cropping a
  simulated body costs an annotation.
  """
  movable = tuple(
      item for item in sorted(scene_config.objects, key=lambda o: o.object_id)
      if item.object_id != FLOOR_ID
  )
  radii = [_circumradius(item.shape, item.size) for item in movable]
  if positions is None:
    layout = np.array([item.position for item in movable], dtype=np.float64)
    return _corner_targets(layout, radii)

  index = {object_id: row for row, object_id in enumerate(object_ids or ())}
  columns = [index[item.object_id] for item in movable]
  return _corner_targets(np.asarray(positions)[:, columns, :], radii)



def _build_visual_scene(
    ranges: Mapping[str, Any],
    scene_config: SceneConfig,
    camera: appearance.CameraRenderSpec,
    families: Mapping[str, str],
    rngs: Mapping[str, np.random.Generator],
    master_seed: int,
    index: int,
) -> appearance.VisualSceneSpec:
  """Realizes appearance for the families the physics already committed to.

  ``appearance_sampling.sample_visual_scene`` picks its own material families,
  which would leave a body that looks like rubber but slides like metal. Here the
  family is chosen once, before the ``SceneConfig`` is built, and both the
  friction and the shader read the same choice.
  """
  objects = []
  for object_config in sorted(scene_config.objects, key=lambda item: item.object_id):
    family = families.get(object_config.object_id)
    if family is None:
      # The floor never entered the physics material draw; give it its own.
      family = _weighted_choice(rngs["appearance"], _family_weights(ranges))
    color = appearance_sampling.sample_color(ranges, rngs["appearance"])
    texture = appearance_sampling.sample_texture(ranges, rngs["texture"], family)
    material = materials.sample_material(
        rngs["appearance"], family, color, texture
    )
    objects.append(appearance.VisualObjectSpec(
        object_id=object_config.object_id,
        source_kind="procedural",
        asset=None,
        collision_proxy_id=object_config.object_id,
        scale=(1.0, 1.0, 1.0),
        origin_offset=_ZERO3,
        alignment_quaternion=_IDENTITY_QUATERNION,
        material=material,
    ))

  return appearance.VisualSceneSpec(
      objects=tuple(objects),
      camera=camera,
      lights=appearance_sampling.sample_lights(ranges, rngs["lighting"]),
      background=appearance_sampling.sample_background(ranges, rngs["background"]),
      render_seed=int(derive_seed(master_seed, index, "render") % (2 ** 31)),
      frame_steps=appearance.frame_steps_for(scene_config),
  )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _build_kubric_scene(scene_config: SceneConfig):
  """Builds the Kubric scene. Imported lazily: sampling must not need Bullet."""
  import kubric as kb  # pylint: disable=import-outside-toplevel

  frame_start, frame_end = scene_config.frame_range
  scene = kb.Scene(
      frame_start=frame_start,
      frame_end=frame_end - 1,
      frame_rate=scene_config.frame_rate,
      step_rate=scene_config.step_rate,
      gravity=scene_config.gravity,
  )
  assets = {}
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
  return scene, assets


def run_branch(
    scene_config: SceneConfig,
    branch: str,
    steps: int,
    *,
    initial_velocities: Mapping[str, Sequence[float]] = (),
) -> SimulationLog:
  """Rolls out ``scene_config`` for ``steps`` free-body physics steps.

  ``initial_velocities`` is applied once, before the first step, and never
  touched again — that is the whole intervention. Bodies not named there start
  at rest.
  """
  from interventions.kinematic_simulator import (  # pylint: disable=import-outside-toplevel
      KinematicDragSimulator,
  )

  overrides = dict(initial_velocities or {})
  unknown = set(overrides) - {item.object_id for item in scene_config.objects}
  if unknown:
    raise ValueError(
        "initial_velocities names absent objects: {}".format(sorted(unknown))
    )

  scene, assets = _build_kubric_scene(scene_config)
  with tempfile.TemporaryDirectory(prefix="kubric-velocity-") as scratch:
    with KinematicDragSimulator(scene, scratch_dir=Path(scratch)) as simulator:
      client = simulator.bullet_client
      ordered = tuple(sorted(scene_config.objects, key=lambda item: item.object_id))
      object_ids = tuple(item.object_id for item in ordered)
      bodies = tuple(
          int(assets[item.object_id].linked_objects[simulator]) for item in ordered
      )
      body_to_object_id = dict(zip(bodies, object_ids))

      for item, body in zip(ordered, bodies):
        client.changeDynamics(
            body,
            -1,
            mass=0.0 if item.static else item.mass,
            lateralFriction=item.friction,
            restitution=item.restitution,
        )
        velocity = overrides.get(item.object_id, item.linear_velocity)
        client.resetBaseVelocity(
            body,
            linearVelocity=[float(value) for value in velocity],
            angularVelocity=[float(value) for value in item.angular_velocity],
        )

      step_rate = float(scene_config.step_rate)
      logger = ContactLogger(body_to_object_id, step_rate)
      states = np.empty((steps, len(ordered), 13), dtype=np.float64)

      for step in range(steps):
        simulator.step_passive()
        logger.log(step, client.getContactPoints())
        for row, body in enumerate(bodies):
          position, quaternion_xyzw = client.getBasePositionAndOrientation(body)
          linear, angular = client.getBaseVelocity(body)
          states[step, row, 0:3] = position
          states[step, row, 3] = quaternion_xyzw[3]
          states[step, row, 4:7] = quaternion_xyzw[0:3]
          states[step, row, 7:10] = linear
          states[step, row, 10:13] = angular

      return SimulationLog(
          branch=branch,
          object_ids=object_ids,
          steps=tuple(range(steps)),
          states=states,
          contacts=tuple(logger.records),
          step_rate=step_rate,
          metadata={
              "trust_model": VELOCITY_TRUST_MODEL,
              "dt": 1.0 / step_rate,
              "intervention_kind": "initial_linear_velocity",
              "initial_velocities": {
                  key: [float(value) for value in vector]
                  for key, vector in sorted(overrides.items())
              },
              "scene_config_sha256": hashlib.sha256(
                  json.dumps(
                      to_jsonable(scene_config), sort_keys=True, separators=(",", ":")
                  ).encode("utf-8")
              ).hexdigest(),
          },
      )


def generate_triplet(spec: VelocityInstanceSpec) -> Dict[str, SimulationLog]:
  """Runs all three branches of ``spec`` in three fresh Bullet worlds."""
  logs = {
      "factual": run_branch(
          spec.scene_config,
          "factual",
          spec.rollout_steps,
          initial_velocities={spec.subject_id: spec.factual_velocity},
      ),
      "counterfactual": run_branch(
          spec.scene_config,
          "counterfactual",
          spec.rollout_steps,
          initial_velocities={spec.subject_id: spec.counterfactual_velocity},
      ),
      "subject_removed": run_branch(
          spec.scene_without_subject(),
          "subject_removed",
          spec.rollout_steps,
      ),
  }
  return logs


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def _temporal_graph(log: SimulationLog) -> TemporalGraph:
  return contact_log_to_temporal_graph(log.contacts, log.step_rate)


def _state_divergence(
    factual: SimulationLog,
    other: SimulationLog,
    *,
    position_epsilon: float,
    velocity_epsilon: float,
    quaternion_epsilon: float,
) -> Tuple[str, ...]:
  """Object ids whose trajectories differ, restricted to bodies in both logs."""
  shared = [
      object_id
      for object_id in factual.object_ids
      if object_id in other.object_ids
  ]
  factual_index = {name: i for i, name in enumerate(factual.object_ids)}
  other_index = {name: i for i, name in enumerate(other.object_ids)}
  horizon = min(factual.states.shape[0], other.states.shape[0])
  diverged = []
  for object_id in shared:
    left = factual.states[:horizon, factual_index[object_id]]
    right = other.states[:horizon, other_index[object_id]]
    position = np.max(np.linalg.norm(left[:, 0:3] - right[:, 0:3], axis=-1))
    linear = np.max(np.linalg.norm(left[:, 7:10] - right[:, 7:10], axis=-1))
    angular = np.max(np.linalg.norm(left[:, 10:13] - right[:, 10:13], axis=-1))
    dots = np.abs(np.sum(left[:, 3:7] * right[:, 3:7], axis=-1))
    rotation = float(np.max(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))))
    if (
        position > position_epsilon
        or linear > velocity_epsilon
        or angular > velocity_epsilon
        or rotation > quaternion_epsilon
    ):
      diverged.append(object_id)
  return tuple(sorted(diverged))


def _reachable_from(
    graph: TemporalGraph, source: str, exclude: Sequence[str] = ()
) -> Dict[str, Tuple[str, ...]]:
  """Forward-in-time contact reachability from ``source``.

  ``graph_extraction.temporal_reachability`` needs two aligned branches and an
  intervention start step. Here the intervention is at step 0 and the removal
  branch is not aligned, so reachability is computed on one graph directly.
  """
  blocked = set(exclude)
  frontier = {source: (0, (source,))}
  changed = True
  while changed:
    changed = False
    for edge in graph.edges:
      for near, far in ((edge.object_a, edge.object_b), (edge.object_b, edge.object_a)):
        if near not in frontier or far in blocked:
          continue
        arrival, path = frontier[near]
        if edge.end_step <= arrival:
          continue
        candidate = (edge.start_step, path + (far,))
        if far not in frontier or candidate[0] < frontier[far][0]:
          frontier[far] = candidate
          changed = True
  return {
      node: path for node, (_, path) in frontier.items() if node != source
  }


def pair_ground_truth(
    factual: SimulationLog,
    counterfactual: SimulationLog,
    subject_id: str,
    qc: Mapping[str, Any] = (),
) -> GroundTruth:
  """Graph delta and affected sets for the velocity-replacement counterfactual.

  The intervention lands at step 0, so there is no shared prefix and no
  "identical until the intervention" guarantee to check. Everything downstream
  of the subject is therefore in scope from the first step.
  """
  settings = dict(qc or {})
  factual_graph = _temporal_graph(factual)
  counterfactual_graph = _temporal_graph(counterfactual)
  delta = graph_delta(factual_graph, counterfactual_graph)

  paths: Dict[str, Tuple[str, ...]] = {}
  for graph in (factual_graph, counterfactual_graph):
    for node, path in _reachable_from(graph, subject_id, exclude=(FLOOR_ID,)).items():
      if node not in paths or len(path) < len(paths[node]):
        paths[node] = path

  state_ids = _state_divergence(
      factual,
      counterfactual,
      position_epsilon=float(settings.get("position_epsilon", 1e-3)),
      velocity_epsilon=float(settings.get("velocity_epsilon", 1e-3)),
      quaternion_epsilon=float(settings.get("quaternion_epsilon", 1e-3)),
  )
  hard = tuple(sorted(set(paths) & set(state_ids)))
  soft = tuple(sorted(set(state_ids) - set(hard) - {subject_id}))
  return GroundTruth(
      graph_delta=delta,
      hard_affected=hard,
      soft_affected=soft,
      propagation_path={node: paths[node] for node in hard},
  )


def removal_ground_truth(
    factual: SimulationLog,
    removed: SimulationLog,
    subject_id: str,
    qc: Mapping[str, Any] = (),
) -> GroundTruth:
  """Graph delta and affected sets for the subject-removed branch.

  The two branches do not share an object set, so the factual graph is first
  stripped of every edge and node incident on the subject. What survives the
  delta is the contact structure that the subject's mere presence caused.
  """
  settings = dict(qc or {})
  factual_graph = _temporal_graph(factual)
  stripped = TemporalGraph(
      nodes=tuple(node for node in factual_graph.nodes if node != subject_id),
      edges=tuple(
          edge
          for edge in factual_graph.edges
          if subject_id not in (edge.object_a, edge.object_b)
      ),
  )
  removed_graph = _temporal_graph(removed)
  delta = graph_delta(stripped, removed_graph)

  paths = _reachable_from(factual_graph, subject_id, exclude=(FLOOR_ID,))
  state_ids = _state_divergence(
      factual,
      removed,
      position_epsilon=float(settings.get("position_epsilon", 1e-3)),
      velocity_epsilon=float(settings.get("velocity_epsilon", 1e-3)),
      quaternion_epsilon=float(settings.get("quaternion_epsilon", 1e-3)),
  )
  hard = tuple(sorted(set(paths) & set(state_ids)))
  soft = tuple(sorted(set(state_ids) - set(hard) - {subject_id}))
  return GroundTruth(
      graph_delta=delta,
      hard_affected=hard,
      soft_affected=soft,
      propagation_path={node: paths[node] for node in hard},
  )


# ---------------------------------------------------------------------------
# Motion classification and QC
# ---------------------------------------------------------------------------


def motion_modes(
    log: SimulationLog, qc: Mapping[str, Any] = ()
) -> Dict[str, Dict[str, Any]]:
  """Labels each body ``rolling``, ``sliding``, ``both``, or ``static``.

  The label is measured, not assumed. Rolling is integrated angular travel about
  a horizontal axis; sliding is horizontal travel that the rolling cannot
  account for. A body can be both, which is the requested motion mix.
  """
  settings = dict(qc or {})
  min_revolutions = float(settings.get("rolling_min_revolutions", 0.35))
  min_travel = float(settings.get("sliding_min_travel", 0.10))
  dt = 1.0 / log.step_rate
  result: Dict[str, Dict[str, Any]] = {}
  for index, object_id in enumerate(log.object_ids):
    track = log.states[:, index]
    travel = float(np.sum(np.linalg.norm(np.diff(track[:, 0:3], axis=0), axis=-1)))
    horizontal = np.linalg.norm(track[:, 10:12], axis=-1)
    revolutions = float(np.sum(horizontal) * dt / (2.0 * math.pi))
    rolls = revolutions >= min_revolutions
    slides = travel >= min_travel
    if not slides and not rolls:
      mode = "static"
    elif rolls and slides:
      mode = "both"
    elif rolls:
      mode = "rolling"
    else:
      mode = "sliding"
    result[object_id] = {
        "mode": mode,
        "path_length_m": travel,
        "revolutions": revolutions,
    }
  return result


@dataclass(frozen=True)
class QCResult:
  """Whether an instance is publishable, and why not when it is not."""

  accepted: bool
  reasons: Tuple[str, ...]
  metrics: Mapping[str, Any]


def evaluate_qc(
    spec: VelocityInstanceSpec,
    logs: Mapping[str, SimulationLog],
    pair_truth: GroundTruth,
    qc: Mapping[str, Any] = (),
) -> QCResult:
  """Rejects rollouts that cannot support the intended supervision."""
  settings = dict(qc or {})
  reasons: List[str] = []
  metrics: Dict[str, Any] = {}

  linear_ceiling = float(settings.get("linear_velocity_ceiling", 60.0))
  angular_ceiling = float(settings.get("angular_velocity_ceiling", 120.0))

  for branch, log in sorted(logs.items()):
    if not np.all(np.isfinite(log.states)):
      reasons.append("nonfinite_state:{}".format(branch))
      continue
    linear = float(np.max(np.linalg.norm(log.states[:, :, 7:10], axis=-1)))
    angular = float(np.max(np.linalg.norm(log.states[:, :, 10:13], axis=-1)))
    metrics["{}_max_linear_velocity".format(branch)] = linear
    metrics["{}_max_angular_velocity".format(branch)] = angular
    if linear > linear_ceiling:
      reasons.append("linear_velocity_ceiling:{}".format(branch))
    if angular > angular_ceiling:
      reasons.append("angular_velocity_ceiling:{}".format(branch))

    lower, upper = spec.scene_config.scene_bounds
    positions = log.states[:, :, 0:3]
    for axis in range(3):
      if np.any(positions[:, :, axis] < lower[axis] - 1e-6) or np.any(
          positions[:, :, axis] > upper[axis] + 1e-6
      ):
        reasons.append("out_of_bounds:{}".format(branch))
        break

  factual = logs["factual"]
  subject_contacts = tuple(
      record
      for record in factual.contacts
      if spec.subject_id in (record.object_a, record.object_b)
      and FLOOR_ID not in (record.object_a, record.object_b)
  )
  metrics["factual_subject_contacts"] = len(subject_contacts)
  if settings.get("require_factual_contact", True) and not subject_contacts:
    reasons.append("no_factual_interaction")

  min_effect = float(settings.get("min_counterfactual_effect", 0.0))
  displacement = _max_non_subject_displacement(
      factual, logs["counterfactual"], spec.subject_id
  )
  metrics["counterfactual_max_displacement_m"] = displacement
  if min_effect > 0.0 and displacement < min_effect:
    reasons.append("null_effect")

  metrics["removal_max_displacement_m"] = _max_non_subject_displacement(
      factual, logs["subject_removed"], spec.subject_id
  )
  metrics["hard_affected"] = list(pair_truth.hard_affected)
  metrics["soft_affected"] = list(pair_truth.soft_affected)

  modes = motion_modes(factual, settings)
  moving = {
      name: value["mode"]
      for name, value in modes.items()
      if name != FLOOR_ID and value["mode"] != "static"
  }
  metrics["motion_modes"] = {
      name: value["mode"] for name, value in sorted(modes.items())
  }
  if not any(mode in ("rolling", "both") for mode in moving.values()):
    reasons.append("no_rolling_motion")
  if not any(mode in ("sliding", "both") for mode in moving.values()):
    reasons.append("no_sliding_motion")

  return QCResult(
      accepted=not reasons, reasons=tuple(sorted(set(reasons))), metrics=metrics
  )


def _max_non_subject_displacement(
    factual: SimulationLog, other: SimulationLog, subject_id: str
) -> float:
  factual_index = {name: i for i, name in enumerate(factual.object_ids)}
  other_index = {name: i for i, name in enumerate(other.object_ids)}
  horizon = min(factual.states.shape[0], other.states.shape[0])
  best = 0.0
  for object_id in factual.object_ids:
    if object_id in (subject_id, FLOOR_ID) or object_id not in other_index:
      continue
    left = factual.states[:horizon, factual_index[object_id], 0:3]
    right = other.states[:horizon, other_index[object_id], 0:3]
    best = max(best, float(np.max(np.linalg.norm(left - right, axis=-1))))
  return best


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def instance_summary(
    spec: VelocityInstanceSpec,
    logs: Mapping[str, SimulationLog],
    pair_truth: GroundTruth,
    removal_truth: GroundTruth,
    qc_result: QCResult,
) -> Dict[str, Any]:
  """The JSON record published beside every instance's artifacts."""
  factual = logs["factual"]
  return {
      "schema_version": "1.0",
      "trust_model": VELOCITY_TRUST_MODEL,
      "instance_id": spec.instance_id,
      "index": spec.index,
      "master_seed": spec.master_seed,
      "subject_id": spec.subject_id,
      "branches": list(BRANCHES),
      "intervention": {
          "kind": "initial_linear_velocity",
          "applied_at_step": 0,
          "factual_velocity": list(spec.factual_velocity),
          "counterfactual_velocity": list(spec.counterfactual_velocity),
          "factual_azimuth": spec.factual_azimuth,
          "counterfactual_azimuth": spec.counterfactual_azimuth,
          "turn_radians": spec.counterfactual_azimuth - spec.factual_azimuth,
      },
      "scene": {
          "object_ids": list(factual.object_ids),
          "roles": dict(sorted(spec.roles.items())),
          "material_families": dict(sorted(spec.families.items())),
          "mass_kg": {
              item.object_id: item.mass
              for item in spec.scene_config.objects
              if not item.static
          },
          "frame_range": list(spec.scene_config.frame_range),
          "frame_rate": spec.scene_config.frame_rate,
          "step_rate": spec.scene_config.step_rate,
          "rollout_steps": spec.rollout_steps,
          "frame_steps": list(appearance.frame_steps_for(spec.scene_config)),
      },
      "visual_scene_hash": spec.visual_scene_hash,
      "camera": None if spec.visual_scene is None else {
          "motion": "fixed",
          "position": list(spec.visual_scene.camera.positions[0]),
          "look_at": list(spec.visual_scene.camera.look_ats[0]),
          "focal_length": spec.visual_scene.camera.focal_length,
          "sensor_width": spec.visual_scene.camera.sensor_width,
      },
      "ground_truth": {
          "counterfactual": to_jsonable(pair_truth),
          "subject_removed": to_jsonable(removal_truth),
      },
      "motion": {
          branch: motion_modes(log) for branch, log in sorted(logs.items())
      },
      "qc": {
          "accepted": qc_result.accepted,
          "reasons": list(qc_result.reasons),
          "metrics": qc_result.metrics,
      },
  }


__all__ = [
    "BRANCHES",
    "FLOOR_ID",
    "QCResult",
    "SUBJECT_ID",
    "VELOCITY_TRUST_MODEL",
    "VelocityInstanceSpec",
    "evaluate_qc",
    "generate_triplet",
    "instance_id_for",
    "instance_summary",
    "load_ranges",
    "motion_modes",
    "pair_ground_truth",
    "refit_camera",
    "removal_ground_truth",
    "run_branch",
    "sample_instance",
    "validate_ranges",
]
