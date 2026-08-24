"""Purpose: one immutable eleven-object contract for physics/replay/Blender/FFmpeg.

Public API: DemoObjectSpec, DemoSceneSpec, FORKED_RACK_SPEC,
validate_demo_spec(), canonical_spec_payload(), spec_sha256(), and
demo_spec_summary().
Dependencies: standard-library only; importing this module never loads Kubric,
PyBullet, Blender, NumPy, or TensorFlow.
Trust boundary: the digest detects local specification drift, not a signature
or producer attestation.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

BALL_RADIUS = 0.22
BALL_MASS = 1.0
BALL_FRICTION = 0.02
BALL_RESTITUTION = 0.65
_EXPECTED_IDS = ("breaker", "floor", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05", "rack_06", "side_01", "side_02", "target")
_MAIN_IDS = ("breaker", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05", "rack_06")
_SIDE_IDS = ("side_01", "side_02")


@dataclass(frozen=True)
class DemoObjectSpec:
    """Immutable physical and presentation metadata for one scene object."""
    object_id: str
    shape: str
    size: float | tuple[float, float, float]
    mass: float
    position: tuple[float, float, float]
    static: bool
    friction: float
    restitution: float
    visual_role: str
    group: str | None
    color: tuple[float, float, float]
    ball_number: int | None = None
    striped: bool = False
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DemoSceneSpec:
    """Immutable scene-wide contract consumed by all demo backends."""
    version: str
    objects: tuple[DemoObjectSpec, ...]
    seed: int
    scene_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    gravity: tuple[float, float, float]
    frame_range: tuple[int, int]
    frame_rate: int
    step_rate: int
    intervention_window: tuple[int, int]
    intervention_recipe: str
    intervention_magnitude: float
    push_mass: float
    target_id: str
    path_start: tuple[float, float, float]
    path_end: tuple[float, float, float]

    @property
    def num_steps(self) -> int:
        """Return integral physics steps covered by the frame range."""
        return ((self.frame_range[1] - self.frame_range[0]) * self.step_rate // self.frame_rate)

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Return object IDs in their stored, canonical order."""
        return tuple(obj.object_id for obj in self.objects)

    @property
    def ball_ids(self) -> tuple[str, ...]:
        """Return IDs whose visual role is ball."""
        return tuple(obj.object_id for obj in self.objects if obj.visual_role == "ball")

    @property
    def main_ball_ids(self) -> tuple[str, ...]:
        """Return ball IDs assigned to the main group."""
        return tuple(obj.object_id for obj in self.objects if obj.group == "main")

    @property
    def side_ball_ids(self) -> tuple[str, ...]:
        """Return ball IDs assigned to the side group."""
        return tuple(obj.object_id for obj in self.objects if obj.group == "side")


def _finite_vector(value: Sequence[object], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} finite values")
    try:
        values = tuple(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain {length} finite values") from None
    if len(values) != length or any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(v) for v in values)


def _number(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite real")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def validate_demo_spec(spec: DemoSceneSpec) -> None:
    """Validate the complete structural and numeric demo contract."""
    if not isinstance(spec, DemoSceneSpec):
        raise TypeError("spec must be DemoSceneSpec")
    if spec.version != "forked_rack_v1":
        raise ValueError("version must be forked_rack_v1")
    if isinstance(spec.objects, (str, bytes)) or not isinstance(spec.objects, Sequence):
        raise TypeError("objects must be a sequence of DemoObjectSpec")
    seen = set()
    for obj in spec.objects:
        if not isinstance(obj, DemoObjectSpec):
            raise TypeError("objects must contain DemoObjectSpec")
        if not isinstance(obj.object_id, str) or not obj.object_id:
            raise ValueError("object_id must be nonempty")
        if obj.object_id in seen:
            raise ValueError("object IDs must be unique; duplicate found")
        seen.add(obj.object_id)
        if obj.shape not in ("cube", "sphere"):
            raise ValueError("unsupported shape")
        if isinstance(obj.size, bool):
            raise ValueError("size must not be bool")
        if isinstance(obj.size, numbers.Real):
            _number(obj.size, "size", positive=True)
        else:
            vals = _finite_vector(obj.size, 3, "size (three components)")
            if any(v <= 0 for v in vals):
                raise ValueError("size must be positive")
        _number(obj.mass, "mass", positive=True)
        position = _finite_vector(obj.position, 3, "position")
        if obj.static is not True and obj.static is not False:
            raise ValueError("static must be bool")
        _number(obj.friction, "friction", nonnegative=True)
        restitution = _number(obj.restitution, "restitution")
        if not 0 <= restitution <= 1:
            raise ValueError("restitution must be in [0, 1]")
        if obj.visual_role not in ("floor", "target", "ball"):
            raise ValueError("visual_role unsupported")
        if obj.group not in (None, "main", "side"):
            raise ValueError("group unsupported")
        color = _finite_vector(obj.color, 3, "color")
        if any(not 0 <= v <= 1 for v in color):
            raise ValueError("color must be in [0, 1]")
        if not isinstance(obj.striped, bool):
            raise ValueError("striped must be bool")
        quat = _finite_vector(obj.quaternion, 4, "quaternion")
        norm = math.sqrt(sum(v * v for v in quat))
        if norm == 0 or abs(norm - 1.0) > 1e-12:
            raise ValueError("quaternion must be nonzero normalized")
        if obj.ball_number is not None and (isinstance(obj.ball_number, bool) or not isinstance(obj.ball_number, int)):
            raise ValueError("ball numbers must be integers")
        if obj.visual_role == "ball":
            if obj.group not in ("main", "side") or obj.shape != "sphere" or obj.static:
                raise ValueError("balls must be dynamic sphere objects")
        elif obj.group is not None or obj.ball_number is not None:
            raise ValueError("floor and target cannot have group or ball number")
        if obj.visual_role in ("floor", "target") and (not obj.static or obj.shape != "cube"):
            raise ValueError("floor and target must be static cube")
    roles = tuple(obj.visual_role for obj in spec.objects)
    if roles.count("floor") != 1:
        raise ValueError("floor required")
    if roles.count("target") != 1:
        raise ValueError("target role required exactly once")
    if roles.count("ball") != 9:
        raise ValueError("spec must contain exactly nine balls")
    if len(spec.objects) != 11 or len(spec.ball_ids) != 9:
        raise ValueError("spec must contain exactly nine balls")
    if spec.object_ids.count(spec.target_id) != 1 or spec.target_id != "target":
        raise ValueError("target must be target")
    numbers_seen = tuple(obj.ball_number for obj in spec.objects if obj.visual_role == "ball")
    if set(numbers_seen) != set(range(1, 10)) or len(numbers_seen) != 9:
        raise ValueError("ball numbers must be exactly 1..9")
    if spec.main_ball_ids != _MAIN_IDS or spec.side_ball_ids != _SIDE_IDS:
        raise ValueError("group memberships must be canonical")
    if spec.object_ids != _EXPECTED_IDS:
        raise ValueError("objects must use canonical order")
    for name, value in (("frame_range", spec.frame_range), ("intervention_window", spec.intervention_window)):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
            raise ValueError(f"{name} must contain integer endpoints")
    start, end = spec.frame_range
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (spec.frame_rate, spec.step_rate)):
        raise ValueError("frame rates must be integers")
    if end <= start or spec.frame_rate <= 0 or spec.step_rate <= 0:
        raise ValueError("frame rates and span must be positive")
    if (end - start) * spec.step_rate % spec.frame_rate:
        raise ValueError("physics steps must be integral")
    if spec.num_steps != 200:
        raise ValueError("spec must have 200 steps")
    win_start, win_end = spec.intervention_window
    if not (0 <= win_start < win_end <= spec.num_steps):
        raise ValueError("intervention window invalid")
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int) or spec.seed != 0:
        raise ValueError("seed must be 0")
    gravity = _finite_vector(spec.gravity, 3, "gravity")
    if isinstance(spec.scene_bounds, (str, bytes)) or not isinstance(spec.scene_bounds, Sequence) or len(spec.scene_bounds) != 2:
        raise ValueError("scene bounds must contain two vectors")
    bounds = tuple(_finite_vector(v, 3, "scene bounds") for v in spec.scene_bounds)
    if any(a >= b for a, b in zip(bounds[0], bounds[1])):
        raise ValueError("scene bounds must be increasing")
    path_start = _finite_vector(spec.path_start, 3, "path_start")
    _finite_vector(spec.path_end, 3, "path_end")
    target = next(obj for obj in spec.objects if obj.object_id == "target")
    if tuple(target.position) != tuple(path_start):
        raise ValueError("target position must equal path_start")
    if spec.intervention_recipe != "create_collision":
        raise ValueError("intervention recipe invalid")
    _number(spec.intervention_magnitude, "intervention magnitude", positive=True)
    _number(spec.push_mass, "push mass", positive=True)


OBJECTS = (
    DemoObjectSpec("breaker", "sphere", BALL_RADIUS, BALL_MASS, (0.25, 0.60, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.95, 0.45, 0.08), 1),
    DemoObjectSpec("floor", "cube", (4.0, 4.0, 0.25), 1.0, (0, 0, -0.25), True, 0, 0, "floor", None, (0.055, 0.19, 0.12)),
    DemoObjectSpec("rack_01", "sphere", BALL_RADIUS, BALL_MASS, (0.72, 0.60, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.15, 0.55, 0.92), 2),
    DemoObjectSpec("rack_02", "sphere", BALL_RADIUS, BALL_MASS, (1.18, 0.37, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.84, 0.16, 0.20), 3),
    DemoObjectSpec("rack_03", "sphere", BALL_RADIUS, BALL_MASS, (1.18, 0.83, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.48, 0.22, 0.78), 4),
    DemoObjectSpec("rack_04", "sphere", BALL_RADIUS, BALL_MASS, (2.00, 1.32, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.96, 0.78, 0.08), 5),
    DemoObjectSpec("rack_05", "sphere", BALL_RADIUS, BALL_MASS, (1.64, 0.60, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.10, 0.62, 0.30), 6),
    DemoObjectSpec("rack_06", "sphere", BALL_RADIUS, BALL_MASS, (1.64, 1.06, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "main", (0.50, 0.18, 0.10), 7),
    DemoObjectSpec("side_01", "sphere", BALL_RADIUS, BALL_MASS, (0.25, -0.48, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "side", (0.08, 0.42, 0.90), 8),
    DemoObjectSpec("side_02", "sphere", BALL_RADIUS, BALL_MASS, (0.72, -0.48, BALL_RADIUS), False, BALL_FRICTION, BALL_RESTITUTION, "ball", "side", (0.82, 0.12, 0.18), 9, True),
    DemoObjectSpec("target", "cube", 0.18, 2.0, (-2.0, -0.25, 0.18), True, 0.02, 0.65, "target", None, (0.58, 0.27, 0.08)),
)

FORKED_RACK_SPEC = DemoSceneSpec("forked_rack_v1", OBJECTS, 0, ((-4.5, -4.5, -1.0), (4.5, 4.5, 2.0)), (0, 0, 0), (0, 20), 24, 240, (40, 160), "create_collision", 1.2, 2.0, "target", (-2.0, -0.25, 0.18), (2.0, -0.25, 0.18))

validate_demo_spec(FORKED_RACK_SPEC)


def canonical_spec_payload(spec: DemoSceneSpec) -> Mapping[str, object]:
    """Return validated JSON-compatible data without reordering objects."""
    validate_demo_spec(spec)
    return asdict(spec)


def spec_sha256(spec: DemoSceneSpec) -> str:
    """Return the canonical lowercase SHA-256 digest of a scene spec."""
    payload = canonical_spec_payload(spec)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def demo_spec_summary(spec: DemoSceneSpec) -> dict[str, object]:
    """Return object count, digest, source frame count, and version."""
    validate_demo_spec(spec)
    return {"object_count": len(spec.objects), "sha256": spec_sha256(spec), "source_frames": spec.num_steps, "version": spec.version}
