# Dramatic Forked-Rack Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-object intervention demo with a deterministic eleven-object forked-rack billiards scene that contrasts a small normal chain, a large trajectory-changed chain, and a no-chain target-removal branch, while documenting the module surface primarily through docstrings.

**Architecture:** Add one pure-Python shared demo-spec module and make the physics generator, Blender renderer, and FFmpeg compositor validate and consume it. Keep the public `interventions/` behavior and three-branch trust model unchanged; only the fixed demo fixture, replay contract, visuals, overlays, and documentation change.

**Tech Stack:** Python 3.11 (`thesis` Conda env), dataclasses, JSON/SHA-256, NumPy, Kubric, PyBullet, Blender/Cycles in `kubricdockerhub/kubruntudev`, FFmpeg/FFprobe, pytest.

---

## Preconditions and execution rules

- Execute from an isolated workspace created with `using-git-worktrees` unless
  the user explicitly declines isolation.
- Use `/home/pineapple/miniconda3/envs/thesis/bin/python` for Python and pytest.
- Set `MPLCONFIGDIR=/tmp/kubric-mpl` and treat warnings as errors.
- Do not edit any file under `kubric/`.
- Do not stage `output/`, `.blend`, `.mp4`, `.npy`, or calibration scratch data.
- Implement every task red-green-refactor and obtain spec review followed by
  code-quality review before moving to the next task.
- The accepted design is
  `docs/superpowers/specs/2026-08-24-dramatic-forked-rack-demo-design.md`.

## File map

- Create `scripts/trajectory_demo_spec.py`: offline-safe dataclasses, the fixed
  eleven-object scene, grouping/timing/path values, canonical serialization,
  and spec digest.
- Create `tests/test_trajectory_demo_spec.py`: pure spec/validation/hash tests.
- Modify `scripts/demo_collision_intervention.py`: build the public pair from
  the shared spec, enforce small/large/no-chain outcomes, write spec identity,
  and remove four-object assumptions.
- Modify `tests/test_demo_collision_intervention.py`: real physics, removal,
  bundle, determinism, and workflow assertions for 200x11 replays.
- Modify `scripts/render_demo_branches_blender.py`: dynamic collider/material/
  decoration construction and shared-spec preflight.
- Modify `tests/test_render_demo_branches_blender.py`: eleven-object offline
  replay/scene tests and stale-spec rejection.
- Modify `run_demo.sh`: invoke the renderer as a package module so it can import
  the shared spec inside Docker.
- Modify `scripts/compose_intervention_demo.py`: spec validation, separate small
  and large chain cues, compact ending summary, and 272-frame output validation.
- Modify `tests/test_compose_intervention_demo.py`: summary/cue/filter/media
  tests for the new contract while retaining all safety regressions.
- Create `tests/test_module_documentation.py`: AST-based module/public-docstring
  contract without importing heavy backends.
- Modify all `interventions/*.py` module docstrings and missing exported-symbol
  docstrings.
- Modify `scripts/__init__.py`, `scripts/generate_dataset.py`,
  `scripts/generate_instance.py`, and the four demo-related script docstrings.
- Modify `docs/trajectory_interventions.md`: concise module table, new demo
  behavior, new duration/shapes, and stale-bundle migration note.
- Create `notes/session-logs/2026-08-24-dramatic-forked-rack-demo.md` only after
  final verification.

### Task 1: Add the shared eleven-object demo specification

**Files:**
- Create: `scripts/trajectory_demo_spec.py`
- Create: `tests/test_trajectory_demo_spec.py`

- [ ] **Step 1: Write the failing canonical-spec tests**

Create `tests/test_trajectory_demo_spec.py` with these core assertions:

```python
import dataclasses
import hashlib
import json
import math
import numbers

import pytest

from scripts import trajectory_demo_spec as demo_spec


EXPECTED_IDS = (
    "breaker",
    "floor",
    "rack_01",
    "rack_02",
    "rack_03",
    "rack_04",
    "rack_05",
    "rack_06",
    "side_01",
    "side_02",
    "target",
)


def test_forked_rack_spec_has_exact_contract():
  spec = demo_spec.FORKED_RACK_SPEC
  assert spec.version == "forked_rack_v1"
  assert spec.object_ids == EXPECTED_IDS
  assert len(spec.objects) == 11
  assert spec.ball_ids == (
      "breaker", "rack_01", "rack_02", "rack_03", "rack_04",
      "rack_05", "rack_06", "side_01", "side_02",
  )
  assert spec.main_ball_ids == (
      "breaker", "rack_01", "rack_02", "rack_03",
      "rack_04", "rack_05", "rack_06",
  )
  assert spec.side_ball_ids == ("side_01", "side_02")
  assert spec.num_steps == 200
  assert spec.frame_range == (0, 20)
  assert spec.frame_rate == 24
  assert spec.step_rate == 240
  assert spec.intervention_window == (40, 160)
  assert spec.target_id == "target"
  assert all(item.quaternion == (1.0, 0.0, 0.0, 0.0)
             for item in spec.objects)
  assert tuple(item.object_id for item in spec.objects) == EXPECTED_IDS
  assert tuple(
      item.object_id for item in spec.objects if item.group == "main"
  ) == spec.main_ball_ids
  assert tuple(
      item.object_id for item in spec.objects if item.group == "side"
  ) == spec.side_ball_ids


def test_spec_identity_is_canonical_and_deterministic():
  payload = demo_spec.canonical_spec_payload(demo_spec.FORKED_RACK_SPEC)
  encoded = json.dumps(
      payload, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")
  assert demo_spec.spec_sha256(demo_spec.FORKED_RACK_SPEC) == hashlib.sha256(
      encoded
  ).hexdigest()
  assert demo_spec.demo_spec_summary(demo_spec.FORKED_RACK_SPEC) == {
      "object_count": 11,
      "sha256": hashlib.sha256(encoded).hexdigest(),
      "source_frames": 200,
      "version": "forked_rack_v1",
  }
  assert tuple(item["object_id"] for item in payload["objects"]) == EXPECTED_IDS


def test_spec_is_frozen_and_rejects_duplicate_ids():
  with pytest.raises(dataclasses.FrozenInstanceError):
    demo_spec.FORKED_RACK_SPEC.num_steps = 10
  duplicate = dataclasses.replace(
      demo_spec.FORKED_RACK_SPEC,
      objects=(demo_spec.FORKED_RACK_SPEC.objects[0],) * 11,
  )
  with pytest.raises(ValueError, match="unique|duplicate"):
    demo_spec.validate_demo_spec(duplicate)


def test_spec_rejects_noncanonical_object_order():
  reordered = dataclasses.replace(
      demo_spec.FORKED_RACK_SPEC,
      objects=tuple(reversed(demo_spec.FORKED_RACK_SPEC.objects)),
  )
  with pytest.raises(ValueError, match="canonical.*order"):
    demo_spec.validate_demo_spec(reordered)


def test_initial_quaternion_is_part_of_the_digest():
  rotated = _replace_object(
      demo_spec.FORKED_RACK_SPEC,
      "breaker",
      quaternion=(0.0, 0.0, 0.0, 1.0),
  )
  demo_spec.validate_demo_spec(rotated)
  assert demo_spec.spec_sha256(rotated) != demo_spec.spec_sha256(
      demo_spec.FORKED_RACK_SPEC
  )


def _replace_object(spec, object_id, **changes):
  return dataclasses.replace(
      spec,
      objects=tuple(
          dataclasses.replace(item, **changes)
          if item.object_id == object_id else item
          for item in spec.objects
      ),
  )


@pytest.mark.parametrize(
    ("mutant", "message"),
    (
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", object_id=""),
         "object_id"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", shape="capsule"),
         "shape"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", size=0.0),
         "size"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", size=True),
         "size"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker",
                         size=(0.22, 0.22)),
         "three components"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", mass=0.0),
         "mass"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker",
                         position=(math.nan, 0.6, 0.22)),
         "finite"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker",
                         quaternion=(0.0, 0.0, 0.0, 0.0)),
         "quaternion"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker",
                         visual_role="prop"),
         "visual_role"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", group="side"),
         "group"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker",
                         ball_number=True),
         "ball numbers"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", static=True),
         "dynamic sphere"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "breaker", shape="cube"),
         "dynamic sphere"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "target", static=False),
         "static cube"),
        (_replace_object(demo_spec.FORKED_RACK_SPEC, "floor", shape="sphere"),
         "static cube"),
        (dataclasses.replace(demo_spec.FORKED_RACK_SPEC,
                             target_id="missing_target"),
         "target"),
        (dataclasses.replace(
            demo_spec.FORKED_RACK_SPEC,
            objects=tuple(item for item in demo_spec.FORKED_RACK_SPEC.objects
                          if item.object_id != "floor")),
         "floor"),
        (dataclasses.replace(
            demo_spec.FORKED_RACK_SPEC,
            objects=tuple(item for item in demo_spec.FORKED_RACK_SPEC.objects
                          if item.object_id != "side_02")),
         "nine balls"),
        (dataclasses.replace(demo_spec.FORKED_RACK_SPEC,
                             intervention_window=(-1, 160)),
         "window"),
        (dataclasses.replace(demo_spec.FORKED_RACK_SPEC, seed=-1),
         "seed"),
        (dataclasses.replace(demo_spec.FORKED_RACK_SPEC,
                             frame_range=(0, 21)),
         "200"),
        (dataclasses.replace(demo_spec.FORKED_RACK_SPEC,
                             frame_range=(0, 8), step_rate=25),
         "integral"),
    ),
)
def test_spec_rejects_every_invalid_structural_case(mutant, message):
  with pytest.raises((TypeError, ValueError), match=message):
    demo_spec.validate_demo_spec(mutant)
```

- [ ] **Step 2: Run the tests and capture RED**

Run:

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_trajectory_demo_spec.py
```

Expected: collection fails because `scripts.trajectory_demo_spec` does not
exist.

- [ ] **Step 3: Implement frozen spec types, fixed values, and digest helpers**

Create `scripts/trajectory_demo_spec.py` with:

```python
"""Defines the backend-neutral fixed scene used by the trajectory demo.

Purpose: Keep physics, replay validation, Blender presentation, and FFmpeg
annotations on one immutable eleven-object contract.
Public API: DemoObjectSpec, DemoSceneSpec, FORKED_RACK_SPEC,
validate_demo_spec(), canonical_spec_payload(), spec_sha256(), and
demo_spec_summary().
Dependencies: Python standard library only; importing this module never loads
Kubric, PyBullet, Blender, NumPy, or TensorFlow.
Trust boundary: The digest detects local spec drift; it is not a signature or
producer attestation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DemoObjectSpec:
  """One physics collider and its deterministic presentation role."""

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
  """Complete fixed forked-rack contract shared by every demo stage."""

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
    return ((self.frame_range[1] - self.frame_range[0]) * self.step_rate
            // self.frame_rate)

  @property
  def object_ids(self) -> tuple[str, ...]:
    return tuple(item.object_id for item in self.objects)

  @property
  def ball_ids(self) -> tuple[str, ...]:
    return tuple(
        item.object_id for item in self.objects if item.visual_role == "ball"
    )

  @property
  def main_ball_ids(self) -> tuple[str, ...]:
    return tuple(
        item.object_id for item in self.objects if item.group == "main"
    )

  @property
  def side_ball_ids(self) -> tuple[str, ...]:
    return tuple(
        item.object_id for item in self.objects if item.group == "side"
    )
```

Use these exact fixed values. They were calibrated in two independent thesis
Python processes and produced byte-identical paths, states, contacts, and
removal replays; runtime generation performs no search:

```python
BALL_RADIUS = 0.22
BALL_MASS = 1.0
BALL_FRICTION = 0.02
BALL_RESTITUTION = 0.65

OBJECTS = (
    DemoObjectSpec("breaker", "sphere", BALL_RADIUS, BALL_MASS,
                   (0.25, 0.60, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.95, 0.45, 0.08), 1, False),
    DemoObjectSpec("floor", "cube", (4.0, 4.0, 0.25), 1.0,
                   (0.0, 0.0, -0.25), True, 0.0, 0.0, "floor", None,
                   (0.055, 0.19, 0.12)),
    DemoObjectSpec("rack_01", "sphere", BALL_RADIUS, BALL_MASS,
                   (0.72, 0.60, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.15, 0.55, 0.92), 2, False),
    DemoObjectSpec("rack_02", "sphere", BALL_RADIUS, BALL_MASS,
                   (1.18, 0.37, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.84, 0.16, 0.20), 3, False),
    DemoObjectSpec("rack_03", "sphere", BALL_RADIUS, BALL_MASS,
                   (1.18, 0.83, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.48, 0.22, 0.78), 4, False),
    DemoObjectSpec("rack_04", "sphere", BALL_RADIUS, BALL_MASS,
                   (2.00, 1.32, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.96, 0.78, 0.08), 5, False),
    DemoObjectSpec("rack_05", "sphere", BALL_RADIUS, BALL_MASS,
                   (1.64, 0.60, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.10, 0.62, 0.30), 6, False),
    DemoObjectSpec("rack_06", "sphere", BALL_RADIUS, BALL_MASS,
                   (1.64, 1.06, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "main",
                   (0.50, 0.18, 0.10), 7, False),
    DemoObjectSpec("side_01", "sphere", BALL_RADIUS, BALL_MASS,
                   (0.25, -0.48, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "side",
                   (0.08, 0.42, 0.90), 8, False),
    DemoObjectSpec("side_02", "sphere", BALL_RADIUS, BALL_MASS,
                   (0.72, -0.48, BALL_RADIUS), False, BALL_FRICTION,
                   BALL_RESTITUTION, "ball", "side",
                   (0.82, 0.12, 0.18), 9, True),
    DemoObjectSpec("target", "cube", 0.18, 2.0,
                   (-2.0, -0.25, 0.18), True, 0.02, 0.65, "target", None,
                   (0.58, 0.27, 0.08)),
)

FORKED_RACK_SPEC = DemoSceneSpec(
    version="forked_rack_v1",
    objects=OBJECTS,
    seed=0,
    scene_bounds=((-4.5, -4.5, -1.0), (4.5, 4.5, 2.0)),
    gravity=(0.0, 0.0, 0.0),
    frame_range=(0, 20),
    frame_rate=24,
    step_rate=240,
    intervention_window=(40, 160),
    intervention_recipe="create_collision",
    intervention_magnitude=1.2,
    push_mass=2.0,
    target_id="target",
    path_start=(-2.0, -0.25, 0.18),
    path_end=(2.0, -0.25, 0.18),
)
```

Add the exact validator before constructing `FORKED_RACK_SPEC`:

```python
_EXPECTED_IDS = (
    "breaker", "floor", "rack_01", "rack_02", "rack_03", "rack_04",
    "rack_05", "rack_06", "side_01", "side_02", "target",
)
_EXPECTED_MAIN = (
    "breaker", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05",
    "rack_06",
)
_EXPECTED_SIDE = ("side_01", "side_02")


def _finite_vector(value: Sequence[float], length: int, name: str) -> None:
  if (isinstance(value, (str, bytes)) or len(value) != length or
      not all(isinstance(item, numbers.Real) and not isinstance(item, bool)
              and math.isfinite(float(item)) for item in value)):
    raise ValueError(f"{name} must contain {length} finite values")


def validate_demo_spec(spec: DemoSceneSpec) -> None:
  """Rejects any scene that is not the complete canonical forked-rack contract."""
  if not isinstance(spec, DemoSceneSpec):
    raise TypeError("spec must be a DemoSceneSpec")
  if spec.version != "forked_rack_v1":
    raise ValueError("version must be forked_rack_v1")

  ids = tuple(item.object_id for item in spec.objects)
  for item in spec.objects:
    if not item.object_id:
      raise ValueError("object_id must not be empty")
    if item.shape not in {"cube", "sphere"}:
      raise ValueError(f"unsupported shape for {item.object_id}")
    if item.visual_role not in {"floor", "target", "ball"}:
      raise ValueError(f"unsupported visual_role for {item.object_id}")
    if item.group not in {None, "main", "side"}:
      raise ValueError(f"unsupported group for {item.object_id}")
    if isinstance(item.size, bool):
      raise TypeError(f"size for {item.object_id} must be numeric")
    if isinstance(item.size, numbers.Real):
      size = (item.size,)
    else:
      if isinstance(item.size, (str, bytes)) or len(item.size) != 3:
        raise ValueError(f"size for {item.object_id} must have three components")
      size = item.size
    if not all(isinstance(v, numbers.Real) and not isinstance(v, bool)
               and math.isfinite(float(v)) and float(v) > 0 for v in size):
      raise ValueError(f"size for {item.object_id} must be finite and positive")
    if (not isinstance(item.mass, numbers.Real) or isinstance(item.mass, bool)
        or not math.isfinite(item.mass) or item.mass <= 0):
      raise ValueError(f"mass for {item.object_id} must be finite and positive")
    if (not isinstance(item.friction, numbers.Real) or
        isinstance(item.friction, bool) or not math.isfinite(item.friction) or
        item.friction < 0):
      raise ValueError(f"friction for {item.object_id} must be finite and nonnegative")
    if (not isinstance(item.restitution, numbers.Real) or
        isinstance(item.restitution, bool) or
        not math.isfinite(item.restitution) or
        not 0 <= item.restitution <= 1):
      raise ValueError(f"restitution for {item.object_id} must lie in [0, 1]")
    _finite_vector(item.position, 3, f"{item.object_id}.position")
    _finite_vector(item.color, 3, f"{item.object_id}.color")
    _finite_vector(item.quaternion, 4, f"{item.object_id}.quaternion")
    quaternion_norm = math.hypot(*item.quaternion)
    if quaternion_norm == 0.0 or abs(quaternion_norm - 1.0) > 1e-12:
      raise ValueError(f"quaternion for {item.object_id} must be normalized")
    if not all(0 <= component <= 1 for component in item.color):
      raise ValueError(f"color for {item.object_id} must lie in [0, 1]")
    if not isinstance(item.static, bool) or not isinstance(item.striped, bool):
      raise TypeError("static and striped must be bool values")

  if len(set(ids)) != len(ids):
    raise ValueError("object IDs must be unique; duplicate found")
  balls = tuple(item for item in spec.objects if item.visual_role == "ball")
  floors = tuple(item for item in spec.objects if item.visual_role == "floor")
  targets = tuple(item for item in spec.objects if item.visual_role == "target")
  if len(balls) != 9:
    raise ValueError("the scene must contain exactly nine balls")
  if len(floors) != 1:
    raise ValueError("the scene must contain exactly one floor")
  if len(targets) != 1 or targets[0].object_id != spec.target_id:
    raise ValueError("the scene must contain exactly one declared target")
  if {item.ball_number for item in balls} != set(range(1, 10)):
    raise ValueError("ball numbers must be unique integers 1 through 9")
  if any(not isinstance(item.ball_number, int) or
         isinstance(item.ball_number, bool) for item in balls):
    raise TypeError("ball numbers must be integers, not bool or float")
  if any(item.group not in {"main", "side"} for item in balls):
    raise ValueError("every ball must have main or side group membership")
  if any(item.shape != "sphere" or item.static for item in balls):
    raise ValueError("every ball must be a dynamic sphere")
  if any(item.group is not None or item.ball_number is not None
         for item in floors + targets):
    raise ValueError("only balls may have group membership or ball numbers")
  if floors[0].shape != "cube" or not floors[0].static:
    raise ValueError("floor must be a static cube")
  if targets[0].shape != "cube" or not targets[0].static:
    raise ValueError("target must be a static cube")
  if ids != _EXPECTED_IDS:
    raise ValueError("objects must use the exact canonical object order")
  if spec.main_ball_ids != _EXPECTED_MAIN or spec.side_ball_ids != _EXPECTED_SIDE:
    raise ValueError("main/side group membership differs from the canonical groups")

  timing = (*spec.frame_range, spec.frame_rate, spec.step_rate,
            *spec.intervention_window)
  if not all(isinstance(value, int) and not isinstance(value, bool)
             for value in timing):
    raise TypeError("timing values must be integers")
  frame_span = spec.frame_range[1] - spec.frame_range[0]
  numerator = frame_span * spec.step_rate
  if frame_span <= 0 or spec.frame_rate <= 0 or spec.step_rate <= 0:
    raise ValueError("timing rates and frame span must be positive")
  if numerator % spec.frame_rate:
    raise ValueError("timing must produce an integral number of physics steps")
  if numerator // spec.frame_rate != 200:
    raise ValueError("timing must produce exactly 200 physics steps")
  start, end = spec.intervention_window
  if not 0 <= start < end <= 200:
    raise ValueError("intervention window must lie inside [0, 200]")
  if not isinstance(spec.seed, int) or isinstance(spec.seed, bool) or spec.seed != 0:
    raise ValueError("seed must be the canonical integer 0")

  _finite_vector(spec.path_start, 3, "path_start")
  _finite_vector(spec.path_end, 3, "path_end")
  _finite_vector(spec.gravity, 3, "gravity")
  _finite_vector(spec.scene_bounds[0], 3, "scene_bounds lower")
  _finite_vector(spec.scene_bounds[1], 3, "scene_bounds upper")
  if not all(low < high for low, high in zip(*spec.scene_bounds)):
    raise ValueError("scene bounds must be strictly increasing")
  target = next(item for item in spec.objects if item.object_id == spec.target_id)
  if target.position != spec.path_start:
    raise ValueError("target position must exactly equal path_start")
  if spec.intervention_recipe != "create_collision":
    raise ValueError("intervention recipe must be create_collision")
  if not math.isfinite(spec.intervention_magnitude) or spec.intervention_magnitude <= 0:
    raise ValueError("intervention magnitude must be finite and positive")
  if not math.isfinite(spec.push_mass) or spec.push_mass <= 0:
    raise ValueError("push_mass must be finite and positive")
```

Call `validate_demo_spec(FORKED_RACK_SPEC)` immediately after construction; it
must never sort a mutant into validity. Then implement the canonical helpers:

```python
validate_demo_spec(FORKED_RACK_SPEC)
```

```python
def canonical_spec_payload(spec: DemoSceneSpec) -> Mapping[str, object]:
  """Returns validated JSON-safe data while preserving canonical object order."""
  validate_demo_spec(spec)
  return asdict(spec)


def spec_sha256(spec: DemoSceneSpec) -> str:
  """Returns the lowercase SHA-256 drift digest for the full scene contract."""
  encoded = json.dumps(
      canonical_spec_payload(spec),
      sort_keys=True,
      separators=(",", ":"),
      allow_nan=False,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def demo_spec_summary(spec: DemoSceneSpec) -> dict[str, object]:
  """Returns the compact exact identity persisted beside every replay."""
  return {
      "object_count": len(spec.objects),
      "sha256": spec_sha256(spec),
      "source_frames": spec.num_steps,
      "version": spec.version,
  }
```

- [ ] **Step 4: Run the pure tests and capture GREEN**

Run the command from Step 2. Expected: all spec tests pass with no warning.

- [ ] **Step 5: Commit the shared spec**

```bash
git add scripts/trajectory_demo_spec.py tests/test_trajectory_demo_spec.py
git commit -m "feat(demo): define forked rack spec"
```

### Task 2: Convert and calibrate the real physics fixture

**Files:**
- Modify: `scripts/demo_collision_intervention.py`
- Modify: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Replace four-object fixture assertions with failing chain tests**

Update the real fixture tests to assert:

```python
def _unique_dynamic_pairs(records):
  return {
      tuple(sorted((record.object_a, record.object_b)))
      for record in demo.dynamic_contacts(records)
  }


def test_build_demo_inputs_uses_shared_forked_rack_spec():
  scene, intervention, factual_path = demo.build_demo_inputs()
  spec = demo.FORKED_RACK_SPEC
  objects = {item.object_id: item for item in scene.objects}
  assert tuple(sorted(objects)) == spec.object_ids
  assert len(objects) == 11
  assert factual_path.shape == (200, 7)
  np.testing.assert_array_equal(factual_path[0, :3], spec.path_start)
  np.testing.assert_array_equal(factual_path[-1, :3], spec.path_end)
  np.testing.assert_array_equal(
      factual_path[:, 3:], np.tile((1.0, 0.0, 0.0, 0.0), (200, 1))
  )
  assert scene.frame_range == (0, 20)
  assert intervention.time_window == (40.0, 160.0)


def test_demo_has_small_normal_and_large_changed_chain(generated_demo):
  assert generated_demo.demo_spec == demo.FORKED_RACK_SPEC
  normal_pairs = _unique_dynamic_pairs(generated_demo.normal.contacts)
  changed_pairs = _unique_dynamic_pairs(generated_demo.changed.contacts)
  assert 2 <= len(normal_pairs) <= 3
  assert ("side_01", "target") in normal_pairs
  assert {"side_01", "side_02"}.issubset(
      {endpoint for pair in normal_pairs for endpoint in pair}
  )
  assert not any(
      endpoint in demo.FORKED_RACK_SPEC.main_ball_ids
      for pair in normal_pairs for endpoint in pair
  )
  assert 7 <= len(changed_pairs) <= 9
  assert ("breaker", "target") in changed_pairs
  changed_main = {
      endpoint
      for pair in changed_pairs
      for endpoint in pair
      if endpoint in demo.FORKED_RACK_SPEC.main_ball_ids
  }
  assert len(changed_main) >= 6
  assert not any(
      endpoint in demo.FORKED_RACK_SPEC.side_ball_ids
      for pair in changed_pairs for endpoint in pair
  )
  assert len(changed_pairs) >= len(normal_pairs) + 5
  hard = set(generated_demo.ground_truth.hard_affected)
  soft = set(generated_demo.ground_truth.soft_affected)
  assert hard.isdisjoint(soft)
  assert set(generated_demo.ground_truth.propagation_path) == hard
  assert all(
      path[0] == "target" and path[-1] == affected
      for affected, path in generated_demo.ground_truth.propagation_path.items()
  )
```

Update common-prefix expectations to `(40, 160)`, step count to 200, object
count to 11, and expected object order to `FORKED_RACK_SPEC.object_ids`. Add:

```python
def test_removed_branch_has_no_post_removal_dynamic_chain(generated_demo):
  assert not demo.dynamic_contacts(tuple(
      record for record in generated_demo.removed.contacts
      if record.step >= 40
  ))
```

Delete the obsolete Pillow schematic tests
`test_dynamic_contact_records_filters_floor_and_object`,
`test_render_branch_video_writes_readable_mp4`, and
`test_render_branch_video_draws_impact_ring_only_on_contact_frames`, together
with their `_synthetic_branch()` helper; Blender is the only supported
presentation path for this demo.

- [ ] **Step 2: Run the focused tests and capture RED**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_demo_collision_intervention.py \
  -k 'shared_forked or small_normal or post_removal_dynamic or diverges_only'
```

Expected: failures show the current 120-step/four-object/no-normal-contact
fixture.

- [ ] **Step 3: Adapt the generator to the shared spec**

Import `FORKED_RACK_SPEC`, `DemoSceneSpec`, and `demo_spec_summary`. Replace
module constants with values derived from the default spec:

```python
_DEMO_SPEC = FORKED_RACK_SPEC
_DEMO_SEED = _DEMO_SPEC.seed
_NUM_STEPS = _DEMO_SPEC.num_steps


def build_demo_inputs(
    spec: DemoSceneSpec = _DEMO_SPEC,
) -> tuple[SceneConfig, Intervention, np.ndarray]:
  objects = tuple(ObjectConfig(
      item.object_id,
      item.shape,
      size=item.size,
      mass=item.mass,
      position=item.position,
      quaternion=item.quaternion,
      static=item.static,
      friction=item.friction,
      restitution=item.restitution,
  ) for item in spec.objects)
  scene = SceneConfig(
      objects=objects,
      seed=spec.seed,
      scene_bounds=spec.scene_bounds,
      gravity=spec.gravity,
      frame_range=spec.frame_range,
      frame_rate=spec.frame_rate,
      step_rate=spec.step_rate,
  )
  intervention = Intervention(
      target_id=spec.target_id,
      recipe=spec.intervention_recipe,
      magnitude=spec.intervention_magnitude,
      time_window=spec.intervention_window,
      push_mass=spec.push_mass,
  )
  factual_path = np.zeros((spec.num_steps, 7), dtype=np.float64)
  factual_path[:, :3] = np.linspace(spec.path_start, spec.path_end, spec.num_steps)
  factual_path[:, 3] = 1.0
  return scene, intervention, factual_path
```

Extend the result with the immutable spec that actually produced it:

```python
@dataclass(frozen=True)
class DemoResult:
  """Carries the canonical spec and its three validated replay branches."""

  demo_spec: DemoSceneSpec
  scene_config: SceneConfig
  intervention: Intervention
  normal: SimulationLog
  changed: SimulationLog
  removed: RemovedBranch
  ground_truth: GroundTruth

  @property
  def intervention_window(self) -> tuple[int, int]:
    """Returns the half-open integer intervention window."""
    return tuple(int(value) for value in self.intervention.time_window)
```

Replace `generate_demo()` with the identity-carrying version:

```python
def generate_demo(seed: int = _DEMO_SEED) -> DemoResult:
  """Generates the three deterministic branches bound to the canonical spec."""
  if seed != _DEMO_SPEC.seed:
    raise ValueError("fixed deterministic demo seed is 0")
  scene, intervention, factual_path = build_demo_inputs(_DEMO_SPEC)
  normal, changed = generate_paired_instance(
      scene,
      intervention.target_id,
      intervention,
      seed,
      factual_path=factual_path,
  )
  ground_truth = extract_pair_ground_truth(
      scene, intervention, normal, changed
  )
  provenance = {
      key: normal.metadata[key]
      for key in ("scene_config_sha256", "intervention_sha256")
      if key in normal.metadata
  }
  removed = _run_removed_branch(
      scene, intervention, factual_path, provenance
  )
  _validate_demo_outcomes(normal, changed, ground_truth)
  _validate_removed_branch(removed, normal, intervention)
  return DemoResult(
      demo_spec=_DEMO_SPEC,
      scene_config=scene,
      intervention=intervention,
      normal=normal,
      changed=changed,
      removed=removed,
      ground_truth=ground_truth,
  )
```

Add a canonical-pair helper and replace `_validate_demo_outcomes()` with:

```python
def _unique_dynamic_pairs(records) -> frozenset[tuple[str, str]]:
  return frozenset(
      tuple(sorted((str(_record_value(record, "object_a")),
                    str(_record_value(record, "object_b")))))
      for record in dynamic_contacts(records)
  )


def _validate_demo_outcomes(normal, changed, ground_truth) -> None:
  normal_pairs = _unique_dynamic_pairs(normal.contacts)
  changed_pairs = _unique_dynamic_pairs(changed.contacts)
  normal_endpoints = {name for pair in normal_pairs for name in pair}
  changed_endpoints = {name for pair in changed_pairs for name in pair}
  if not 2 <= len(normal_pairs) <= 3:
    raise RuntimeError(f"normal pair count drifted: {sorted(normal_pairs)!r}")
  if ("side_01", "target") not in normal_pairs:
    raise RuntimeError(f"normal misses target|side_01: {sorted(normal_pairs)!r}")
  if not set(_DEMO_SPEC.side_ball_ids) <= normal_endpoints:
    raise RuntimeError(f"normal misses a side ball: {sorted(normal_pairs)!r}")
  if set(_DEMO_SPEC.main_ball_ids) & normal_endpoints:
    raise RuntimeError(f"normal reached the main group: {sorted(normal_pairs)!r}")
  if not 7 <= len(changed_pairs) <= 9:
    raise RuntimeError(f"changed pair count drifted: {sorted(changed_pairs)!r}")
  if ("breaker", "target") not in changed_pairs:
    raise RuntimeError(f"changed misses breaker|target: {sorted(changed_pairs)!r}")
  if len(set(_DEMO_SPEC.main_ball_ids) & changed_endpoints) < 6:
    raise RuntimeError(f"changed reaches too few main balls: {sorted(changed_pairs)!r}")
  if set(_DEMO_SPEC.side_ball_ids) & changed_endpoints:
    raise RuntimeError(f"changed reached the side group: {sorted(changed_pairs)!r}")
  if len(changed_pairs) < len(normal_pairs) + 5:
    raise RuntimeError("changed chain is not at least five pairs larger")
  if set(ground_truth.hard_affected) & set(ground_truth.soft_affected):
    raise RuntimeError("hard and soft affected sets overlap")
  if set(ground_truth.propagation_path) != set(ground_truth.hard_affected):
    raise RuntimeError("propagation-path keys differ from hard affected IDs")
  if any(path[0] != _DEMO_SPEC.target_id or path[-1] != affected
         for affected, path in ground_truth.propagation_path.items()):
    raise RuntimeError("ground-truth propagation path is not target-rooted")
```

At the end of `_validate_removed_branch()`, reject every post-removal dynamic
contact, not only target contacts:

```python
post_removal = dynamic_contacts(tuple(
    record for record in removed.contacts if record.step >= removed_step
))
if post_removal:
  raise RuntimeError(
      f"removed branch contains a post-removal dynamic contact: {post_removal!r}"
  )
```

Update `_validate_demo_bundle()` to use `(result.demo_spec.num_steps,
len(result.demo_spec.object_ids), 13)` instead of `(120, 4, 13)`. Remove
the schematic-only `BranchResult` type, `imageio`, Pillow, `_CANVAS_SIZE`,
`_WORLD_BOUNDS`, `_world_to_canvas()`,
`_draw_circle()`, `_dynamic_contact_records()`, and `_render_branch_video()`;
none participates in the supported realistic Blender workflow.

- [ ] **Step 4: Run the calibrated fixture and require the exact outcome**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_demo_collision_intervention.py \
  -k 'shared_forked or small_normal or post_removal_dynamic or diverges_only' \
  -vv
```

Expected exact deterministic diagnostics:

```text
normal (2): side_01|side_02, side_01|target
trajectory_changed (9): breaker|target, rack_01|rack_03,
rack_01|target, rack_02|rack_03, rack_02|rack_05, rack_02|target,
rack_03|rack_06, rack_04|rack_06, rack_05|rack_06
target_removed after step 40: no dynamic pairs
hard affected: breaker, rack_01, rack_02, rack_03, rack_04, rack_05,
rack_06, side_01, side_02
soft affected: none
```

Any mismatch is a regression: stop and diagnose it; do not search at runtime or
weaken the approved thresholds. `target.position` and `path_start` remain the
same coupled value `(-2.0, -0.25, 0.18)`.

- [ ] **Step 5: Lock determinism and complete the generator suite**

Run the full generator test twice:

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_demo_collision_intervention.py
```

Expected: both runs pass and the existing exact determinism tests observe
byte-identical states, presence, contacts, and ground truth.

- [ ] **Step 6: Commit the calibrated physics fixture**

```bash
git add scripts/demo_collision_intervention.py \
  tests/test_demo_collision_intervention.py
git commit -m "feat(demo): add forked rack chain reaction"
```

### Task 3: Bind the replay bundle to the shared spec

**Files:**
- Modify: `scripts/demo_collision_intervention.py`
- Modify: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Add failing summary identity and stale-result tests**

Add:

```python
from scripts import trajectory_demo_spec as demo_spec


def test_bundle_summary_binds_exact_demo_spec(generated_demo, tmp_path):
  summary = demo.write_demo_bundle(tmp_path, generated_demo)
  assert summary["demo_spec"] == demo_spec.demo_spec_summary(
      demo_spec.FORKED_RACK_SPEC
  )
  assert np.load(tmp_path / "normal_states.npy", allow_pickle=False).shape == (
      200, 11, 13
  )
  assert np.load(
      tmp_path / "target_removed_presence.npy", allow_pickle=False
  ).shape == (200, 11)


def test_bundle_rejects_result_from_different_spec(generated_demo, tmp_path):
  wrong_spec = dataclasses.replace(
      demo_spec.FORKED_RACK_SPEC, version="forked_rack_v2"
  )
  wrong_result = dataclasses.replace(generated_demo, demo_spec=wrong_spec)
  with pytest.raises(
      ValueError, match="demo result spec identity differs from canonical"
  ):
    demo.write_demo_bundle(tmp_path, wrong_result)
  assert not any(tmp_path.iterdir())
```

Update exact summary-key tests to include `demo_spec` and exact output file
shape expectations to 200x11.

- [ ] **Step 2: Run the focused tests and capture RED**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_demo_collision_intervention.py \
  -k 'binds_exact_demo_spec or different_spec or summary_has_exact'
```

Expected: summary assertions fail because `demo_spec` is absent, and the stale
result is not yet rejected.

- [ ] **Step 3: Persist and validate the identity before writes**

Keep the public API narrow:

```python
def write_demo_bundle(
    output_dir: str | Path,
    result: DemoResult,
) -> dict[str, object]:
```

At the start of `_validate_demo_bundle(result)`, bind the stored identity and
reuse the generator adapter for exact geometry/timing/path comparison:

```python
if result.demo_spec != _DEMO_SPEC:
  raise ValueError("demo result spec identity differs from canonical")
validate_demo_spec(result.demo_spec)
expected_scene, expected_intervention, expected_path = build_demo_inputs(
    result.demo_spec
)
if result.scene_config != expected_scene:
  raise ValueError("demo scene differs from the stored demo spec")
if result.intervention != expected_intervention:
  raise ValueError("demo intervention differs from the stored demo spec")
if not np.array_equal(result.normal.commanded_path, expected_path):
  raise ValueError("factual commanded path differs from the stored demo spec")
```

Retain the existing branch/provenance/ground-truth checks and derive every array
shape from `result.demo_spec`. Build the summary with:

```python
summary = {
    "branches": branch_summaries,
    "demo_spec": demo_spec_summary(result.demo_spec),
    "ground_truth": result.ground_truth.to_dict(),
    "intervention_end": end,
    "intervention_start": start,
    "intervention_window": [start, end],
    "object_ids": list(result.normal.object_ids),
    "seed": int(result.scene_config.seed),
    "step_rate": float(result.scene_config.step_rate),
}
```

Perform every validation and canonical JSON encoding before `output.mkdir()` so
a mismatched spec cannot create an empty output directory.

- [ ] **Step 4: Run bundle tests and capture GREEN**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_demo_collision_intervention.py
```

Expected: all generator/bundle tests pass; two writes are byte-identical.

- [ ] **Step 5: Commit the bound bundle contract**

```bash
git add scripts/demo_collision_intervention.py \
  tests/test_demo_collision_intervention.py
git commit -m "feat(demo): bind replays to scene spec"
```

### Task 4: Make the Blender renderer construct all eleven objects

**Files:**
- Modify: `scripts/render_demo_branches_blender.py`
- Modify: `tests/test_render_demo_branches_blender.py`
- Modify: `run_demo.sh`
- Modify: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Convert test fixtures to the shared object contract**

Import the shared module in the renderer tests and define:

```python
from scripts import trajectory_demo_spec as demo_spec


_OBJECT_IDS = demo_spec.FORKED_RACK_SPEC.object_ids


def _synthetic_states(num_steps=demo_spec.FORKED_RACK_SPEC.num_steps):
  states = np.zeros((num_steps, len(_OBJECT_IDS), 13), dtype=np.float64)
  states[:, :, 3] = 1.0
  for index, object_id in enumerate(_OBJECT_IDS):
    initial = next(
        item.position for item in demo_spec.FORKED_RACK_SPEC.objects
        if item.object_id == object_id
    )
    states[:, index, 0:3] = initial
  return states
```

Make `_write_bundle()` include the exact `demo_spec` summary. Add failing tests:

```python
def test_scene_specs_cover_every_shared_object_once():
  colliders = render_script._scene_specs()["colliders"]
  assert tuple(colliders) == _OBJECT_IDS
  assert colliders["breaker"]["kind"] == "sphere"
  assert colliders["target"]["kind"] == "cube"


def test_material_specs_number_all_nine_balls():
  materials = render_script._material_specs()
  balls = [
      object_id for object_id in _OBJECT_IDS
      if materials[object_id]["material"] == "lacquer"
  ]
  assert tuple(balls) == demo_spec.FORKED_RACK_SPEC.ball_ids
  assert {materials[name]["number"] for name in balls} == set(range(1, 10))
  assert {
      name: materials[name]["striped"] for name in balls
  } == {
      item.object_id: item.striped
      for item in demo_spec.FORKED_RACK_SPEC.objects
      if item.visual_role == "ball"
  }


def test_load_replay_accepts_exact_current_demo_spec(tmp_path):
  states, _, _ = _write_bundle(tmp_path)
  replay = render_script._load_replay(tmp_path, "normal")
  assert replay.states.shape == (200, 11, 13)
  np.testing.assert_array_equal(replay.states, states)


def test_load_replay_rejects_stale_demo_spec(tmp_path):
  _write_bundle(tmp_path)
  summary_path = tmp_path / "summary.json"
  summary = json.loads(summary_path.read_text("utf-8"))
  summary["demo_spec"]["sha256"] = "0" * 64
  summary_path.write_text(json.dumps(summary), encoding="utf-8")
  with pytest.raises(ValueError, match="demo_spec.sha256 mismatch"):
    render_script._load_replay(tmp_path, "normal")


def test_load_replay_rejects_truncated_states_before_max_frame_slice(tmp_path):
  _write_bundle(tmp_path)
  states_path = tmp_path / "normal_states.npy"
  states = np.load(states_path, allow_pickle=False)
  np.save(states_path, states[:-1], allow_pickle=False)
  with pytest.raises(ValueError, match=r"states.*\(200, 11, 13\)"):
    render_script._load_replay(tmp_path, "normal")


def test_camera_contract_contains_fixture_and_rejects_clipped_extent(tmp_path):
  _write_bundle(tmp_path)
  replay = render_script._load_replay(tmp_path, "normal")
  render_script._validate_camera_containment((replay,), (640, 540))
  camera = np.asarray(render_script._CAMERA_POSITION)
  forward = np.asarray(render_script._CAMERA_LOOK_AT) - camera
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
  right /= np.linalg.norm(right)
  depth = 10.0
  half_width = depth * 36.0 / (2.0 * render_script._CAMERA_FOCAL_LENGTH)
  near_edge = camera + forward * depth + right * (0.90 * half_width)
  clipped_states = replay.states.copy()
  clipped_states[:, replay.object_ids.index("breaker"), :3] = near_edge
  clipped = dataclasses.replace(replay, states=clipped_states)
  with pytest.raises(ValueError, match="camera framing"):
    render_script._validate_camera_containment((clipped,), (640, 540))


def test_run_demo_invokes_renderer_as_package_module(tmp_path):
  script, root, call_log, environment = _workflow_sandbox(tmp_path)
  completed = _run_workflow(script, root, environment, "intervention")
  assert completed.returncode == 0, completed.stderr
  render_call = next(
      line for line in call_log.read_text("utf-8").splitlines()
      if line.startswith("docker run ")
  )
  assert "python3 -m scripts.render_demo_branches_blender" in render_call
```

Place `test_run_demo_invokes_renderer_as_package_module` in
`tests/test_demo_collision_intervention.py`, beside `_workflow_sandbox()` and
`_run_workflow()`. All preceding tests in this Step go in
`tests/test_render_demo_branches_blender.py`.

Update invalid shape cases from `(frames, 4, 13)` to `(200, 11, 13)`. The
`_write_bundle()` fixture always writes 200 frames; truncation happens only in
`_prepare_replay()` after the full replay has passed `_load_replay()`. Update
the existing slice assertions so `max_frames=100` returns 100 frames and
`max_frames=500` returns the complete 200-frame replay.

Mechanically update the remaining old-fixture assertions as follows:

```python
# Summary windows that must lie outside the 200-frame replay.
@pytest.mark.parametrize(("start", "end"), ((201, 202), (40, 201)))

# WXYZ pose test: index 0 is now breaker.
states[0, 0, 3:7] = (0.4, 0.1, 0.2, 0.3)
assert render_script._pose_at(states, 0, 0) == (
    (0.25, 0.60, 0.22), (0.4, 0.1, 0.2, 0.3)
)

# Every manually constructed Replay uses the canonical IDs and 11 columns.
object_ids = demo_spec.FORKED_RACK_SPEC.object_ids
presence = np.ones((num_steps, 11), dtype=np.bool_)
```

Replace old `upper_ball`/`lower_ball` material assertions with
`breaker`/`side_02`, and replace the four-entry collider literal with exact
equality to `_collider_specs()`. Tests that deliberately compare already
prepared 5-frame and 4-frame `Replay` values may retain those short time axes;
only persisted `_load_replay()` input is required to have 200 frames.

- [ ] **Step 2: Run renderer tests and capture RED**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_render_demo_branches_blender.py \
  tests/test_demo_collision_intervention.py \
  -k 'shared_object or all_nine or current_demo_spec or stale_demo_spec or truncated_states or camera_contract or invokes_renderer_as_package'
```

Expected: current renderer requires the four old IDs and lacks `demo_spec`.

- [ ] **Step 3: Replace four-object constants with spec-derived maps**

Import only the pure spec module at import time:

```python
from scripts.trajectory_demo_spec import (
    FORKED_RACK_SPEC,
    demo_spec_summary,
)

_DEMO_SPEC = FORKED_RACK_SPEC
_CANONICAL_OBJECT_IDS = _DEMO_SPEC.object_ids
```

Generate collider and material specs:

```python
def _collider_specs() -> Dict[str, Dict[str, Any]]:
  return {
      item.object_id: {"kind": item.shape, "scale": item.size}
      for item in _DEMO_SPEC.objects
  }


def _material_specs() -> Dict[str, Dict[str, Any]]:
  specs = {
      "rail": {
          "material": "wood", "color": (0.16, 0.045, 0.012, 1.0),
          "light_color": (0.43, 0.13, 0.025, 1.0), "roughness": 0.30,
          "grain_scale": 3.5,
      },
      "backdrop": {
          "material": "matte", "color": (0.055, 0.06, 0.072, 1.0),
          "roughness": 0.82,
      },
      "band": {
          "material": "lacquer", "color": (0.94, 0.94, 0.90, 1.0),
          "roughness": 0.16, "metallic": 0.0,
      },
      "number": {
          "material": "matte", "color": (0.012, 0.012, 0.012, 1.0),
          "roughness": 0.45,
      },
  }
  for item in _DEMO_SPEC.objects:
    rgba = (*item.color, 1.0)
    if item.visual_role == "ball":
      specs[item.object_id] = {
          "material": "lacquer",
          "color": rgba,
          "roughness": 0.16,
          "metallic": 0.04,
          "number": item.ball_number,
          "striped": item.striped,
      }
    elif item.visual_role == "target":
      specs[item.object_id] = {
          "material": "wood", "color": rgba,
          "light_color": (0.72, 0.31, 0.075, 1.0),
          "roughness": 0.30, "grain_scale": 5.5,
      }
    else:
      specs[item.object_id] = {
          "material": "felt", "color": rgba, "roughness": 0.88,
          "noise_scale": 92.0, "bump_strength": 0.16,
      }
  return copy.deepcopy(specs)
```

Make `_scene_specs()` include `_collider_specs()`. Extend `_load_summary()` exact
keys with `demo_spec`. Compare its exact keys and values to
`demo_spec_summary(_DEMO_SPEC)` and raise
`ValueError("demo_spec.<field> mismatch")` for the first differing field.
Every replay shape check requires exactly
`(_DEMO_SPEC.num_steps, len(_CANONICAL_OBJECT_IDS), 13)` and the matching full
presence shape before `_prepare_replay()` applies `--max-frames`.

Use this identity helper from `_load_summary()`:

```python
def _validate_demo_spec_identity(value: Any) -> Mapping[str, Any]:
  if not isinstance(value, dict):
    raise TypeError("demo_spec must be a JSON object")
  expected = demo_spec_summary(_DEMO_SPEC)
  if set(value) != set(expected):
    raise ValueError("demo_spec keys mismatch")
  for field, expected_value in expected.items():
    if value[field] != expected_value:
      raise ValueError(f"demo_spec.{field} mismatch")
  return value
```

- [ ] **Step 4: Write failing decoration, DOF, and camera tests**

Add these offline assertions before changing Blender construction:

```python
def test_camera_dof_uses_approved_deterministic_values():
  assert render_script._camera_dof_spec() == {
      "use_dof": True,
      "focus_distance": 12.0,
      "aperture_fstop": 5.6,
  }


def test_material_contract_distinguishes_solid_and_striped_balls():
  materials = render_script._material_specs()
  assert materials["side_01"]["number"] == 8
  assert materials["side_01"]["striped"] is False
  assert materials["side_02"]["number"] == 9
  assert materials["side_02"]["striped"] is True
```

Run the focused tests and require RED on the old DOF/four-object material
contract:

```bash
/home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest -q \
  tests/test_render_demo_branches_blender.py \
  -k 'approved_deterministic or distinguishes_solid or camera_contract'
```

- [ ] **Step 5: Generalize decorations and camera preflight**

Change `_add_ball_decorations()` so `number` is an integer, the white torus is
created only when `striped` is true, the number badge/decal are always created,
and `decal.data.body = str(number)`. Return every created object in one tuple;
all are attached through `_parent_local()` before visibility keyframes.

In `_build_and_render_branch_in_scratch()`, iterate every shared object whose
`visual_role == "ball"`, smooth its collider mesh, and call:

```python
decorations[name] = _add_ball_decorations(
    bpy,
    blender_assets[name],
    name,
    material_specs[name]["number"],
    material_specs[name]["striped"],
    band_material,
    number_material,
)
```

Keep `_round_target()`. The existing visibility loop must continue to pass
`(blender_assets[name],) + decorations.get(name, ())`, which hides the target
parent and every child decoration together.

Use these shared presentation values:

```python
_CAMERA_POSITION = (7.6, -9.2, 8.4)
_CAMERA_LOOK_AT = (0.35, 0.30, -0.02)
_CAMERA_FOCAL_LENGTH = 55.0
```

Set deterministic depth of field to `focus_distance=12.0` and
`aperture_fstop=5.6`. Add this conservative pure NumPy pinhole preflight. A
cube uses its rotation-invariant enclosing-sphere radius, so arbitrary replay
WXYZ rotation cannot invalidate the bound:

```python
def _collider_radius(item) -> float:
  if item.visual_role == "floor":
    return 0.0  # the felt surface is intentionally full-bleed
  if item.shape == "sphere":
    return float(item.size)
  scale = ((float(item.size),) * 3
           if isinstance(item.size, (int, float)) else item.size)
  return math.sqrt(sum(float(component) ** 2 for component in scale))


def _validate_camera_containment(
    replays: Sequence[Replay], resolution: Tuple[int, int]
) -> None:
  camera = np.asarray(_CAMERA_POSITION, dtype=np.float64)
  forward = np.asarray(_CAMERA_LOOK_AT, dtype=np.float64) - camera
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  width, height = resolution
  item_by_id = {item.object_id: item for item in _DEMO_SPEC.objects}
  for replay in replays:
    for object_index, object_id in enumerate(replay.object_ids):
      radius = _collider_radius(item_by_id[object_id])
      for center in replay.states[:, object_index, _POSITION_SLICE]:
        relative = center - camera
        depth = float(relative @ forward)
        near_depth = depth - radius
        if near_depth <= 0.0:
          raise ValueError(f"camera framing excludes {object_id}")
        half_width = near_depth * 36.0 / (2.0 * _CAMERA_FOCAL_LENGTH)
        half_height = half_width * height / width
        if (abs(float(relative @ right)) + radius > 0.94 * half_width or
            abs(float(relative @ up)) + radius > 0.94 * half_height):
          raise ValueError(f"camera framing excludes {object_id}")
```

Call `_validate_camera_containment(replays, resolution)` in `main()` after
`_preflight_replays()` and before `_require_imageio_ffmpeg()` or any Blender
initialization. The calibrated replay was checked independently with exact
sphere/cube extrema: maximum normalized horizontal extent is `0.4939730326`
(`target`, frame 0) and vertical extent is `0.4417458285` (`side_01`, normal
frame 199), both far inside the `0.94` gate.

Finally change the Docker renderer command in `run_demo.sh` from direct script
execution to package execution while retaining the existing imageio bootstrap:

```bash
python3 -m scripts.render_demo_branches_blender \
  --states-dir /workspace/output/demo_collision_intervention \
  --branches normal trajectory_changed target_removed
```

- [ ] **Step 6: Run the complete offline renderer suite**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_render_demo_branches_blender.py
```

Expected: all tests pass without importing Kubric, PyBullet, or Blender during
module import.

- [ ] **Step 7: Run a Docker one-frame smoke for all branches**

Generate the bundle, then run:

```bash
./run_demo.sh intervention-physics-only
DEMO_UID="$(id -u)"
DEMO_GID="$(id -g)"
DEMO_ROOT="$(pwd)"
DEMO_IMAGE_ID="$(docker image inspect --format '{{.Id}}' \
  kubricdockerhub/kubruntudev)"
printf '%s\n' "${DEMO_IMAGE_ID}"
docker run --rm --user "${DEMO_UID}:${DEMO_GID}" \
  --volume "${DEMO_ROOT}:/workspace" "${DEMO_IMAGE_ID}" \
  sh -lc 'set -eu; cd /workspace; if ! python3 -c "import imageio_ffmpeg" >/dev/null 2>&1; then python3 -m pip install --quiet --disable-pip-version-check --no-cache-dir --target /tmp/kubric-demo-imageio imageio-ffmpeg; export PYTHONPATH="/tmp/kubric-demo-imageio${PYTHONPATH:+:$PYTHONPATH}"; fi; python3 -m scripts.render_demo_branches_blender \
    --states-dir output/demo_collision_intervention \
    --branches normal trajectory_changed target_removed \
    --resolution 320 180 --samples 8 --max-frames 1'
```

Expected: three H.264/yuv420p 320x180, 24-fps, one-frame videos; no missing
object/material/decor error.

- [ ] **Step 8: Commit the dynamic renderer**

```bash
git add run_demo.sh scripts/render_demo_branches_blender.py \
  tests/test_render_demo_branches_blender.py \
  tests/test_demo_collision_intervention.py
git commit -m "feat(demo): render eleven-object rack"
```

### Task 5: Add small/large chain cues and compact final metadata

**Files:**
- Modify: `scripts/compose_intervention_demo.py`
- Modify: `tests/test_compose_intervention_demo.py`

- [ ] **Step 1: Update the synthetic summary and write failing overlay tests**

Make every test summary contain the exact shared `demo_spec`, 200 source frames,
window `[40, 160)`, these two normal pairs and nine changed pairs, a
`side_01|target` removed graph record, and a `breaker|target` added record:

```python
from scripts import trajectory_demo_spec as demo_spec


NORMAL_PAIRS = {"side_01|side_02": 1, "side_01|target": 1}
CHANGED_PAIRS = {
    "breaker|target": 1,
    "rack_01|rack_03": 1,
    "rack_01|target": 1,
    "rack_02|rack_03": 1,
    "rack_02|rack_05": 1,
    "rack_02|target": 1,
    "rack_03|rack_06": 1,
    "rack_04|rack_06": 1,
    "rack_05|rack_06": 1,
}
```

Keep these maps in the synthetic `_summary()` fixture; use a normal cue step of
`88`, changed cue step of `70`, and removal step `40`. Then add:

All valid summary/source fixtures now use `frame_count=200`, source duration
`200 / 24`, and valid composed output `frame_count=272`, duration `272 / 24`.
Change intentional source-frame mismatch cases to `199`. Change the real FFmpeg
fixture from six frames to 200 frames (`-frames:v 200`, duration `200/24`) and
rename its test to `test_real_ffmpeg_composes_full_demo_frame_count`; require a
272-frame, approximately 11.333333-second output. This keeps the strict spec
identity meaningful instead of weakening it for tests.

```python
def test_overlay_texts_bind_small_and_large_chain_events(compositor):
  texts = compositor._overlay_texts(
      _summary(), source_duration=200 / 24, source_fps=24.0
  )
  assert texts["normal_chain"].startswith("SMALL CHAIN → SIDE 01")
  assert texts["changed_chain"].startswith("LARGE CHAIN → BREAKER")


def test_summary_overlay_formats_synthetic_counts_compactly(compositor):
  graph, affected, propagation = compositor._summary_overlay_lines(_summary())
  assert graph == "GRAPH DELTA added=1 removed=1 changed=0"
  assert affected == "AFFECTED hard=9 soft=0"
  assert propagation.startswith("MAX PROPAGATION 7 HOPS ")
  assert propagation.count(";") == 0
  assert len(propagation) < 110


def test_load_summary_accepts_exact_current_demo_spec(compositor, tmp_path):
  payload = _summary()
  (tmp_path / "summary.json").write_text(json.dumps(payload), "utf-8")
  loaded = compositor._load_summary(tmp_path)
  assert loaded["demo_spec"] == demo_spec.demo_spec_summary(
      demo_spec.FORKED_RACK_SPEC
  )


def test_load_summary_rejects_stale_demo_spec(compositor, tmp_path):
  payload = _summary()
  payload["demo_spec"]["version"] = "forked_rack_v0"
  (tmp_path / "summary.json").write_text(json.dumps(payload), "utf-8")
  with pytest.raises(ValueError, match="demo_spec.version mismatch"):
    compositor._load_summary(tmp_path)


def test_filter_times_both_chain_cues_for_nine_tenths(compositor, tmp_path):
  summary = _summary()
  overlay_dir = tmp_path / "overlays"
  overlay_dir.mkdir()
  files = compositor._write_overlay_textfiles(
      overlay_dir, summary, source_duration=200 / 24, source_fps=24.0
  )
  graph = compositor._build_filter(
      summary, _font(tmp_path), source_duration=200 / 24,
      source_fps=24.0, overlay_files=files,
  )
  normal_time = compositor._event_time(88, 24.0)
  changed_time = compositor._event_time(70, 24.0)
  assert f"between(t,{normal_time:.6f},{normal_time + 0.9:.6f})" in graph
  assert f"between(t,{changed_time:.6f},{changed_time + 0.9:.6f})" in graph
  assert str(files["normal_chain"]) in graph
  assert str(files["changed_chain"]) in graph
```

Retain the punctuation, apostrophe, symlink-ancestor, hardlink, CFR/PTS,
duration, timeout, failed-output-preservation, and real synthetic-media tests.

- [ ] **Step 2: Run the focused compositor tests and capture RED**

```bash
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_compose_intervention_demo.py \
  -k 'small_and_large or synthetic_counts or current_demo_spec or stale_demo_spec or both_chain_cues'
```

Expected: exact-spec acceptance fails on the current top-level schema; cue and
compact-summary tests fail on the old one-cue/full-name overlay contract.

- [ ] **Step 3: Validate the shared spec and select branch-specific events**

Add `demo_spec` to `_TOP_LEVEL_KEYS`, validate exact equality with
`demo_spec_summary(FORKED_RACK_SPEC)` using the same field-specific mismatch
helper as Task 4, and replace `_contact_cue_event()` with:

```python
def _chain_cue_event(
    summary: Mapping[str, Any], branch: str
) -> tuple[str, int]:
  if branch == "normal":
    buckets = ("removed",)
    expected_peer = "side_01"
  elif branch == "trajectory_changed":
    buckets = ("added", "changed")
    expected_peer = "breaker"
  else:
    raise ValueError("chain cue branch must be normal or trajectory_changed")
  target = FORKED_RACK_SPEC.target_id
  branch_pairs = summary["branches"][branch]["contact_pairs"]
  branch_steps = set(summary["branches"][branch]["contact_steps"])
  changed_pairs = summary["branches"]["trajectory_changed"]["contact_pairs"]
  candidates = []
  for bucket in buckets:
    for record in summary["ground_truth"]["graph_delta"][bucket]:
      endpoints = (record["object_a"], record["object_b"])
      pair_key = "|".join(sorted(endpoints))
      if target in endpoints and expected_peer in endpoints:
        normal_only = branch != "normal" or pair_key not in changed_pairs
        if (normal_only and pair_key in branch_pairs and
            record["start_step"] in branch_steps):
          candidates.append((record["start_step"], expected_peer))
  if not candidates:
    raise ValueError(f"summary cannot bind the {branch} chain cue")
  step, peer = min(candidates)
  return peer, step
```

Validate normal has 2-3 unique pairs, changed has 7-9, changed exceeds normal by
at least five, normal contains no main-group endpoint, changed reaches at least
six main-group IDs and contains no side-group endpoint. For removal, reject only
`contact_steps >= intervention_start`; prefix contacts remain legal. Invoke
`_chain_cue_event(summary, "normal")` and
`_chain_cue_event(summary, "trajectory_changed")` from `_load_summary()` so
both cues are bound during preflight.

Implement those checks in one helper called by `_load_summary()`:

```python
def _validate_demo_branch_contract(summary: Mapping[str, Any]) -> None:
  branches = summary["branches"]
  normal_pairs = set(branches["normal"]["contact_pairs"])
  changed_pairs = set(branches["trajectory_changed"]["contact_pairs"])
  if not 2 <= len(normal_pairs) <= 3:
    raise ValueError("normal must contain 2-3 unique dynamic pairs")
  if not 7 <= len(changed_pairs) <= 9:
    raise ValueError("trajectory_changed must contain 7-9 unique dynamic pairs")
  if len(changed_pairs) < len(normal_pairs) + 5:
    raise ValueError("trajectory_changed must exceed normal by at least 5 pairs")
  normal_endpoints = {name for pair in normal_pairs for name in pair.split("|")}
  changed_endpoints = {
      name for pair in changed_pairs for name in pair.split("|")
  }
  if set(FORKED_RACK_SPEC.main_ball_ids) & normal_endpoints:
    raise ValueError("normal contact pairs must not reach the main group")
  if set(FORKED_RACK_SPEC.side_ball_ids) & changed_endpoints:
    raise ValueError("trajectory_changed pairs must not reach the side group")
  if len(set(FORKED_RACK_SPEC.main_ball_ids) & changed_endpoints) < 6:
    raise ValueError("trajectory_changed must reach at least six main balls")
  start = summary["intervention_start"]
  if any(step >= start for step in branches["target_removed"]["contact_steps"]):
    raise ValueError("target_removed has a post-removal dynamic contact")
```

- [ ] **Step 4: Build compact ending text and two timed cues**

Implement:

```python
def _summary_overlay_lines(summary):
  truth = summary["ground_truth"]
  delta = truth["graph_delta"]
  graph = (
      f"GRAPH DELTA added={len(delta['added'])} "
      f"removed={len(delta['removed'])} changed={len(delta['changed'])}"
  )
  affected = (
      f"AFFECTED hard={len(truth['hard_affected'])} "
      f"soft={len(truth['soft_affected'])}"
  )
  ranked = sorted(
      (
          (len(path) - 1, affected_id, path)
          for affected_id, path in truth["propagation_path"].items()
      ),
      key=lambda item: (-item[0], item[1]),
  )
  if not ranked:
    return graph, affected, "MAX PROPAGATION 0 HOPS none"
  hops, _, path = ranked[0]
  return graph, affected, (
      f"MAX PROPAGATION {hops} HOPS {' > '.join(path)}"
  )
```

Replace the one-cue portion of `_overlay_texts()` with:

```python
normal_object, normal_step = _chain_cue_event(summary, "normal")
changed_object, changed_step = _chain_cue_event(
    summary, "trajectory_changed"
)
normal_time = _event_time(normal_step, source_fps)
changed_time = _event_time(changed_step, source_fps)
normal_object = normal_object.replace("_", " ").upper()
changed_object = changed_object.replace("_", " ").upper()
```

Its returned mapping uses these two keys instead of `contact`:

```python
"normal_chain": f"SMALL CHAIN → {normal_object} {normal_time:.3f}s",
"changed_chain": f"LARGE CHAIN → {changed_object} {changed_time:.3f}s",
```

In `_build_filter()`, compute both event times and enables:

```python
_, normal_step = _chain_cue_event(summary, "normal")
_, changed_step = _chain_cue_event(summary, "trajectory_changed")
normal_time = _event_time(normal_step, source_fps)
changed_time = _event_time(changed_step, source_fps)
normal_enable = f"between(t,{normal_time:.6f},{normal_time + 0.9:.6f})"
changed_enable = f"between(t,{changed_time:.6f},{changed_time + 0.9:.6f})"
```

Replace the old changed-panel `contact` drawtext with two drawtexts:

```python
_drawtext(
    font_path, textfile=overlay_files["normal_chain"],
    x="(640-text_w)/2", y="120", size=26,
    color="0xffd166", enable=normal_enable, box=True,
),
_drawtext(
    font_path, textfile=overlay_files["changed_chain"],
    x="640+(640-text_w)/2", y="120", size=26,
    color="0xffd166", enable=changed_enable, box=True,
),
```

Update `_validate_event_steps()` to require the media frame count equals
`summary["demo_spec"]["source_frames"]` and bounds-check both selected steps:

```python
if frame_count != summary["demo_spec"]["source_frames"]:
  raise ValueError("source frame count differs from demo_spec.source_frames")
_, normal_step = _chain_cue_event(summary, "normal")
_, changed_step = _chain_cue_event(summary, "trajectory_changed")
named_steps = {
    "intervention_start": summary["intervention_start"],
    "intervention_end": summary["intervention_end"],
    "normal chain event": normal_step,
    "trajectory_changed chain event": changed_step,
    "removed_step": summary["branches"]["target_removed"]["removed_step"],
}
```

Keep the existing per-name upper-bound loop after this mapping. The overlay-file
writer automatically creates `normal_chain.txt` and `changed_chain.txt` from the
new mapping. Keep removal timing independent. Output validation continues to
derive frames from source `frame_count + 72`; for the real demo this is 272
frames and 11.333333 seconds.

- [ ] **Step 5: Run the full compositor suite and real synthetic integration**

```bash
/home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_compose_intervention_demo.py
```

Expected: all tests pass, including pre-existing source-overwrite and PTS
regressions.

- [ ] **Step 6: Commit compositor changes**

```bash
git add scripts/compose_intervention_demo.py \
  tests/test_compose_intervention_demo.py
git commit -m "feat(demo): annotate forked chain contrast"
```

### Task 6: Document module responsibilities in code

**Files:**
- Create: `tests/test_module_documentation.py`
- Modify: `interventions/__init__.py`
- Modify: `interventions/dataset.py`
- Modify: `interventions/graph_extraction.py`
- Modify: `interventions/kinematic_simulator.py`
- Modify: `interventions/logging.py`
- Modify: `interventions/schema.py`
- Modify: `interventions/tagging.py`
- Modify: `interventions/trajectory.py`
- Modify: `interventions/twin_runner.py`
- Modify: `scripts/__init__.py`
- Modify: `scripts/generate_dataset.py`
- Modify: `scripts/generate_instance.py`
- Modify: `scripts/trajectory_demo_spec.py`
- Modify: `scripts/demo_collision_intervention.py`
- Modify: `scripts/render_demo_branches_blender.py`
- Modify: `scripts/compose_intervention_demo.py`
- Modify: `docs/trajectory_interventions.md`

- [ ] **Step 1: Write an AST-only documentation contract**

Create `tests/test_module_documentation.py`:

```python
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    *sorted((ROOT / "interventions").glob("*.py")),
    ROOT / "scripts" / "__init__.py",
    ROOT / "scripts" / "generate_dataset.py",
    ROOT / "scripts" / "generate_instance.py",
    ROOT / "scripts" / "trajectory_demo_spec.py",
    ROOT / "scripts" / "demo_collision_intervention.py",
    ROOT / "scripts" / "render_demo_branches_blender.py",
    ROOT / "scripts" / "compose_intervention_demo.py",
)
HEADINGS = ("Purpose:", "Public API:", "Dependencies:", "Trust boundary:")


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_module_docstrings_explain_contract(path):
  tree = ast.parse(path.read_text("utf-8"), filename=str(path))
  doc = ast.get_docstring(tree, clean=False)
  assert doc is not None, path
  for heading in HEADINGS:
    assert heading in doc, f"{path}: missing {heading}"


@pytest.mark.parametrize("path", MODULES, ids=lambda path: path.name)
def test_public_top_level_definitions_have_docstrings(path):
  tree = ast.parse(path.read_text("utf-8"), filename=str(path))
  public = tuple(
      node for node in tree.body
      if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
      and not node.name.startswith("_")
  )
  missing = tuple(node.name for node in public if not ast.get_docstring(node))
  assert not missing, f"{path}: public definitions missing docstrings: {missing}"


REQUIRED_ENTRY_POINTS = {
    ROOT / "scripts" / "trajectory_demo_spec.py": {
        "DemoObjectSpec", "DemoSceneSpec", "validate_demo_spec",
        "canonical_spec_payload", "spec_sha256", "demo_spec_summary",
    },
    ROOT / "scripts" / "demo_collision_intervention.py": {
        "build_demo_inputs", "generate_demo", "write_demo_bundle", "main",
    },
    ROOT / "scripts" / "render_demo_branches_blender.py": {"main"},
    ROOT / "scripts" / "compose_intervention_demo.py": {
        "compose_intervention_demo", "main",
    },
}


@pytest.mark.parametrize(
    ("path", "required"), REQUIRED_ENTRY_POINTS.items(),
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_required_script_entries_exist_and_are_documented(path, required):
  tree = ast.parse(path.read_text("utf-8"), filename=str(path))
  definitions = {
      node.name: node for node in tree.body
      if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
  }
  assert required <= set(definitions), f"{path}: missing {required - set(definitions)}"
  missing = {name for name in required if not ast.get_docstring(definitions[name])}
  assert not missing, f"{path}: required entries missing docstrings: {missing}"
```

- [ ] **Step 2: Run the documentation test and capture RED**

```bash
/home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_module_documentation.py
```

Expected: current short module docstrings lack one or more required headings and
some declared public symbols lack docstrings.

- [ ] **Step 3: Write structured module docstrings**

Each module docstring must use the four exact headings and cover this content:

| Module | Purpose / Public API / dependency and trust emphasis |
|---|---|
| `schema.py` | Validated backend-neutral values; JSON conversion; no backend imports; schemas establish structure, not simulation origin. |
| `trajectory.py` | Sample/validate/perturb paths; NumPy/SciPy only; recipe names are heuristic candidates until physics/QC confirms them. |
| `logging.py` | Immutable state/contact generations and readers; no Bullet import; body resolver and caller input trust. |
| `graph_extraction.py` | Aggregate contacts, temporal graphs, deltas, affected sets; consumes canonical logs; graph output is oracle-relative, not causal proof beyond inputs. |
| `tagging.py` | Deterministic tags and propagation descriptors; pure metadata transformation. |
| `kinematic_simulator.py` | Kubric/PyBullet adapter and lifecycle; deferred backend loading; private snapshots are simulator-bound. |
| `twin_runner.py` | Fresh-world factual/counterfactual orchestration and provenance checks; public canonical pair path; caller logs remain unattested. |
| `dataset.py` | Deterministic attempts, QC, journaling, balance, splits, publication/readers; filesystem atomicity and resume trust. |
| `interventions/__init__.py` | Stable exported surface and import behavior; points consumers to canonical entry points. |
| `generate_instance.py` | One inspectable CLI attempt and exit/JSON contract. |
| `generate_dataset.py` | Resumable batch CLI and capacity/error exit contract. |
| `trajectory_demo_spec.py` | Pure forked-rack contract and drift digest, explicitly not an attestation. |
| `demo_collision_intervention.py` | Public pair + demo-only removal + atomic replay bundle. |
| `render_demo_branches_blender.py` | Exact offline-validated replay presentation; lazy backend imports; visuals do not alter physics. |
| `compose_intervention_demo.py` | Strict media preflight and atomic three-panel composition; no Kubric dependency. |
| `scripts/__init__.py` | Marks script helpers importable for tests/CLI modules; no public runtime API promise. |

Use concise behavior-oriented docstrings on every exported symbol reported by
the failing AST test. Do not add docstrings that merely restate a function name;
state its invariant, return contract, or trust implication.

Use these exact missing entry-point docstrings (the other public definitions
already have behavior-oriented docstrings or are created with them above):

```python
# scripts/generate_dataset.py: main
"""Runs resumable batch generation and returns its stable CLI exit status."""

# scripts/generate_instance.py: main
"""Runs one inspectable generation attempt and returns its CLI exit status."""

# scripts/demo_collision_intervention.py: main
"""Generates and atomically publishes the canonical three-branch replay."""

# scripts/render_demo_branches_blender.py: main
"""Preflights requested replays, renders them atomically, and returns zero."""

# scripts/compose_intervention_demo.py: main
"""Validates CLI inputs, composes the comparison atomically, and returns zero."""
```

- [ ] **Step 4: Update the concise external module table and demo section**

In `docs/trajectory_interventions.md`, replace the architecture bullets with a
table containing `Module`, `Responsibility`, `Primary entry points`, and
`Trust/dependency note`. Update the demo section to state:

- eleven objects and exact groups;
- normal 2-3 pairs, changed 7-9 pairs/at least six downstream balls, removed no
  post-removal chain;
- calibrated fixture evidence: normal 2 pairs, changed 9 pairs/all seven main
  balls, graph delta added=15/removed=4/changed=0, hard=9, soft=0;
- 200x11 states, 200x11 presence, `[40,160)`, `forked_rack_v1` digest;
- 272-frame, approximately 11.333333-second final media;
- old four-object bundles fail preflight and must be regenerated;
- removal and Milestone F boundaries are unchanged.

- [ ] **Step 5: Run documentation and offline-import gates**

```bash
/home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_module_documentation.py tests/test_offline_imports.py
git diff --check
```

Expected: all pass; AST test itself imports no intervention or rendering
backend.

- [ ] **Step 6: Commit documentation**

```bash
git add interventions/__init__.py interventions/dataset.py \
  interventions/graph_extraction.py interventions/kinematic_simulator.py \
  interventions/logging.py interventions/schema.py interventions/tagging.py \
  interventions/trajectory.py interventions/twin_runner.py \
  scripts/__init__.py scripts/generate_dataset.py scripts/generate_instance.py \
  scripts/trajectory_demo_spec.py scripts/demo_collision_intervention.py \
  scripts/render_demo_branches_blender.py \
  scripts/compose_intervention_demo.py docs/trajectory_interventions.md \
  tests/test_module_documentation.py
git commit -m "docs(interventions): document module contracts"
```

Before committing, confirm the staged list contains no generated file and no
path under `kubric/`.

### Task 7: Run the complete workflow and record the verified artifact

**Files:**
- Modify: `docs/trajectory_interventions.md`
- Create: `notes/session-logs/2026-08-24-dramatic-forked-rack-demo.md`

- [ ] **Step 1: Run all automated tests on the final code**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests test/test_scene.py test/test_pybullet.py
```

Expected: zero failures and zero warnings. Record the exact count and duration.

- [ ] **Step 2: Run compile, shell, scope, and packaging checks**

```bash
PYTHONPYCACHEPREFIX=/tmp/kubric-forked-rack-pycache \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m compileall \
  -q interventions scripts tests
bash -n run_demo.sh
git diff --check
git diff main...HEAD --name-only
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -c \
  'import interventions; from scripts.trajectory_demo_spec import FORKED_RACK_SPEC; assert len(FORKED_RACK_SPEC.objects) == 11'
```

Expected: commands exit zero; changed-path list has no `kubric/` entry.

- [ ] **Step 3: Generate the final replay bundle through the shell workflow**

```bash
./run_demo.sh intervention-physics-only
```

Validate the summary and arrays with the thesis Python:

```bash
/home/pineapple/miniconda3/envs/thesis/bin/python -c \
  'import json, pathlib, numpy as np; p=pathlib.Path("output/demo_collision_intervention"); s=json.loads((p/"summary.json").read_text()); assert s["demo_spec"]["version"]=="forked_rack_v1"; assert np.load(p/"normal_states.npy", allow_pickle=False).shape==(200,11,13); assert len(s["branches"]["normal"]["contact_pairs"])==2; assert len(s["branches"]["trajectory_changed"]["contact_pairs"])==9; assert all(step<40 for step in s["branches"]["target_removed"]["contact_steps"]); d=s["ground_truth"]["graph_delta"]; assert (len(d["added"]),len(d["removed"]),len(d["changed"]))==(15,4,0); assert len(s["ground_truth"]["hard_affected"])==9 and not s["ground_truth"]["soft_affected"]; print(s["branches"])'
```

Expected: exact calibrated result: normal 2 pairs, changed 9 pairs, no removed
contact at/after step 40, graph delta added=15/removed=4/changed=0, hard=9,
soft=0.

- [ ] **Step 4: Render and compose the complete artifact**

Run the supported full command:

```bash
./run_demo.sh intervention
```

Expected canonical outputs:

```text
output/demo_collision_intervention/normal_blender.mp4
output/demo_collision_intervention/trajectory_changed_blender.mp4
output/demo_collision_intervention/target_removed_blender.mp4
output/demo_collision_intervention/trajectory_intervention_demo.mp4
```

- [ ] **Step 5: Validate media and extract visual checkpoints**

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,duration \
  -show_entries format=duration,size -of json \
  output/demo_collision_intervention/trajectory_intervention_demo.mp4
ffmpeg -v error -i \
  output/demo_collision_intervention/trajectory_intervention_demo.mp4 \
  -f null -
sha256sum output/demo_collision_intervention/*_blender.mp4 \
  output/demo_collision_intervention/trajectory_intervention_demo.mp4
```

Require H.264/yuv420p, 1920x720, CFR 24, 272 decoded frames, approximately
11.333333 seconds, and no decode error.

Extract frames at: pre-intervention, first normal chain contact, first changed
chain contact, after target removal, maximum rack propagation, and ending hold.
Inspect with the image-view tool and require:

- all eleven objects visible and unclipped before intervention;
- exact shared prefix;
- small normal side-chain readable;
- changed breaker/rack propagation visibly reaches at least six balls;
- target absent only in removed after step 40;
- no teleport, decoration detachment, label overlap, or branch desynchronization;
- compact ending metadata fits inside the lower band.

- [ ] **Step 6: Record final evidence and commit the handoff**

Update `docs/trajectory_interventions.md` with exact final media metadata/hash
and visual result. Create the session log with branch/HEAD, architecture,
calibrated constants, test counts, Docker image ID, bundle/media hashes, trust
boundaries, no-`kubric/` confirmation, and remaining Milestone F work.

```bash
git add docs/trajectory_interventions.md \
  notes/session-logs/2026-08-24-dramatic-forked-rack-demo.md
git commit -m "docs(demo): record forked rack handoff"
```

- [ ] **Step 7: Use verification-before-completion and finish the branch**

Run the full test/media/scope gate fresh after the handoff commit. Then use
`finishing-a-development-branch` to offer local merge, push/PR, keep, or discard.
Do not push, merge, delete branches, or stage generated media without the
user's explicit choice.
