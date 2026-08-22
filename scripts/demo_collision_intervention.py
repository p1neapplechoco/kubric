#!/usr/bin/env python3
"""Generate deterministic replay data for three collision-demo branches."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

import imageio
import numpy as np
from PIL import Image, ImageDraw

from interventions import (
    ContactLogger,
    ContactRecord,
    GroundTruth,
    Intervention,
    KinematicSimulator,
    ObjectConfig,
    SceneConfig,
    SimulationLog,
    extract_pair_ground_truth,
    generate_paired_instance,
)
from interventions.twin_runner import _build_scene, _configure_physics


_OUTPUT_DIR = Path("output/demo_collision_intervention")
_DEMO_SEED = 0
_NUM_STEPS = 120
_BUNDLE_BRANCHES = ("normal", "trajectory_changed", "target_removed")
_CANVAS_SIZE = (960, 544)
_WORLD_BOUNDS = (-4.5, 4.5, -4.5, 4.5)


@dataclass(frozen=True)
class BranchResult:
  branch: str
  states: np.ndarray
  contact_records: Sequence[Mapping[str, object] | ContactRecord]


def _freeze_demo_metadata(value: Mapping[str, object]) -> Mapping[str, object]:
  """Copies the small demo metadata tree into immutable containers."""
  if not isinstance(value, Mapping):
    raise TypeError("metadata must be a mapping")

  def freeze(item):
    if isinstance(item, Mapping):
      if not all(isinstance(key, str) for key in item):
        raise TypeError("metadata keys must be strings")
      return MappingProxyType({key: freeze(item[key]) for key in sorted(item)})
    if isinstance(item, (tuple, list)):
      return tuple(freeze(child) for child in item)
    return item

  return freeze(value)


@dataclass(frozen=True, eq=False)
class RemovedBranch:
  """Fixed-shape demo replay for a physically removed target.

  ``states`` stores XYZ + WXYZ quaternion + linear/angular velocity rows. A
  false ``presence`` entry means the corresponding finite row is only a replay
  placeholder and no longer represents a body in the physics world.
  """

  branch: str
  object_ids: tuple[str, ...]
  steps: tuple[int, ...]
  states: np.ndarray
  presence: np.ndarray
  contacts: tuple[ContactRecord, ...]
  metadata: Mapping[str, object] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not isinstance(self.branch, str) or not self.branch.strip():
      raise ValueError("branch must be a non-empty string")
    if isinstance(self.object_ids, (str, bytes, set, frozenset)):
      raise TypeError("object_ids must be an ordered iterable")
    object_ids = tuple(self.object_ids)
    if not all(isinstance(item, str) and item.strip() for item in object_ids):
      raise ValueError("object_ids must contain non-empty strings")
    if len(set(object_ids)) != len(object_ids):
      raise ValueError("object_ids must be unique")

    if isinstance(self.steps, (str, bytes, set, frozenset)):
      raise TypeError("steps must be an ordered iterable")
    steps = tuple(self.steps)
    if not all(isinstance(step, int) and not isinstance(step, bool) for step in steps):
      raise TypeError("steps must contain integers")
    if any(step < 0 for step in steps) or any(
        right <= left for left, right in zip(steps, steps[1:])
    ):
      raise ValueError("steps must be nonnegative and strictly increasing")

    try:
      states = np.array(self.states, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
      raise ValueError("states must be a numeric array") from error
    expected_states = (len(steps), len(object_ids), 13)
    if states.shape != expected_states:
      raise ValueError("states must have shape {!r}".format(expected_states))
    if not np.isfinite(states).all():
      raise ValueError("states must contain only finite values")

    untyped_presence = np.asarray(self.presence)
    if untyped_presence.dtype.kind != "b":
      raise TypeError("presence must contain Boolean values")
    presence = np.array(
        untyped_presence, dtype=np.bool_, order="C", copy=True
    )
    expected_presence = (len(steps), len(object_ids))
    if presence.shape != expected_presence:
      raise ValueError(
          "presence must have shape {!r}".format(expected_presence)
      )

    if isinstance(self.contacts, (str, bytes)):
      raise TypeError("contacts must be an iterable of ContactRecord values")
    contacts = tuple(self.contacts)
    if not all(isinstance(record, ContactRecord) for record in contacts):
      raise TypeError("contacts must contain only ContactRecord values")
    known_ids = set(object_ids)
    if any(
        record.object_a not in known_ids or record.object_b not in known_ids
        for record in contacts
    ):
      raise ValueError("contact endpoints must use known object_ids")
    if contacts and (
        not steps
        or any(record.step < steps[0] or record.step > steps[-1]
               for record in contacts)
    ):
      raise ValueError("contact steps must lie within the logged step range")

    states = np.frombuffer(
        states.tobytes(order="C"), dtype=states.dtype
    ).reshape(states.shape)
    presence = np.frombuffer(
        presence.tobytes(order="C"), dtype=presence.dtype
    ).reshape(presence.shape)
    object.__setattr__(self, "object_ids", object_ids)
    object.__setattr__(self, "steps", steps)
    object.__setattr__(self, "states", states)
    object.__setattr__(self, "presence", presence)
    object.__setattr__(self, "contacts", contacts)
    object.__setattr__(self, "metadata", _freeze_demo_metadata(self.metadata))


@dataclass(frozen=True)
class DemoResult:
  scene_config: SceneConfig
  intervention: Intervention
  normal: SimulationLog
  changed: SimulationLog
  removed: RemovedBranch
  ground_truth: GroundTruth

  @property
  def intervention_window(self) -> tuple[int, int]:
    return tuple(int(value) for value in self.intervention.time_window)


def build_demo_inputs() -> tuple[SceneConfig, Intervention, np.ndarray]:
  """Builds the deterministic public-API fixture used by the demo."""
  floor = ObjectConfig(
      "floor",
      "cube",
      size=(4.0, 4.0, 0.25),
      mass=1.0,
      position=(0.0, 0.0, -0.25),
      static=True,
      friction=0.0,
      restitution=0.0,
  )
  target = ObjectConfig(
      "target",
      "cube",
      size=0.18,
      mass=2.0,
      position=(-1.0, 0.0, 0.18),
      static=True,
      friction=0.0,
      restitution=0.0,
  )
  upper_ball = ObjectConfig(
      "upper_ball",
      "sphere",
      size=0.26,
      mass=1.0,
      position=(0.0, 0.45, 0.26),
      friction=0.0,
      restitution=0.0,
  )
  lower_ball = ObjectConfig(
      "lower_ball",
      "sphere",
      size=0.26,
      mass=1.0,
      position=(0.0, -0.45, 0.26),
      friction=0.0,
      restitution=0.0,
  )
  scene = SceneConfig(
      objects=(floor, target, upper_ball, lower_ball),
      seed=0,
      scene_bounds=((-4.5, -4.5, -1.0), (4.5, 4.5, 2.0)),
      gravity=(0.0, 0.0, 0.0),
      frame_range=(0, 12),
      frame_rate=24,
      step_rate=240,
  )
  intervention = Intervention(
      target_id="target",
      recipe="create_collision",
      magnitude=0.35,
      time_window=(24, 96),
      push_mass=2.0,
  )
  factual_path = np.zeros((_NUM_STEPS, 7), dtype=np.float64)
  factual_path[:, 0] = np.linspace(-1.0, 1.0, _NUM_STEPS)
  factual_path[:, 2] = 0.18
  factual_path[:, 3] = 1.0
  return scene, intervention, factual_path


def _record_value(record: Mapping[str, object] | ContactRecord, field: str) -> Any:
  if isinstance(record, Mapping):
    return record[field]
  return getattr(record, field)


def dynamic_contacts(
    contact_records: Sequence[Mapping[str, object] | ContactRecord],
    object_id: str | None = None,
) -> tuple[Mapping[str, object] | ContactRecord, ...]:
  """Returns contacts that do not involve the static floor."""
  return tuple(
      record
      for record in contact_records
      if "floor" not in (
          _record_value(record, "object_a"),
          _record_value(record, "object_b"),
      )
      and (
          object_id is None
          or object_id in (
              _record_value(record, "object_a"),
              _record_value(record, "object_b"),
          )
      )
  )


def _validate_demo_outcomes(
    normal: SimulationLog,
    changed: SimulationLog,
    ground_truth: GroundTruth,
) -> None:
  if dynamic_contacts(normal.contacts):
    raise RuntimeError("normal branch unexpectedly contains a dynamic contact")
  if not any(
      {_record_value(record, "object_a"), _record_value(record, "object_b")}
      == {"target", "upper_ball"}
      for record in changed.contacts
  ):
    raise RuntimeError("changed branch did not create the target|upper_ball contact")
  if ground_truth.hard_affected != ("upper_ball",):
    raise RuntimeError("changed branch did not hard-affect only upper_ball")


def _run_removed_branch(
    scene_config: SceneConfig,
    intervention: Intervention,
    factual_path: np.ndarray,
    provenance: Mapping[str, object],
) -> RemovedBranch:
  """Runs the demo-only physical deletion branch in a fresh Bullet world.

  This deliberately reuses ``twin_runner``'s private scene/config mapping and
  the simulator's private snapshot helper so the presentation branch matches
  the public pair without adding deletion semantics to the dataset schema.
  """
  removed_step = int(intervention.time_window[0])
  target_id = intervention.target_id
  scene, assets = _build_scene(scene_config)
  with tempfile.TemporaryDirectory(prefix="kubric-demo-removal-") as scratch:
    with KinematicSimulator(scene, scratch_dir=Path(scratch)) as simulator:
      _configure_physics(simulator, scene_config, assets)
      prefix = simulator.run_with_intervention(
          assets[target_id],
          factual_path[:removed_step],
          push_mass=intervention.push_mass,
          branch="target_removed",
          start_step=0,
          write_keyframes=False,
      )
      object_ids = prefix.object_ids
      target_index = object_ids.index(target_id)
      states = np.empty(
          (len(factual_path), len(object_ids), 13), dtype=np.float64
      )
      states[:removed_step] = prefix.states
      states[removed_step:, target_index] = prefix.states[-1, target_index]
      presence = np.ones(
          (len(factual_path), len(object_ids)), dtype=np.bool_
      )
      presence[removed_step:, target_index] = False

      scene.remove(assets[target_id])
      live_ids = tuple(
          object_id for object_id in object_ids if object_id != target_id
      )
      live_assets = tuple(assets[object_id] for object_id in live_ids)
      live_indices = tuple(object_ids.index(object_id) for object_id in live_ids)
      body_to_object_id = {
          int(asset.linked_objects[simulator]): object_id
          for object_id, asset in zip(live_ids, live_assets)
      }
      contact_logger = ContactLogger(
          body_to_object_id, scene_config.step_rate
      )

      for step in range(removed_step, len(factual_path)):
        simulator.step_passive()
        contact_logger.log(step, simulator.bullet_client.getContactPoints())
        states[step, live_indices] = simulator._snapshot(live_assets)

  metadata = {
      "trust_model": "demo_only_removal_v1",
      "target_id": target_id,
      "removed_step": removed_step,
      "scene_seed": scene_config.seed,
      "intervention_recipe": intervention.recipe,
      "push_mass": intervention.push_mass,
  }
  metadata.update(provenance)
  return RemovedBranch(
      branch="target_removed",
      object_ids=object_ids,
      steps=tuple(range(len(factual_path))),
      states=states,
      presence=presence,
      contacts=tuple(prefix.contacts) + tuple(contact_logger.records),
      metadata=metadata,
  )


def _validate_removed_branch(
    removed: RemovedBranch,
    normal: SimulationLog,
    intervention: Intervention,
) -> None:
  """Rejects replay corruption before exposing the demo result."""
  if not isinstance(removed, RemovedBranch):
    raise RuntimeError("removed branch has the wrong container type")
  removed_step = int(intervention.time_window[0])
  target_id = intervention.target_id
  if removed.object_ids != normal.object_ids:
    raise RuntimeError("removed branch object IDs differ from normal")
  if removed.steps != normal.steps:
    raise RuntimeError("removed branch steps differ from normal")
  if removed.states.shape != normal.states.shape:
    raise RuntimeError("removed branch state shape differs from normal")
  if removed.presence.shape != removed.states.shape[:2]:
    raise RuntimeError("removed branch presence shape differs from states")
  if not np.isfinite(removed.states).all():
    raise RuntimeError("removed branch states must be finite")

  target_index = normal.object_ids.index(target_id)
  non_target_indices = tuple(
      index
      for index, object_id in enumerate(normal.object_ids)
      if object_id != target_id
  )
  if not np.array_equal(
      removed.states[:removed_step, non_target_indices],
      normal.states[:removed_step, non_target_indices],
  ):
    raise RuntimeError("removed branch non-target prefix differs from normal")
  if not np.array_equal(
      removed.states[:removed_step, target_index],
      normal.states[:removed_step, target_index],
  ):
    raise RuntimeError("removed branch target prefix differs from normal")

  expected_presence = np.ones_like(removed.presence, dtype=np.bool_)
  expected_presence[removed_step:, target_index] = False
  if not np.array_equal(removed.presence, expected_presence):
    raise RuntimeError("removed branch presence does not match removal timing")
  expected_target = np.broadcast_to(
      removed.states[removed_step - 1, target_index],
      removed.states[removed_step:, target_index].shape,
  )
  if not np.array_equal(removed.states[removed_step:, target_index], expected_target):
    raise RuntimeError("removed branch did not retain the target's last state")

  normal_prefix_contacts = tuple(
      record for record in normal.contacts if record.step < removed_step
  )
  removed_prefix_contacts = tuple(
      record for record in removed.contacts if record.step < removed_step
  )
  if removed_prefix_contacts != normal_prefix_contacts:
    raise RuntimeError("removed branch contact prefix differs from normal")
  if any(
      record.step >= removed_step
      and target_id in (record.object_a, record.object_b)
      for record in removed.contacts
  ):
    raise RuntimeError("removed target appears in a post-removal contact")
  if (
      removed.metadata.get("trust_model") != "demo_only_removal_v1"
      or removed.metadata.get("target_id") != target_id
      or removed.metadata.get("removed_step") != removed_step
  ):
    raise RuntimeError("removed branch metadata is inconsistent")


def generate_demo(seed: int = _DEMO_SEED) -> DemoResult:
  """Generates factual and changed branches through the public pair runner."""
  if seed != _DEMO_SEED:
    raise ValueError("fixed deterministic demo seed is 0")
  scene, intervention, factual_path = build_demo_inputs()
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
      scene_config=scene,
      intervention=intervention,
      normal=normal,
      changed=changed,
      removed=removed,
      ground_truth=ground_truth,
  )


def _contact_pairs(
    contact_records: Sequence[Mapping[str, object] | ContactRecord],
) -> dict[str, int]:
  counts: dict[str, int] = {}
  for record in contact_records:
    key = "|".join(sorted((
        str(_record_value(record, "object_a")),
        str(_record_value(record, "object_b")),
    )))
    counts[key] = counts.get(key, 0) + 1
  return counts


def _validate_demo_bundle(result: DemoResult) -> None:
  """Validates the fixed three-branch replay contract before any writes."""
  if not isinstance(result, DemoResult):
    raise TypeError("result must be a DemoResult")
  expected_source_branches = {
      "normal": "factual",
      "trajectory_changed": "counterfactual",
      "target_removed": "target_removed",
  }
  source_branches = {
      "normal": result.normal,
      "trajectory_changed": result.changed,
      "target_removed": result.removed,
  }
  for role, expected_branch in expected_source_branches.items():
    if source_branches[role].branch != expected_branch:
      raise ValueError(
          f"{role} branch must be {expected_branch!r}, got "
          f"{source_branches[role].branch!r}"
      )

  scene_object_ids = tuple(
      item.object_id for item in result.scene_config.objects
  )
  object_ids = result.normal.object_ids
  expected_shape = (_NUM_STEPS, len(object_ids), 13)
  expected_presence_shape = expected_shape[:2]
  if (
      expected_shape != (_NUM_STEPS, 4, 13)
      or set(object_ids) != set(scene_object_ids)
  ):
    raise ValueError("normal object_ids must match the four demo scene objects")
  if result.scene_config.seed != _DEMO_SEED:
    raise ValueError("demo scene seed must be 0")
  if result.intervention_window != (24, 96):
    raise ValueError("demo intervention window must be (24, 96)")

  for role, branch in source_branches.items():
    if branch.object_ids != object_ids:
      raise ValueError(f"{role} object_ids differ from scene object order")
    if branch.steps != tuple(range(_NUM_STEPS)):
      raise ValueError(f"{role} steps must be exactly range(120)")
    if branch.states.shape != expected_shape:
      raise ValueError(
          f"{role} states must have shape {expected_shape!r}"
      )
  if result.removed.presence.shape != expected_presence_shape:
    raise ValueError(
        "target_removed presence must have shape "
        f"{expected_presence_shape!r}"
    )
  if (
      result.normal.step_rate != result.scene_config.step_rate
      or result.changed.step_rate != result.scene_config.step_rate
  ):
    raise ValueError("branch step_rate differs from scene step_rate")

  _validate_demo_outcomes(result.normal, result.changed, result.ground_truth)
  _validate_removed_branch(result.removed, result.normal, result.intervention)


def _canonical_json_bytes(value: object) -> bytes:
  return (
      json.dumps(
          value,
          ensure_ascii=False,
          allow_nan=False,
          sort_keys=True,
          separators=(",", ":"),
      )
      + "\n"
  ).encode("utf-8")


def _atomic_write(path: Path, write_payload: Callable[[BinaryIO], None]) -> None:
  descriptor, temporary_name = tempfile.mkstemp(
      dir=path.parent,
      prefix=f".{path.name}.",
      suffix=".tmp",
  )
  try:
    with os.fdopen(descriptor, "wb") as handle:
      write_payload(handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary_name, path)
  except BaseException:
    try:
      os.unlink(temporary_name)
    except FileNotFoundError:
      pass
    raise


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
  def save(handle: BinaryIO) -> None:
    np.save(handle, value, allow_pickle=False)

  _atomic_write(path, save)


def _atomic_save_json(path: Path, payload: bytes) -> None:
  def save(handle: BinaryIO) -> None:
    handle.write(payload)

  _atomic_write(path, save)


def _branch_contact_summary(
    records: Sequence[Mapping[str, object] | ContactRecord],
) -> dict[str, object]:
  contacts = dynamic_contacts(records)
  return {
      "contact_pairs": _contact_pairs(contacts),
      "contact_steps": sorted({
          int(_record_value(record, "step")) for record in contacts
      }),
  }


def write_demo_bundle(
    output_dir: str | Path, result: DemoResult
) -> dict[str, object]:
  """Atomically writes the canonical three-branch replay bundle."""
  _validate_demo_bundle(result)
  output = Path(output_dir)
  branches = {
      "normal": (
          result.normal.states,
          np.ones(result.normal.states.shape[:2], dtype=np.bool_),
          result.normal.contacts,
      ),
      "trajectory_changed": (
          result.changed.states,
          np.ones(result.changed.states.shape[:2], dtype=np.bool_),
          result.changed.contacts,
      ),
      "target_removed": (
          result.removed.states,
          result.removed.presence,
          result.removed.contacts,
      ),
  }
  if tuple(branches) != _BUNDLE_BRANCHES:  # Defensive against accidental drift.
    raise RuntimeError("canonical demo branch order changed")

  contact_payload = {
      branch_name: [record.to_dict() for record in records]
      for branch_name, (_, _, records) in branches.items()
  }
  start, end = result.intervention_window
  branch_summaries = {
      branch_name: _branch_contact_summary(records)
      for branch_name, (_, _, records) in branches.items()
  }
  branch_summaries["target_removed"].update({
      "removed_step": int(result.removed.metadata["removed_step"]),
      "target_id": str(result.removed.metadata["target_id"]),
      "trust_model": str(result.removed.metadata["trust_model"]),
  })
  summary = {
      "branches": branch_summaries,
      "ground_truth": result.ground_truth.to_dict(),
      "intervention_end": end,
      "intervention_start": start,
      "intervention_window": [start, end],
      "object_ids": list(result.normal.object_ids),
      "seed": int(result.scene_config.seed),
      "step_rate": float(result.scene_config.step_rate),
  }
  contacts_bytes = _canonical_json_bytes(contact_payload)
  summary_bytes = _canonical_json_bytes(summary)

  output.mkdir(parents=True, exist_ok=True)
  for branch_name, (states, presence, _) in branches.items():
    _atomic_save_array(output / f"{branch_name}_states.npy", states)
    _atomic_save_array(output / f"{branch_name}_presence.npy", presence)
  _atomic_save_json(output / "contacts.json", contacts_bytes)
  _atomic_save_json(output / "summary.json", summary_bytes)
  return summary


def _world_to_canvas(x: float, y: float) -> tuple[float, float]:
  x0, x1, y0, y1 = _WORLD_BOUNDS
  width, height = _CANVAS_SIZE
  px = (x - x0) / (x1 - x0) * (width - 1)
  py = (1.0 - (y - y0) / (y1 - y0)) * (height - 1)
  return px, py


def _draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, fill, outline):
  draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline)


def _dynamic_contact_records(
    contact_records: Sequence[Mapping[str, object] | ContactRecord],
    object_id: str | None = None,
) -> list[Mapping[str, object] | ContactRecord]:
  return list(dynamic_contacts(contact_records, object_id))


def _render_branch_video(
    output: Path, branch: BranchResult | SimulationLog
) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  if isinstance(branch, SimulationLog):
    object_ids = branch.object_ids
    contact_records = branch.contacts
  else:
    object_ids = ("floor", "pusher", "upper_ball", "lower_ball")
    contact_records = branch.contact_records
  target_id = "target" if "target" in object_ids else "pusher"
  contact_steps = {
      int(_record_value(record, "step"))
      for record in dynamic_contacts(contact_records, target_id)
  }
  colors = {
      "floor": (70, 70, 70),
      target_id: (50, 120, 255),
      "upper_ball": (255, 140, 70),
      "lower_ball": (90, 200, 120),
  }
  radii = {
      "floor": 18,
      target_id: 14,
      "upper_ball": 16,
      "lower_ball": 16,
  }
  frame_map = {
      name: {
          "index": object_ids.index(name),
          "color": colors[name],
          "radius": radii[name],
      }
      for name in ("floor", target_id, "upper_ball", "lower_ball")
  }
  states = branch.states
  trails = {name: [] for name in frame_map if name != "floor"}
  writer = imageio.get_writer(output, fps=24, codec="libx264", quality=8)
  try:
    for step in range(len(states)):
      image = Image.new("RGB", _CANVAS_SIZE, (245, 245, 240))
      draw = ImageDraw.Draw(image)

      draw.rectangle((0, 0, _CANVAS_SIZE[0] - 1, _CANVAS_SIZE[1] - 1), outline=(210, 210, 205), width=2)
      x0, x1, y0, y1 = _WORLD_BOUNDS
      left_top = _world_to_canvas(x0, y1)
      right_bottom = _world_to_canvas(x1, y0)
      draw.rectangle((left_top[0], left_top[1], right_bottom[0], right_bottom[1]), outline=(180, 180, 175), width=3)
      draw.text((20, 18), f"{branch.branch} | frame {step + 1:03d}/{len(states):03d}", fill=(30, 30, 30))

      for name, spec in frame_map.items():
        index = spec["index"]
        position = states[step, index]
        x, y = _world_to_canvas(float(position[0]), float(position[1]))
        if name != "floor":
          trails[name].append((x, y))
          if len(trails[name]) > 1:
            draw.line(trails[name][-40:], fill=spec["color"], width=3)
        radius = spec["radius"]
        fill = spec["color"]
        outline = (25, 25, 25)
        if name == target_id and step in contact_steps:
          outline = (220, 30, 30)
          draw.ellipse(
              (x - radius - 7, y - radius - 7, x + radius + 7, y + radius + 7),
              outline=(220, 30, 30),
              width=4,
          )
        _draw_circle(draw, x, y, radius, fill, outline)
        if name != "floor":
          label = name.replace("_", " ")
          draw.text((x + radius + 4, y - radius - 2), label, fill=(25, 25, 25))

      writer.append_data(np.asarray(image))
  finally:
    writer.close()


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--output", default=str(_OUTPUT_DIR))
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  output = Path(args.output)
  result = generate_demo()
  summary = write_demo_bundle(output, result)
  print(_canonical_json_bytes(summary).decode("utf-8"), end="")
  return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually.
  raise SystemExit(main())
