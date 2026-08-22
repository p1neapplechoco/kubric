#!/usr/bin/env python3
"""Demo a velocity-based trajectory intervention that changes collision outcome."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

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


def _contact_steps(
    contact_records: Sequence[Mapping[str, object] | ContactRecord],
    object_id: str,
) -> tuple[int, ...]:
  steps = sorted(
      {
          int(_record_value(record, "step"))
          for record in contact_records
          if object_id in (
              _record_value(record, "object_a"),
              _record_value(record, "object_b"),
          )
      }
  )
  return tuple(steps)


def _final_position(states: np.ndarray, object_index: int) -> tuple[float, float, float]:
  return tuple(float(value) for value in states[-1, object_index, 0:3])


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
  output.mkdir(parents=True, exist_ok=True)

  result = generate_demo()
  factual = result.normal
  counterfactual = result.changed
  factual_contacts = dynamic_contacts(factual.contacts)
  counterfactual_contacts = dynamic_contacts(counterfactual.contacts)
  upper_ball = factual.object_ids.index("upper_ball")
  lower_ball = factual.object_ids.index("lower_ball")

  summary = {
      "output": str(output),
      "seed": _DEMO_SEED,
      "branches": {
          "factual": {
              "contact_pairs": _contact_pairs(factual_contacts),
              "contact_steps_upper_ball": _contact_steps(factual_contacts, "upper_ball"),
              "contact_steps_lower_ball": _contact_steps(factual_contacts, "lower_ball"),
              "final_position_upper_ball": _final_position(factual.states, upper_ball),
              "final_position_lower_ball": _final_position(factual.states, lower_ball),
          },
          "counterfactual": {
              "contact_pairs": _contact_pairs(counterfactual_contacts),
              "contact_steps_upper_ball": _contact_steps(counterfactual_contacts, "upper_ball"),
              "contact_steps_lower_ball": _contact_steps(counterfactual_contacts, "lower_ball"),
              "final_position_upper_ball": _final_position(counterfactual.states, upper_ball),
              "final_position_lower_ball": _final_position(counterfactual.states, lower_ball),
          },
      },
      "comparison": {
          "upper_ball_final_displacement": float(np.linalg.norm(
              np.asarray(_final_position(counterfactual.states, upper_ball))
              - np.asarray(_final_position(factual.states, upper_ball))
          )),
          "lower_ball_final_displacement": float(np.linalg.norm(
              np.asarray(_final_position(counterfactual.states, lower_ball))
              - np.asarray(_final_position(factual.states, lower_ball))
          )),
      },
      "object_ids": list(factual.object_ids),
  }

  np.save(output / "factual_states.npy", factual.states)
  np.save(output / "counterfactual_states.npy", counterfactual.states)
  (output / "factual_contacts.jsonl").write_text(
      "\n".join(
          json.dumps(record.to_dict(), sort_keys=True) for record in factual.contacts
      )
      + "\n",
      encoding="utf-8",
  )
  (output / "counterfactual_contacts.jsonl").write_text(
      "\n".join(
          json.dumps(record.to_dict(), sort_keys=True)
          for record in counterfactual.contacts
      )
      + "\n",
      encoding="utf-8",
  )
  _render_branch_video(output / "factual.mp4", factual)
  _render_branch_video(output / "counterfactual.mp4", counterfactual)
  (output / "summary.json").write_text(
      json.dumps(summary, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
  )

  print(json.dumps(summary, sort_keys=True))
  return 0


if __name__ == "__main__":  # pragma: no cover - exercised manually.
  raise SystemExit(main())
