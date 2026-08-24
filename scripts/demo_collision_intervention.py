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

import numpy as np

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
from scripts.trajectory_demo_spec import (
    FORKED_RACK_SPEC,
    DemoSceneSpec,
    demo_spec_summary,
    spec_sha256,
    validate_demo_spec,
)


_OUTPUT_DIR = Path("output/demo_collision_intervention")
_DEMO_SPEC = FORKED_RACK_SPEC
_DEMO_SPEC_SHA256 = spec_sha256(_DEMO_SPEC)
_DEMO_SEED = _DEMO_SPEC.seed
_BUNDLE_BRANCHES = ("normal", "trajectory_changed", "target_removed")


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
    return tuple(int(value) for value in self.intervention.time_window)


def build_demo_inputs(
    spec: DemoSceneSpec = _DEMO_SPEC,
) -> tuple[SceneConfig, Intervention, np.ndarray]:
  """Builds the deterministic public-API fixture used by the demo."""
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
  factual_path[:, :3] = np.linspace(
      spec.path_start, spec.path_end, spec.num_steps
  )
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


def _unique_dynamic_pairs(
    records: Sequence[Mapping[str, object] | ContactRecord],
) -> frozenset[tuple[str, str]]:
  return frozenset(
      tuple(sorted((
          str(_record_value(record, "object_a")),
          str(_record_value(record, "object_b")),
      )))
      for record in dynamic_contacts(records)
  )


def _validate_demo_outcomes(
    normal: SimulationLog,
    changed: SimulationLog,
    ground_truth: GroundTruth,
) -> None:
  normal_pairs = _unique_dynamic_pairs(normal.contacts)
  changed_pairs = _unique_dynamic_pairs(changed.contacts)
  normal_endpoints = {endpoint for pair in normal_pairs for endpoint in pair}
  changed_endpoints = {endpoint for pair in changed_pairs for endpoint in pair}
  main_balls = set(_DEMO_SPEC.main_ball_ids)
  side_balls = set(_DEMO_SPEC.side_ball_ids)

  if not 2 <= len(normal_pairs) <= 3:
    raise RuntimeError(
        "normal branch must contain 2..3 dynamic pairs; got "
        f"{sorted(normal_pairs)!r}"
    )
  if ("side_01", "target") not in normal_pairs:
    raise RuntimeError(
        "normal branch lacks side_01|target; got "
        f"{sorted(normal_pairs)!r}"
    )
  if not side_balls.issubset(normal_endpoints):
    raise RuntimeError(
        "normal branch does not reach both side balls; got "
        f"{sorted(normal_pairs)!r}"
    )
  if main_balls & normal_endpoints:
    raise RuntimeError(
        "normal branch reaches main balls; got "
        f"{sorted(normal_pairs)!r}"
    )
  if not 7 <= len(changed_pairs) <= 9:
    raise RuntimeError(
        "changed branch must contain 7..9 dynamic pairs; got "
        f"{sorted(changed_pairs)!r}"
    )
  if ("breaker", "target") not in changed_pairs:
    raise RuntimeError(
        "changed branch lacks breaker|target; got "
        f"{sorted(changed_pairs)!r}"
    )
  if len(main_balls & changed_endpoints) < 6:
    raise RuntimeError(
        "changed branch reaches fewer than six main balls; got "
        f"{sorted(changed_pairs)!r}"
    )
  if side_balls & changed_endpoints:
    raise RuntimeError(
        "changed branch reaches side balls; got "
        f"{sorted(changed_pairs)!r}"
    )
  if len(changed_pairs) < len(normal_pairs) + 5:
    raise RuntimeError(
        "changed branch must add at least five pairs; normal="
        f"{sorted(normal_pairs)!r}, changed={sorted(changed_pairs)!r}"
    )

  hard = set(ground_truth.hard_affected)
  soft = set(ground_truth.soft_affected)
  if hard & soft:
    raise RuntimeError(
        f"hard and soft affected overlap: {sorted(hard & soft)!r}"
    )
  propagation_path = ground_truth.propagation_path
  if set(propagation_path) != hard:
    raise RuntimeError(
        "propagation paths must cover exactly hard affected IDs; hard="
        f"{sorted(hard)!r}, paths={sorted(propagation_path)!r}"
    )
  for affected, path in propagation_path.items():
    if not path or path[0] != _DEMO_SPEC.target_id or path[-1] != affected:
      raise RuntimeError(
          "propagation path must start at target and end at affected ID; "
          f"{affected!r}: {path!r}"
      )


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
  post_removal = dynamic_contacts(tuple(
      record for record in removed.contacts if record.step >= removed_step
  ))
  if post_removal:
    raise RuntimeError(
        f"removed branch contains a post-removal dynamic contact: {post_removal!r}"
    )
  post_removal_target_contacts = tuple(
      record
      for record in removed.contacts
      if record.step >= removed_step
      and target_id in (record.object_a, record.object_b)
  )
  if post_removal_target_contacts:
    raise RuntimeError(
        "removed branch contains a post-removal target contact: "
        f"{post_removal_target_contacts!r}"
    )
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
  try:
    validate_demo_spec(result.demo_spec)
    result_spec_sha256 = spec_sha256(result.demo_spec)
  except (TypeError, ValueError) as error:
    raise ValueError("demo result spec identity differs from canonical") from error
  if result_spec_sha256 != _DEMO_SPEC_SHA256:
    raise ValueError("demo result spec identity differs from canonical")
  expected_scene, expected_intervention, expected_path = build_demo_inputs(
      result.demo_spec
  )
  if result.scene_config != expected_scene:
    raise ValueError("demo scene differs from the stored demo spec")
  if result.intervention != expected_intervention:
    raise ValueError("demo intervention differs from the stored demo spec")
  if not np.array_equal(result.normal.commanded_path, expected_path):
    raise ValueError("factual commanded path differs from the stored demo spec")
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

  canonical_object_ids = result.demo_spec.object_ids
  object_ids = result.normal.object_ids
  expected_shape = (
      result.demo_spec.num_steps, len(result.demo_spec.object_ids), 13
  )
  expected_presence_shape = expected_shape[:2]
  if object_ids != canonical_object_ids:
    raise ValueError(
        "normal object order must match the canonical demo scene object IDs"
    )
  if result.scene_config.seed != _DEMO_SEED:
    raise ValueError("demo scene seed must be 0")
  if result.intervention_window != _DEMO_SPEC.intervention_window:
    raise ValueError(
        "demo intervention window must be "
        f"{_DEMO_SPEC.intervention_window!r}"
    )

  for role, branch in source_branches.items():
    if branch.object_ids != object_ids:
      raise ValueError(f"{role} object order differs from normal")
    if branch.steps != tuple(range(result.demo_spec.num_steps)):
      raise ValueError(
          f"{role} steps must be exactly range({result.demo_spec.num_steps})"
      )
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

  recomputed_ground_truth = extract_pair_ground_truth(
      result.scene_config,
      result.intervention,
      result.normal,
      result.changed,
  )
  if recomputed_ground_truth != result.ground_truth:
    raise ValueError("ground_truth does not match the validated public pair")
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
    output_dir: str | Path,
    result: DemoResult,
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
      "demo_spec": demo_spec_summary(result.demo_spec),
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
