"""Tests for the collision demo's physics branches and video rendering."""

import dataclasses
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import pytest

pytest.importorskip("pybullet")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

import imageio.v2 as imageio  # noqa: E402  (guarded above)
import numpy as np  # noqa: E402

_SCRIPT_NAME = "demo_collision_intervention"
_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / f"{_SCRIPT_NAME}.py"
)
_SPEC = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SCRIPT_NAME] = demo
_SPEC.loader.exec_module(demo)

_NUM_STEPS = 6


@pytest.fixture(scope="module")
def generated_demo():
  return demo.generate_demo(seed=0)


def test_build_demo_inputs_matches_verified_public_pair_fixture():
  scene, intervention, factual_path = demo.build_demo_inputs()
  objects = {item.object_id: item for item in scene.objects}

  assert tuple(objects) == ("floor", "target", "upper_ball", "lower_ball")
  assert objects["floor"].shape == "cube"
  assert objects["floor"].size == (4.0, 4.0, 0.25)
  assert objects["floor"].position == (0.0, 0.0, -0.25)
  assert objects["floor"].static
  assert objects["target"].shape == "cube"
  assert objects["target"].size == (0.18, 0.18, 0.18)
  assert objects["target"].position == (-1.0, 0.0, 0.18)
  assert objects["target"].mass == 2.0
  assert objects["target"].static
  assert objects["upper_ball"].shape == "sphere"
  assert objects["upper_ball"].size == (0.26, 0.26, 0.26)
  assert objects["upper_ball"].position == (0.0, 0.45, 0.26)
  assert not objects["upper_ball"].static
  assert objects["lower_ball"].shape == "sphere"
  assert objects["lower_ball"].size == (0.26, 0.26, 0.26)
  assert objects["lower_ball"].position == (0.0, -0.45, 0.26)
  assert not objects["lower_ball"].static
  assert all(item.friction == 0.0 for item in objects.values())
  assert all(item.restitution == 0.0 for item in objects.values())

  assert scene.seed == 0
  assert scene.gravity == (0.0, 0.0, 0.0)
  assert scene.scene_bounds == ((-4.5, -4.5, -1.0), (4.5, 4.5, 2.0))
  assert scene.frame_range == (0, 12)
  assert scene.frame_rate == 24
  assert scene.step_rate == 240
  assert intervention.target_id == "target"
  assert intervention.recipe == "create_collision"
  assert intervention.magnitude == 0.35
  assert intervention.time_window == (24.0, 96.0)
  assert intervention.push_mass == 2.0

  assert factual_path.shape == (120, 7)
  np.testing.assert_array_equal(factual_path[:, 0], np.linspace(-1.0, 1.0, 120))
  np.testing.assert_array_equal(factual_path[:, 1], np.zeros(120))
  np.testing.assert_array_equal(factual_path[:, 2], np.full(120, 0.18))
  np.testing.assert_array_equal(
      factual_path[:, 3:], np.tile((1.0, 0.0, 0.0, 0.0), (120, 1))
  )


def test_generate_demo_uses_public_pair_and_expected_ground_truth(generated_demo):
  result = generated_demo

  assert result.normal.branch == "factual"
  assert result.changed.branch == "counterfactual"
  assert isinstance(result.removed, demo.RemovedBranch)
  assert result.removed.branch == "target_removed"
  assert result.ground_truth.hard_affected == ("upper_ball",)
  assert result.ground_truth.soft_affected == ()
  assert not demo.dynamic_contacts(result.normal.contacts)
  assert any(
      {record.object_a, record.object_b} == {"target", "upper_ball"}
      for record in result.changed.contacts
  )
  with pytest.raises(FrozenInstanceError):
    result.changed = result.normal


def test_changed_branch_diverges_only_inside_intervention_window(generated_demo):
  result = generated_demo
  start, end = result.intervention_window

  assert (start, end) == (24, 96)
  np.testing.assert_array_equal(
      result.normal.commanded_path[:start], result.changed.commanded_path[:start]
  )
  np.testing.assert_array_equal(
      result.normal.commanded_path[end:], result.changed.commanded_path[end:]
  )
  assert np.any(
      result.normal.commanded_path[start:end]
      != result.changed.commanded_path[start:end]
  )
  np.testing.assert_array_equal(
      result.normal.states[:start], result.changed.states[:start]
  )
  assert tuple(record for record in result.normal.contacts if record.step < start) == tuple(
      record for record in result.changed.contacts if record.step < start
  )
  target_index = result.normal.object_ids.index("target")
  np.testing.assert_array_equal(
      result.normal.states[0, target_index, 3:7], (1.0, 0.0, 0.0, 0.0)
  )


@pytest.mark.parametrize("seed", (6, 7))
def test_generate_demo_rejects_nonzero_seed(seed):
  with pytest.raises(ValueError, match="fixed deterministic demo seed is 0"):
    demo.generate_demo(seed=seed)


@pytest.mark.parametrize("seed", (6, 7))
def test_cli_rejects_nonzero_seed(seed):
  with pytest.raises(SystemExit):
    demo._parser().parse_args(["--seed", str(seed)])


def test_fixed_seed_demo_is_exactly_repeatable(generated_demo):
  repeated = demo.generate_demo(seed=0)

  for branch_name in ("normal", "changed"):
    first_branch = getattr(generated_demo, branch_name)
    repeated_branch = getattr(repeated, branch_name)
    np.testing.assert_array_equal(first_branch.states, repeated_branch.states)
    np.testing.assert_array_equal(
        first_branch.commanded_path, repeated_branch.commanded_path
    )
    assert first_branch.contacts == repeated_branch.contacts
  np.testing.assert_array_equal(
      generated_demo.removed.states, repeated.removed.states
  )
  np.testing.assert_array_equal(
      generated_demo.removed.presence, repeated.removed.presence
  )
  assert generated_demo.removed.contacts == repeated.removed.contacts
  assert generated_demo.removed.metadata == repeated.removed.metadata
  assert generated_demo.ground_truth == repeated.ground_truth


def test_demo_result_requires_a_frozen_removed_branch(generated_demo):
  hints = get_type_hints(demo.DemoResult)

  assert hints["removed"] is demo.RemovedBranch
  assert (
      demo.DemoResult.__dataclass_fields__["removed"].default
      is dataclasses.MISSING
  )
  with pytest.raises(FrozenInstanceError):
    generated_demo.removed.branch = "corrupted"
  with pytest.raises(ValueError, match="read-only"):
    generated_demo.removed.states[0, 0, 0] = 1.0
  with pytest.raises(TypeError):
    generated_demo.removed.metadata["trust_model"] = "corrupted"


@pytest.mark.parametrize("field", ("states", "presence"))
def test_removed_branch_arrays_cannot_be_unfrozen_or_alias_inputs(
    generated_demo, field
):
  states_input = np.array(generated_demo.removed.states, copy=True)
  presence_input = np.array(generated_demo.removed.presence, copy=True)
  branch = dataclasses.replace(
      generated_demo.removed,
      states=states_input,
      presence=presence_input,
  )

  assert not np.shares_memory(branch.states, states_input)
  assert not np.shares_memory(branch.presence, presence_input)
  with pytest.raises(ValueError):
    getattr(branch, field).setflags(write=True)


def test_removed_branch_has_exact_prefix_and_presence_mask(generated_demo):
  result = generated_demo
  removed = result.removed
  start, _ = result.intervention_window
  target = result.normal.object_ids.index("target")
  non_target = tuple(
      index
      for index, object_id in enumerate(result.normal.object_ids)
      if object_id != "target"
  )

  assert removed.object_ids == result.normal.object_ids
  assert removed.steps == result.normal.steps == tuple(range(120))
  assert removed.states.shape == result.normal.states.shape == (120, 4, 13)
  assert removed.presence.shape == (120, 4)
  assert removed.presence.dtype == np.bool_
  assert np.isfinite(removed.states).all()
  np.testing.assert_array_equal(
      removed.states[:start], result.normal.states[:start]
  )
  np.testing.assert_array_equal(
      removed.states[:start, non_target], result.normal.states[:start, non_target]
  )
  assert removed.presence[:start, target].all()
  assert not removed.presence[start:, target].any()
  assert removed.presence[:, non_target].all()
  np.testing.assert_array_equal(
      removed.states[start:, target],
      np.broadcast_to(
          removed.states[start - 1, target],
          removed.states[start:, target].shape,
      ),
  )
  assert removed.metadata["trust_model"] == "demo_only_removal_v1"
  assert removed.metadata["target_id"] == "target"
  assert removed.metadata["removed_step"] == start


def test_removed_target_has_no_post_removal_contacts(generated_demo):
  removed = generated_demo.removed
  start, _ = generated_demo.intervention_window
  known_ids = set(removed.object_ids)

  assert isinstance(removed.contacts, tuple)
  assert all(
      {record.object_a, record.object_b} <= known_ids
      for record in removed.contacts
  )
  assert all(
      record.step < start
      for record in removed.contacts
      if "target" in (record.object_a, record.object_b)
  )


def test_generate_demo_rejects_a_corrupted_removed_prefix(
    generated_demo, monkeypatch
):
  states = np.array(generated_demo.removed.states, copy=True)
  non_target = next(
      index
      for index, object_id in enumerate(generated_demo.removed.object_ids)
      if object_id != "target"
  )
  states[0, non_target, 0] += 0.01
  corrupted = dataclasses.replace(generated_demo.removed, states=states)

  monkeypatch.setattr(
      demo,
      "generate_paired_instance",
      lambda *args, **kwargs: (generated_demo.normal, generated_demo.changed),
  )
  monkeypatch.setattr(
      demo,
      "extract_pair_ground_truth",
      lambda *args, **kwargs: generated_demo.ground_truth,
  )
  monkeypatch.setattr(
      demo, "_run_removed_branch", lambda *args, **kwargs: corrupted
  )

  with pytest.raises(RuntimeError, match="prefix"):
    demo.generate_demo(seed=0)


def test_write_demo_bundle_roundtrips_all_branches(generated_demo, tmp_path):
  demo.write_demo_bundle(tmp_path, generated_demo)

  expected = {
      "normal": (
          generated_demo.normal.states,
          np.ones(generated_demo.normal.states.shape[:2], dtype=np.bool_),
          generated_demo.normal.contacts,
      ),
      "trajectory_changed": (
          generated_demo.changed.states,
          np.ones(generated_demo.changed.states.shape[:2], dtype=np.bool_),
          generated_demo.changed.contacts,
      ),
      "target_removed": (
          generated_demo.removed.states,
          generated_demo.removed.presence,
          generated_demo.removed.contacts,
      ),
  }
  for branch_name, (expected_states, expected_presence, _) in expected.items():
    states = np.load(
        tmp_path / f"{branch_name}_states.npy", allow_pickle=False
    )
    presence = np.load(
        tmp_path / f"{branch_name}_presence.npy", allow_pickle=False
    )
    assert states.shape == (120, 4, 13)
    assert presence.shape == (120, 4)
    assert presence.dtype == np.bool_
    np.testing.assert_array_equal(states, expected_states)
    np.testing.assert_array_equal(presence, expected_presence)

  contacts = json.loads((tmp_path / "contacts.json").read_text("utf-8"))
  assert contacts == {
      branch_name: [record.to_dict() for record in records]
      for branch_name, (_, _, records) in expected.items()
  }


def test_write_demo_bundle_summary_has_exact_event_metadata(
    generated_demo, tmp_path
):
  demo.write_demo_bundle(tmp_path, generated_demo)
  summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))

  assert set(summary) == {
      "branches",
      "ground_truth",
      "intervention_end",
      "intervention_start",
      "intervention_window",
      "object_ids",
      "seed",
      "step_rate",
  }
  assert summary["object_ids"] == list(generated_demo.normal.object_ids)
  assert summary["step_rate"] == generated_demo.scene_config.step_rate == 240
  assert summary["seed"] == 0
  assert summary["intervention_start"] == 24
  assert summary["intervention_end"] == 96
  assert summary["intervention_window"] == [24, 96]
  assert summary["ground_truth"] == generated_demo.ground_truth.to_dict()
  assert summary["ground_truth"]["hard_affected"] == ["upper_ball"]
  assert summary["ground_truth"]["soft_affected"] == []
  assert set(summary["ground_truth"]) == {
      "graph_delta",
      "hard_affected",
      "propagation_path",
      "schema_version",
      "soft_affected",
  }

  sources = {
      "normal": generated_demo.normal.contacts,
      "trajectory_changed": generated_demo.changed.contacts,
      "target_removed": generated_demo.removed.contacts,
  }
  assert set(summary["branches"]) == set(sources)
  for branch_name, records in sources.items():
    dynamic = demo.dynamic_contacts(records)
    expected_fields = {"contact_pairs", "contact_steps"}
    if branch_name == "target_removed":
      expected_fields.update(("removed_step", "target_id", "trust_model"))
    assert set(summary["branches"][branch_name]) == expected_fields
    assert summary["branches"][branch_name]["contact_steps"] == sorted(
        {record.step for record in dynamic}
    )
    assert summary["branches"][branch_name]["contact_pairs"] == (
        demo._contact_pairs(dynamic)
    )

  removal = summary["branches"]["target_removed"]
  assert removal["removed_step"] == 24
  assert removal["target_id"] == "target"
  assert removal["trust_model"] == "demo_only_removal_v1"


def test_write_demo_bundle_is_byte_identical_and_canonical(
    generated_demo, tmp_path
):
  demo.write_demo_bundle(tmp_path, generated_demo)
  first = {
      path.name: path.read_bytes()
      for path in sorted(tmp_path.iterdir())
  }

  assert set(first) == {
      "contacts.json",
      "normal_presence.npy",
      "normal_states.npy",
      "summary.json",
      "target_removed_presence.npy",
      "target_removed_states.npy",
      "trajectory_changed_presence.npy",
      "trajectory_changed_states.npy",
  }
  for filename in ("contacts.json", "summary.json"):
    decoded = json.loads(first[filename])
    expected = (
        json.dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert first[filename] == expected

  demo.write_demo_bundle(tmp_path, generated_demo)
  second = {
      path.name: path.read_bytes()
      for path in sorted(tmp_path.iterdir())
  }
  assert second == first


def test_write_demo_bundle_rejects_wrong_source_branch_before_writing(
    generated_demo, tmp_path
):
  wrong_normal = dataclasses.replace(generated_demo.normal, branch="normal")
  corrupted = dataclasses.replace(generated_demo, normal=wrong_normal)
  output = tmp_path / "bundle"

  with pytest.raises(ValueError, match="normal branch.*factual"):
    demo.write_demo_bundle(output, corrupted)
  assert not output.exists()


def test_write_demo_bundle_rejects_jointly_permuted_object_order(
    generated_demo, tmp_path
):
  canonical_ids = generated_demo.normal.object_ids
  permuted_ids = tuple(
      item.object_id for item in generated_demo.scene_config.objects
  )
  permutation = tuple(canonical_ids.index(item) for item in permuted_ids)
  assert permuted_ids != tuple(sorted(permuted_ids))

  normal = dataclasses.replace(
      generated_demo.normal,
      object_ids=permuted_ids,
      states=np.take(generated_demo.normal.states, permutation, axis=1),
  )
  changed = dataclasses.replace(
      generated_demo.changed,
      object_ids=permuted_ids,
      states=np.take(generated_demo.changed.states, permutation, axis=1),
  )
  removed = dataclasses.replace(
      generated_demo.removed,
      object_ids=permuted_ids,
      states=np.take(generated_demo.removed.states, permutation, axis=1),
      presence=np.take(generated_demo.removed.presence, permutation, axis=1),
  )
  corrupted = dataclasses.replace(
      generated_demo,
      normal=normal,
      changed=changed,
      removed=removed,
  )
  output = tmp_path / "bundle"

  with pytest.raises(ValueError, match="object order"):
    demo.write_demo_bundle(output, corrupted)
  assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    ("preintervention_state", "commanded_path", "prefix_contacts"),
)
def test_write_demo_bundle_revalidates_the_public_pair_before_writing(
    generated_demo, tmp_path, mutation
):
  changed = generated_demo.changed
  start, _ = generated_demo.intervention_window

  if mutation == "preintervention_state":
    states = np.array(changed.states, copy=True)
    non_target = next(
        index
        for index, object_id in enumerate(changed.object_ids)
        if object_id != generated_demo.intervention.target_id
    )
    states[0, non_target, 0] += 0.01
    changed = dataclasses.replace(changed, states=states)
  elif mutation == "commanded_path":
    commanded_path = np.array(changed.commanded_path, copy=True)
    commanded_path[start + 1, 1] += 0.01
    changed = dataclasses.replace(changed, commanded_path=commanded_path)
  else:
    prefix_index = next(
        index
        for index, record in enumerate(changed.contacts)
        if record.step < start
    )
    contacts = tuple(
        record
        for index, record in enumerate(changed.contacts)
        if index != prefix_index
    )
    changed = dataclasses.replace(changed, contacts=contacts)

  corrupted = dataclasses.replace(generated_demo, changed=changed)
  output = tmp_path / mutation
  with pytest.raises(ValueError):
    demo.write_demo_bundle(output, corrupted)
  assert not output.exists()


def test_write_demo_bundle_rejects_stale_ground_truth(generated_demo, tmp_path):
  stale_ground_truth = dataclasses.replace(
      generated_demo.ground_truth,
      propagation_path={},
  )
  corrupted = dataclasses.replace(
      generated_demo,
      ground_truth=stale_ground_truth,
  )
  output = tmp_path / "bundle"

  with pytest.raises(ValueError, match="ground_truth"):
    demo.write_demo_bundle(output, corrupted)
  assert not output.exists()


def test_main_generates_then_writes_the_replay_bundle(
    generated_demo, tmp_path, monkeypatch, capsys
):
  calls = []
  expected_summary = {"seed": 0, "branches": {}}

  def fake_generate_demo():
    calls.append(("generate",))
    return generated_demo

  def fake_write_demo_bundle(output_dir, result):
    calls.append(("write", output_dir, result))
    return expected_summary

  monkeypatch.setattr(demo, "generate_demo", fake_generate_demo)
  monkeypatch.setattr(demo, "write_demo_bundle", fake_write_demo_bundle)

  assert demo.main(["--output", str(tmp_path)]) == 0
  assert calls == [
      ("generate",),
      ("write", tmp_path, generated_demo),
  ]
  assert json.loads(capsys.readouterr().out) == expected_summary


def _contact(step, object_a, object_b):
  return {
      "step": step,
      "object_a": object_a,
      "object_b": object_b,
      "contact_distance": -0.01,
      "normal_force": 3.5,
  }


def _contact_records():
  records = [_contact(step, "floor", "pusher") for step in range(_NUM_STEPS)]
  records.append(_contact(2, "pusher", "upper_ball"))
  records.append(_contact(3, "pusher", "upper_ball"))
  return records


def _synthetic_branch():
  states = np.zeros((_NUM_STEPS, 4, 13), dtype=np.float64)
  states[:, :, 3] = 1.0
  states[:, 0, 0:3] = (0.0, 0.0, -0.25)
  for step in range(_NUM_STEPS):
    states[step, 1, 0] = -0.6 + 0.2 * step
    states[step, 1, 1] = 0.0
  states[:, 2, 0:3] = (0.0, 0.45, 0.26)
  states[:, 3, 0:3] = (0.0, -0.45, 0.26)
  return demo.BranchResult(
      branch="counterfactual",
      states=states,
      contact_records=_contact_records(),
  )


def _strong_red_mask(frame):
  channels = frame.astype(int)
  return (
      (channels[:, :, 0] > 140)
      & (channels[:, :, 1] < 110)
      & (channels[:, :, 2] < 110)
  )


def test_dynamic_contact_records_filters_floor_and_object():
  records = _contact_records()
  dynamic = demo._dynamic_contact_records(records)
  assert len(dynamic) == 2
  assert all("floor" not in (r["object_a"], r["object_b"]) for r in dynamic)

  assert len(demo._dynamic_contact_records(records, "upper_ball")) == 2
  assert demo._dynamic_contact_records(records, "lower_ball") == []
  assert demo._dynamic_contact_records(records, "pusher") == dynamic


def test_render_branch_video_writes_readable_mp4(tmp_path):
  output = tmp_path / "counterfactual.mp4"
  demo._render_branch_video(output, _synthetic_branch())

  assert output.exists()
  assert output.stat().st_size > 0
  reader = imageio.get_reader(output)
  try:
    metadata = reader.get_meta_data()
    frame_count = reader.count_frames()
    frames = [reader.get_data(index) for index in range(min(frame_count, 2))]
  finally:
    reader.close()
  assert tuple(metadata["size"]) == demo._CANVAS_SIZE
  assert metadata["fps"] == 24.0
  assert frame_count == _NUM_STEPS
  expected_shape = (demo._CANVAS_SIZE[1], demo._CANVAS_SIZE[0], 3)
  assert len(frames) == 2
  assert all(frame.shape == expected_shape for frame in frames)


def test_render_branch_video_draws_impact_ring_only_on_contact_frames(tmp_path):
  output = tmp_path / "counterfactual.mp4"
  demo._render_branch_video(output, _synthetic_branch())

  reader = imageio.get_reader(output)
  try:
    impact_frame = reader.get_data(2)
    idle_frame = reader.get_data(0)
  finally:
    reader.close()
  impact_red = int(_strong_red_mask(impact_frame).sum())
  idle_red = int(_strong_red_mask(idle_frame).sum())
  assert impact_red > 100
  assert idle_red == 0
