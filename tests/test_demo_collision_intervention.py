"""Tests for the collision demo's physics branches and video rendering."""

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import pytest

pytest.importorskip("pybullet")
import numpy as np  # noqa: E402

_SCRIPT_NAME = "demo_collision_intervention"
_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / f"{_SCRIPT_NAME}.py"
)
_SPEC = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SCRIPT_NAME] = demo
_SPEC.loader.exec_module(demo)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUN_DEMO = _PROJECT_ROOT / "run_demo.sh"
_DEMO_DOCS = _PROJECT_ROOT / "docs" / "trajectory_interventions.md"
_FULL_REQUIREMENTS = _PROJECT_ROOT / "requirements_full.txt"


@pytest.fixture(scope="module")
def generated_demo():
  return demo.generate_demo(seed=0)


def _write_executable(path: Path, contents: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(contents, encoding="utf-8")
  path.chmod(0o755)


def _workflow_sandbox(tmp_path: Path, *, docker: bool = True):
  root = tmp_path / "repo"
  root.mkdir()
  script = root / "run_demo.sh"
  script.write_bytes(_RUN_DEMO.read_bytes())
  script.chmod(0o755)

  fake_bin = tmp_path / "bin"
  fake_bin.mkdir()
  for command in ("bash", "dirname", "id", "ls", "mkdir"):
    executable = shutil.which(command)
    assert executable is not None
    (fake_bin / command).symlink_to(executable)

  conda_base = tmp_path / "conda"
  thesis_python = conda_base / "envs" / "thesis" / "bin" / "python"
  call_log = tmp_path / "calls.log"
  _write_executable(
      fake_bin / "conda",
      """#!/bin/sh
if [ "$1" = "info" ] && [ "$2" = "--base" ]; then
  printf '%s\n' "$FAKE_CONDA_BASE"
  exit 0
fi
exit 1
""",
  )
  _write_executable(
      thesis_python,
      """#!/bin/sh
printf 'python %s\n' "$*" >> "$CALL_LOG"
output="$FAKE_REPO_ROOT/output/demo_collision_intervention"
case " $* " in
  *" -m scripts.demo_collision_intervention "*)
    mkdir -p "$output"
    for name in normal trajectory_changed target_removed; do
      printf x > "$output/${name}_states.npy"
      printf x > "$output/${name}_presence.npy"
    done
    printf '{}\n' > "$output/contacts.json"
    printf '{}\n' > "$output/summary.json"
    ;;
  *" -m scripts.compose_intervention_demo "*)
    printf x > "$output/trajectory_intervention_demo.mp4"
    ;;
esac
""",
  )
  if docker:
    _write_executable(
        fake_bin / "docker",
        """#!/bin/sh
printf 'docker %s\n' "$*" >> "$CALL_LOG"
if [ "$1" = "info" ]; then
  exit "${FAKE_DOCKER_INFO_STATUS:-0}"
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  exit "${FAKE_DOCKER_IMAGE_STATUS:-0}"
fi
if [ "$1" = "run" ]; then
  if [ "${FAKE_DOCKER_RUN_STATUS:-0}" -ne 0 ]; then
    exit "$FAKE_DOCKER_RUN_STATUS"
  fi
  output="$FAKE_REPO_ROOT/output/demo_collision_intervention"
  mkdir -p "$output"
  for name in normal trajectory_changed target_removed; do
    printf x > "$output/${name}_blender.mp4"
  done
fi
""",
    )

  environment = os.environ.copy()
  environment.update({
      "CALL_LOG": str(call_log),
      "FAKE_CONDA_BASE": str(conda_base),
      "FAKE_REPO_ROOT": str(root),
      "PATH": str(fake_bin),
  })
  return script, root, call_log, environment


def _run_workflow(script: Path, root: Path, environment, mode: str):
  return subprocess.run(
      [str(script), mode],
      cwd=root,
      env=environment,
      check=False,
      capture_output=True,
      text=True,
  )


def test_run_demo_intervention_orders_generate_render_and_compose(tmp_path):
  script, root, call_log, environment = _workflow_sandbox(tmp_path)

  completed = _run_workflow(script, root, environment, "intervention")

  assert completed.returncode == 0, completed.stderr
  calls = call_log.read_text("utf-8").splitlines()
  generate = next(
      index
      for index, call in enumerate(calls)
      if call.startswith("python -m scripts.demo_collision_intervention ")
  )
  render = next(
      index for index, call in enumerate(calls) if call.startswith("docker run ")
  )
  compose = next(
      index
      for index, call in enumerate(calls)
      if call.startswith("python -m scripts.compose_intervention_demo ")
  )
  assert generate < render < compose
  assert "--branches normal trajectory_changed target_removed" in calls[render]
  assert 'python3 -c "import imageio_ffmpeg"' in calls[render]
  assert "--target /tmp/kubric-demo-imageio" in calls[render]
  assert "export HOME=" not in calls[render]
  assert (
      str(root / "output/demo_collision_intervention/trajectory_intervention_demo.mp4")
      in completed.stdout
  )


def test_run_demo_intervention_fails_explicitly_without_docker(tmp_path):
  script, root, call_log, environment = _workflow_sandbox(
      tmp_path, docker=False
  )

  completed = _run_workflow(script, root, environment, "intervention")

  assert completed.returncode != 0
  assert "Docker" in completed.stderr
  calls = call_log.read_text("utf-8")
  assert "scripts.demo_collision_intervention" in calls
  assert "scripts.compose_intervention_demo" not in calls


def test_run_demo_intervention_fails_explicitly_when_blender_render_fails(
    tmp_path,
):
  script, root, call_log, environment = _workflow_sandbox(tmp_path)
  environment["FAKE_DOCKER_RUN_STATUS"] = "17"

  completed = _run_workflow(script, root, environment, "intervention")

  assert completed.returncode != 0
  assert "Blender" in completed.stderr
  assert "scripts.compose_intervention_demo" not in call_log.read_text("utf-8")


def test_run_demo_physics_only_does_not_require_docker_or_compose(tmp_path):
  script, root, call_log, environment = _workflow_sandbox(
      tmp_path, docker=False
  )

  completed = _run_workflow(
      script, root, environment, "intervention-physics-only"
  )

  assert completed.returncode == 0, completed.stderr
  calls = call_log.read_text("utf-8").splitlines()
  assert len(calls) == 1
  assert calls[0].startswith("python -m scripts.demo_collision_intervention ")


def test_demo_documentation_describes_three_branch_video_contract():
  documentation = _DEMO_DOCS.read_text("utf-8")

  for branch in ("normal", "trajectory_changed", "target_removed"):
    assert branch in documentation
  assert "trajectory_intervention_demo.mp4" in documentation
  assert "demo_only_removal_v1" in documentation
  assert "procedural" in documentation.lower()
  assert "ffprobe" in documentation
  assert "Milestone E" in documentation
  assert "Milestone F" in documentation


def test_full_requirements_declares_direct_video_encoder_dependency():
  requirements = {
      line.strip()
      for line in _FULL_REQUIREMENTS.read_text("utf-8").splitlines()
      if line.strip() and not line.lstrip().startswith("#")
  }

  assert "imageio-ffmpeg" in requirements


def _unique_dynamic_pairs(records):
  return {
      tuple(sorted((record.object_a, record.object_b)))
      for record in demo.dynamic_contacts(records)
  }


def test_build_demo_inputs_uses_shared_forked_rack_spec():
  scene, intervention, factual_path = demo.build_demo_inputs()
  spec = demo.FORKED_RACK_SPEC
  objects = {item.object_id: item for item in scene.objects}

  assert tuple(item.object_id for item in scene.objects) == spec.object_ids
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
      endpoint for pair in changed_pairs for endpoint in pair
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
  with pytest.raises(FrozenInstanceError):
    generated_demo.changed = generated_demo.normal


def test_calibrated_forked_rack_has_exact_chain_outcomes(generated_demo):
  assert _unique_dynamic_pairs(generated_demo.normal.contacts) == {
      ("side_01", "side_02"),
      ("side_01", "target"),
  }
  assert _unique_dynamic_pairs(generated_demo.changed.contacts) == {
      ("breaker", "target"),
      ("rack_01", "rack_03"),
      ("rack_01", "target"),
      ("rack_02", "rack_03"),
      ("rack_02", "rack_05"),
      ("rack_02", "target"),
      ("rack_03", "rack_06"),
      ("rack_04", "rack_06"),
      ("rack_05", "rack_06"),
  }
  assert generated_demo.ground_truth.hard_affected == (
      "breaker", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05",
      "rack_06", "side_01", "side_02",
  )
  assert generated_demo.ground_truth.soft_affected == ()


def test_changed_branch_diverges_only_inside_intervention_window(generated_demo):
  result = generated_demo
  start, end = result.intervention_window

  assert (start, end) == (40, 160)
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
  assert removed.steps == result.normal.steps == tuple(range(200))
  assert removed.states.shape == result.normal.states.shape == (200, 11, 13)
  assert removed.presence.shape == (200, 11)
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


def test_removed_branch_has_no_post_removal_dynamic_chain(generated_demo):
  removed = generated_demo.removed
  start, _ = generated_demo.intervention_window
  known_ids = set(removed.object_ids)

  assert isinstance(removed.contacts, tuple)
  assert all(
      {record.object_a, record.object_b} <= known_ids
      for record in removed.contacts
  )
  assert not demo.dynamic_contacts(tuple(
      record for record in removed.contacts if record.step >= 40
  ))


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
    assert states.shape == (200, 11, 13)
    assert presence.shape == (200, 11)
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
  assert summary["intervention_start"] == 40
  assert summary["intervention_end"] == 160
  assert summary["intervention_window"] == [40, 160]
  assert summary["ground_truth"] == generated_demo.ground_truth.to_dict()
  assert summary["ground_truth"]["hard_affected"] == [
      "breaker", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05",
      "rack_06", "side_01", "side_02",
  ]
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
  assert removal["removed_step"] == 40
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
  permuted_ids = tuple(reversed(canonical_ids))
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
    prefix_index = 0
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
