"""Offline tests for the procedural Blender branch-replay renderer."""

import builtins
import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_NAME = "render_demo_branches_blender"
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / f"{_SCRIPT_NAME}.py"
_OBJECT_IDS = ("floor", "lower_ball", "target", "upper_ball")
_ALLOWED_BRANCHES = ("normal", "trajectory_changed", "target_removed")

_SPEC = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
render_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SCRIPT_NAME] = render_script
_SPEC.loader.exec_module(render_script)


def _synthetic_states(num_steps=5):
  states = np.zeros((num_steps, 4, 13), dtype=np.float64)
  states[:, :, 3] = 1.0  # identity quaternion in SimulationLog WXYZ order
  states[:, 0, 0:3] = (0.0, 0.0, -0.25)
  for step in range(num_steps):
    states[step, 1, 0:3] = (-0.5 + 0.1 * step, -0.45, 0.26)
  states[:, 2, 0:3] = (0.0, 0.0, 0.18)
  states[:, 3, 0:3] = (0.0, 0.45, 0.26)
  return states


def _write_bundle(
    directory,
    *,
    branch="normal",
    states=None,
    presence=None,
    object_ids=_OBJECT_IDS,
):
  if states is None:
    states = _synthetic_states()
  if presence is None:
    presence = np.ones(states.shape[:2], dtype=np.bool_)
  np.save(directory / f"{branch}_states.npy", states)
  np.save(directory / f"{branch}_presence.npy", presence)
  summary = {
      "branches": {
          name: {} for name in _ALLOWED_BRANCHES
      },
      "ground_truth": {"schema_version": "1.0"},
      "intervention_end": 4,
      "intervention_start": 2,
      "intervention_window": [2, 4],
      "object_ids": list(object_ids),
      "seed": 0,
      "step_rate": 240,
  }
  (directory / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
  )
  return states, presence, summary


def test_load_replay_roundtrips_states_presence_and_summary(tmp_path):
  states, presence, summary = _write_bundle(tmp_path)

  replay = render_script._load_replay(tmp_path, "normal")

  assert replay.branch == "normal"
  assert replay.object_ids == _OBJECT_IDS
  assert replay.summary == summary
  np.testing.assert_array_equal(replay.states, states)
  np.testing.assert_array_equal(replay.presence, presence)


def test_load_replay_requires_matching_presence(tmp_path):
  states = _synthetic_states()
  _write_bundle(
      tmp_path,
      states=states,
      presence=np.ones((4, 4), dtype=np.bool_),
  )

  with pytest.raises(ValueError, match="presence"):
    render_script._load_replay(tmp_path, "normal")


def test_load_replay_requires_boolean_presence(tmp_path):
  states = _synthetic_states()
  _write_bundle(
      tmp_path,
      states=states,
      presence=np.ones(states.shape[:2], dtype=np.uint8),
  )

  with pytest.raises(TypeError, match="Boolean"):
    render_script._load_replay(tmp_path, "normal")


@pytest.mark.parametrize(
    "states",
    (
        np.zeros((5, 4, 12), dtype=np.float64),
        np.zeros((5, 3, 13), dtype=np.float64),
        np.zeros((5, 4, 13, 1), dtype=np.float64),
    ),
)
def test_load_replay_rejects_invalid_state_shape(tmp_path, states):
  _write_bundle(
      tmp_path,
      states=states,
      presence=np.ones(states.shape[:2], dtype=np.bool_),
  )

  with pytest.raises(ValueError, match="states"):
    render_script._load_replay(tmp_path, "normal")


def test_load_replay_rejects_empty_or_nonfinite_states(tmp_path):
  states = _synthetic_states(num_steps=0)
  _write_bundle(tmp_path, states=states)
  with pytest.raises(ValueError, match="frame"):
    render_script._load_replay(tmp_path, "normal")

  states = _synthetic_states()
  states[2, 1, 0] = np.nan
  _write_bundle(tmp_path, states=states)
  with pytest.raises(ValueError, match="non-finite"):
    render_script._load_replay(tmp_path, "normal")


def test_load_replay_requires_canonical_summary_object_order(tmp_path):
  _write_bundle(
      tmp_path,
      object_ids=("floor", "target", "lower_ball", "upper_ball"),
  )

  with pytest.raises(ValueError, match="object_ids"):
    render_script._load_replay(tmp_path, "normal")


@pytest.mark.parametrize("missing_key", ("ground_truth", "intervention_end"))
def test_load_replay_requires_exact_summary_keys(tmp_path, missing_key):
  _, _, summary = _write_bundle(tmp_path)
  del summary[missing_key]
  (tmp_path / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
  )

  with pytest.raises(ValueError, match="summary|keys|missing"):
    render_script._load_replay(tmp_path, "normal")


def test_load_replay_rejects_unknown_summary_keys(tmp_path):
  _, _, summary = _write_bundle(tmp_path)
  summary["resolution"] = [640, 540]
  (tmp_path / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
  )

  with pytest.raises(ValueError, match="summary|keys|unexpected"):
    render_script._load_replay(tmp_path, "normal")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("step_rate", 0),
        ("intervention_start", -1),
        ("intervention_window", [1, 4]),
        ("ground_truth", []),
        ("seed", True),
    ),
)
def test_load_replay_validates_summary_types_and_ranges(
    tmp_path, field, value
):
  _, _, summary = _write_bundle(tmp_path)
  summary[field] = value
  (tmp_path / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
  )

  with pytest.raises((TypeError, ValueError), match=field):
    render_script._load_replay(tmp_path, "normal")


@pytest.mark.parametrize(("start", "end"), ((5, 6), (2, 6)))
def test_load_replay_requires_intervention_window_within_frames(
    tmp_path, start, end
):
  _, _, summary = _write_bundle(tmp_path)
  summary["intervention_start"] = start
  summary["intervention_end"] = end
  summary["intervention_window"] = [start, end]
  (tmp_path / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
  )

  with pytest.raises(ValueError, match="intervention.*frame"):
    render_script._load_replay(tmp_path, "normal")


@pytest.mark.parametrize("branch", ("factual", "counterfactual", "NORMAL", ""))
def test_load_replay_rejects_noncanonical_branch_names(tmp_path, branch):
  _write_bundle(tmp_path)

  with pytest.raises(ValueError, match="branch"):
    render_script._load_replay(tmp_path, branch)


def test_prepare_replay_slices_both_arrays_and_validates_max_frames(tmp_path):
  states, presence, _ = _write_bundle(tmp_path)
  replay = render_script._load_replay(tmp_path, "normal")

  sliced = render_script._prepare_replay(replay, 3)
  np.testing.assert_array_equal(sliced.states, states[:3])
  np.testing.assert_array_equal(sliced.presence, presence[:3])
  assert sliced.object_ids == replay.object_ids
  assert render_script._prepare_replay(replay, None) is replay
  assert len(render_script._prepare_replay(replay, 100).states) == len(states)

  for invalid in (0, -1, 1.5, True):
    with pytest.raises((TypeError, ValueError), match="max-frames"):
      render_script._prepare_replay(replay, invalid)


def test_pose_at_reads_simulation_log_wxyz_without_reordering():
  states = _synthetic_states(num_steps=1)
  states[0, 1, 3:7] = (0.4, 0.1, 0.2, 0.3)

  position, quaternion = render_script._pose_at(states, 1, 0)

  assert position == (-0.5, -0.45, 0.26)
  assert quaternion == (0.4, 0.1, 0.2, 0.3)


def test_material_specs_are_deterministic_and_realistic():
  first = render_script._material_specs()
  second = render_script._material_specs()

  assert first == second
  assert first is not second
  assert first["target"]["material"] == "wood"
  assert first["target"]["grain_scale"] > 1.0
  assert first["floor"]["material"] == "felt"
  assert first["floor"]["roughness"] > 0.8
  assert first["upper_ball"]["material"] == "lacquer"
  assert first["lower_ball"]["material"] == "lacquer"
  assert first["upper_ball"]["roughness"] < 0.25
  assert first["lower_ball"]["roughness"] < 0.25


def test_scene_specs_preserve_colliders_and_define_three_area_lights():
  specs = render_script._scene_specs()

  assert specs["colliders"] == {
      "floor": {"kind": "cube", "scale": (4.0, 4.0, 0.25)},
      "lower_ball": {"kind": "sphere", "scale": 0.26},
      "target": {"kind": "cube", "scale": (0.18, 0.18, 0.18)},
      "upper_ball": {"kind": "sphere", "scale": 0.26},
  }
  assert tuple(light["role"] for light in specs["lights"]) == (
      "key",
      "fill",
      "rim",
  )
  assert all(light["kind"] == "area" for light in specs["lights"])
  assert specs["renderer"] == {
      "engine": "CYCLES",
      "adaptive_sampling": True,
      "denoising": True,
      "transparent": False,
  }


def test_scene_specs_define_deterministic_camera_depth_of_field():
  first = render_script._scene_specs()["camera"]
  second = render_script._scene_specs()["camera"]
  expected_distance = float(
      np.linalg.norm(
          np.asarray(first["position"]) - np.asarray(first["look_at"])
      )
  )

  assert first == second
  assert first["dof"] == {
      "use_dof": True,
      "focus_distance": pytest.approx(expected_distance),
      "aperture_fstop": 4.0,
  }


def test_configure_camera_dof_updates_blender_camera_data():
  class FakeDof:
    use_dof = False
    focus_distance = 0.0
    aperture_fstop = 0.0

  class FakeCameraData:
    dof = FakeDof()

  class FakeCamera:
    data = FakeCameraData()

  camera = FakeCamera()
  spec = {
      "use_dof": True,
      "focus_distance": 9.5,
      "aperture_fstop": 4.0,
  }

  render_script._configure_camera_dof(camera, spec)

  assert camera.data.dof.use_dof is True
  assert camera.data.dof.focus_distance == 9.5
  assert camera.data.dof.aperture_fstop == 4.0


def test_encoder_backend_uses_imageio_when_blender_lacks_ffmpeg():
  assert render_script._encoder_backend(("PNG", "AVI_JPEG")) == "imageio"
  assert render_script._encoder_backend(("PNG", "FFMPEG")) == "blender"


def test_visibility_transitions_hide_at_first_absent_frame_and_stay_hidden():
  present = np.ones(120, dtype=np.bool_)
  present[24:] = False

  transitions = render_script._visibility_transitions(present)

  assert transitions == ((0, False), (24, True))


def test_visibility_transitions_keep_always_present_objects_visible():
  present = np.ones(120, dtype=np.bool_)

  assert render_script._visibility_transitions(present) == ((0, False),)


@pytest.mark.parametrize(
    ("presence", "exception"),
    (
        (np.ones((2, 1), dtype=np.bool_), ValueError),
        (np.ones(2, dtype=np.uint8), TypeError),
        (np.ones(0, dtype=np.bool_), ValueError),
    ),
)
def test_visibility_transitions_validate_input(presence, exception):
  with pytest.raises(exception, match="presence"):
    render_script._visibility_transitions(presence)


def test_parser_defaults_and_exact_branch_validation():
  args = render_script._parser().parse_args([])
  assert args.branches == list(render_script._ALLOWED_BRANCHES)
  assert args.resolution == [640, 540]
  assert args.fps == 24
  assert args.samples in (32, 64)
  assert args.max_frames is None
  assert not args.save_blend

  parsed = render_script._parser().parse_args(
      ["--branches", "normal", "target_removed"]
  )
  assert parsed.branches == ["normal", "target_removed"]

  with pytest.raises(SystemExit):
    render_script._parser().parse_args(["--branches", "factual"])


@pytest.mark.parametrize(
    "arguments",
    (
        ("--resolution", "0", "540"),
        ("--resolution", "640", "-1"),
        ("--fps", "0"),
        ("--samples", "0"),
        ("--max-frames", "0"),
        ("--max-frames", "-1"),
    ),
)
def test_parser_rejects_nonpositive_numeric_options(arguments):
  with pytest.raises(SystemExit):
    render_script._parser().parse_args(arguments)


def test_parser_rejects_duplicate_branches():
  with pytest.raises(SystemExit):
    render_script._parser().parse_args(
        ["--branches", "normal", "target_removed", "normal"]
    )


def test_branch_output_names_are_canonical():
  assert render_script._BRANCH_FILENAMES == {
      "normal": "normal_blender.mp4",
      "trajectory_changed": "trajectory_changed_blender.mp4",
      "target_removed": "target_removed_blender.mp4",
  }


def test_main_fails_fast_when_imageio_ffmpeg_is_unavailable(
    tmp_path, monkeypatch
):
  _write_bundle(tmp_path, branch="normal")
  render_calls = []
  real_import = builtins.__import__

  def blocked_import(name, *args, **kwargs):
    if name == "imageio_ffmpeg":
      raise ModuleNotFoundError(name=name)
    return real_import(name, *args, **kwargs)

  def fake_render(*args, **kwargs):
    render_calls.append((args, kwargs))
    return {}

  monkeypatch.setattr(builtins, "__import__", blocked_import)
  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)

  with pytest.raises(ImportError, match="imageio-ffmpeg"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
    ])
  assert render_calls == []


def test_main_rejects_corrupt_preintervention_state_before_render(
    tmp_path, monkeypatch
):
  normal = _synthetic_states(num_steps=5)
  changed = normal.copy()
  changed[0, 1, 0] += 0.01
  _write_bundle(tmp_path, branch="normal", states=normal)
  _write_bundle(tmp_path, branch="trajectory_changed", states=changed)
  render_calls = []
  monkeypatch.setattr(
      render_script,
      "_build_and_render_branch",
      lambda *args, **kwargs: render_calls.append((args, kwargs)),
  )

  with pytest.raises(ValueError, match="synchronized.*common prefix"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
        "trajectory_changed",
    ])
  assert render_calls == []


def test_main_rejects_absent_preintervention_object_before_render(
    tmp_path, monkeypatch
):
  states = _synthetic_states(num_steps=5)
  normal_presence = np.ones(states.shape[:2], dtype=np.bool_)
  changed_presence = normal_presence.copy()
  changed_presence[0, 2] = False
  _write_bundle(
      tmp_path,
      branch="normal",
      states=states,
      presence=normal_presence,
  )
  _write_bundle(
      tmp_path,
      branch="trajectory_changed",
      states=states,
      presence=changed_presence,
  )
  render_calls = []
  monkeypatch.setattr(
      render_script,
      "_build_and_render_branch",
      lambda *args, **kwargs: render_calls.append((args, kwargs)),
  )

  with pytest.raises(ValueError, match="pre-intervention presence"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
        "trajectory_changed",
    ])
  assert render_calls == []


def test_main_preflights_synchronized_frame_counts_before_rendering(
    tmp_path, monkeypatch
):
  _write_bundle(
      tmp_path,
      branch="normal",
      states=_synthetic_states(num_steps=5),
  )
  _write_bundle(
      tmp_path,
      branch="trajectory_changed",
      states=_synthetic_states(num_steps=4),
  )
  rendered = []

  def fake_render(branch, replay, output, *args):
    rendered.append((branch, len(replay.states)))
    output.write_text("rendered too early", encoding="utf-8")
    return {"branch": branch, "output": str(output)}

  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)

  with pytest.raises(ValueError, match="synchronized|frame count"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
        "trajectory_changed",
    ])

  assert rendered == []
  assert not (tmp_path / "normal_blender.mp4").exists()
  assert not (tmp_path / "trajectory_changed_blender.mp4").exists()


def test_main_rejects_source_length_mismatch_before_max_frames(
    tmp_path, monkeypatch
):
  _write_bundle(
      tmp_path,
      branch="normal",
      states=_synthetic_states(num_steps=5),
  )
  _write_bundle(
      tmp_path,
      branch="trajectory_changed",
      states=_synthetic_states(num_steps=4),
  )
  rendered = []

  def fake_render(branch, replay, output, *args):
    rendered.append((branch, len(replay.states)))
    output.write_text("rendered too early", encoding="utf-8")
    return {"branch": branch, "output": str(output)}

  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)

  with pytest.raises(ValueError, match="synchronized|frame count"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
        "trajectory_changed",
        "--max-frames",
        "4",
    ])

  assert rendered == []
  assert not (tmp_path / "normal_blender.mp4").exists()
  assert not (tmp_path / "trajectory_changed_blender.mp4").exists()


def test_main_slices_synchronized_sources_after_preflight(tmp_path, monkeypatch):
  for branch in ("normal", "trajectory_changed"):
    _write_bundle(
        tmp_path,
        branch=branch,
        states=_synthetic_states(num_steps=5),
    )
  rendered = []

  def fake_render(branch, replay, output, *args):
    rendered.append((branch, len(replay.states), replay.steps))
    output.write_bytes(b"synthetic video")
    return {"branch": branch, "output": str(output)}

  monkeypatch.setattr(render_script, "_require_imageio_ffmpeg", lambda: None)
  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)
  monkeypatch.setattr(
      render_script,
      "_verify_rendered_mp4",
      lambda path: None,
      raising=False,
  )

  assert render_script.main([
      "--states-dir",
      str(tmp_path),
      "--branches",
      "normal",
      "trajectory_changed",
      "--max-frames",
      "4",
  ]) == 0
  assert rendered == [
      ("normal", 4, tuple(range(4))),
      ("trajectory_changed", 4, tuple(range(4))),
  ]


def test_main_allows_a_single_requested_branch(tmp_path, monkeypatch):
  _write_bundle(tmp_path, branch="normal")
  rendered = []

  def fake_render(branch, replay, output, *args):
    rendered.append((branch, replay.object_ids, replay.steps))
    output.write_bytes(b"synthetic video")
    return {"branch": branch, "output": str(output)}

  monkeypatch.setattr(render_script, "_require_imageio_ffmpeg", lambda: None)
  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)
  monkeypatch.setattr(
      render_script,
      "_verify_rendered_mp4",
      lambda path: None,
      raising=False,
  )

  assert render_script.main([
      "--states-dir",
      str(tmp_path),
      "--branches",
      "normal",
  ]) == 0
  assert rendered == [(
      "normal",
      _OBJECT_IDS,
      tuple(range(5)),
  )]


def test_main_keeps_existing_outputs_when_a_later_render_fails(
    tmp_path, monkeypatch
):
  states = _synthetic_states(num_steps=5)
  for branch in ("normal", "trajectory_changed"):
    _write_bundle(tmp_path, branch=branch, states=states)
  normal_output = tmp_path / "normal_blender.mp4"
  changed_output = tmp_path / "trajectory_changed_blender.mp4"
  normal_output.write_bytes(b"old normal")
  changed_output.write_bytes(b"old changed")
  render_paths = []

  def fake_render(branch, replay, output, *args):
    render_paths.append(output)
    if branch == "trajectory_changed":
      raise RuntimeError("second branch failed")
    output.write_bytes(b"new normal")
    return {"branch": branch, "output": str(output)}

  monkeypatch.setattr(render_script, "_require_imageio_ffmpeg", lambda: None)
  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)
  monkeypatch.setattr(
      render_script,
      "_verify_rendered_mp4",
      lambda path: None,
      raising=False,
  )

  with pytest.raises(RuntimeError, match="second branch failed"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
        "trajectory_changed",
    ])

  assert normal_output.read_bytes() == b"old normal"
  assert changed_output.read_bytes() == b"old changed"
  assert all(path not in (normal_output, changed_output) for path in render_paths)
  assert not any(
      path.name.startswith(".blender-replays-")
      for path in tmp_path.iterdir()
  )


def test_main_does_not_accept_a_stale_final_as_a_new_render(
    tmp_path, monkeypatch
):
  _write_bundle(tmp_path, branch="normal")
  final_output = tmp_path / "normal_blender.mp4"
  final_output.write_bytes(b"stale video")
  monkeypatch.setattr(render_script, "_require_imageio_ffmpeg", lambda: None)
  monkeypatch.setattr(
      render_script,
      "_build_and_render_branch",
      lambda branch, replay, output, *args: {
          "branch": branch,
          "output": str(output),
      },
  )

  with pytest.raises(RuntimeError, match="new|missing|nonempty"):
    render_script.main([
        "--states-dir",
        str(tmp_path),
        "--branches",
        "normal",
    ])
  assert final_output.read_bytes() == b"stale video"


def test_main_publishes_only_verified_temporary_outputs(tmp_path, monkeypatch):
  _write_bundle(tmp_path, branch="normal")
  final_output = tmp_path / "normal_blender.mp4"
  final_output.write_bytes(b"old video")
  rendered_paths = []
  verified_paths = []

  def fake_render(branch, replay, output, *args):
    rendered_paths.append(output)
    output.write_bytes(b"new video")
    return {"branch": branch, "output": str(output)}

  monkeypatch.setattr(render_script, "_require_imageio_ffmpeg", lambda: None)
  monkeypatch.setattr(render_script, "_build_and_render_branch", fake_render)
  monkeypatch.setattr(
      render_script,
      "_verify_rendered_mp4",
      lambda path: verified_paths.append(path),
      raising=False,
  )

  assert render_script.main([
      "--states-dir",
      str(tmp_path),
      "--branches",
      "normal",
  ]) == 0
  assert final_output.read_bytes() == b"new video"
  assert rendered_paths == verified_paths
  assert rendered_paths[0] != final_output
  assert not rendered_paths[0].exists()


def test_verify_rendered_mp4_uses_ffprobe(tmp_path, monkeypatch):
  video = tmp_path / "new.mp4"
  video.write_bytes(b"new video")
  calls = []

  def fake_run(command, **kwargs):
    calls.append((command, kwargs))
    return subprocess.CompletedProcess(command, 0, stdout="video\n", stderr="")

  monkeypatch.setattr(shutil, "which", lambda executable: "/usr/bin/ffprobe")
  monkeypatch.setattr(subprocess, "run", fake_run)

  render_script._verify_rendered_mp4(video)

  assert calls
  assert calls[0][0][0] == "/usr/bin/ffprobe"
  assert str(video) in calls[0][0]


def test_temporary_scratch_is_removed_after_render_error():
  observed = []

  def fail_in_scratch(scratch_dir):
    observed.append(scratch_dir)
    assert scratch_dir.is_dir()
    (scratch_dir / "partial.exr").write_bytes(b"partial")
    raise RuntimeError("render failed")

  with pytest.raises(RuntimeError, match="render failed"):
    render_script._run_with_temporary_scratch(fail_in_scratch)

  assert len(observed) == 1
  assert not observed[0].exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "object_ids",
            ("floor", "target", "lower_ball", "upper_ball"),
            "object_ids",
        ),
        ("steps", (0, 1, 2, 4, 5), "step arrays"),
        ("summary", {"step_rate": 120}, "metadata"),
    ),
)
def test_synchronization_rejects_cross_branch_contract_drift(
    tmp_path, field, value, message
):
  _write_bundle(tmp_path)
  baseline = render_script._load_replay(tmp_path, "normal")
  peer = dataclasses.replace(baseline, branch="trajectory_changed")
  if field == "summary":
    value = {**baseline.summary, **value}
  peer = dataclasses.replace(peer, **{field: value})

  with pytest.raises(ValueError, match=f"synchronized.*{message}"):
    render_script._validate_synchronized_replays((baseline, peer))


def test_module_imports_when_optional_backends_are_blocked():
  script = r'''
import importlib.abc
import importlib.util
import sys


class BackendBlocker(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname.split(".", 1)[0] in {"kubric", "pybullet", "bpy"}:
      raise ModuleNotFoundError(
          "blocked optional backend dependency: " + fullname,
          name=fullname,
      )
    return None


sys.meta_path.insert(0, BackendBlocker())

spec = importlib.util.spec_from_file_location(
    "render_demo_branches_blender", "scripts/render_demo_branches_blender.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert module._load_replay is not None
assert module._material_specs is not None
assert module.main is not None
assert not any(
    name.split(".", 1)[0] in {"kubric", "pybullet", "bpy"}
    for name in sys.modules
)
'''

  result = subprocess.run(
      [sys.executable, "-c", script],
      cwd=_PROJECT_ROOT,
      check=False,
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stderr
