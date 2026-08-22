"""Tests for the ffmpeg-only intervention comparison compositor."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_NAME = "compose_intervention_demo"
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / f"{_SCRIPT_NAME}.py"


def _import_script():
  if not _SCRIPT_PATH.is_file():
    return None
  spec = importlib.util.spec_from_file_location(_SCRIPT_NAME, _SCRIPT_PATH)
  module = importlib.util.module_from_spec(spec)
  sys.modules[_SCRIPT_NAME] = module
  spec.loader.exec_module(module)
  return module


compose_script = _import_script()


@pytest.fixture
def compositor():
  if compose_script is None:
    pytest.skip(f"missing compositor: {_SCRIPT_PATH}")
  return compose_script


def _summary(*, frame_count=120):
  return {
      "branches": {
          "normal": {
              "contact_pairs": {},
              "contact_steps": [],
          },
          "trajectory_changed": {
              "contact_pairs": {"target|upper_ball": 2},
              "contact_steps": [48, 49],
          },
          "target_removed": {
              "contact_pairs": {},
              "contact_steps": [],
              "removed_step": 24,
              "target_id": "target",
              "trust_model": "demo_only_removal_v1",
          },
      },
      "ground_truth": {
          "graph_delta": {
              "added": [{
                  "object_a": "target",
                  "object_b": "upper_ball",
                  "start_step": 48,
                  "end_step": 59,
              }],
              "changed": [],
              "removed": [],
              "schema_version": "1.0",
          },
          "hard_affected": ["upper_ball"],
          "propagation_path": {
              "upper_ball": ["target", "upper_ball"],
          },
          "schema_version": "1.0",
          "soft_affected": [],
      },
      "intervention_end": min(96, frame_count - 1),
      "intervention_start": min(24, frame_count - 2),
      "intervention_window": [
          min(24, frame_count - 2),
          min(96, frame_count - 1),
      ],
      "object_ids": ["floor", "lower_ball", "target", "upper_ball"],
      "seed": 0,
      "step_rate": 240.0,
  }


def _write_summary(directory, *, frame_count=120):
  payload = _summary(frame_count=frame_count)
  if frame_count < 50:
    payload["branches"]["trajectory_changed"]["contact_steps"] = [2, 3]
    payload["branches"]["target_removed"]["removed_step"] = 1
  (directory / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )
  return payload


def _touch_sources(directory):
  for branch in ("normal", "trajectory_changed", "target_removed"):
    (directory / f"{branch}_blender.mp4").write_bytes(b"video")


def _font(tmp_path):
  path = tmp_path / "Test Font.ttf"
  path.write_bytes(b"font")
  return path


def test_compositor_script_exists():
  assert compose_script is not None, f"missing compositor: {_SCRIPT_PATH}"


def test_module_imports_without_kubric(compositor):
  assert compositor.main is not None
  assert compositor._probe_video is not None
  script = f'''\
import importlib.abc
import importlib.util
import sys


class BackendBlocker(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == "kubric" or fullname.startswith("kubric."):
      raise ModuleNotFoundError("blocked Kubric import", name=fullname)
    return None


sys.meta_path.insert(0, BackendBlocker())
spec = importlib.util.spec_from_file_location(
    "isolated_compose_intervention_demo", {str(_SCRIPT_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.main is not None
assert not any(
    name == "kubric" or name.startswith("kubric.") for name in sys.modules
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


def test_filter_has_layout_labels_timeline_cues_and_summary(
    compositor, tmp_path
):
  overlay_dir = tmp_path / "overlays"
  overlay_dir.mkdir()
  overlay_files = compositor._write_overlay_textfiles(
      overlay_dir,
      _summary(),
      source_duration=5.0,
      source_fps=24.0,
  )
  filter_graph = compositor._build_filter(
      _summary(),
      _font(tmp_path),
      source_duration=5.0,
      source_fps=24.0,
      overlay_files=overlay_files,
  )

  assert "hstack=inputs=3" in filter_graph
  assert "pad=1920:720:0:90" in filter_graph
  for label in ("NORMAL", "TRAJECTORY CHANGED", "TARGET REMOVED"):
    assert f"text='{label}'" in filter_graph
  assert filter_graph.count("textfile=") == len(overlay_files)
  assert filter_graph.count("reload=0") == len(overlay_files)
  assert "text='CONTACT" not in filter_graph
  assert "text='GRAPH DELTA" not in filter_graph
  assert overlay_files["intervention"].read_text("utf-8") == (
      "INTERVENTION 2.000s"
  )
  assert overlay_files["contact"].read_text("utf-8") == (
      "CONTACT → UPPER BALL 3.000s"
  )
  assert overlay_files["removal"].read_text("utf-8") == (
      "TARGET REMOVED 2.000s"
  )
  assert overlay_files["graph"].read_text("utf-8") == (
      "GRAPH DELTA added=1 removed=0 changed=0"
  )
  assert overlay_files["affected"].read_text("utf-8") == (
      "HARD upper_ball   |   SOFT none"
  )
  assert "target > upper_ball" in overlay_files["propagation"].read_text(
      "utf-8"
  )


def test_contact_cue_prefers_changed_contact_peer_among_hard_affected(
    compositor, tmp_path
):
  summary = _summary()
  summary["ground_truth"]["hard_affected"] = ["lower_ball", "upper_ball"]
  overlay_files = compositor._write_overlay_textfiles(
      tmp_path,
      summary,
      source_duration=5.0,
      source_fps=24.0,
  )

  assert overlay_files["contact"].read_text("utf-8") == (
      "CONTACT → UPPER BALL 3.000s"
  )


def _render_textfile_frame(compositor, ffmpeg, font, textfile):
  drawtext = compositor._drawtext(
      font,
      textfile=textfile,
      x="8",
      y="8",
      size=24,
  )
  return subprocess.run(
      [
          ffmpeg,
          "-hide_banner",
          "-loglevel",
          "error",
          "-f",
          "lavfi",
          "-i",
          "color=c=black:s=320x64:r=24:d=0.1",
          "-vf",
          drawtext,
          "-frames:v",
          "1",
          "-pix_fmt",
          "gray",
          "-f",
          "rawvideo",
          "-",
      ],
      check=False,
      capture_output=True,
  )


def test_drawtext_textfile_renders_apostrophe_as_a_visible_glyph(
    compositor, tmp_path
):
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    pytest.skip("ffmpeg is required for drawtext parser coverage")
  with_apostrophe = tmp_path / "with-apostrophe.txt"
  without_apostrophe = tmp_path / "without-apostrophe.txt"
  with_apostrophe.write_text("target's path", encoding="utf-8")
  without_apostrophe.write_text("targets path", encoding="utf-8")
  font = compositor._resolve_font(None)

  rendered_with = _render_textfile_frame(
      compositor, ffmpeg, font, with_apostrophe
  )
  rendered_without = _render_textfile_frame(
      compositor, ffmpeg, font, without_apostrophe
  )

  assert rendered_with.returncode == 0, rendered_with.stderr.decode("utf-8")
  assert rendered_without.returncode == 0, rendered_without.stderr.decode(
      "utf-8"
  )
  assert rendered_with.stdout != rendered_without.stdout


def test_drawtext_escapes_apostrophe_in_textfile_path(compositor, tmp_path):
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    pytest.skip("ffmpeg is required for drawtext path coverage")
  apostrophe_dir = tmp_path / "apostrophe's directory"
  apostrophe_dir.mkdir()
  textfile = apostrophe_dir / "cue.txt"
  textfile.write_text("CONTACT → UPPER BALL", encoding="utf-8")

  rendered = _render_textfile_frame(
      compositor,
      ffmpeg,
      compositor._resolve_font(None),
      textfile,
  )

  assert rendered.returncode == 0, rendered.stderr.decode("utf-8")
  assert len(rendered.stdout) == 320 * 64


def test_full_filter_escapes_summary_metacharacters(compositor, tmp_path):
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    pytest.skip("ffmpeg is required for filter parser coverage")
  summary = _summary()
  affected = "target's 100% [path]: a,b\\c"
  summary["ground_truth"]["hard_affected"] = [affected]
  summary["ground_truth"]["propagation_path"] = {
      affected: ["target", affected]
  }
  overlay_dir = tmp_path / "special-overlays"
  overlay_dir.mkdir()
  overlay_files = compositor._write_overlay_textfiles(
      overlay_dir,
      summary,
      source_duration=0.25,
      source_fps=24.0,
  )
  filter_graph = compositor._build_filter(
      summary,
      compositor._resolve_font(None),
      source_duration=0.25,
      source_fps=24.0,
      overlay_files=overlay_files,
  )
  command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
  for color in ("red", "green", "blue"):
    command.extend((
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=160x120:r=24:d=0.25",
    ))
  command.extend((
      "-filter_complex",
      filter_graph,
      "-map",
      "[outv]",
      "-frames:v",
      "1",
      "-f",
      "null",
      "-",
  ))

  result = subprocess.run(
      command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert result.returncode == 0, result.stderr


def test_probe_video_parses_positive_stream_metadata(
    compositor, tmp_path, monkeypatch
):
  video = tmp_path / "input.mp4"
  video.write_bytes(b"video")
  payload = {
      "streams": [{
          "codec_name": "h264",
          "width": 640,
          "height": 540,
          "pix_fmt": "yuv420p",
          "avg_frame_rate": "24/1",
          "r_frame_rate": "24/1",
          "nb_frames": "120",
          "duration": "5.000000",
      }],
      "format": {"duration": "5.000000"},
  }
  calls = []

  def fake_run(command, **kwargs):
    calls.append((command, kwargs))
    return subprocess.CompletedProcess(
        command, 0, stdout=json.dumps(payload), stderr=""
    )

  monkeypatch.setattr(subprocess, "run", fake_run)

  info = compositor._probe_video(video, ffprobe="/tools/ffprobe")

  assert info.width == 640
  assert info.height == 540
  assert info.fps == Fraction(24, 1)
  assert isinstance(info.fps, Fraction)
  assert info.frame_count == 120
  assert info.duration == 5.0
  assert info.codec_name == "h264"
  assert info.pix_fmt == "yuv420p"
  assert calls[0][0][0] == "/tools/ffprobe"
  assert calls[0][1]["check"] is True


def test_probe_video_prefers_counted_frames_over_declared_frames(
    compositor, tmp_path, monkeypatch
):
  video = tmp_path / "input.mp4"
  video.write_bytes(b"video")
  payload = {
      "streams": [{
          "codec_name": "h264",
          "width": 640,
          "height": 540,
          "pix_fmt": "yuv420p",
          "avg_frame_rate": "24/1",
          "r_frame_rate": "24/1",
          "nb_frames": "120",
          "nb_read_frames": "119",
          "duration": "5.0",
      }],
      "format": {"duration": "5.0"},
  }
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda command, **kwargs: subprocess.CompletedProcess(
          command, 0, stdout=json.dumps(payload), stderr=""
      ),
  )

  info = compositor._probe_video(video, ffprobe="/tools/ffprobe")

  assert info.frame_count == 119


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("width", 0),
        ("height", -1),
        ("avg_frame_rate", "0/0"),
        ("nb_frames", "0"),
        ("duration", "nan"),
    ),
)
def test_probe_video_rejects_nonpositive_or_nonfinite_metadata(
    compositor, tmp_path, monkeypatch, field, value
):
  video = tmp_path / "input.mp4"
  video.write_bytes(b"video")
  stream = {
      "codec_name": "h264",
      "width": 640,
      "height": 540,
      "pix_fmt": "yuv420p",
      "avg_frame_rate": "24/1",
      "r_frame_rate": "24/1",
      "nb_frames": "120",
      "duration": "5.0",
  }
  stream[field] = value
  payload = {"streams": [stream], "format": {"duration": "5.0"}}
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda command, **kwargs: subprocess.CompletedProcess(
          command, 0, stdout=json.dumps(payload), stderr=""
      ),
  )

  with pytest.raises(ValueError, match="width|height|fps|frame|duration"):
    compositor._probe_video(video, ffprobe="/tools/ffprobe")


def test_source_probe_mismatch_is_rejected_before_ffmpeg_or_output(
    compositor, tmp_path, monkeypatch
):
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  output = tmp_path / "final.mp4"
  font = _font(tmp_path)
  baseline = compositor.VideoInfo(
      width=640,
      height=540,
      fps=24.0,
      frame_count=120,
      duration=5.0,
      codec_name="h264",
      pix_fmt="yuv420p",
  )

  def fake_probe(path, ffprobe=None):
    if Path(path).name == "target_removed_blender.mp4":
      return compositor.VideoInfo(
          width=641,
          height=540,
          fps=24.0,
          frame_count=120,
          duration=5.0,
          codec_name="h264",
          pix_fmt="yuv420p",
      )
    return baseline

  monkeypatch.setattr(compositor, "_probe_video", fake_probe)
  monkeypatch.setattr(
      compositor.shutil,
      "which",
      lambda name: f"/tools/{name}",
  )
  monkeypatch.setattr(
      compositor,
      "_run_ffmpeg",
      lambda command: pytest.fail("ffmpeg ran before mismatch rejection"),
  )

  with pytest.raises(ValueError, match="width|synchronized|match"):
    compositor.compose_intervention_demo(tmp_path, output, font)

  assert not output.exists()


@pytest.mark.parametrize("codec_name", ("vp9", "mpeg4"))
def test_equal_non_h264_sources_are_rejected_before_ffmpeg_or_output(
    compositor, tmp_path, monkeypatch, codec_name
):
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  output = tmp_path / "final.mp4"
  font = _font(tmp_path)
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=120,
      duration=5.0,
      codec_name=codec_name,
      pix_fmt="yuv420p",
  )
  monkeypatch.setattr(compositor, "_probe_video", lambda *args, **kwargs: info)
  monkeypatch.setattr(
      compositor.shutil,
      "which",
      lambda name: f"/tools/{name}",
  )
  monkeypatch.setattr(
      compositor,
      "_run_ffmpeg",
      lambda command: pytest.fail("ffmpeg ran for a non-H264 source"),
  )

  with pytest.raises(ValueError, match="H264|h264|codec"):
    compositor.compose_intervention_demo(tmp_path, output, font)

  assert not output.exists()


def _synchronized_infos(compositor, *, rates=None, durations=None):
  branch_names = ("normal", "trajectory_changed", "target_removed")
  rates = rates or (Fraction(24, 1),) * 3
  durations = durations or (5.0,) * 3
  return {
      branch: compositor.VideoInfo(
          width=640,
          height=540,
          fps=rate,
          frame_count=120,
          duration=duration,
          codec_name="h264",
          pix_fmt="yuv420p",
      )
      for branch, rate, duration in zip(branch_names, rates, durations)
  }


def _validate_fake_infos(compositor, monkeypatch, infos):
  sources = {
      branch: Path(f"/{branch}.mp4") for branch in infos
  }
  monkeypatch.setattr(
      compositor,
      "_probe_video",
      lambda path, ffprobe=None: infos[Path(path).stem],
  )
  return compositor._validate_synchronized_sources(sources, "/tools/ffprobe")


def test_source_fps_compares_normalized_fractions_exactly(
    compositor, monkeypatch
):
  equivalent = _synchronized_infos(
      compositor,
      rates=(Fraction(24, 1), Fraction(48, 2), Fraction(240, 10)),
  )

  reference, _ = _validate_fake_infos(compositor, monkeypatch, equivalent)

  assert reference.fps == Fraction(24, 1)

  mismatched = _synchronized_infos(
      compositor,
      rates=(
          Fraction(24, 1),
          Fraction(240000001, 10000000),
          Fraction(24, 1),
      ),
  )
  with pytest.raises(ValueError, match="fps|frame rate|synchronized"):
    _validate_fake_infos(compositor, monkeypatch, mismatched)


def test_source_duration_rejects_difference_above_one_microsecond(
    compositor, monkeypatch
):
  infos = _synchronized_infos(
      compositor,
      durations=(5.0, 5.0000011, 5.0),
  )

  with pytest.raises(ValueError, match="duration|synchronized"):
    _validate_fake_infos(compositor, monkeypatch, infos)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.pop("ground_truth"), "summary|ground_truth|keys"),
        (
            lambda value: value["branches"].__setitem__(
                "trajectory_changed", []
            ),
            "trajectory_changed|object",
        ),
        (
            lambda value: value["ground_truth"].__setitem__(
                "hard_affected", "upper_ball"
            ),
            "hard_affected",
        ),
        (
            lambda value: value["branches"]["trajectory_changed"].__setitem__(
                "contact_steps", []
            ),
            "contact_steps|contact",
        ),
        (
            lambda value: value.__setitem__("intervention_start", True),
            "intervention_start",
        ),
        (
            lambda value: value["ground_truth"]["graph_delta"]["added"][
                0
            ].pop("start_step"),
            "start_step|required",
        ),
        (
            lambda value: value["ground_truth"].__setitem__(
                "schema_version", "2.0"
            ),
            "schema_version",
        ),
        (
            lambda value: value["branches"]["target_removed"].__setitem__(
                "trust_model", "unknown"
            ),
            "trust_model",
        ),
    ),
)
def test_load_summary_rejects_missing_or_malformed_fields(
    compositor, tmp_path, mutation, message
):
  payload = _summary()
  mutation(payload)
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  with pytest.raises((TypeError, ValueError), match=message):
    compositor._load_summary(tmp_path)


def test_load_summary_reports_missing_and_invalid_json(compositor, tmp_path):
  with pytest.raises(FileNotFoundError, match="summary.json"):
    compositor._load_summary(tmp_path)

  (tmp_path / "summary.json").write_text("{", encoding="utf-8")
  with pytest.raises(ValueError, match="valid JSON"):
    compositor._load_summary(tmp_path)


def test_failed_encode_preserves_existing_output_and_removes_staging(
    compositor, tmp_path, monkeypatch
):
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  font = _font(tmp_path)
  output = tmp_path / "trajectory_intervention_demo.mp4"
  output.write_bytes(b"existing video")
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=24.0,
      frame_count=120,
      duration=5.0,
      codec_name="h264",
      pix_fmt="yuv420p",
  )
  monkeypatch.setattr(compositor, "_probe_video", lambda *args, **kwargs: info)
  monkeypatch.setattr(
      compositor.shutil,
      "which",
      lambda name: f"/tools/{name}",
  )

  def fail_encode(command):
    filter_graph = command[command.index("-filter_complex") + 1]
    textfiles = [Path(value) for value in re.findall(
        r"textfile=([^:]+):reload=0", filter_graph
    )]
    assert textfiles
    assert all(path.is_file() for path in textfiles)
    assert any(
        path.read_text("utf-8").startswith("CONTACT → UPPER BALL")
        for path in textfiles
    )
    overlay_directories = {path.parent for path in textfiles}
    assert len(overlay_directories) == 1
    observed_overlay_dirs.extend(overlay_directories)
    Path(command[-1]).write_bytes(b"partial")
    raise RuntimeError("synthetic ffmpeg failure")

  observed_overlay_dirs = []
  monkeypatch.setattr(compositor, "_run_ffmpeg", fail_encode)

  with pytest.raises(RuntimeError, match="synthetic ffmpeg failure"):
    compositor.compose_intervention_demo(tmp_path, output, font)

  assert output.read_bytes() == b"existing video"
  assert observed_overlay_dirs
  assert all(not path.exists() for path in observed_overlay_dirs)
  assert sorted(path.name for path in tmp_path.iterdir()) == [
      "Test Font.ttf",
      "normal_blender.mp4",
      "summary.json",
      "target_removed_blender.mp4",
      "trajectory_changed_blender.mp4",
      "trajectory_intervention_demo.mp4",
  ]


def test_input_and_output_symlinks_are_rejected(
    compositor, tmp_path, monkeypatch
):
  if not hasattr(os, "symlink"):
    pytest.skip("symlinks are unavailable")
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  font = _font(tmp_path)
  real_source = tmp_path / "normal_blender.mp4"
  real_source.rename(tmp_path / "normal-real.mp4")
  real_source.symlink_to(tmp_path / "normal-real.mp4")
  monkeypatch.setattr(
      compositor.shutil,
      "which",
      lambda name: f"/tools/{name}",
  )

  with pytest.raises(ValueError, match="symlink"):
    compositor.compose_intervention_demo(tmp_path, font=font)

  real_source.unlink()
  (tmp_path / "normal-real.mp4").rename(real_source)
  output = tmp_path / "trajectory_intervention_demo.mp4"
  target = tmp_path / "elsewhere.mp4"
  target.write_bytes(b"old")
  output.symlink_to(target)
  with pytest.raises(ValueError, match="symlink"):
    compositor.compose_intervention_demo(tmp_path, output, font)
  assert target.read_bytes() == b"old"


def _make_synthetic_video(ffmpeg, path, color):
  result = subprocess.run(
      [
          ffmpeg,
          "-hide_banner",
          "-loglevel",
          "error",
          "-f",
          "lavfi",
          "-i",
          f"color=c={color}:s=160x120:r=24:d=0.25",
          "-frames:v",
          "6",
          "-c:v",
          "libx264",
          "-pix_fmt",
          "yuv420p",
          "-movflags",
          "+faststart",
          str(path),
      ],
      check=False,
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, result.stderr


def test_real_ffmpeg_composes_six_frame_sources(compositor, tmp_path):
  ffmpeg = shutil.which("ffmpeg")
  ffprobe = shutil.which("ffprobe")
  if ffmpeg is None or ffprobe is None:
    pytest.skip("ffmpeg and ffprobe are required for integration coverage")
  font = compositor._resolve_font(None)
  for name, color in (
      ("normal", "red"),
      ("trajectory_changed", "green"),
      ("target_removed", "blue"),
  ):
    _make_synthetic_video(ffmpeg, tmp_path / f"{name}_blender.mp4", color)
  _write_summary(tmp_path, frame_count=6)
  output = tmp_path / "comparison.mp4"

  metadata = compositor.compose_intervention_demo(tmp_path, output, font)
  info = compositor._probe_video(output, ffprobe=ffprobe)

  assert metadata == {
      "duration": pytest.approx(3.25, abs=0.01),
      "frame_count": 78,
      "output": str(output),
      "size": [1920, 720],
  }
  assert info.width == 1920
  assert info.height == 720
  assert info.fps == pytest.approx(24.0)
  assert info.frame_count == 78
  assert info.duration == pytest.approx(3.25, abs=0.01)
  assert info.codec_name == "h264"
  assert info.pix_fmt == "yuv420p"


def test_cli_prints_compact_json_metadata(compositor, tmp_path, monkeypatch, capsys):
  expected = {
      "duration": 8.0,
      "frame_count": 192,
      "output": str(tmp_path / "result.mp4"),
      "size": [1920, 720],
  }
  monkeypatch.setattr(
      compositor,
      "compose_intervention_demo",
      lambda states_dir, output, font: expected,
  )

  assert compositor.main([
      "--states-dir",
      str(tmp_path),
      "--output",
      str(tmp_path / "result.mp4"),
  ]) == 0

  captured = capsys.readouterr()
  assert captured.err == ""
  assert captured.out == json.dumps(
      expected, sort_keys=True, separators=(",", ":")
  ) + "\n"
