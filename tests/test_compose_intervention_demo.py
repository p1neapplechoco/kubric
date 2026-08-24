"""Tests for the ffmpeg-only intervention comparison compositor."""

import hashlib
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

from scripts import trajectory_demo_spec as demo_spec

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

NORMAL_PAIRS = {
    "side_01|side_02": 1,
    "side_01|target": 1,
}
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


@pytest.fixture
def compositor():
  if compose_script is None:
    pytest.skip(f"missing compositor: {_SCRIPT_PATH}")
  return compose_script


def _summary():
  return {
      "branches": {
          "normal": {
              "contact_pairs": NORMAL_PAIRS.copy(),
              "contact_steps": [88],
          },
          "trajectory_changed": {
              "contact_pairs": CHANGED_PAIRS.copy(),
              "contact_steps": [70],
          },
          "target_removed": {
              "contact_pairs": {},
              "contact_steps": [],
              "removed_step": 40,
              "target_id": "target",
              "trust_model": "demo_only_removal_v1",
          },
      },
      "demo_spec": demo_spec.demo_spec_summary(demo_spec.FORKED_RACK_SPEC),
      "ground_truth": {
          "graph_delta": {
              "added": [{
                  "object_a": "breaker",
                  "object_b": "target",
                  "start_step": 70,
                  "end_step": 71,
              }],
              "changed": [],
              "removed": [{
                  "object_a": "side_01",
                  "object_b": "target",
                  "start_step": 88,
                  "end_step": 89,
              }],
              "schema_version": "1.0",
          },
          "hard_affected": [
              "breaker",
              "rack_01",
              "rack_02",
              "rack_03",
              "rack_04",
              "rack_05",
              "rack_06",
              "side_01",
              "side_02",
          ],
          "propagation_path": {
              "breaker": ["target", "breaker"],
              "rack_01": ["target", "breaker", "rack_01"],
              "rack_02": ["target", "breaker", "rack_01", "rack_02"],
              "rack_03": [
                  "target", "breaker", "rack_01", "rack_02", "rack_03"
              ],
              "rack_04": [
                  "target", "breaker", "rack_01", "rack_02", "rack_03",
                  "rack_04",
              ],
              "rack_05": [
                  "target", "breaker", "rack_01", "rack_02", "rack_03",
                  "rack_04", "rack_05",
              ],
              "rack_06": [
                  "target", "breaker", "rack_01", "rack_02", "rack_03",
                  "rack_04", "rack_05", "rack_06",
              ],
              "side_01": ["target", "side_01"],
              "side_02": ["target", "side_01", "side_02"],
          },
          "schema_version": "1.0",
          "soft_affected": [],
      },
      "intervention_end": 160,
      "intervention_start": 40,
      "intervention_window": [40, 160],
      "object_ids": list(demo_spec.FORKED_RACK_SPEC.object_ids),
      "seed": 0,
      "step_rate": 240.0,
  }


def _write_summary(directory):
  payload = _summary()
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


def _cfr_frames(frame_count, *, ticks_per_frame=512):
  return [
      {"best_effort_timestamp": index * ticks_per_frame}
      for index in range(frame_count)
  ]


def test_compositor_script_exists():
  assert compose_script is not None, f"missing compositor: {_SCRIPT_PATH}"


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
  payload = _write_summary(tmp_path)

  loaded = compositor._load_summary(tmp_path)

  assert loaded["demo_spec"] == payload["demo_spec"]
  assert loaded["demo_spec"] == demo_spec.demo_spec_summary(
      demo_spec.FORKED_RACK_SPEC
  )


def test_load_summary_rejects_stale_demo_spec(compositor, tmp_path):
  payload = _summary()
  payload["demo_spec"]["version"] = "forked_rack_v0"
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  with pytest.raises(ValueError, match=r"demo_spec\.version mismatch"):
    compositor._load_summary(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.__setitem__(
                "object_ids", list(reversed(value["object_ids"]))
            ),
            "object_ids mismatch",
        ),
        (
            lambda value: value["branches"]["target_removed"].__setitem__(
                "target_id", "breaker"
            ),
            "target_id mismatch",
        ),
        (lambda value: value.__setitem__("seed", 1), "seed mismatch"),
        (
            lambda value: value.__setitem__("step_rate", 241.0),
            "step_rate mismatch",
        ),
        (
            lambda value: (
                value.__setitem__("intervention_start", 41),
                value.__setitem__("intervention_window", [41, 160]),
            ),
            "intervention_start mismatch",
        ),
        (
            lambda value: (
                value.__setitem__("intervention_end", 159),
                value.__setitem__("intervention_window", [40, 159]),
            ),
            "intervention_end mismatch",
        ),
        (
            lambda value: value["branches"]["target_removed"].__setitem__(
                "removed_step", 41
            ),
            "removed_step mismatch",
        ),
    ),
)
def test_load_summary_rejects_demo_contract_metadata_mismatches(
    compositor, tmp_path, mutation, message
):
  payload = _summary()
  mutation(payload)
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  with pytest.raises(ValueError, match=message):
    compositor._load_summary(tmp_path)


@pytest.mark.parametrize(
    ("branch", "bad_pair", "message"),
    (
        ("normal", "side_01", "exactly two"),
        ("normal", "side_01|side_02|target", "exactly two"),
        ("normal", "ghost|side_01", "canonical object_ids"),
        ("normal", "side_01|side_01", "distinct"),
        ("normal", "target|side_01", "canonically ordered"),
        ("trajectory_changed", "target|breaker", "canonically ordered"),
    ),
)
def test_load_summary_rejects_malformed_contact_pair_keys(
    compositor, tmp_path, branch, bad_pair, message
):
  payload = _summary()
  pairs = payload["branches"][branch]["contact_pairs"]
  original = next(iter(pairs))
  pairs[bad_pair] = pairs.pop(original)
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  with pytest.raises(ValueError, match=message):
    compositor._load_summary(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["branches"]["normal"].__setitem__(
                "contact_pairs",
                {
                    "floor|side_01": 1,
                    "floor|side_02": 1,
                    "floor|target": 1,
                    "side_01|target": 1,
                },
            ),
            "normal branch must contain 2 to 3",
        ),
        (
            lambda value: value["branches"]["trajectory_changed"].__setitem__(
                "contact_pairs",
                dict(list(CHANGED_PAIRS.items())[:6]),
            ),
            "trajectory_changed branch must contain 7 to 9",
        ),
        (
            lambda value: value["branches"]["normal"].__setitem__(
                "contact_pairs",
                {"breaker|target": 1, "side_01|target": 1},
            ),
            "normal branch.*main-group",
        ),
        (
            lambda value: value["branches"]["trajectory_changed"].__setitem__(
                "contact_pairs",
                {
                    **{
                        pair: count
                        for pair, count in CHANGED_PAIRS.items()
                        if pair != "rack_05|rack_06"
                    },
                    "side_01|target": 1,
                },
            ),
            "trajectory_changed branch.*side-group",
        ),
        (
            lambda value: value["branches"]["trajectory_changed"].__setitem__(
                "contact_pairs",
                {
                    "breaker|target": 1,
                    "floor|rack_01": 1,
                    "floor|target": 1,
                    "rack_01|target": 1,
                    "rack_02|target": 1,
                    "rack_03|target": 1,
                    "rack_04|target": 1,
                },
            ),
            "trajectory_changed branch.*at least 6 main-group",
        ),
    ),
)
def test_load_summary_rejects_invalid_chain_pair_ranges_and_groups(
    compositor, tmp_path, mutation, message
):
  payload = _summary()
  mutation(payload)
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  with pytest.raises(ValueError, match=message):
    compositor._load_summary(tmp_path)


def test_load_summary_allows_only_prefix_contacts_for_removed_branch(
    compositor, tmp_path
):
  payload = _summary()
  removed = payload["branches"]["target_removed"]
  removed["contact_pairs"] = {"side_01|side_02": 1}
  removed["contact_steps"] = [39]
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  assert compositor._load_summary(tmp_path) == payload


@pytest.mark.parametrize("step", (40, 41, 199))
def test_load_summary_rejects_post_removal_contacts(
    compositor, tmp_path, step
):
  payload = _summary()
  removed = payload["branches"]["target_removed"]
  removed["contact_pairs"] = {"side_01|side_02": 1}
  removed["contact_steps"] = [step]
  (tmp_path / "summary.json").write_text(
      json.dumps(payload), encoding="utf-8"
  )

  with pytest.raises(ValueError, match="target_removed branch.*at or after"):
    compositor._load_summary(tmp_path)


def test_filter_times_both_chain_cues_for_nine_tenths(
    compositor, tmp_path
):
  overlay_dir = tmp_path / "overlays"
  overlay_dir.mkdir()
  summary = _summary()
  overlay_files = compositor._write_overlay_textfiles(
      overlay_dir,
      summary,
      source_duration=200 / 24,
      source_fps=24.0,
  )

  font = _font(tmp_path)
  filter_graph = compositor._build_filter(
      summary,
      font,
      source_duration=200 / 24,
      source_fps=24.0,
      overlay_files=overlay_files,
  )

  normal_time = compositor._event_time(88, 24.0)
  changed_time = compositor._event_time(70, 24.0)
  assert (
      f"between(t,{normal_time:.6f},{normal_time + 0.9:.6f})"
      in filter_graph
  )
  assert (
      f"between(t,{changed_time:.6f},{changed_time + 0.9:.6f})"
      in filter_graph
  )
  assert compositor._escape_filter_path(
      overlay_files["normal_chain"]
  ) in filter_graph
  assert compositor._escape_filter_path(
      overlay_files["changed_chain"]
  ) in filter_graph
  assert compositor._drawtext(
      font,
      textfile=overlay_files["normal_chain"],
      x="(640-text_w)/2",
      y="120",
      size=26,
      color="0xffd166",
      enable=f"between(t,{normal_time:.6f},{normal_time + 0.9:.6f})",
      box=True,
  ) in filter_graph
  assert compositor._drawtext(
      font,
      textfile=overlay_files["changed_chain"],
      x="640+(640-text_w)/2",
      y="120",
      size=26,
      color="0xffd166",
      enable=f"between(t,{changed_time:.6f},{changed_time + 0.9:.6f})",
      box=True,
  ) in filter_graph


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
      source_duration=200 / 24,
      source_fps=24.0,
  )
  filter_graph = compositor._build_filter(
      _summary(),
      _font(tmp_path),
      source_duration=200 / 24,
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
      "INTERVENTION 2.667s"
  )
  assert overlay_files["normal_chain"].read_text("utf-8") == (
      "SMALL CHAIN → SIDE 01 4.667s"
  )
  assert overlay_files["changed_chain"].read_text("utf-8") == (
      "LARGE CHAIN → BREAKER 3.917s"
  )
  assert overlay_files["removal"].read_text("utf-8") == (
      "TARGET REMOVED 2.667s"
  )
  assert overlay_files["graph"].read_text("utf-8") == (
      "GRAPH DELTA added=1 removed=1 changed=0"
  )
  assert overlay_files["affected"].read_text("utf-8") == (
      "AFFECTED hard=9 soft=0"
  )
  assert "target > breaker > rack_01" in overlay_files[
      "propagation"
  ].read_text(
      "utf-8"
  )


def test_overlay_textfiles_write_both_branch_specific_chain_cues(
    compositor, tmp_path
):
  overlay_files = compositor._write_overlay_textfiles(
      tmp_path,
      _summary(),
      source_duration=200 / 24,
      source_fps=24.0,
  )

  assert overlay_files["normal_chain"].read_text("utf-8") == (
      "SMALL CHAIN → SIDE 01 4.667s"
  )
  assert overlay_files["changed_chain"].read_text("utf-8") == (
      "LARGE CHAIN → BREAKER 3.917s"
  )


def _summary_with_earlier_unrelated_contact():
  summary = _summary()
  summary["branches"]["trajectory_changed"]["contact_steps"] = [12, 70]
  summary["ground_truth"]["graph_delta"]["added"].insert(0, {
      "object_a": "rack_01",
      "object_b": "rack_03",
      "start_step": 12,
      "end_step": 13,
  })
  return summary


def test_changed_chain_time_and_peer_come_from_same_graph_event(
    compositor, tmp_path
):
  summary = _summary_with_earlier_unrelated_contact()
  overlay_files = compositor._write_overlay_textfiles(
      tmp_path,
      summary,
      source_duration=200 / 24,
      source_fps=24.0,
  )
  filter_graph = compositor._build_filter(
      summary,
      _font(tmp_path),
      source_duration=200 / 24,
      source_fps=24.0,
      overlay_files=overlay_files,
  )

  assert overlay_files["changed_chain"].read_text("utf-8") == (
      "LARGE CHAIN → BREAKER 3.917s"
  )
  assert "enable='between(t,3.916667,4.816667)'" in filter_graph
  assert "enable='between(t,1.500000,2.400000)'" not in filter_graph


def test_event_bounds_validate_the_selected_changed_chain_event(compositor):
  summary = _summary_with_earlier_unrelated_contact()
  summary["branches"]["trajectory_changed"]["contact_steps"] = [12, 200]
  graph_event = summary["ground_truth"]["graph_delta"]["added"][1]
  graph_event["start_step"] = 200
  graph_event["end_step"] = 201

  with pytest.raises(ValueError, match="trajectory_changed chain event"):
    compositor._validate_event_steps(summary, frame_count=200)


def test_load_summary_rejects_graph_contact_step_mismatch(
    compositor, tmp_path
):
  summary = _summary_with_earlier_unrelated_contact()
  summary["ground_truth"]["graph_delta"]["added"][1]["start_step"] = 69
  (tmp_path / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
  )

  with pytest.raises(ValueError, match="bind|chain|step"):
    compositor._load_summary(tmp_path)


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
  textfile.write_text("LARGE CHAIN → BREAKER", encoding="utf-8")

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
  summary["ground_truth"]["hard_affected"].append(affected)
  summary["ground_truth"]["propagation_path"] = {
      **summary["ground_truth"]["propagation_path"],
      affected: [
          "target", "breaker", "rack_01", "rack_02", "rack_03",
          "rack_04", "rack_05", "rack_06", affected,
      ],
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
      "frames": _cfr_frames(200),
      "streams": [{
          "codec_name": "h264",
          "width": 640,
          "height": 540,
          "pix_fmt": "yuv420p",
          "avg_frame_rate": "24/1",
          "r_frame_rate": "24/1",
          "time_base": "1/12288",
          "nb_frames": "200",
          "duration": "8.333333",
      }],
      "format": {"duration": "8.333333"},
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
  assert info.frame_count == 200
  assert info.duration == pytest.approx(200 / 24, abs=1e-6)
  assert info.codec_name == "h264"
  assert info.pix_fmt == "yuv420p"
  assert calls[0][0][0] == "/tools/ffprobe"
  assert "-show_frames" in calls[0][0]
  assert calls[0][1]["check"] is True


def test_probe_video_prefers_counted_frames_over_declared_frames(
    compositor, tmp_path, monkeypatch
):
  video = tmp_path / "input.mp4"
  video.write_bytes(b"video")
  payload = {
      "frames": _cfr_frames(199),
      "streams": [{
          "codec_name": "h264",
          "width": 640,
          "height": 540,
          "pix_fmt": "yuv420p",
          "avg_frame_rate": "24/1",
          "r_frame_rate": "24/1",
          "time_base": "1/12288",
          "nb_frames": "200",
          "nb_read_frames": "199",
          "duration": "8.333333",
      }],
      "format": {"duration": "8.333333"},
  }
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda command, **kwargs: subprocess.CompletedProcess(
          command, 0, stdout=json.dumps(payload), stderr=""
      ),
  )

  info = compositor._probe_video(video, ffprobe="/tools/ffprobe")

  assert info.frame_count == 199


def test_probe_video_rejects_vfr_like_rate_metadata(
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
          "r_frame_rate": "30/1",
          "nb_frames": "200",
          "duration": "8.333333",
      }],
      "format": {"duration": "8.333333"},
  }
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda command, **kwargs: subprocess.CompletedProcess(
          command, 0, stdout=json.dumps(payload), stderr=""
      ),
  )

  with pytest.raises(ValueError, match="CFR|frame rate|r_frame_rate|VFR"):
    compositor._probe_video(video, ffprobe="/tools/ffprobe")


def test_probe_video_rejects_irregular_pts_when_aggregate_rates_equal(
    compositor, tmp_path, monkeypatch
):
  video = tmp_path / "input.mp4"
  video.write_bytes(b"video")
  frames = _cfr_frames(6)
  frames[3]["best_effort_timestamp"] += 128
  payload = {
      "frames": frames,
      "streams": [{
          "codec_name": "h264",
          "width": 640,
          "height": 540,
          "pix_fmt": "yuv420p",
          "avg_frame_rate": "24/1",
          "r_frame_rate": "24/1",
          "time_base": "1/12288",
          "nb_frames": "6",
          "nb_read_frames": "6",
          "duration": "0.25",
      }],
      "format": {"duration": "0.25"},
  }
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda command, **kwargs: subprocess.CompletedProcess(
          command, 0, stdout=json.dumps(payload), stderr=""
      ),
  )

  with pytest.raises(ValueError, match="CFR|timestamp|PTS|lattice"):
    compositor._probe_video(video, ffprobe="/tools/ffprobe")


def test_probe_video_rejects_time_base_too_coarse_for_24fps_lattice(
    compositor, tmp_path, monkeypatch
):
  video = tmp_path / "input.mp4"
  video.write_bytes(b"video")
  payload = {
      "frames": [
          {"best_effort_timestamp": timestamp}
          for timestamp in (0, 0, 1, 1)
      ],
      "streams": [{
          "codec_name": "h264",
          "width": 640,
          "height": 540,
          "pix_fmt": "yuv420p",
          "avg_frame_rate": "24/1",
          "r_frame_rate": "24/1",
          "time_base": "1/12",
          "nb_frames": "4",
          "nb_read_frames": "4",
          "duration": "0.166667",
      }],
      "format": {"duration": "0.166667"},
  }
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda command, **kwargs: subprocess.CompletedProcess(
          command, 0, stdout=json.dumps(payload), stderr=""
      ),
  )

  with pytest.raises(ValueError, match="time base|CFR|timestamp|lattice"):
    compositor._probe_video(video, ffprobe="/tools/ffprobe")


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
      "time_base": "1/12288",
      "nb_frames": "200",
      "duration": "8.333333",
  }
  stream[field] = value
  payload = {
      "frames": _cfr_frames(200),
      "streams": [stream],
      "format": {"duration": "8.333333"},
  }
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
      frame_count=200,
      duration=200 / 24,
      codec_name="h264",
      pix_fmt="yuv420p",
  )

  def fake_probe(path, ffprobe=None):
    if Path(path).name == "target_removed_blender.mp4":
      return compositor.VideoInfo(
          width=641,
          height=540,
          fps=24.0,
          frame_count=200,
          duration=200 / 24,
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


def test_synchronized_199_frame_sources_rejected_before_ffmpeg_or_output(
    compositor, tmp_path, monkeypatch
):
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  output = tmp_path / "final.mp4"
  font = _font(tmp_path)
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=199,
      duration=199 / 24,
      codec_name="h264",
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
      lambda command: pytest.fail("ffmpeg ran for 199-frame sources"),
  )

  with pytest.raises(
      ValueError, match=r"source frame count differs.*source_frames"
  ):
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
      frame_count=200,
      duration=200 / 24,
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
  durations = durations or (200 / 24,) * 3
  return {
      branch: compositor.VideoInfo(
          width=640,
          height=540,
          fps=rate,
          frame_count=200,
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


def test_sources_must_match_renderer_contract_of_exactly_24_fps(
    compositor, monkeypatch
):
  infos = _synchronized_infos(
      compositor,
      rates=(Fraction(23, 1),) * 3,
  )

  with pytest.raises(ValueError, match="24|renderer|frame rate|fps"):
    _validate_fake_infos(compositor, monkeypatch, infos)


def test_composed_timing_uses_source_frame_count_not_float_duration(
    compositor
):
  source = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=200,
      duration=200 / 24 + 0.021,
      codec_name="h264",
      pix_fmt="yuv420p",
  )
  composed = compositor.VideoInfo(
      width=1920,
      height=720,
      fps=Fraction(24, 1),
      frame_count=272,
      duration=272 / 24,
      codec_name="h264",
      pix_fmt="yuv420p",
  )

  compositor._validate_composed_video(composed, source)


def test_source_duration_rejects_difference_above_one_microsecond(
    compositor, monkeypatch
):
  infos = _synchronized_infos(
      compositor,
      durations=(200 / 24, 200 / 24 + 0.0000011, 200 / 24),
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


def test_compose_quantizes_overlay_duration_from_source_frame_count(
    compositor, tmp_path, monkeypatch
):
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  font = _font(tmp_path)
  output = tmp_path / "comparison.mp4"
  source_info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=200,
      duration=200 / 24 + 0.021,
      codec_name="h264",
      pix_fmt="yuv420p",
  )
  composed_info = compositor.VideoInfo(
      width=1920,
      height=720,
      fps=Fraction(24, 1),
      frame_count=272,
      duration=272 / 24,
      codec_name="h264",
      pix_fmt="yuv420p",
  )

  def fake_probe(path, ffprobe=None):
    return composed_info if Path(path).name.endswith(".tmp.mp4") else source_info

  def inspect_encode(command):
    filter_graph = command[command.index("-filter_complex") + 1]
    textfiles = [Path(value) for value in re.findall(
        r"textfile=([^:]+):reload=0", filter_graph
    )]
    assert any(path.read_text("utf-8") == "11.333s" for path in textfiles)
    Path(command[-1]).write_bytes(b"encoded")

  monkeypatch.setattr(compositor, "_probe_video", fake_probe)
  monkeypatch.setattr(
      compositor.shutil,
      "which",
      lambda name: f"/tools/{name}",
  )
  monkeypatch.setattr(compositor, "_run_ffmpeg", inspect_encode)

  metadata = compositor.compose_intervention_demo(tmp_path, output, font)

  assert metadata["frame_count"] == 272
  assert output.read_bytes() == b"encoded"


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
      frame_count=200,
      duration=200 / 24,
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
        path.read_text("utf-8").startswith("LARGE CHAIN → BREAKER")
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


def test_ffmpeg_has_bounded_timeout_and_reports_timeout_cleanly(
    compositor, monkeypatch
):
  observed = {}

  def time_out(command, **kwargs):
    observed.update(kwargs)
    raise subprocess.TimeoutExpired(
        command, kwargs.get("timeout"), stderr="synthetic hang"
    )

  monkeypatch.setattr(subprocess, "run", time_out)

  with pytest.raises(RuntimeError, match="ffmpeg.*timed out|timed out.*ffmpeg"):
    compositor._run_ffmpeg(["/tools/ffmpeg", "-version"])

  assert 0 < observed["timeout"] <= 3600
  assert observed["check"] is True
  assert observed["capture_output"] is True
  assert observed["text"] is True


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


def test_states_dir_rejects_symlink_ancestor_before_ffmpeg(
    compositor, tmp_path, monkeypatch
):
  if not hasattr(os, "symlink"):
    pytest.skip("symlinks are unavailable")
  real_root = tmp_path / "real"
  states = real_root / "states"
  states.mkdir(parents=True)
  _write_summary(states)
  _touch_sources(states)
  font = _font(tmp_path)
  alias = tmp_path / "alias"
  alias.symlink_to(real_root, target_is_directory=True)
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=200,
      duration=200 / 24,
      codec_name="h264",
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
      lambda command: pytest.fail("ffmpeg ran through a symlink ancestor"),
  )

  with pytest.raises(ValueError, match="symlink"):
    compositor.compose_intervention_demo(alias / "states", font=font)


def test_output_rejects_symlink_ancestor_canonical_source_alias_before_ffmpeg(
    compositor, tmp_path, monkeypatch
):
  if not hasattr(os, "symlink"):
    pytest.skip("symlinks are unavailable")
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  (tmp_path / "nested").mkdir()
  font = _font(tmp_path)
  alias = tmp_path.parent / f"{tmp_path.name}-alias"
  alias.symlink_to(tmp_path, target_is_directory=True)
  output = alias / "nested" / ".." / "normal_blender.mp4"
  source = tmp_path / "normal_blender.mp4"
  source_bytes = source.read_bytes()
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=200,
      duration=200 / 24,
      codec_name="h264",
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
      lambda command: pytest.fail("ffmpeg ran for a canonical source alias"),
  )

  try:
    with pytest.raises(ValueError, match="symlink|source|overwrite|alias"):
      compositor.compose_intervention_demo(tmp_path, output, font)
  finally:
    alias.unlink(missing_ok=True)

  assert source.read_bytes() == source_bytes


def test_output_rejects_source_hardlink_before_ffmpeg(
    compositor, tmp_path, monkeypatch
):
  if not hasattr(os, "link"):
    pytest.skip("hard links are unavailable")
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  font = _font(tmp_path)
  source = tmp_path / "normal_blender.mp4"
  source_bytes = source.read_bytes()
  output = tmp_path / "hardlink-output.mp4"
  os.link(source, output)
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=200,
      duration=200 / 24,
      codec_name="h264",
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
      lambda command: pytest.fail("ffmpeg ran for a source hardlink"),
  )

  with pytest.raises(ValueError, match="source|overwrite|same file|hardlink"):
    compositor.compose_intervention_demo(tmp_path, output, font)

  assert source.read_bytes() == source_bytes


def test_output_is_revalidated_before_publish_and_preserves_source_hash(
    compositor, tmp_path, monkeypatch
):
  if not hasattr(os, "link"):
    pytest.skip("hard links are unavailable")
  _write_summary(tmp_path)
  _touch_sources(tmp_path)
  font = _font(tmp_path)
  source = tmp_path / "normal_blender.mp4"
  source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
  output = tmp_path / "late-hardlink-output.mp4"
  info = compositor.VideoInfo(
      width=640,
      height=540,
      fps=Fraction(24, 1),
      frame_count=200,
      duration=200 / 24,
      codec_name="h264",
      pix_fmt="yuv420p",
  )
  composed_info = compositor.VideoInfo(
      width=1920,
      height=720,
      fps=Fraction(24, 1),
      frame_count=272,
      duration=272 / 24,
      codec_name="h264",
      pix_fmt="yuv420p",
  )

  def fake_probe(path, ffprobe=None):
    if Path(path).name in {
        "normal_blender.mp4",
        "trajectory_changed_blender.mp4",
        "target_removed_blender.mp4",
    }:
      return info
    return composed_info

  monkeypatch.setattr(compositor, "_probe_video", fake_probe)
  monkeypatch.setattr(
      compositor.shutil,
      "which",
      lambda name: f"/tools/{name}",
  )

  def create_late_alias(command):
    Path(command[-1]).write_bytes(b"encoded staging video")
    os.link(source, output)

  monkeypatch.setattr(compositor, "_run_ffmpeg", create_late_alias)

  with pytest.raises(ValueError, match="source|overwrite|same file|hardlink"):
    compositor.compose_intervention_demo(tmp_path, output, font)

  assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


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
          f"color=c={color}:s=160x120:r=24:d={200 / 24}",
          "-frames:v",
          "200",
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


def test_real_ffmpeg_composes_full_demo_frame_count(compositor, tmp_path):
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
  _write_summary(tmp_path)
  output = tmp_path / "comparison.mp4"

  metadata = compositor.compose_intervention_demo(tmp_path, output, font)
  info = compositor._probe_video(output, ffprobe=ffprobe)

  assert metadata == {
      "duration": pytest.approx(272 / 24, abs=0.01),
      "frame_count": 272,
      "output": str(output),
      "size": [1920, 720],
  }
  assert info.width == 1920
  assert info.height == 720
  assert info.fps == pytest.approx(24.0)
  assert info.frame_count == 272
  assert info.duration == pytest.approx(272 / 24, abs=0.01)
  assert info.codec_name == "h264"
  assert info.pix_fmt == "yuv420p"


def test_cli_prints_compact_json_metadata(compositor, tmp_path, monkeypatch, capsys):
  expected = {
      "duration": 272 / 24,
      "frame_count": 272,
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
