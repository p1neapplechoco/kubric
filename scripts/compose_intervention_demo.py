"""Compose three canonical Blender replays into one comparison MP4.

Purpose: validate synchronized branch media/metadata and atomically produce the
labelled 1920x720 comparison with contact/removal and graph-summary overlays.
Public API: VideoInfo, compose_intervention_demo(), and main().
Dependencies: Python metadata validation plus external ffprobe/ffmpeg binaries;
trajectory_demo_spec supplies the exact canonical scene identity.
Trust boundary: exact digest, replay, event, codec, frame, and duration contracts
gate composition; the compositor does not rerun physics or attest source origin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.trajectory_demo_spec import FORKED_RACK_SPEC, demo_spec_summary


_BRANCH_FILES = (
    ("normal", "normal_blender.mp4"),
    ("trajectory_changed", "trajectory_changed_blender.mp4"),
    ("target_removed", "target_removed_blender.mp4"),
)
_TOP_LEVEL_KEYS = frozenset({
    "branches",
    "demo_spec",
    "ground_truth",
    "intervention_end",
    "intervention_start",
    "intervention_window",
    "object_ids",
    "seed",
    "step_rate",
})
_GROUND_TRUTH_KEYS = frozenset({
    "graph_delta",
    "hard_affected",
    "propagation_path",
    "schema_version",
    "soft_affected",
})
_GRAPH_DELTA_KEYS = frozenset({
    "added",
    "changed",
    "removed",
    "schema_version",
})
_CONTACT_KEYS = frozenset({"contact_pairs", "contact_steps"})
_REMOVAL_KEYS = _CONTACT_KEYS | frozenset({
    "removed_step",
    "target_id",
    "trust_model",
})
_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)
_OUTPUT_WIDTH = 1920
_OUTPUT_HEIGHT = 720
_PANEL_WIDTH = 640
_PANEL_HEIGHT = 540
_OUTPUT_FPS = Fraction(24, 1)
_START_HOLD_DURATION = Fraction(1, 1)
_END_HOLD_DURATION = Fraction(2, 1)
_CHAIN_CUE_DURATION = Fraction(9, 10)
_REMOVAL_CUE_DURATION = Fraction(3, 4)
_START_HOLD = float(_START_HOLD_DURATION)
_END_HOLD = float(_END_HOLD_DURATION)
_SOURCE_DURATION_TOLERANCE = 1e-6
_FFMPEG_TIMEOUT_SECONDS = 600
_SCHEMA_VERSION = "1.0"
_REMOVAL_TRUST_MODEL = "demo_only_removal_v1"


@dataclass(frozen=True)
class VideoInfo:
  """Validated metadata for the first video stream in an MP4."""

  width: int
  height: int
  fps: Fraction
  frame_count: int
  duration: float
  codec_name: str
  pix_fmt: str


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
  actual = set(value)
  if actual != expected:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ValueError(
        f"{name} keys are invalid; missing={missing!r}, "
        f"unexpected={unexpected!r}"
    )


def _require_object(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise TypeError(f"{name} must be a JSON object")
  return value


def _validate_demo_spec_identity(value: Any) -> Mapping[str, Any]:
  """Reject summary metadata produced from a different scene contract."""
  if not isinstance(value, dict):
    raise TypeError("demo_spec must be a JSON object")
  expected = demo_spec_summary(FORKED_RACK_SPEC)
  if set(value) != set(expected):
    raise ValueError("demo_spec keys mismatch")
  for field, expected_value in expected.items():
    if value[field] != expected_value:
      raise ValueError(f"demo_spec.{field} mismatch")
  return value


def _require_nonempty_string(value: Any, name: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise TypeError(f"{name} must be a nonempty string")
  return value


def _require_integer(value: Any, name: str, *, minimum: int = 0) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ValueError(f"{name} must be an integer >= {minimum}")
  return value


def _require_string_list(value: Any, name: str) -> list[str]:
  if not isinstance(value, list):
    raise TypeError(f"{name} must be a JSON array of strings")
  result = [
      _require_nonempty_string(item, f"{name} item") for item in value
  ]
  if len(set(result)) != len(result):
    raise ValueError(f"{name} must not contain duplicate identifiers")
  return result


def _require_step_list(value: Any, name: str) -> list[int]:
  if not isinstance(value, list):
    raise TypeError(f"{name} must be a JSON array of steps")
  result = [
      _require_integer(item, f"{name} item") for item in value
  ]
  if result != sorted(set(result)):
    raise ValueError(f"{name} must contain sorted, unique steps")
  return result


def _validate_contact_pair_key(pair: str, name: str) -> tuple[str, str]:
  endpoints = pair.split("|")
  if len(endpoints) != 2:
    raise ValueError(
        f"{name} must contain exactly two object IDs separated by '|'"
    )
  object_a, object_b = endpoints
  canonical_ids = set(FORKED_RACK_SPEC.object_ids)
  if object_a not in canonical_ids or object_b not in canonical_ids:
    raise ValueError(f"{name} endpoints must name canonical object_ids")
  if object_a == object_b:
    raise ValueError(f"{name} endpoints must be distinct")
  if object_a > object_b:
    raise ValueError(f"{name} endpoints must be canonically ordered")
  return object_a, object_b


def _validate_contact_summary(
    value: Any, name: str, *, removal: bool = False
) -> dict[str, Any]:
  branch = _require_object(value, name)
  _require_exact_keys(
      branch, _REMOVAL_KEYS if removal else _CONTACT_KEYS, name
  )
  pairs = _require_object(branch["contact_pairs"], f"{name}.contact_pairs")
  for pair, count in pairs.items():
    _require_nonempty_string(pair, f"{name}.contact_pairs key")
    _validate_contact_pair_key(pair, f"{name}.contact_pairs[{pair!r}]")
    _require_integer(count, f"{name}.contact_pairs[{pair!r}]", minimum=1)
  _require_step_list(branch["contact_steps"], f"{name}.contact_steps")
  if removal:
    _require_integer(branch["removed_step"], f"{name}.removed_step")
    _require_nonempty_string(branch["target_id"], f"{name}.target_id")
    if branch["trust_model"] != _REMOVAL_TRUST_MODEL:
      raise ValueError(
          f"{name}.trust_model must be {_REMOVAL_TRUST_MODEL!r}"
      )
  return branch


def _validate_graph_records(graph_delta: Mapping[str, Any]) -> None:
  owners: dict[tuple[str, str, int, int], str] = {}
  for bucket in ("added", "removed", "changed"):
    records = graph_delta[bucket]
    if not isinstance(records, list):
      raise TypeError(
          f"ground_truth.graph_delta.{bucket} must be a JSON array of objects"
      )
    for index, record in enumerate(records):
      name = f"ground_truth.graph_delta.{bucket}[{index}]"
      if not isinstance(record, dict):
        raise TypeError(f"{name} must be a JSON object")
      required = {"object_a", "object_b", "start_step", "end_step"}
      missing = sorted(required - set(record))
      if missing:
        raise ValueError(f"{name} missing required fields {missing!r}")
      object_a = _require_nonempty_string(record["object_a"], f"{name}.object_a")
      object_b = _require_nonempty_string(record["object_b"], f"{name}.object_b")
      if object_a >= object_b:
        raise ValueError(
            f"{name} endpoints must be distinct and canonically ordered"
        )
      start = _require_integer(record["start_step"], f"{name}.start_step")
      end = _require_integer(record["end_step"], f"{name}.end_step")
      if end <= start:
        raise ValueError(f"{name} steps must satisfy start_step < end_step")
      identity = (object_a, object_b, start, end)
      previous = owners.get(identity)
      if previous is not None:
        raise ValueError(
            f"graph edge {identity!r} appears in both {previous} and {bucket}"
        )
      owners[identity] = bucket


def _validate_ground_truth(value: Any) -> dict[str, Any]:
  truth = _require_object(value, "ground_truth")
  _require_exact_keys(truth, _GROUND_TRUTH_KEYS, "ground_truth")
  graph_delta = _require_object(truth["graph_delta"], "ground_truth.graph_delta")
  _require_exact_keys(
      graph_delta, _GRAPH_DELTA_KEYS, "ground_truth.graph_delta"
  )
  if graph_delta["schema_version"] != _SCHEMA_VERSION:
    raise ValueError(
        "ground_truth.graph_delta.schema_version must be "
        f"{_SCHEMA_VERSION!r}"
    )
  _validate_graph_records(graph_delta)

  hard = _require_string_list(
      truth["hard_affected"], "ground_truth.hard_affected"
  )
  soft = _require_string_list(
      truth["soft_affected"], "ground_truth.soft_affected"
  )
  if set(hard).intersection(soft):
    raise ValueError("ground_truth hard_affected and soft_affected must be disjoint")
  paths = _require_object(
      truth["propagation_path"], "ground_truth.propagation_path"
  )
  for affected, path in paths.items():
    _require_nonempty_string(
        affected, "ground_truth.propagation_path key"
    )
    items = _require_string_list(
        path, f"ground_truth.propagation_path[{affected!r}]"
    )
    if not items or items[-1] != affected:
      raise ValueError(
          "ground_truth propagation paths must be nonempty and end at their key"
      )
  if set(paths) != set(hard):
    raise ValueError(
        "ground_truth.propagation_path keys must equal hard_affected"
    )
  if truth["schema_version"] != _SCHEMA_VERSION:
    raise ValueError(
        f"ground_truth.schema_version must be {_SCHEMA_VERSION!r}"
    )
  return truth


def _contact_pair_endpoints(pairs: Mapping[str, Any]) -> set[str]:
  return {
      endpoint
      for pair in pairs
      for endpoint in pair.split("|")
  }


def _validate_demo_branch_contract(summary: Mapping[str, Any]) -> None:
  """Validate the small normal fork and large changed fork signatures."""
  branches = summary["branches"]
  normal_pairs = branches["normal"]["contact_pairs"]
  changed_pairs = branches["trajectory_changed"]["contact_pairs"]
  normal_count = len(normal_pairs)
  changed_count = len(changed_pairs)
  if not 2 <= normal_count <= 3:
    raise ValueError("normal branch must contain 2 to 3 unique contact pairs")
  if not 7 <= changed_count <= 9:
    raise ValueError(
        "trajectory_changed branch must contain 7 to 9 unique contact pairs"
    )
  if changed_count < normal_count + 5:
    raise ValueError(
        "trajectory_changed branch must contain at least 5 more unique "
        "contact pairs than normal"
    )
  normal_endpoints = _contact_pair_endpoints(normal_pairs)
  changed_endpoints = _contact_pair_endpoints(changed_pairs)
  if normal_endpoints.intersection(FORKED_RACK_SPEC.main_ball_ids):
    raise ValueError(
        "normal branch contact pairs must not include main-group balls"
    )
  if changed_endpoints.intersection(FORKED_RACK_SPEC.side_ball_ids):
    raise ValueError(
        "trajectory_changed branch contact pairs must not include "
        "side-group balls"
    )
  if len(changed_endpoints.intersection(FORKED_RACK_SPEC.main_ball_ids)) < 6:
    raise ValueError(
        "trajectory_changed branch contact pairs must reach at least 6 "
        "main-group balls"
    )
  intervention_start = summary["intervention_start"]
  removed_steps = branches["target_removed"]["contact_steps"]
  if any(step >= intervention_start for step in removed_steps):
    raise ValueError(
        "target_removed branch must not contain contact steps at or after "
        "intervention_start"
    )


def _require_regular_input(path: Path, name: str) -> None:
  _reject_symlink_components(path, name)
  if not path.exists():
    raise FileNotFoundError(f"missing {name}: {path}")
  if not path.is_file():
    raise ValueError(f"{name} must be a regular file: {path}")


def _reject_symlink_components(path: str | Path, name: str) -> None:
  """Reject every existing symlink traversed by a lexical path."""
  candidate = Path(path)
  if not candidate.is_absolute():
    candidate = Path.cwd() / candidate
  current = Path(candidate.anchor)
  for component in candidate.parts[1:]:
    if component in ("", "."):
      continue
    if component == "..":
      current = current.parent
      continue
    current = current / component
    try:
      mode = current.lstat().st_mode
    except FileNotFoundError:
      continue
    if stat.S_ISLNK(mode):
      raise ValueError(f"{name} must not traverse a symlink: {current}")


def _load_summary(states_dir: str | Path) -> dict[str, Any]:
  """Load and strictly validate the generator's summary contract."""
  directory = Path(states_dir)
  summary_path = directory / "summary.json"
  _require_regular_input(summary_path, "summary.json")
  try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
  except (json.JSONDecodeError, UnicodeDecodeError) as error:
    raise ValueError(f"{summary_path} must contain valid JSON") from error
  summary = _require_object(summary, "summary")
  _require_exact_keys(summary, _TOP_LEVEL_KEYS, "summary")
  _validate_demo_spec_identity(summary["demo_spec"])

  object_ids = _require_string_list(summary["object_ids"], "object_ids")
  if object_ids != list(FORKED_RACK_SPEC.object_ids):
    raise ValueError("object_ids mismatch")
  seed = summary["seed"]
  _require_integer(seed, "seed")
  if seed != FORKED_RACK_SPEC.seed:
    raise ValueError("seed mismatch")
  step_rate = summary["step_rate"]
  if (
      isinstance(step_rate, bool)
      or not isinstance(step_rate, (int, float))
      or not math.isfinite(step_rate)
      or step_rate <= 0
  ):
    raise ValueError("step_rate must be a positive finite number")
  if step_rate != FORKED_RACK_SPEC.step_rate:
    raise ValueError("step_rate mismatch")

  start = _require_integer(summary["intervention_start"], "intervention_start")
  end = _require_integer(summary["intervention_end"], "intervention_end")
  if end <= start:
    raise ValueError("intervention_end must be greater than intervention_start")
  window = summary["intervention_window"]
  if window != [start, end]:
    raise ValueError(
        f"intervention_window must exactly equal {[start, end]!r}"
    )
  expected_start, expected_end = FORKED_RACK_SPEC.intervention_window
  if start != expected_start:
    raise ValueError("intervention_start mismatch")
  if end != expected_end:
    raise ValueError("intervention_end mismatch")

  branches = _require_object(summary["branches"], "branches")
  expected_branches = frozenset(name for name, _ in _BRANCH_FILES)
  _require_exact_keys(branches, expected_branches, "branches")
  _validate_contact_summary(branches["normal"], "branches.normal")
  changed = _validate_contact_summary(
      branches["trajectory_changed"], "branches.trajectory_changed"
  )
  if not changed["contact_steps"]:
    raise ValueError(
        "branches.trajectory_changed.contact_steps must contain a contact cue"
    )
  removed = _validate_contact_summary(
      branches["target_removed"], "branches.target_removed", removal=True
  )
  if removed["target_id"] != FORKED_RACK_SPEC.target_id:
    raise ValueError("branches.target_removed.target_id mismatch")
  if removed["removed_step"] != expected_start:
    raise ValueError("branches.target_removed.removed_step mismatch")
  _validate_ground_truth(summary["ground_truth"])
  _validate_demo_branch_contract(summary)
  _chain_cue_event(summary, "normal")
  _chain_cue_event(summary, "trajectory_changed")
  return summary


def _positive_int(value: Any, name: str) -> int:
  if isinstance(value, bool):
    raise ValueError(f"video {name} must be a positive integer")
  try:
    result = int(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"video {name} must be a positive integer") from error
  if result <= 0 or str(value).strip() not in {str(result), f"+{result}"}:
    raise ValueError(f"video {name} must be a positive integer")
  return result


def _positive_float(value: Any, name: str) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"video {name} must be a positive number") from error
  if not math.isfinite(result) or result <= 0:
    raise ValueError(f"video {name} must be a positive finite number")
  return result


def _positive_fps(value: Any) -> Fraction:
  if not isinstance(value, str):
    raise ValueError("video fps must be a positive rational number")
  try:
    result = Fraction(value)
  except (ValueError, ZeroDivisionError) as error:
    raise ValueError("video fps must be a positive rational number") from error
  if result <= 0:
    raise ValueError("video fps must be a positive rational number")
  return result


def _probe_video(path: str | Path, ffprobe: str | None = None) -> VideoInfo:
  """Probe and validate one video's first stream using ffprobe JSON output."""
  video_path = Path(path)
  _require_regular_input(video_path, "video input")
  executable = ffprobe or shutil.which("ffprobe")
  if executable is None:
    raise RuntimeError("ffprobe was not found on PATH")
  command = [
      executable,
      "-v",
      "error",
      "-count_frames",
      "-select_streams",
      "v:0",
      "-show_frames",
      "-show_entries",
      (
          "stream=codec_name,width,height,pix_fmt,avg_frame_rate,"
          "r_frame_rate,time_base,nb_frames,nb_read_frames,duration:"
          "frame=best_effort_timestamp:format=duration"
      ),
      "-of",
      "json",
      str(video_path),
  ]
  try:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
  except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
    detail = getattr(error, "stderr", "") or str(error)
    raise RuntimeError(f"ffprobe failed for {video_path}: {detail.strip()}") from error
  try:
    payload = json.loads(completed.stdout)
  except json.JSONDecodeError as error:
    raise ValueError(f"ffprobe returned invalid JSON for {video_path}") from error
  if not isinstance(payload, dict):
    raise ValueError(f"ffprobe returned no JSON object for {video_path}")
  streams = payload.get("streams")
  if not isinstance(streams, list) or not streams:
    raise ValueError(f"{video_path} has no video stream")
  stream = streams[0]
  if not isinstance(stream, dict):
    raise ValueError(f"{video_path} has malformed video stream metadata")

  width = _positive_int(stream.get("width"), "width")
  height = _positive_int(stream.get("height"), "height")
  fps = _positive_fps(stream.get("avg_frame_rate"))
  nominal_fps = _positive_fps(stream.get("r_frame_rate"))
  if nominal_fps != fps:
    raise ValueError(
        f"video must be CFR; avg_frame_rate={fps}, "
        f"r_frame_rate={nominal_fps}"
    )
  frame_value = stream.get("nb_read_frames")
  if frame_value in (None, "N/A"):
    frame_value = stream.get("nb_frames")
  frame_count = _positive_int(frame_value, "frame count")
  time_base = _positive_fps(stream.get("time_base"))
  frame_period = Fraction(1, 1) / fps
  if time_base > frame_period:
    raise ValueError(
        f"video time base {time_base} is too coarse for CFR {fps}"
    )
  frames = payload.get("frames")
  if not isinstance(frames, list) or len(frames) != frame_count:
    raise ValueError(
        f"video frame timestamps must contain {frame_count} entries"
    )
  # An integer PTS can round a desired presentation time by at most half of
  # one stream time-base tick.  Rational arithmetic avoids float drift here.
  timestamp_tolerance = time_base / 2
  for index, frame in enumerate(frames):
    if not isinstance(frame, dict):
      raise ValueError(f"video frame timestamp {index} is malformed")
    timestamp = _require_integer(
        frame.get("best_effort_timestamp"),
        f"video frame timestamp {index}",
    )
    actual_time = timestamp * time_base
    expected_time = index * frame_period
    if abs(actual_time - expected_time) > timestamp_tolerance:
      raise ValueError(
          "video must be CFR on a contiguous timestamp lattice; "
          f"frame {index} has PTS {actual_time}, expected {expected_time}"
      )
  duration_value = stream.get("duration")
  if duration_value in (None, "N/A"):
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
      raise ValueError(f"video duration is missing for {video_path}")
    duration_value = format_info.get("duration")
  duration = _positive_float(duration_value, "duration")
  codec_name = _require_nonempty_string(
      stream.get("codec_name"), "video codec_name"
  )
  pix_fmt = _require_nonempty_string(stream.get("pix_fmt"), "video pix_fmt")
  return VideoInfo(
      width=width,
      height=height,
      fps=fps,
      frame_count=frame_count,
      duration=duration,
      codec_name=codec_name,
      pix_fmt=pix_fmt,
  )


def _resolve_font(font: str | Path | None) -> Path:
  """Return an explicit font or the first deterministic system fallback."""
  if font is not None:
    path = Path(font).expanduser()
    if not path.is_file():
      raise FileNotFoundError(f"font file does not exist: {path}")
    return path
  for candidate in _FONT_CANDIDATES:
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(
      "no supported font found; pass --font with a Noto or DejaVu TTF"
  )


def _escape_drawtext(text: str) -> str:
  """Escape text for a single-quoted ffmpeg drawtext option."""
  replacements = (
      ("\\", "\\\\"),
      (":", "\\:"),
      (",", "\\,"),
      ("%", "\\%"),
      ("'", "'\\''"),
      ("[", "\\["),
      ("]", "\\]"),
  )
  escaped = str(text)
  for source, replacement in replacements:
    escaped = escaped.replace(source, replacement)
  return escaped


def _escape_filter_path(path: str | Path) -> str:
  """Escape a path through both filtergraph and drawtext option parsers."""
  escaped = str(path).replace("\\", "\\\\\\\\")
  for character in ("'", ":", ",", ";", "[", "]", "%"):
    escaped = escaped.replace(character, "\\\\\\" + character)
  return escaped


def _drawtext(
    font: Path,
    text: str | None = None,
    *,
    textfile: str | Path | None = None,
    x: str,
    y: str,
    size: int,
    color: str = "white",
    enable: str | None = None,
    box: bool = False,
) -> str:
  if (text is None) == (textfile is None):
    raise ValueError("drawtext requires exactly one of text or textfile")
  options = [f"fontfile={_escape_filter_path(font)}"]
  if textfile is not None:
    options.extend((
        f"textfile={_escape_filter_path(textfile)}",
        "reload=0",
    ))
  else:
    options.append(f"text='{_escape_drawtext(text)}'")
  options.extend([
      "expansion=none",
      f"x={x}",
      f"y={y}",
      f"fontsize={size}",
      f"fontcolor={color}",
  ])
  if box:
    options.extend(("box=1", "boxcolor=black@0.72", "boxborderw=10"))
  if enable is not None:
    options.append(f"enable='{enable}'")
  return "drawtext=" + ":".join(options)


def _event_time(step: int, source_fps: float) -> float:
  return _START_HOLD + step / source_fps


def _exact_duration_frames(duration: Fraction) -> int:
  frames = duration * _OUTPUT_FPS
  if frames.denominator != 1:
    raise ValueError("duration must equal an integral number of output frames")
  return frames.numerator


def _event_output_frame(step: int) -> int:
  """Map one source event step to its exact post-hold output frame."""
  return _exact_duration_frames(_START_HOLD_DURATION) + step


def _frame_window_enable(start_frame: int, duration: Fraction) -> str:
  """Return an inclusive expression for a half-open frame-duration window."""
  duration_frames = duration * _OUTPUT_FPS
  displayed_frames = (
      duration_frames.numerator + duration_frames.denominator - 1
  ) // duration_frames.denominator
  end_frame = start_frame + displayed_frames - 1
  return f"between(n,{start_frame},{end_frame})"


def _summary_overlay_lines(summary: Mapping[str, Any]) -> tuple[str, str, str]:
  truth = summary["ground_truth"]
  graph_delta = truth["graph_delta"]
  graph_line = (
      "GRAPH DELTA "
      f"added={len(graph_delta['added'])} "
      f"removed={len(graph_delta['removed'])} "
      f"changed={len(graph_delta['changed'])}"
  )
  affected_line = (
      f"AFFECTED hard={len(truth['hard_affected'])} "
      f"soft={len(truth['soft_affected'])}"
  )
  paths = truth["propagation_path"]
  if not paths:
    propagation_line = "MAX PROPAGATION 0 HOPS none"
  else:
    _, path = min(
        paths.items(), key=lambda item: (-(len(item[1]) - 1), item[0])
    )
    hops = len(path) - 1
    propagation_line = f"MAX PROPAGATION {hops} HOPS {' > '.join(path)}"
  return graph_line, affected_line, propagation_line


def _chain_cue_event(
    summary: Mapping[str, Any], branch: str
) -> tuple[str, int]:
  """Return the branch-specific peer and step bound to one graph delta."""
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
        if (
            normal_only
            and pair_key in branch_pairs
            and record["start_step"] in branch_steps
        ):
          candidates.append((record["start_step"], expected_peer))
  if not candidates:
    raise ValueError(f"summary cannot bind the {branch} chain cue")
  step, peer = min(candidates)
  return peer, step


def _overlay_texts(
    summary: Mapping[str, Any],
    *,
    source_duration: float,
    source_fps: float,
) -> dict[str, str]:
  total_duration = source_duration + _START_HOLD + _END_HOLD
  intervention_time = _event_time(summary["intervention_start"], source_fps)
  normal_object, normal_step = _chain_cue_event(summary, "normal")
  changed_object, changed_step = _chain_cue_event(
      summary, "trajectory_changed"
  )
  normal_time = _event_time(normal_step, source_fps)
  changed_time = _event_time(changed_step, source_fps)
  removal_step = summary["branches"]["target_removed"]["removed_step"]
  removal_time = _event_time(removal_step, source_fps)
  normal_object = normal_object.replace("_", " ").upper()
  changed_object = changed_object.replace("_", " ").upper()
  graph_line, affected_line, propagation_line = _summary_overlay_lines(summary)
  return {
      "intervention": f"INTERVENTION {intervention_time:.3f}s",
      "duration": f"{total_duration:.3f}s",
      "normal_chain": f"SMALL CHAIN → {normal_object} {normal_time:.3f}s",
      "changed_chain": (
          f"LARGE CHAIN → {changed_object} {changed_time:.3f}s"
      ),
      "removal": f"TARGET REMOVED {removal_time:.3f}s",
      "graph": graph_line,
      "affected": affected_line,
      "propagation": propagation_line,
  }


def _write_overlay_textfiles(
    directory: str | Path,
    summary: Mapping[str, Any],
    *,
    source_duration: float,
    source_fps: float,
) -> dict[str, Path]:
  overlay_dir = Path(directory)
  if overlay_dir.is_symlink() or not overlay_dir.is_dir():
    raise ValueError(f"overlay directory must be a real directory: {overlay_dir}")
  texts = _overlay_texts(
      summary,
      source_duration=source_duration,
      source_fps=source_fps,
  )
  result = {}
  for name, value in texts.items():
    path = overlay_dir / f"{name}.txt"
    path.write_text(value, encoding="utf-8")
    result[name] = path
  return result


def _build_filter(
    summary: Mapping[str, Any],
    font: str | Path,
    *,
    source_duration: float = 5.0,
    source_fps: float = 24.0,
    overlay_files: Mapping[str, Path],
) -> str:
  """Build the deterministic layout and annotation filter graph."""
  if not math.isfinite(source_duration) or source_duration <= 0:
    raise ValueError("source_duration must be positive and finite")
  if not math.isfinite(source_fps) or source_fps <= 0:
    raise ValueError("source_fps must be positive and finite")
  font_path = Path(font)
  expected_overlays = frozenset(_overlay_texts(
      summary,
      source_duration=source_duration,
      source_fps=source_fps,
  ))
  if set(overlay_files) != expected_overlays:
    raise ValueError(
        "overlay_files must contain exactly "
        f"{sorted(expected_overlays)!r}"
    )
  if not all(Path(path).is_file() for path in overlay_files.values()):
    raise FileNotFoundError("every overlay text file must exist")
  total_duration = source_duration + _START_HOLD + _END_HOLD
  intervention_time = _event_time(
      summary["intervention_start"], source_fps
  )
  _, normal_step = _chain_cue_event(summary, "normal")
  _, changed_step = _chain_cue_event(summary, "trajectory_changed")
  removal_step = summary["branches"]["target_removed"]["removed_step"]
  marker_x = 80 + round(1760 * intervention_time / total_duration)
  final_start_frame = _event_output_frame(
      summary["demo_spec"]["source_frames"]
  )
  final_enable = f"gte(n,{final_start_frame})"
  normal_enable = _frame_window_enable(
      _event_output_frame(normal_step), _CHAIN_CUE_DURATION
  )
  changed_enable = _frame_window_enable(
      _event_output_frame(changed_step), _CHAIN_CUE_DURATION
  )
  removal_enable = _frame_window_enable(
      _event_output_frame(removal_step), _REMOVAL_CUE_DURATION
  )

  chains = []
  for index in range(3):
    chains.append(
        f"[{index}:v]"
        f"fps={int(_OUTPUT_FPS)},"
        f"scale={_PANEL_WIDTH}:{_PANEL_HEIGHT}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={_PANEL_WIDTH}:{_PANEL_HEIGHT}:"
        "(ow-iw)/2:(oh-ih)/2:color=0x10151d,"
        "setsar=1,setpts=PTS-STARTPTS,"
        f"tpad=start_mode=clone:start_duration={_START_HOLD:g}:"
        f"stop_mode=clone:stop_duration={_END_HOLD:g}"
        f"[panel{index}]"
    )

  overlays = [
      _drawtext(
          font_path, "NORMAL", x="(640-text_w)/2", y="(90-text_h)/2",
          size=28, color="0x8bd5ca"
      ),
      _drawtext(
          font_path, "TRAJECTORY CHANGED",
          x="640+(640-text_w)/2", y="(90-text_h)/2",
          size=28, color="0xf5a97f"
      ),
      _drawtext(
          font_path, "TARGET REMOVED",
          x="1280+(640-text_w)/2", y="(90-text_h)/2",
          size=28, color="0xc6a0f6"
      ),
      "drawbox=x=80:y=671:w=1760:h=4:color=0x5b6078:t=fill",
      (
          f"drawbox=x={marker_x}:y=653:w=4:h=40:"
          "color=0xed8796:t=fill"
      ),
      _drawtext(
          font_path,
          textfile=overlay_files["intervention"],
          x=str(max(8, marker_x - 110)), y="627", size=18,
          color="0xed8796",
      ),
      _drawtext(
          font_path, "0s", x="80", y="687", size=16, color="0xa5adcb"
      ),
      _drawtext(
          font_path, textfile=overlay_files["duration"], x="1810", y="687",
          size=16, color="0xa5adcb"
      ),
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
      _drawtext(
          font_path, textfile=overlay_files["removal"],
          x="1280+(640-text_w)/2", y="120", size=26,
          color="0xff8fab", enable=removal_enable, box=True,
      ),
  ]
  overlays.extend((
      _drawtext(
          font_path, textfile=overlay_files["graph"],
          x="(w-text_w)/2", y="530", size=22,
          color="white", enable=final_enable, box=True,
      ),
      _drawtext(
          font_path, textfile=overlay_files["affected"],
          x="(w-text_w)/2", y="566", size=20,
          color="white", enable=final_enable, box=True,
      ),
      _drawtext(
          font_path, textfile=overlay_files["propagation"],
          x="(w-text_w)/2", y="602", size=18,
          color="white", enable=final_enable, box=True,
      ),
  ))
  layout = (
      "[panel0][panel1][panel2]hstack=inputs=3:shortest=1,"
      f"pad={_OUTPUT_WIDTH}:{_OUTPUT_HEIGHT}:0:90:color=0x181926,"
      + ",".join(overlays)
      + "[outv]"
  )
  chains.append(layout)
  return ";".join(chains)


def _find_tool(name: str) -> str:
  executable = shutil.which(name)
  if executable is None:
    raise RuntimeError(f"{name} was not found on PATH")
  return executable


def _source_paths(states_dir: Path) -> dict[str, Path]:
  _reject_symlink_components(states_dir, "states directory")
  if not states_dir.is_dir():
    raise FileNotFoundError(f"states directory does not exist: {states_dir}")
  result = {
      branch: states_dir / filename for branch, filename in _BRANCH_FILES
  }
  for branch, path in result.items():
    _require_regular_input(path, f"{branch} source")
  return result


def _normalized_fps(value: Any) -> Fraction:
  if isinstance(value, bool):
    raise ValueError("video fps must be a positive rational number")
  try:
    result = value if isinstance(value, Fraction) else Fraction(str(value))
  except (TypeError, ValueError, ZeroDivisionError) as error:
    raise ValueError("video fps must be a positive rational number") from error
  if result <= 0:
    raise ValueError("video fps must be a positive rational number")
  return result


def _validate_synchronized_sources(
    sources: Mapping[str, Path], ffprobe: str
) -> tuple[VideoInfo, dict[str, VideoInfo]]:
  infos = {
      branch: _probe_video(path, ffprobe=ffprobe)
      for branch, path in sources.items()
  }
  reference_name = _BRANCH_FILES[0][0]
  reference = infos[reference_name]
  for branch, info in infos.items():
    if info.codec_name.lower() != "h264":
      raise ValueError(
          f"source video {branch} codec must be H264; got {info.codec_name!r}"
      )
    if _normalized_fps(info.fps) != _OUTPUT_FPS:
      raise ValueError(
          f"source video {branch} must match the renderer frame rate of "
          f"24 fps; got {info.fps!r}"
      )
  for branch, info in infos.items():
    if branch == reference_name:
      continue
    for field in ("width", "height", "frame_count"):
      if getattr(info, field) != getattr(reference, field):
        raise ValueError(
            f"source videos must have synchronized {field}; "
            f"{reference_name}={getattr(reference, field)!r}, "
            f"{branch}={getattr(info, field)!r}"
        )
    if _normalized_fps(info.fps) != _normalized_fps(reference.fps):
      raise ValueError(
          "source videos must have synchronized fps; "
          f"{reference_name}={reference.fps!r}, {branch}={info.fps!r}"
      )
    if not math.isclose(
        info.duration,
        reference.duration,
        rel_tol=0.0,
        abs_tol=_SOURCE_DURATION_TOLERANCE,
    ):
      raise ValueError(
          "source videos must have synchronized duration; "
          f"{reference_name}={reference.duration!r}, "
          f"{branch}={info.duration!r}"
      )
  return reference, infos


def _validate_event_steps(summary: Mapping[str, Any], frame_count: int) -> None:
  if frame_count != summary["demo_spec"]["source_frames"]:
    raise ValueError(
        "source frame count differs from demo_spec.source_frames"
    )
  _, normal_step = _chain_cue_event(summary, "normal")
  _, changed_step = _chain_cue_event(summary, "trajectory_changed")
  named_steps = {
      "intervention_start": summary["intervention_start"],
      "intervention_end": summary["intervention_end"],
      "normal chain event": normal_step,
      "trajectory_changed chain event": changed_step,
      "removed_step": summary["branches"]["target_removed"]["removed_step"],
  }
  for name, step in named_steps.items():
    upper_bound = frame_count if name == "intervention_end" else frame_count - 1
    if step > upper_bound:
      raise ValueError(
          f"summary {name}={step} lies outside {frame_count} source frames"
      )


def _run_ffmpeg(command: Sequence[str]) -> None:
  try:
    subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
  except subprocess.TimeoutExpired as error:
    detail = error.stderr or ""
    if isinstance(detail, bytes):
      detail = detail.decode("utf-8", errors="replace")
    suffix = f": {detail.strip()}" if detail.strip() else ""
    raise RuntimeError(
        "ffmpeg composition timed out after "
        f"{_FFMPEG_TIMEOUT_SECONDS} seconds{suffix}"
    ) from error
  except (OSError, subprocess.CalledProcessError) as error:
    detail = getattr(error, "stderr", "") or str(error)
    if isinstance(detail, bytes):
      detail = detail.decode("utf-8", errors="replace")
    raise RuntimeError(f"ffmpeg composition failed: {detail.strip()}") from error


def _validate_composed_video(
    info: VideoInfo, source: VideoInfo
) -> None:
  hold_frames = _exact_duration_frames(
      _START_HOLD_DURATION + _END_HOLD_DURATION
  )
  expected_frames = source.frame_count + hold_frames
  expected_duration = expected_frames / float(_OUTPUT_FPS)
  if (info.width, info.height) != (_OUTPUT_WIDTH, _OUTPUT_HEIGHT):
    raise ValueError(
        "composed video size must be 1920x720; "
        f"got {info.width}x{info.height}"
    )
  if info.codec_name.lower() != "h264":
    raise ValueError(f"composed video codec must be H264; got {info.codec_name!r}")
  if info.pix_fmt != "yuv420p":
    raise ValueError(
        f"composed video pixel format must be yuv420p; got {info.pix_fmt!r}"
    )
  if _normalized_fps(info.fps) != _OUTPUT_FPS:
    raise ValueError(f"composed video frame rate must be 24; got {info.fps!r}")
  if info.frame_count != expected_frames:
    raise ValueError(
        f"composed video must have {expected_frames} frames; "
        f"got {info.frame_count}"
    )
  if not math.isclose(
      info.duration, expected_duration, rel_tol=0.0, abs_tol=0.01
  ):
    raise ValueError(
        f"composed video duration must be {expected_duration:.6f}s; "
        f"got {info.duration:.6f}s"
    )


def _validate_output_path(output: Path, sources: Mapping[str, Path]) -> None:
  _reject_symlink_components(output, "output")
  if output.exists() and not output.is_file():
    raise ValueError(f"output must be a regular file path: {output}")
  canonical_output = output.resolve(strict=False)
  for source in sources.values():
    _require_regular_input(source, "source video")
    if canonical_output == source.resolve(strict=True):
      raise ValueError("output must not overwrite a source video")
    if output.exists() and os.path.samefile(output, source):
      raise ValueError("output must not be the same file as a source video")


def compose_intervention_demo(
    states_dir: str | Path = "output/demo_collision_intervention",
    output: str | Path | None = None,
    font: str | Path | None = None,
) -> dict[str, Any]:
  """Compose, verify, and atomically publish the comparison video."""
  directory = Path(states_dir)
  output_path = (
      directory / "trajectory_intervention_demo.mp4"
      if output is None
      else Path(output)
  )
  sources = _source_paths(directory)
  _validate_output_path(output_path, sources)
  summary = _load_summary(directory)
  font_path = _resolve_font(font)
  ffmpeg = _find_tool("ffmpeg")
  ffprobe = _find_tool("ffprobe")
  source_info, _ = _validate_synchronized_sources(sources, ffprobe)
  _validate_event_steps(summary, source_info.frame_count)
  source_duration = source_info.frame_count / float(_OUTPUT_FPS)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = tempfile.NamedTemporaryFile(
      prefix=f".{output_path.stem}.",
      suffix=".tmp.mp4",
      dir=output_path.parent,
      delete=False,
  )
  staging = Path(temporary.name)
  temporary.close()
  try:
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}.overlays.",
        dir=output_path.parent,
    ) as overlay_name:
      overlay_files = _write_overlay_textfiles(
          overlay_name,
          summary,
          source_duration=source_duration,
          source_fps=source_info.fps,
      )
      filter_graph = _build_filter(
          summary,
          font_path,
          source_duration=source_duration,
          source_fps=source_info.fps,
          overlay_files=overlay_files,
      )
      command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
      for _, path in _BRANCH_FILES:
        command.extend(("-i", str(directory / path)))
      command.extend((
          "-filter_complex",
          filter_graph,
          "-map",
          "[outv]",
          "-an",
          "-c:v",
          "libx264",
          "-pix_fmt",
          "yuv420p",
          "-r",
          str(int(_OUTPUT_FPS)),
          "-movflags",
          "+faststart",
          str(staging),
      ))
      _run_ffmpeg(command)
      composed_info = _probe_video(staging, ffprobe=ffprobe)
      _validate_composed_video(composed_info, source_info)
      _validate_output_path(output_path, sources)
      os.replace(staging, output_path)
  except BaseException:
    staging.unlink(missing_ok=True)
    raise

  return {
      "duration": composed_info.duration,
      "frame_count": composed_info.frame_count,
      "output": str(output_path),
      "size": [composed_info.width, composed_info.height],
  }


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Compose the three trajectory-intervention Blender replays."
  )
  parser.add_argument(
      "--states-dir",
      type=Path,
      default=Path("output/demo_collision_intervention"),
      help="directory containing summary.json and the three Blender MP4s",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=None,
      help="output MP4 (default: STATES_DIR/trajectory_intervention_demo.mp4)",
  )
  parser.add_argument(
      "--font",
      type=Path,
      default=None,
      help="TTF/OTF font for overlays (default: deterministic Noto/DejaVu)",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Validates CLI inputs, composes the comparison atomically, and returns zero."""
  args = _parser().parse_args(argv)
  metadata = compose_intervention_demo(args.states_dir, args.output, args.font)
  print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
