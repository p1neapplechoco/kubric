#!/usr/bin/env python3
"""Render the three intervention-demo replays with procedural Blender assets.

The replay files contain position XYZ followed by quaternion WXYZ and velocity
columns. Kubric, PyBullet, and Blender are deliberately imported only on the
rendering path so validation helpers stay usable in an ordinary Python process.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from scripts.trajectory_demo_spec import FORKED_RACK_SPEC, demo_spec_summary

_DEFAULT_STATES_DIR = Path("output/demo_collision_intervention")
_ALLOWED_BRANCHES = ("normal", "trajectory_changed", "target_removed")
_BRANCH_FILENAMES = {
    "normal": "normal_blender.mp4",
    "trajectory_changed": "trajectory_changed_blender.mp4",
    "target_removed": "target_removed_blender.mp4",
}
_FRAME_RATE = FORKED_RACK_SPEC.frame_rate

# State layout matches SimulationLog: position XYZ, quaternion WXYZ, then linear
# and angular velocity. The ordering is persisted in summary.json and fixed for
# this deterministic demo.
_DEMO_SPEC = FORKED_RACK_SPEC
_CANONICAL_OBJECT_IDS = _DEMO_SPEC.object_ids
_STATE_STRIDE = 13
_POSITION_SLICE = slice(0, 3)
_QUATERNION_WXYZ_SLICE = slice(3, 7)
_SYNCHRONIZED_SUMMARY_KEYS = (
    "branches",
    "demo_spec",
    "ground_truth",
    "intervention_end",
    "intervention_start",
    "intervention_window",
    "object_ids",
    "seed",
    "step_rate",
)

_CAMERA_POSITION = (7.6, -9.2, 8.4)
_CAMERA_LOOK_AT = (0.35, 0.30, -0.02)
_CAMERA_FOCAL_LENGTH = 55.0
_CAMERA_CLIP_START = 0.1
_CAMERA_CLIP_END = 1000.0
_AMBIENT_ILLUMINATION = (0.035, 0.04, 0.05)


@dataclass(frozen=True)
class Replay:
  """Validated render inputs for one synchronized branch."""

  branch: str
  object_ids: Tuple[str, ...]
  steps: Tuple[int, ...]
  states: np.ndarray
  presence: np.ndarray
  summary: Mapping[str, Any]


def _camera_dof_spec() -> Dict[str, Any]:
  """Returns the deterministic focus settings for the shared camera."""
  return {
      "use_dof": True,
      "focus_distance": 12.0,
      "aperture_fstop": 5.6,
  }


def _collider_specs() -> Dict[str, Dict[str, Any]]:
  """Returns collider kinds and dimensions in canonical specification order."""
  return {
      item.object_id: {"kind": item.shape, "scale": item.size}
      for item in _DEMO_SPEC.objects
  }


def _material_specs() -> Dict[str, Dict[str, Any]]:
  """Returns fresh deterministic specifications for all procedural materials."""
  specs = {
      "rail": {
          "material": "wood",
          "color": (0.16, 0.045, 0.012, 1.0),
          "light_color": (0.43, 0.13, 0.025, 1.0),
          "roughness": 0.3,
          "grain_scale": 3.5,
      },
      "backdrop": {
          "material": "matte",
          "color": (0.055, 0.06, 0.072, 1.0),
          "roughness": 0.82,
      },
      "band": {
          "material": "lacquer",
          "color": (0.94, 0.94, 0.9, 1.0),
          "roughness": 0.16,
          "metallic": 0.0,
      },
      "number": {
          "material": "matte",
          "color": (0.012, 0.012, 0.012, 1.0),
          "roughness": 0.45,
      },
  }
  for item in _DEMO_SPEC.objects:
    color = (*item.color, 1.0)
    if item.visual_role == "ball":
      specs[item.object_id] = {
          "material": "lacquer",
          "color": color,
          "roughness": 0.16,
          "metallic": 0.04,
          "number": item.ball_number,
          "striped": item.striped,
      }
    elif item.visual_role == "target":
      specs[item.object_id] = {
          "material": "wood",
          "color": color,
          "light_color": (0.72, 0.31, 0.075, 1.0),
          "roughness": 0.30,
          "grain_scale": 5.5,
      }
    else:
      specs[item.object_id] = {
          "material": "felt",
          "color": color,
          "roughness": 0.88,
          "noise_scale": 92.0,
          "bump_strength": 0.16,
      }
  return copy.deepcopy(specs)


def _scene_specs() -> Dict[str, Any]:
  """Returns the collider and studio contract without importing Blender."""
  specs = {
      "colliders": _collider_specs(),
      "camera": {
          "position": _CAMERA_POSITION,
          "look_at": _CAMERA_LOOK_AT,
          "focal_length": _CAMERA_FOCAL_LENGTH,
          "clip_start": _CAMERA_CLIP_START,
          "clip_end": _CAMERA_CLIP_END,
          "dof": _camera_dof_spec(),
      },
      "lights": (
          {
              "role": "key",
              "kind": "area",
              "position": (-3.8, -4.4, 7.5),
              "color": (1.0, 0.78, 0.58),
              "intensity": 620.0,
              "width": 4.5,
              "height": 4.5,
          },
          {
              "role": "fill",
              "kind": "area",
              "position": (4.8, -1.8, 5.2),
              "color": (0.58, 0.72, 1.0),
              "intensity": 330.0,
              "width": 3.5,
              "height": 3.5,
          },
          {
              "role": "rim",
              "kind": "area",
              "position": (0.6, 5.4, 6.2),
              "color": (1.0, 0.5, 0.24),
              "intensity": 480.0,
              "width": 3.0,
              "height": 3.0,
          },
      ),
      "renderer": {
          "engine": "CYCLES",
          "adaptive_sampling": True,
          "denoising": True,
          "transparent": False,
      },
  }
  return copy.deepcopy(specs)


def _load_summary(states_dir: Path) -> Mapping[str, Any]:
  summary_path = states_dir / "summary.json"
  try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError as error:
    raise ValueError(f"{summary_path} is not valid JSON") from error
  if not isinstance(summary, dict):
    raise ValueError(f"{summary_path} must contain a JSON object")
  expected_keys = set(_SYNCHRONIZED_SUMMARY_KEYS)
  if set(summary) != expected_keys:
    missing = sorted(expected_keys - set(summary))
    unexpected = sorted(set(summary) - expected_keys)
    raise ValueError(
        f"{summary_path} summary keys are invalid; missing={missing!r}, "
        f"unexpected={unexpected!r}"
    )
  object_ids = summary.get("object_ids")
  if object_ids != list(_CANONICAL_OBJECT_IDS):
    raise ValueError(
        f"{summary_path} object_ids must be {list(_CANONICAL_OBJECT_IDS)!r}; "
        f"got {object_ids!r}"
    )
  branches = summary.get("branches")
  if not isinstance(branches, dict) or set(branches) != set(_ALLOWED_BRANCHES):
    raise ValueError(
        f"{summary_path} branches must be exactly {list(_ALLOWED_BRANCHES)!r}"
    )
  if not all(isinstance(value, dict) for value in branches.values()):
    raise TypeError(f"{summary_path} branches values must be JSON objects")

  _validate_demo_spec_identity(summary["demo_spec"])

  ground_truth = summary["ground_truth"]
  if not isinstance(ground_truth, dict):
    raise TypeError(f"{summary_path} ground_truth must be a JSON object")

  seed = summary["seed"]
  if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
    raise ValueError(f"{summary_path} seed must be a nonnegative integer")

  step_rate = summary["step_rate"]
  if (
      isinstance(step_rate, bool)
      or not isinstance(step_rate, (int, float))
      or not math.isfinite(step_rate)
      or step_rate <= 0
  ):
    raise ValueError(f"{summary_path} step_rate must be a positive number")

  start = summary["intervention_start"]
  end = summary["intervention_end"]
  if isinstance(start, bool) or not isinstance(start, int) or start < 0:
    raise ValueError(
        f"{summary_path} intervention_start must be a nonnegative integer"
    )
  if isinstance(end, bool) or not isinstance(end, int) or end <= start:
    raise ValueError(
        f"{summary_path} intervention_end must be an integer greater than "
        "intervention_start"
    )
  window = summary["intervention_window"]
  if window != [start, end]:
    raise ValueError(
        f"{summary_path} intervention_window must equal {[start, end]!r}"
    )
  if start >= _DEMO_SPEC.num_steps or end > _DEMO_SPEC.num_steps:
    raise ValueError(
        f"{summary_path} intervention window must lie within replay frames"
    )
  if seed != _DEMO_SPEC.seed:
    raise ValueError(f"{summary_path} seed mismatch")
  if step_rate != _DEMO_SPEC.step_rate:
    raise ValueError(f"{summary_path} step_rate mismatch")
  if (start, end) != _DEMO_SPEC.intervention_window:
    raise ValueError(f"{summary_path} intervention window mismatch")
  return summary


def _validate_demo_spec_identity(value: Any) -> Mapping[str, Any]:
  """Rejects replay metadata produced from a different scene contract."""
  if not isinstance(value, dict):
    raise TypeError("demo_spec must be a JSON object")
  expected = demo_spec_summary(_DEMO_SPEC)
  if set(value) != set(expected):
    raise ValueError("demo_spec keys mismatch")
  for field, expected_value in expected.items():
    if value[field] != expected_value:
      raise ValueError(f"demo_spec.{field} mismatch")
  return value


def _load_replay(states_dir: Path, branch: str) -> Replay:
  """Loads and validates one branch's states, presence mask, and metadata."""
  if branch not in _ALLOWED_BRANCHES:
    raise ValueError(
        f"branch must be one of {list(_ALLOWED_BRANCHES)!r}; got {branch!r}"
    )

  states_dir = Path(states_dir)
  summary = _load_summary(states_dir)
  states_path = states_dir / f"{branch}_states.npy"
  presence_path = states_dir / f"{branch}_presence.npy"
  states = np.load(states_path, allow_pickle=False)
  presence = np.load(presence_path, allow_pickle=False)

  expected_shape = (
      _DEMO_SPEC.num_steps,
      len(_CANONICAL_OBJECT_IDS),
      _STATE_STRIDE,
  )
  if states.shape != expected_shape:
    raise ValueError(
        f"{states_path} states have shape {states.shape}; expected "
        f"{expected_shape}"
    )
  intervention_start = summary["intervention_start"]
  intervention_end = summary["intervention_end"]
  if (
      intervention_start >= states.shape[0]
      or intervention_end > states.shape[0]
  ):
    raise ValueError(
        f"{states_path} intervention window must lie within its "
        f"{states.shape[0]} replay frames"
    )
  if states.dtype.kind not in "fiu":
    raise TypeError(f"{states_path} states must be numeric")
  if not np.isfinite(states).all():
    raise ValueError(f"{states_path} contains non-finite states")
  quaternion_norms = np.linalg.norm(
      states[..., _QUATERNION_WXYZ_SLICE], axis=-1
  )
  if not np.isclose(
      quaternion_norms, 1.0, atol=1e-6, rtol=0.0
  ).all():
    raise ValueError(
        f"{states_path} quaternion values must be unit normalized"
    )
  if presence.dtype.kind != "b":
    raise TypeError(f"{presence_path} presence must contain Boolean values")
  expected_presence_shape = expected_shape[:2]
  if presence.shape != expected_presence_shape:
    raise ValueError(
        f"{presence_path} presence has shape {presence.shape}; expected "
        f"{expected_presence_shape} to match the states frame/object count"
    )

  return Replay(
      branch=branch,
      object_ids=_CANONICAL_OBJECT_IDS,
      steps=tuple(range(states.shape[0])),
      states=states,
      presence=presence,
      summary=summary,
  )


def _prepare_replay(replay: Replay, max_frames: int | None) -> Replay:
  """Applies an optional positive frame limit to both replay arrays."""
  if max_frames is None:
    return replay
  if isinstance(max_frames, bool) or not isinstance(max_frames, int):
    raise TypeError("--max-frames must be a positive integer")
  if max_frames < 1:
    raise ValueError("--max-frames must be >= 1")
  limit = min(max_frames, len(replay.states))
  return replace(
      replay,
      steps=replay.steps[:limit],
      states=replay.states[:limit],
      presence=replay.presence[:limit],
  )


def _synchronized_metadata(replay: Replay) -> Dict[str, Any]:
  """Projects summary metadata that must agree across render branches."""
  return {
      key: replay.summary.get(key) for key in _SYNCHRONIZED_SUMMARY_KEYS
  }


def _validate_synchronized_replays(replays: Sequence[Replay]) -> None:
  """Rejects requested branches that cannot share one synchronized timeline."""
  if not replays:
    raise ValueError("at least one replay is required for synchronization")

  reference = replays[0]
  reference_frames = len(reference.states)
  if len(reference.steps) != reference_frames:
    raise ValueError(
        f"replay {reference.branch!r} is not synchronized with its step array"
    )
  reference_metadata = _synchronized_metadata(reference)

  for replay in replays:
    intervention_start = replay.summary["intervention_start"]
    if not replay.presence[:intervention_start].all():
      raise ValueError(
          f"replay {replay.branch!r} has false pre-intervention presence"
      )

  for replay in replays[1:]:
    if replay.object_ids != reference.object_ids:
      raise ValueError(
          "requested branch replays are not synchronized: object_ids differ "
          f"between {reference.branch!r} and {replay.branch!r}"
      )
    if len(replay.states) != reference_frames:
      raise ValueError(
          "requested branch replays are not synchronized: frame count "
          f"differs between {reference.branch!r} ({reference_frames}) and "
          f"{replay.branch!r} ({len(replay.states)})"
      )
    if replay.steps != reference.steps:
      raise ValueError(
          "requested branch replays are not synchronized: step arrays differ "
          f"between {reference.branch!r} and {replay.branch!r}"
      )
    if _synchronized_metadata(replay) != reference_metadata:
      raise ValueError(
          "requested branch replays are not synchronized: metadata differs "
          f"between {reference.branch!r} and {replay.branch!r}"
      )
    intervention_start = reference.summary["intervention_start"]
    if not np.array_equal(
        replay.states[:intervention_start],
        reference.states[:intervention_start],
    ) or not np.array_equal(
        replay.presence[:intervention_start],
        reference.presence[:intervention_start],
    ):
      raise ValueError(
          "requested branch replays are not synchronized: common prefix "
          f"differs between {reference.branch!r} and {replay.branch!r}"
      )


def _preflight_replays(
    states_dir: Path,
    branches: Sequence[str],
    max_frames: int | None,
) -> Tuple[Replay, ...]:
  """Validates full source timelines before consistently limiting them."""
  source_replays = tuple(
      _load_replay(states_dir, branch) for branch in branches
  )
  _validate_synchronized_replays(source_replays)
  prepared_replays = tuple(
      _prepare_replay(replay, max_frames) for replay in source_replays
  )
  _validate_synchronized_replays(prepared_replays)
  return prepared_replays


def _collider_radius(item) -> float:
  """Returns a conservative rotation-invariant framing radius."""
  if item.visual_role == "floor":
    return 0.0
  if item.shape == "sphere":
    return float(item.size)
  scale = (
      (float(item.size),) * 3
      if isinstance(item.size, (int, float))
      else item.size
  )
  return math.sqrt(sum(float(component) ** 2 for component in scale))


def _validate_camera_containment(
    replays: Sequence[Replay],
    resolution: Tuple[int, int],
) -> None:
  """Rejects replays whose conservative collider extents leave the frame."""
  camera = np.asarray(_CAMERA_POSITION, dtype=float)
  forward = np.asarray(_CAMERA_LOOK_AT, dtype=float) - camera
  forward /= np.linalg.norm(forward)
  right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
  right /= np.linalg.norm(right)
  up = np.cross(right, forward)
  width, height = resolution
  item_by_id = {item.object_id: item for item in _DEMO_SPEC.objects}
  for replay in replays:
    for object_index, object_id in enumerate(replay.object_ids):
      radius = _collider_radius(item_by_id[object_id])
      for frame, center in enumerate(
          replay.states[:, object_index, _POSITION_SLICE]
      ):
        if not replay.presence[frame, object_index]:
          continue
        relative = center - camera
        depth = float(relative @ forward)
        near_depth = depth - radius
        far_depth = depth + radius
        if (
            near_depth < _CAMERA_CLIP_START
            or far_depth > _CAMERA_CLIP_END
        ):
          raise ValueError(f"camera framing excludes {object_id}")
        half_width = near_depth * 36.0 / (2.0 * _CAMERA_FOCAL_LENGTH)
        half_height = half_width * height / width
        if (
            abs(float(relative @ right)) + radius > 0.94 * half_width
            or abs(float(relative @ up)) + radius > 0.94 * half_height
        ):
          raise ValueError(f"camera framing excludes {object_id}")


def _visibility_transitions(
    presence: np.ndarray,
) -> Tuple[Tuple[int, bool], ...]:
  """Returns ``(zero_based_frame, hidden)`` entries when visibility changes."""
  presence = np.asarray(presence)
  if presence.dtype.kind != "b":
    raise TypeError("presence must contain Boolean values")
  if presence.ndim != 1 or len(presence) < 1:
    raise ValueError("presence must be a non-empty one-dimensional array")
  transitions = [(0, not bool(presence[0]))]
  for frame in range(1, len(presence)):
    if bool(presence[frame]) != bool(presence[frame - 1]):
      transitions.append((frame, not bool(presence[frame])))
  return tuple(transitions)


def _pose_at(states: np.ndarray, object_index: int, step: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
  row = states[step, object_index]
  position = tuple(float(value) for value in row[_POSITION_SLICE])
  quaternion = tuple(float(value) for value in row[_QUATERNION_WXYZ_SLICE])
  return position, quaternion


def _encode_mp4(output: Path, rgba: np.ndarray, frame_rate: int) -> None:
  import imageio.v2 as imageio

  video_frames = rgba[..., :3]
  if np.issubdtype(video_frames.dtype, np.floating):
    video_frames = np.clip(video_frames, 0.0, 1.0)
    video_frames = (video_frames * 255).astype(np.uint8)
  elif video_frames.dtype == np.uint16:
    video_frames = (video_frames / 257).astype(np.uint8)
  elif video_frames.dtype != np.uint8:
    video_frames = video_frames.astype(np.uint8)

  output.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(
      output,
      video_frames,
      fps=frame_rate,
      codec="libx264",
      quality=8,
      macro_block_size=None,
  )


def _principled_input(node, *names):
  for name in names:
    socket = node.inputs.get(name)
    if socket is not None:
      return socket
  raise KeyError(f"Principled BSDF has none of the inputs {names!r}")


def _base_material(bpy, name: str, spec: Mapping[str, Any]):
  material = bpy.data.materials.new(name=name)
  material.use_nodes = True
  nodes = material.node_tree.nodes
  bsdf = nodes.get("Principled BSDF")
  if bsdf is None:
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
  _principled_input(bsdf, "Base Color").default_value = spec["color"]
  _principled_input(bsdf, "Roughness").default_value = spec["roughness"]
  if "metallic" in spec:
    _principled_input(bsdf, "Metallic").default_value = spec["metallic"]
  if spec["material"] == "lacquer":
    _principled_input(bsdf, "Coat Weight", "Clearcoat").default_value = 0.32
    _principled_input(
        bsdf, "Coat Roughness", "Clearcoat Roughness"
    ).default_value = 0.08
  return material, bsdf


def _wood_material(bpy, name: str, spec: Mapping[str, Any]):
  material, bsdf = _base_material(bpy, name, spec)
  tree = material.node_tree
  nodes = tree.nodes
  links = tree.links

  coordinates = nodes.new(type="ShaderNodeTexCoord")
  mapping = nodes.new(type="ShaderNodeMapping")
  mapping.vector_type = "POINT"
  mapping.inputs["Scale"].default_value = (
      spec["grain_scale"],
      spec["grain_scale"] * 13.0,
      spec["grain_scale"],
  )
  noise = nodes.new(type="ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = 1.35
  noise.inputs["Detail"].default_value = 5.0
  noise.inputs["Roughness"].default_value = 0.72
  noise.inputs["Distortion"].default_value = 0.18
  ramp = nodes.new(type="ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].position = 0.22
  ramp.color_ramp.elements[0].color = spec["color"]
  ramp.color_ramp.elements[1].position = 0.78
  ramp.color_ramp.elements[1].color = spec["light_color"]
  bump = nodes.new(type="ShaderNodeBump")
  bump.inputs["Strength"].default_value = 0.16
  bump.inputs["Distance"].default_value = 0.075

  links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
  links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
  links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
  links.new(ramp.outputs["Color"], _principled_input(bsdf, "Base Color"))
  links.new(noise.outputs["Fac"], bump.inputs["Height"])
  links.new(bump.outputs["Normal"], _principled_input(bsdf, "Normal"))
  return material


def _felt_material(bpy, name: str, spec: Mapping[str, Any]):
  material, bsdf = _base_material(bpy, name, spec)
  tree = material.node_tree
  nodes = tree.nodes
  links = tree.links

  coordinates = nodes.new(type="ShaderNodeTexCoord")
  noise = nodes.new(type="ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = spec["noise_scale"]
  noise.inputs["Detail"].default_value = 2.0
  noise.inputs["Roughness"].default_value = 0.82
  ramp = nodes.new(type="ShaderNodeValToRGB")
  ramp.color_ramp.elements[0].color = (0.008, 0.075, 0.028, 1.0)
  ramp.color_ramp.elements[1].color = spec["color"]
  bump = nodes.new(type="ShaderNodeBump")
  bump.inputs["Strength"].default_value = spec["bump_strength"]
  bump.inputs["Distance"].default_value = 0.025

  links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
  links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
  links.new(ramp.outputs["Color"], _principled_input(bsdf, "Base Color"))
  links.new(noise.outputs["Fac"], bump.inputs["Height"])
  links.new(bump.outputs["Normal"], _principled_input(bsdf, "Normal"))
  return material


def _procedural_material(bpy, name: str, spec: Mapping[str, Any]):
  if spec["material"] == "wood":
    return _wood_material(bpy, name, spec)
  if spec["material"] == "felt":
    return _felt_material(bpy, name, spec)
  return _base_material(bpy, name, spec)[0]


def _smooth_mesh(blender_object) -> None:
  if blender_object.type != "MESH":
    return
  for polygon in blender_object.data.polygons:
    polygon.use_smooth = True


def _configure_camera_dof(
    blender_camera,
    spec: Mapping[str, Any],
) -> None:
  """Applies an offline-testable DOF specification to a Blender camera."""
  blender_camera.data.dof.use_dof = spec["use_dof"]
  blender_camera.data.dof.focus_distance = spec["focus_distance"]
  blender_camera.data.dof.aperture_fstop = spec["aperture_fstop"]


def _configure_camera_settings(
    blender_camera,
    spec: Mapping[str, Any],
) -> None:
  """Applies the complete deterministic camera data contract."""
  _configure_camera_dof(blender_camera, spec["dof"])
  blender_camera.data.clip_start = spec["clip_start"]
  blender_camera.data.clip_end = spec["clip_end"]


def _round_target(blender_object) -> None:
  if hasattr(blender_object.data, "use_auto_smooth"):
    blender_object.data.use_auto_smooth = True
  bevel = blender_object.modifiers.new(
      name="Rounded collider edges", type="BEVEL"
  )
  bevel.width = 0.12
  bevel.segments = 5
  bevel.limit_method = "ANGLE"
  if hasattr(bevel, "harden_normals"):
    bevel.harden_normals = True


def _parent_local(child, parent, location, rotation=(0.0, 0.0, 0.0)):
  child.parent = parent
  child.location = location
  child.rotation_mode = "XYZ"
  child.rotation_euler = rotation
  child.scale = (1.0, 1.0, 1.0)
  return child


def _add_ball_decorations(
    bpy,
    parent,
    object_id: str,
    number: int,
    striped: bool,
    band_material,
    number_material,
) -> Tuple[object, ...]:
  """Adds a number badge and an optional stripe in the ball's local frame."""
  decorations = []
  if striped:
    bpy.ops.mesh.primitive_torus_add(
        align="WORLD",
        major_segments=64,
        minor_segments=16,
        location=(0.0, 0.0, 0.0),
        major_radius=0.79,
        minor_radius=0.205,
    )
    band = _parent_local(
        bpy.context.object,
        parent,
        (0.0, 0.0, 0.0),
    )
    band.name = f"{object_id}_white_band"
    band.data.materials.append(band_material)
    _smooth_mesh(band)
    decorations.append(band)

  bpy.ops.mesh.primitive_cylinder_add(
      align="WORLD",
      vertices=48,
      radius=0.34,
      depth=0.026,
      location=(0.0, 0.0, 0.0),
  )
  badge = _parent_local(
      bpy.context.object,
      parent,
      (0.0, -0.982, 0.0),
      (math.pi / 2.0, 0.0, 0.0),
  )
  badge.name = f"{object_id}_number_badge"
  badge.data.materials.append(band_material)
  _smooth_mesh(badge)
  decorations.append(badge)

  bpy.ops.object.text_add(align="WORLD", location=(0.0, 0.0, 0.0))
  decal = _parent_local(
      bpy.context.object,
      parent,
      (0.0, -1.015, 0.0),
      (math.pi / 2.0, 0.0, 0.0),
  )
  decal.name = f"{object_id}_number"
  decal.data.body = str(number)
  decal.data.align_x = "CENTER"
  decal.data.align_y = "CENTER"
  decal.data.size = 0.48
  decal.data.extrude = 0.008
  decal.data.bevel_depth = 0.004
  decal.data.materials.append(number_material)
  decorations.append(decal)
  return tuple(decorations)


def _insert_visibility_keyframes(
    blender_objects: Sequence[object],
    presence: np.ndarray,
    frame_start: int,
) -> None:
  transitions = _visibility_transitions(presence)
  for zero_based_frame, hidden in transitions:
    frame = frame_start + zero_based_frame
    for blender_object in blender_objects:
      blender_object.hide_render = hidden
      blender_object.hide_viewport = hidden
      blender_object.keyframe_insert(data_path="hide_render", frame=frame)
      blender_object.keyframe_insert(data_path="hide_viewport", frame=frame)

  for blender_object in blender_objects:
    animation = blender_object.animation_data
    action = animation.action if animation is not None else None
    if action is None:
      continue
    for curve in action.fcurves:
      if curve.data_path in {"hide_render", "hide_viewport"}:
        for point in curve.keyframe_points:
          point.interpolation = "CONSTANT"


def _encoder_backend(available_formats: Sequence[str]) -> str:
  """Chooses Blender FFmpeg when compiled in, otherwise ImageIO's plugin."""
  return "blender" if "FFMPEG" in available_formats else "imageio"


def _require_imageio_ffmpeg():
  """Imports the fallback encoder before any expensive Blender work."""
  try:
    import imageio_ffmpeg
  except ImportError as error:
    raise ImportError(
        "imageio-ffmpeg is required before rendering Blender branch videos"
    ) from error
  return imageio_ffmpeg


def _verify_rendered_mp4(output: Path) -> None:
  """Rejects missing, empty, or unreadable staged video output."""
  output = Path(output)
  if not output.is_file() or output.stat().st_size < 1:
    raise RuntimeError(
        f"newly rendered MP4 is missing or not nonempty: {output}"
    )

  ffprobe = shutil.which("ffprobe")
  if ffprobe is not None:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(output),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or "video" not in result.stdout.split():
      raise RuntimeError(
          f"ffprobe could not read a video stream from {output}: "
          f"{result.stderr.strip()}"
      )
    return

  imageio_ffmpeg = _require_imageio_ffmpeg()
  command = [
      imageio_ffmpeg.get_ffmpeg_exe(),
      "-v",
      "error",
      "-i",
      str(output),
      "-map",
      "0:v:0",
      "-frames:v",
      "1",
      "-f",
      "null",
      "-",
  ]
  result = subprocess.run(
      command,
      check=False,
      capture_output=True,
      text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(
        f"FFmpeg could not decode a video frame from {output}: "
        f"{result.stderr.strip()}"
    )


def _run_with_temporary_scratch(
    operation: Callable[[Path], Any],
) -> Any:
  """Runs a Blender operation with scratch storage that is always removed."""
  with tempfile.TemporaryDirectory(
      prefix="kubric-blender-scratch-"
  ) as scratch_name:
    return operation(Path(scratch_name))


def _create_replay_scene(
    kb,
    resolution: Tuple[int, int],
    num_frames: int,
    frame_rate: int,
):
  """Creates a render-only timeline with no hidden physics substeps."""
  return kb.Scene(
      resolution=resolution,
      frame_start=1,
      frame_end=num_frames,
      frame_rate=frame_rate,
      step_rate=frame_rate,
  )


def _render_animation_mp4(
    bpy,
    renderer,
    output: Path,
    frame_rate: int,
) -> None:
  """Renders H.264 with Blender FFmpeg or ImageIO's portable fallback."""
  output.parent.mkdir(parents=True, exist_ok=True)
  blender_scene = renderer.blender_scene
  format_items = blender_scene.render.image_settings.bl_rna.properties[
      "file_format"
  ].enum_items
  available_formats = tuple(item.identifier for item in format_items)
  if _encoder_backend(available_formats) == "imageio":
    frames_dict = renderer.render(return_layers=("rgba",))
    _encode_mp4(output, frames_dict["rgba"], frame_rate)
    return

  renderer.set_exr_output_path(None)
  blender_scene.use_nodes = False
  blender_scene.render.filepath = str(output)
  blender_scene.render.image_settings.file_format = "FFMPEG"
  blender_scene.render.ffmpeg.format = "MPEG4"
  blender_scene.render.ffmpeg.codec = "H264"
  blender_scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
  blender_scene.render.ffmpeg.ffmpeg_preset = "GOOD"
  blender_scene.render.ffmpeg.audio_codec = "NONE"
  bpy.ops.render.render(animation=True)
  if not output.is_file() or output.stat().st_size < 1:
    raise RuntimeError(f"Blender did not produce the expected MP4: {output}")


def _build_and_render_branch(
    branch: str,
    replay: Replay,
    output: Path,
    resolution: Tuple[int, int],
    samples_per_pixel: int,
    frame_rate: int,
    save_blend: bool,
) -> Dict[str, object]:
  import kubric as kb
  from kubric.renderer.blender import Blender
  from kubric.safeimport.bpy import bpy

  return _run_with_temporary_scratch(
      lambda scratch_dir: _build_and_render_branch_in_scratch(
          branch,
          replay,
          output,
          resolution,
          samples_per_pixel,
          frame_rate,
          save_blend,
          scratch_dir,
          kb,
          Blender,
          bpy,
      )
  )


def _build_and_render_branch_in_scratch(
    branch: str,
    replay: Replay,
    output: Path,
    resolution: Tuple[int, int],
    samples_per_pixel: int,
    frame_rate: int,
    save_blend: bool,
    scratch_dir: Path,
    kb,
    blender_factory,
    bpy,
) -> Dict[str, object]:
  """Builds one branch using explicitly scoped Blender scratch storage."""

  if replay.branch != branch:
    raise ValueError(
        f"branch {branch!r} does not match replay branch {replay.branch!r}"
    )

  num_frames = len(replay.states)
  scene_specs = _scene_specs()
  material_specs = _material_specs()
  renderer_spec = scene_specs["renderer"]
  camera_spec = scene_specs["camera"]
  scene = _create_replay_scene(
      kb,
      resolution,
      num_frames,
      frame_rate,
  )
  renderer = blender_factory(
      scene,
      scratch_dir=scratch_dir,
      adaptive_sampling=renderer_spec["adaptive_sampling"],
      use_denoising=renderer_spec["denoising"],
      samples_per_pixel=samples_per_pixel,
      background_transparency=renderer_spec["transparent"],
  )
  renderer.blender_scene.render.engine = renderer_spec["engine"]
  renderer.blender_scene.render.film_transparent = renderer_spec["transparent"]

  object_index = {
      name: index for index, name in enumerate(replay.object_ids)
  }
  assets: Dict[str, object] = {}
  for name in replay.object_ids:
    spec = scene_specs["colliders"][name]
    material_spec = material_specs[name]
    material = kb.PrincipledBSDFMaterial(
        name=f"{name}_placeholder",
        color=kb.Color(*material_spec["color"]),
        roughness=material_spec["roughness"],
    )
    initial_position, initial_quaternion = _pose_at(
        replay.states, object_index[name], 0
    )
    if spec["kind"] == "cube":
      asset = kb.Cube(
          name=name,
          scale=spec["scale"],
          position=initial_position,
          quaternion=initial_quaternion,
          material=material,
      )
    else:
      asset = kb.Sphere(
          name=name,
          scale=spec["scale"],
          position=initial_position,
          quaternion=initial_quaternion,
          material=material,
      )
    scene += asset
    assets[name] = asset

  backdrop_spec = material_specs["backdrop"]
  backdrop = kb.Cube(
      name="neutral_backdrop",
      scale=(6.25, 6.25, 0.08),
      position=(0.0, 0.0, -0.62),
      material=kb.PrincipledBSDFMaterial(
          name="backdrop_placeholder",
          color=kb.Color(*backdrop_spec["color"]),
          roughness=backdrop_spec["roughness"],
      ),
  )
  scene += backdrop

  rail_spec = material_specs["rail"]
  rail_layout = (
      ("rail_left", (-4.22, 0.0, 0.13), (0.20, 4.22, 0.22)),
      ("rail_right", (4.22, 0.0, 0.13), (0.20, 4.22, 0.22)),
      ("rail_back", (0.0, 4.22, 0.13), (4.02, 0.20, 0.22)),
      ("rail_front", (0.0, -4.22, 0.13), (4.02, 0.20, 0.22)),
  )
  rails = []
  for rail_name, position, scale in rail_layout:
    rail = kb.Cube(
        name=rail_name,
        position=position,
        scale=scale,
        material=kb.PrincipledBSDFMaterial(
            name=f"{rail_name}_placeholder",
            color=kb.Color(*rail_spec["color"]),
            roughness=rail_spec["roughness"],
        ),
    )
    scene += rail
    rails.append(rail)

  scene.background = kb.Color(0.035, 0.04, 0.05)
  scene.ambient_illumination = kb.Color(*_AMBIENT_ILLUMINATION)
  for light_spec in scene_specs["lights"]:
    light = kb.RectAreaLight(
        name=f"{light_spec['role']}_area",
        position=light_spec["position"],
        color=kb.Color(*light_spec["color"]),
        intensity=light_spec["intensity"],
        width=light_spec["width"],
        height=light_spec["height"],
    )
    light.look_at(camera_spec["look_at"])
    scene += light

  scene.camera = kb.PerspectiveCamera(
      name="camera",
      position=camera_spec["position"],
      look_at=camera_spec["look_at"],
      focal_length=camera_spec["focal_length"],
      sensor_width=36.0,
  )
  _configure_camera_settings(
      scene.camera.linked_objects[renderer],
      camera_spec,
  )

  blender_assets = {
      name: asset.linked_objects[renderer] for name, asset in assets.items()
  }
  for name, blender_object in blender_assets.items():
    blender_object.active_material = _procedural_material(
        bpy,
        f"{name}_procedural",
        material_specs[name],
    )
  blender_assets["floor"].name = "floor_collider"
  blender_assets["target"].name = "target_collider"
  _round_target(blender_assets["target"])

  backdrop.linked_objects[renderer].active_material = _procedural_material(
      bpy, "neutral_backdrop_material", backdrop_spec
  )
  shared_rail_material = _procedural_material(
      bpy, "rail_wood_material", rail_spec
  )
  for rail in rails:
    rail.linked_objects[renderer].active_material = shared_rail_material

  band_material = _procedural_material(
      bpy, "billiard_white_material", material_specs["band"]
  )
  number_material = _procedural_material(
      bpy, "billiard_number_material", material_specs["number"]
  )
  decorations: Dict[str, Tuple[object, ...]] = {}
  for item in _DEMO_SPEC.objects:
    if item.visual_role != "ball":
      continue
    name = item.object_id
    _smooth_mesh(blender_assets[name])
    decorations[name] = _add_ball_decorations(
        bpy,
        blender_assets[name],
        name,
        material_specs[name]["number"],
        material_specs[name]["striped"],
        band_material,
        number_material,
    )

  for name in replay.object_ids:
    asset = assets[name]
    index = object_index[name]
    for step in range(num_frames):
      if not replay.presence[step, index]:
        continue
      frame = scene.frame_start + step
      asset.position, asset.quaternion = _pose_at(
          replay.states, index, step
      )
      asset.keyframe_insert("position", frame)
      asset.keyframe_insert("quaternion", frame)

    visible_objects = (blender_assets[name],) + decorations.get(name, ())
    _insert_visibility_keyframes(
        visible_objects,
        replay.presence[:, index],
        scene.frame_start,
    )

  if save_blend:
    renderer.save_state(str(output.with_suffix(".blend")))
  _render_animation_mp4(bpy, renderer, output, frame_rate)

  return {
      "branch": branch,
      "frames": num_frames,
      "fps": frame_rate,
      "resolution": list(resolution),
      "samples_per_pixel": samples_per_pixel,
      "output": str(output),
  }


def _render_replays_atomically(
    replays: Sequence[Replay],
    states_dir: Path,
    resolution: Tuple[int, int],
    samples_per_pixel: int,
    frame_rate: int,
    save_blend: bool,
) -> List[Dict[str, object]]:
  """Stages every requested render before publishing any final output."""
  states_dir = Path(states_dir)
  results: List[Dict[str, object]] = []
  publications: List[Tuple[Path, Path]] = []
  with tempfile.TemporaryDirectory(
      prefix=".blender-replays-",
      dir=states_dir,
  ) as staging_name:
    staging_dir = Path(staging_name)
    for replay in replays:
      final_output = states_dir / _BRANCH_FILENAMES[replay.branch]
      staged_output = staging_dir / _BRANCH_FILENAMES[replay.branch]
      result = _build_and_render_branch(
          replay.branch,
          replay,
          staged_output,
          resolution,
          samples_per_pixel,
          frame_rate,
          save_blend,
      )
      _verify_rendered_mp4(staged_output)
      published_result = dict(result)
      published_result["output"] = str(final_output)
      results.append(published_result)
      publications.append((staged_output, final_output))

      if save_blend:
        staged_blend = staged_output.with_suffix(".blend")
        if not staged_blend.is_file() or staged_blend.stat().st_size < 1:
          raise RuntimeError(
              f"newly saved Blender scene is missing or empty: {staged_blend}"
          )
        publications.append(
            (staged_blend, final_output.with_suffix(".blend"))
        )

    for staged_output, final_output in publications:
      os.replace(staged_output, final_output)

  return results


def _positive_int(value: str) -> int:
  try:
    parsed = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
        f"expected a positive integer, got {value!r}"
    ) from error
  if parsed < 1:
    raise argparse.ArgumentTypeError(
        f"expected a positive integer, got {value!r}"
    )
  return parsed


class _UniqueBranchesAction(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    if len(values) != len(set(values)):
      parser.error("--branches must not contain duplicates")
    setattr(namespace, self.dest, values)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--states-dir", default=str(_DEFAULT_STATES_DIR))
  parser.add_argument(
      "--branches",
      nargs="+",
      choices=_ALLOWED_BRANCHES,
      default=list(_ALLOWED_BRANCHES),
      action=_UniqueBranchesAction,
  )
  parser.add_argument(
      "--resolution", nargs=2, type=_positive_int, default=[640, 540]
  )
  parser.add_argument("--fps", type=_positive_int, default=_FRAME_RATE)
  parser.add_argument("--samples", type=_positive_int, default=64)
  parser.add_argument(
      "--max-frames",
      type=_positive_int,
      default=None,
      help="render only the first N steps (smoke tests)",
  )
  parser.add_argument("--save-blend", action="store_true")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  states_dir = Path(args.states_dir)
  resolution = (args.resolution[0], args.resolution[1])
  if any(value < 1 for value in resolution):
    raise ValueError("--resolution values must be >= 1")
  if args.samples < 1:
    raise ValueError("--samples must be >= 1")
  if args.fps < 1:
    raise ValueError("--fps must be >= 1")

  replays = _preflight_replays(
      states_dir,
      args.branches,
      args.max_frames,
  )
  _validate_camera_containment(replays, resolution)
  _require_imageio_ffmpeg()
  results = _render_replays_atomically(
      replays,
      states_dir,
      resolution,
      args.samples,
      args.fps,
      args.save_blend,
  )
  for result in results:
    print(json.dumps(result, sort_keys=True), flush=True)

  print(json.dumps({"renders": results}, sort_keys=True))
  return 0


if __name__ == "__main__":  # pragma: no cover - exercised via Docker.
  raise SystemExit(main())
