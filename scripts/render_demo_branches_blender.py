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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_DEFAULT_STATES_DIR = Path("output/demo_collision_intervention")
_ALLOWED_BRANCHES = ("normal", "trajectory_changed", "target_removed")
_BRANCH_FILENAMES = {
    "normal": "normal_blender.mp4",
    "trajectory_changed": "trajectory_changed_blender.mp4",
    "target_removed": "target_removed_blender.mp4",
}
_FRAME_RATE = 24

# State layout matches SimulationLog: position XYZ, quaternion WXYZ, then linear
# and angular velocity. The ordering is persisted in summary.json and fixed for
# this deterministic demo.
_CANONICAL_OBJECT_IDS = ("floor", "lower_ball", "target", "upper_ball")
_STATE_STRIDE = 13
_POSITION_SLICE = slice(0, 3)
_QUATERNION_WXYZ_SLICE = slice(3, 7)
_SYNCHRONIZED_SUMMARY_KEYS = (
    "branches",
    "ground_truth",
    "intervention_end",
    "intervention_start",
    "intervention_window",
    "object_ids",
    "seed",
    "step_rate",
)

# Visual counterparts of the physics demo's scene. Sizes mirror the PyBullet
# collision shapes: cube half-extents and sphere radii in metres.
_COLLIDER_SPECS = {
    "floor": {"kind": "cube", "scale": (4.0, 4.0, 0.25)},
    "lower_ball": {"kind": "sphere", "scale": 0.26},
    "target": {"kind": "cube", "scale": (0.18, 0.18, 0.18)},
    "upper_ball": {"kind": "sphere", "scale": 0.26},
}

_CAMERA_POSITION = (5.9, -7.2, 6.8)
_CAMERA_LOOK_AT = (0.0, 0.0, -0.05)
_CAMERA_FOCAL_LENGTH = 52.0
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


def _material_specs() -> Dict[str, Dict[str, Any]]:
  """Returns fresh deterministic specifications for all procedural materials."""
  specs = {
      "floor": {
          "material": "felt",
          "color": (0.018, 0.165, 0.075, 1.0),
          "roughness": 0.96,
          "noise_scale": 92.0,
          "bump_strength": 0.16,
      },
      "target": {
          "material": "wood",
          "color": (0.31, 0.105, 0.028, 1.0),
          "light_color": (0.72, 0.31, 0.075, 1.0),
          "roughness": 0.34,
          "grain_scale": 5.5,
      },
      "upper_ball": {
          "material": "lacquer",
          "color": (0.82, 0.035, 0.025, 1.0),
          "roughness": 0.12,
          "metallic": 0.05,
          "number": "3",
      },
      "lower_ball": {
          "material": "lacquer",
          "color": (0.035, 0.12, 0.72, 1.0),
          "roughness": 0.1,
          "metallic": 0.08,
          "number": "8",
      },
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
  return copy.deepcopy(specs)


def _scene_specs() -> Dict[str, Any]:
  """Returns the collider and studio contract without importing Blender."""
  specs = {
      "colliders": _COLLIDER_SPECS,
      "camera": {
          "position": _CAMERA_POSITION,
          "look_at": _CAMERA_LOOK_AT,
          "focal_length": _CAMERA_FOCAL_LENGTH,
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
  return summary


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

  if states.ndim != 3 or states.shape[1:] != (
      len(_CANONICAL_OBJECT_IDS), _STATE_STRIDE
  ):
    raise ValueError(
        f"{states_path} states have shape {states.shape}; expected "
        f"(frames, {len(_CANONICAL_OBJECT_IDS)}, {_STATE_STRIDE})"
    )
  if states.shape[0] < 1:
    raise ValueError(f"{states_path} must contain at least one frame")
  if states.dtype.kind not in "fiu":
    raise TypeError(f"{states_path} states must be numeric")
  if not np.isfinite(states).all():
    raise ValueError(f"{states_path} contains non-finite states")
  if presence.dtype.kind != "b":
    raise TypeError(f"{presence_path} presence must contain Boolean values")
  if presence.shape != states.shape[:2]:
    raise ValueError(
        f"{presence_path} presence has shape {presence.shape}; expected "
        f"{states.shape[:2]} to match the states frame/object count"
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


def _preflight_replays(
    states_dir: Path,
    branches: Sequence[str],
    max_frames: int | None,
) -> Tuple[Replay, ...]:
  """Loads and prepares every requested replay before any rendering begins."""
  replays = tuple(
      _prepare_replay(_load_replay(states_dir, branch), max_frames)
      for branch in branches
  )
  _validate_synchronized_replays(replays)
  return replays


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
    number: str,
    band_material,
    number_material,
) -> Tuple[object, ...]:
  """Adds a raised white stripe and number badge in the ball's local frame."""
  import math

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

  bpy.ops.object.text_add(align="WORLD", location=(0.0, 0.0, 0.0))
  decal = _parent_local(
      bpy.context.object,
      parent,
      (0.0, -1.015, 0.0),
      (math.pi / 2.0, 0.0, 0.0),
  )
  decal.name = f"{object_id}_number"
  decal.data.body = number
  decal.data.align_x = "CENTER"
  decal.data.align_y = "CENTER"
  decal.data.size = 0.48
  decal.data.extrude = 0.008
  decal.data.bevel_depth = 0.004
  decal.data.materials.append(number_material)
  return band, badge, decal


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

  if replay.branch != branch:
    raise ValueError(
        f"branch {branch!r} does not match replay branch {replay.branch!r}"
    )

  num_frames = len(replay.states)
  scene_specs = _scene_specs()
  material_specs = _material_specs()
  scene = kb.Scene(
      resolution=resolution,
      frame_start=1,
      frame_end=num_frames,
      frame_rate=frame_rate,
  )
  renderer = Blender(
      scene,
      adaptive_sampling=True,
      use_denoising=True,
      samples_per_pixel=samples_per_pixel,
      background_transparency=False,
  )
  renderer.blender_scene.render.engine = "CYCLES"
  renderer.blender_scene.render.film_transparent = False

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
    light.look_at(_CAMERA_LOOK_AT)
    scene += light

  scene.camera = kb.PerspectiveCamera(
      name="camera",
      position=_CAMERA_POSITION,
      look_at=_CAMERA_LOOK_AT,
      focal_length=_CAMERA_FOCAL_LENGTH,
      sensor_width=36.0,
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
  _smooth_mesh(blender_assets["lower_ball"])
  _smooth_mesh(blender_assets["upper_ball"])

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
  for name in ("lower_ball", "upper_ball"):
    decorations[name] = _add_ball_decorations(
        bpy,
        blender_assets[name],
        name,
        material_specs[name]["number"],
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


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--states-dir", default=str(_DEFAULT_STATES_DIR))
  parser.add_argument(
      "--branches",
      nargs="+",
      choices=_ALLOWED_BRANCHES,
      default=list(_ALLOWED_BRANCHES),
  )
  parser.add_argument("--resolution", nargs=2, type=int, default=[640, 540])
  parser.add_argument("--fps", type=int, default=_FRAME_RATE)
  parser.add_argument("--samples", type=int, default=64)
  parser.add_argument(
      "--max-frames",
      type=int,
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
  results: List[Dict[str, object]] = []
  for replay in replays:
    branch = replay.branch
    output = states_dir / _BRANCH_FILENAMES[branch]
    result = _build_and_render_branch(
        branch,
        replay,
        output,
        resolution,
        args.samples,
        args.fps,
        args.save_blend,
    )
    results.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)

  print(json.dumps({"renders": results}, sort_keys=True))
  return 0


if __name__ == "__main__":  # pragma: no cover - exercised via Docker.
  raise SystemExit(main())
