#!/usr/bin/env python3
"""Render the three branches of a velocity-intervention instance with Blender.

Purpose: turn one ``VelocityInstanceSpec`` plus its rolled-out ``SimulationLog``
branches into the four published outputs — video, mask, graph, and tracking —
from the single ``VisualSceneSpec`` all three branches share.
Public API: enable_gpu, render_profile_from_ranges, render_branch,
render_instance, main.
Dependencies: NumPy and the standard library at import time; Kubric, ``bpy``,
and ``imageio`` are imported lazily inside the render functions, so importing
this module and building a ``RenderProfile`` works without Blender installed.
Trust boundary: pixels do not attest physics. Every annotation written here is
derived from the logged states and contacts, or from Blender's own segmentation
pass; nothing is re-simulated at render time. The per-branch digest reported in
``render.json`` is a claim about the *inputs* handed to Blender, not about the
encoded frames.

Why the GPU handling looks the way it does: setting ``KUBRIC_USE_GPU=true`` sets
``scene.cycles.device = "GPU"`` but never sets
``preferences.addons["cycles"].preferences.compute_device_type``, which is what
actually selects the CUDA/OPTIX backend. On a fresh pip ``bpy`` that stays
``NONE``, the device list is empty, and Cycles falls back to the CPU silently.
The backend must also be chosen *after* the renderer is constructed, because
Blender's constructor calls ``read_factory_settings`` and discards preferences
set beforehand. See ``docs/shared_visual_scene_demo.md`` § Cycles device
selection.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from interventions import appearance, velocity_scenes
from interventions.graph_extraction import contact_log_to_temporal_graph
from interventions.logging import SimulationLog
from interventions.schema import to_jsonable


#: Backends tried in order. OPTIX first: on the RTX cards this repository has
#: been benchmarked on it beats CUDA, and it falls back cleanly when absent.
GPU_BACKENDS: Tuple[str, ...] = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")

#: Layers this harness knows how to publish. ``rgba`` becomes the video,
#: ``segmentation`` becomes the mask, and ``depth`` is stored as raw float.
PUBLISHABLE_LAYERS: Tuple[str, ...] = ("rgba", "segmentation", "depth")

_MASK_PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 60, 50],
        [50, 120, 230],
        [245, 200, 50],
        [60, 175, 95],
        [150, 80, 200],
        [240, 130, 40],
        [40, 200, 205],
        [225, 110, 180],
        [130, 140, 150],
        [190, 230, 70],
    ],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# Render profile
# ---------------------------------------------------------------------------


def render_profile_from_ranges(
    ranges: Mapping[str, Any], *, name: str = "velocity"
) -> appearance.RenderProfile:
  """Builds the ``RenderProfile`` described by a config's ``render:`` block."""
  section = dict(ranges.get("render", {}))
  layers = tuple(section.get("layers", ("rgba", "segmentation", "depth")))
  unknown = [layer for layer in layers if layer not in PUBLISHABLE_LAYERS]
  if unknown:
    raise ValueError(
        "render.layers contains layers this harness cannot publish: {}".format(
            sorted(unknown)
        )
    )
  resolution = tuple(int(value) for value in section.get("resolution", (512, 384)))
  return appearance.RenderProfile(
      name=name,
      resolution=resolution,
      samples_per_pixel=int(section.get("samples_per_pixel", 48)),
      adaptive_sampling=bool(section.get("adaptive_sampling", True)),
      use_denoising=bool(section.get("use_denoising", True)),
      background_transparency=bool(section.get("background_transparency", False)),
      layers=layers,
      device=str(section.get("device", "GPU")).upper(),
  )


# ---------------------------------------------------------------------------
# GPU device selection
# ---------------------------------------------------------------------------


def enable_gpu(
    renderer, preference: Sequence[str] = GPU_BACKENDS
) -> Dict[str, Any]:
  """Selects a Cycles GPU backend and enables its devices.

  Must be called *after* the renderer is constructed. Returns a record of what
  was actually selected, so a caller can publish the device it rendered on
  rather than the device it asked for. Falls back to the CPU — and says so —
  when no backend reports any device.
  """
  import bpy  # pylint: disable=import-outside-toplevel

  addon = bpy.context.preferences.addons.get("cycles")
  if addon is None:
    renderer.blender_scene.cycles.device = "CPU"
    return {"device": "CPU", "backend": None, "devices": [],
            "reason": "cycles addon unavailable"}

  preferences = addon.preferences
  chosen = None
  probed: Dict[str, list] = {}
  for backend in preference:
    try:
      candidates = [device.name for device in
                    preferences.get_devices_for_type(backend)]
    except (TypeError, RuntimeError):
      # Not a valid backend for this build.
      continue
    probed[backend] = candidates
    if candidates and chosen is None:
      chosen = backend

  if chosen is None:
    renderer.blender_scene.cycles.device = "CPU"
    return {"device": "CPU", "backend": None, "devices": [], "probed": probed,
            "reason": "no backend reported a device"}

  preferences.compute_device_type = chosen
  # Populates ``preferences.devices`` for the newly selected backend.
  preferences.get_devices()
  enabled = []
  for device in preferences.devices:
    device.use = device.type == chosen
    if device.use:
      enabled.append(device.name)

  if not enabled:
    renderer.blender_scene.cycles.device = "CPU"
    return {"device": "CPU", "backend": chosen, "devices": [], "probed": probed,
            "reason": "backend selected but no device could be enabled"}

  renderer.blender_scene.cycles.device = "GPU"
  return {"device": "GPU", "backend": chosen, "devices": enabled,
          "probed": probed}


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _socket(node, *names):
  """Returns the first input socket that exists under any of ``names``.

  Blender renamed several Principled BSDF sockets in 4.0 (``Transmission`` ->
  ``Transmission Weight``, ``Emission`` -> ``Emission Color``, ``Specular`` ->
  ``Specular IOR Level``). Probing by name keeps one code path across builds.
  """
  for name in names:
    socket = node.inputs.get(name)
    if socket is not None:
      return socket
  return None


def _set(node, value, *names) -> bool:
  socket = _socket(node, *names)
  if socket is None:
    return False
  socket.default_value = value
  return True


def _rgba(color: Sequence[float]) -> Tuple[float, float, float, float]:
  values = tuple(float(component) for component in color)
  if len(values) == 3:
    return values + (1.0,)
  return values[:4]


def _texture_pattern(bpy, tree, texture: appearance.TextureSpec):
  """Builds the node graph for one sampled texture.

  Returns ``(color_socket, factor_socket)``, or ``(None, None)`` for a solid
  texture, which has no pattern to build.
  """
  if texture.kind == "solid":
    return None, None
  if texture.kind == "image":
    raise ValueError(
        "image textures need a resolvable asset manifest; this pipeline samples "
        "procedural sources only"
    )

  nodes = tree.nodes
  links = tree.links

  coordinates = nodes.new("ShaderNodeTexCoord")
  coordinates.location = (-1000, 0)
  mapping = nodes.new("ShaderNodeMapping")
  mapping.location = (-800, 0)
  # ``rotation`` is stored in turns; Blender wants radians.
  mapping.inputs["Rotation"].default_value = (
      0.0, 0.0, float(texture.rotation) * 2.0 * math.pi
  )
  # Object coordinates rather than generated ones, so the pattern travels with
  # the body instead of swimming across it as the body moves.
  links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])

  scale = max(float(texture.scale), 1e-3)
  detail = float(texture.detail)
  distortion = float(texture.distortion)
  roughness = float(texture.roughness)

  if texture.kind == "checker":
    node = nodes.new("ShaderNodeTexChecker")
    node.location = (-600, 0)
    _set(node, scale, "Scale")
    _set(node, _rgba(texture.colors[0]), "Color1")
    _set(node, _rgba(texture.colors[-1]), "Color2")
    links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    # Checker already carries both colours; no ramp needed.
    return node.outputs["Color"], node.outputs["Fac"]

  if texture.kind == "noise":
    node = nodes.new("ShaderNodeTexNoise")
    _set(node, scale, "Scale")
    _set(node, detail, "Detail")
    _set(node, roughness, "Roughness")
    _set(node, distortion, "Distortion")
  elif texture.kind == "speckle":
    node = nodes.new("ShaderNodeTexVoronoi")
    node.feature = "F1"
    _set(node, scale * 2.0, "Scale")
    _set(node, min(1.0, 0.25 + roughness), "Randomness")
    _set(node, detail, "Detail")
  elif texture.kind in ("wood", "marble"):
    node = nodes.new("ShaderNodeTexWave")
    node.wave_type = "BANDS" if texture.kind == "wood" else "RINGS"
    node.wave_profile = "SIN"
    if texture.kind == "wood":
      node.bands_direction = "X"
    _set(node, scale, "Scale")
    # A wave with no distortion is a barcode; the sampled distortion range is
    # 0-0.4, which is far too subtle to read as grain, so it is amplified here.
    _set(node, 2.0 + distortion * 20.0, "Distortion")
    _set(node, detail, "Detail")
    _set(node, roughness, "Detail Roughness")
  else:
    raise ValueError("unhandled texture kind {!r}".format(texture.kind))

  node.location = (-600, 0)
  links.new(mapping.outputs["Vector"], node.inputs["Vector"])
  factor = node.outputs.get("Fac") or node.outputs.get("Distance")

  ramp = nodes.new("ShaderNodeValToRGB")
  ramp.location = (-400, 0)
  ramp.color_ramp.elements[0].color = _rgba(texture.colors[0])
  ramp.color_ramp.elements[1].color = _rgba(texture.colors[-1])
  links.new(factor, ramp.inputs["Fac"])
  return ramp.outputs["Color"], factor


def build_material(bpy, name: str, spec: appearance.MaterialSpec):
  """Realizes one ``MaterialSpec`` as a Blender material.

  The base BSDF parameters come straight from the spec. The sampled texture, if
  any, drives base colour and modulates roughness so that two bodies of the same
  family still look different.
  """
  material = bpy.data.materials.new(name=name)
  material.use_nodes = True
  tree = material.node_tree
  bsdf = tree.nodes.get("Principled BSDF")
  if bsdf is None:
    raise RuntimeError("new material has no Principled BSDF node")

  base_color = _rgba(spec.base_color)
  _set(bsdf, base_color, "Base Color")
  _set(bsdf, float(spec.metallic), "Metallic")
  _set(bsdf, float(spec.roughness), "Roughness")
  _set(bsdf, float(spec.ior), "IOR")
  _set(bsdf, float(spec.specular), "Specular IOR Level", "Specular")
  _set(bsdf, float(spec.transmission), "Transmission Weight", "Transmission")

  emission = _rgba(spec.emission)
  if _set(bsdf, emission, "Emission Color", "Emission"):
    # Blender 4.x defaults Emission Strength to 1.0, so a black emission colour
    # is already a no-op; setting it explicitly keeps 3.x builds consistent.
    _set(bsdf, 1.0 if any(emission[:3]) else 0.0, "Emission Strength")

  color_socket, factor_socket = _texture_pattern(bpy, tree, spec.texture)
  if color_socket is not None:
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.location = (-200, 100)
    mix.blend_type = "MIX"
    mix.inputs["Fac"].default_value = 0.75
    mix.inputs["Color1"].default_value = base_color
    tree.links.new(color_socket, mix.inputs["Color2"])
    tree.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    roughness_socket = _socket(bsdf, "Roughness")
    if factor_socket is not None and roughness_socket is not None:
      spread = tree.nodes.new("ShaderNodeMapRange")
      spread.location = (-200, -150)
      spread.inputs["From Min"].default_value = 0.0
      spread.inputs["From Max"].default_value = 1.0
      spread.inputs["To Min"].default_value = max(0.0, spec.roughness * 0.6)
      spread.inputs["To Max"].default_value = min(1.0, spec.roughness * 1.4 + 0.05)
      tree.links.new(factor_socket, spread.inputs["Value"])
      tree.links.new(spread.outputs["Result"], roughness_socket)

  return material


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


def _add_light(kb, light: appearance.LightSpec):
  color = kb.Color(*_rgba(light.color)[:3])
  common = {
      "name": light.light_id,
      "position": tuple(light.position),
      "color": color,
      "intensity": float(light.intensity),
  }
  if light.kind == "rect_area":
    asset = kb.RectAreaLight(
        width=float(light.width or 1.0),
        height=float(light.height or 1.0),
        **common,
    )
  elif light.kind == "point":
    asset = kb.PointLight(**common)
  elif light.kind == "directional":
    asset = kb.DirectionalLight(**common)
  elif light.kind == "spot":
    asset = kb.SpotLight(
        spot_size=float(light.spot_size or 0.8),
        spot_blend=float(light.spot_blend or 0.15),
        **common,
    )
  else:
    raise ValueError("unhandled light kind {!r}".format(light.kind))
  asset.look_at(tuple(light.look_at))
  return asset


def _pose(states: np.ndarray, step: int, row: int):
  entry = states[step, row]
  return (
      tuple(float(value) for value in entry[0:3]),
      tuple(float(value) for value in entry[3:7]),
  )


def _build_scene(
    kb,
    spec: velocity_scenes.VelocityInstanceSpec,
    log: SimulationLog,
    profile: appearance.RenderProfile,
):
  """Creates the render-only timeline, assets, lights, camera, and background.

  The timeline is one Blender frame per sampled physics step, with
  ``step_rate == frame_rate``: nothing is simulated here, so a hidden substep
  rate would only invite the illusion that it is.
  """
  visual = spec.visual_scene
  if visual is None:
    raise ValueError("instance has no VisualSceneSpec; appearance must be enabled")

  frame_steps = tuple(visual.frame_steps)
  num_frames = len(frame_steps)
  scene = kb.Scene(
      resolution=tuple(profile.resolution),
      frame_start=1,
      frame_end=num_frames,
      frame_rate=spec.scene_config.frame_rate,
      step_rate=spec.scene_config.frame_rate,
      gravity=spec.scene_config.gravity,
  )

  rendered_ids = tuple(log.object_ids)
  by_id = {item.object_id: item for item in spec.scene_config.objects}
  materials_by_id = {item.object_id: item.material for item in visual.objects}

  assets: Dict[str, Any] = {}
  for row, object_id in enumerate(rendered_ids):
    config = by_id[object_id]
    position, quaternion = _pose(log.states, frame_steps[0], row)
    constructor = kb.Cube if config.shape == "cube" else kb.Sphere
    asset = constructor(
        name=object_id,
        scale=tuple(config.size),
        position=position,
        quaternion=quaternion,
        static=True,
        # The mask's integer labels are the tracking arrays' row indices plus
        # one, so a reader can index states by mask value without a lookup.
        segmentation_id=row + 1,
        material=kb.PrincipledBSDFMaterial(
            name="{}_placeholder".format(object_id),
            color=kb.Color(*_rgba(materials_by_id[object_id].base_color)),
            roughness=float(materials_by_id[object_id].roughness),
        ),
        metadata={"logical_id": object_id, "row": row},
    )
    scene += asset
    assets[object_id] = asset

  background = visual.background
  if background.kind != "color":
    raise ValueError(
        "only solid-colour backgrounds are supported here; got {!r}".format(
            background.kind
        )
    )
  background_color = _rgba(background.color)
  scene.background = kb.Color(*background_color)
  # A uniform-colour world lights the scene as well as showing behind it. The
  # ambient node is set to the same colour so the visible backdrop and the fill
  # it casts cannot disagree; the three-point rig sits on top of that.
  scene.ambient_illumination = kb.Color(*background_color)

  for light in visual.lights:
    scene += _add_light(kb, light)

  camera_position = tuple(visual.camera.positions[0])
  camera_look_at = tuple(visual.camera.look_ats[0])
  scene.camera = kb.PerspectiveCamera(
      name="camera",
      position=camera_position,
      look_at=camera_look_at,
      focal_length=float(visual.camera.focal_length),
      sensor_width=float(visual.camera.sensor_width),
  )
  return scene, assets, rendered_ids, frame_steps


def _keyframe(scene, assets, rendered_ids, log, frame_steps) -> None:
  """Writes one keyframe per rendered frame from the logged states.

  The renderer must already exist: Kubric forwards keyframes to Blender through
  traitlets observers registered when the renderer is constructed, so keyframes
  inserted earlier are silently dropped.
  """
  for row, object_id in enumerate(rendered_ids):
    asset = assets[object_id]
    for offset, step in enumerate(frame_steps):
      asset.position, asset.quaternion = _pose(log.states, step, row)
      frame = scene.frame_start + offset
      asset.keyframe_insert("position", frame)
      asset.keyframe_insert("quaternion", frame)


def _configure_camera(scene, visual: appearance.VisualSceneSpec) -> None:
  """Keyframes the camera path and applies its clipping range.

  For this dataset every entry of ``positions`` is the same point — the camera
  is fixed within a clip and resampled between clips — but the path is
  keyframed rather than assumed static, so the same code renders a moving
  camera if a future config asks for one.
  """
  near, far = visual.camera.clipping_range
  for offset, (position, look_at) in enumerate(
      zip(visual.camera.positions, visual.camera.look_ats)
  ):
    scene.camera.position = tuple(position)
    scene.camera.look_at(tuple(look_at))
    frame = scene.frame_start + offset
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)
  return float(near), float(far)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _to_uint8(frames: np.ndarray) -> np.ndarray:
  frames = frames[..., :3]
  if np.issubdtype(frames.dtype, np.floating):
    return (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
  if frames.dtype == np.uint16:
    return (frames / 257).astype(np.uint8)
  if frames.dtype != np.uint8:
    return frames.astype(np.uint8)
  return frames


def _encode_mp4(output: Path, frames: np.ndarray, frame_rate: int) -> None:
  """Encodes RGB frames to H.264.

  Pip-distributed ``bpy`` wheels list ``FFMPEG`` in the image-format enum
  without a working muxer behind it, so this never asks Blender to write the
  movie; it encodes the returned array through ``imageio-ffmpeg``, which ships
  its own binary and needs nothing on ``PATH``.
  """
  import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel

  output.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(
      output,
      _to_uint8(frames),
      fps=int(frame_rate),
      codec="libx264",
      quality=8,
      macro_block_size=None,
  )
  if not output.is_file() or output.stat().st_size < 1:
    raise RuntimeError("encoder produced no file at {}".format(output))


def _colorize_mask(segmentation: np.ndarray) -> np.ndarray:
  labels = np.asarray(segmentation).reshape(segmentation.shape[:3])
  palette = _MASK_PALETTE
  if int(labels.max(initial=0)) >= len(palette):
    # Deterministic extension rather than a wrap-around, so two labels never
    # share a colour in a preview a human is meant to read.
    extra = int(labels.max()) + 1 - len(palette)
    rng = np.random.default_rng(0xB1A5)
    palette = np.concatenate(
        [palette, rng.integers(40, 240, size=(extra, 3), dtype=np.uint8)]
    )
  return palette[np.clip(labels, 0, len(palette) - 1)]


def _bounding_boxes(
    segmentation: np.ndarray, num_objects: int
) -> Tuple[np.ndarray, np.ndarray]:
  """Per-frame pixel bounding boxes and visible pixel counts, from the mask.

  Boxes are ``(min_row, min_col, max_row, max_col)`` inclusive, and ``-1``
  everywhere an object contributes no pixels — occluded, off-frame, or, in the
  removal branch, absent.
  """
  labels = np.asarray(segmentation).reshape(segmentation.shape[:3])
  num_frames = labels.shape[0]
  boxes = np.full((num_frames, num_objects, 4), -1, dtype=np.int32)
  counts = np.zeros((num_frames, num_objects), dtype=np.int32)
  for frame in range(num_frames):
    plane = labels[frame]
    for row in range(num_objects):
      mask = plane == (row + 1)
      count = int(mask.sum())
      counts[frame, row] = count
      if count == 0:
        continue
      rows = np.nonzero(mask.any(axis=1))[0]
      cols = np.nonzero(mask.any(axis=0))[0]
      boxes[frame, row] = (rows[0], cols[0], rows[-1], cols[-1])
  return boxes, counts


def _tracking(
    scene,
    log: SimulationLog,
    frame_steps: Sequence[int],
    resolution: Tuple[int, int],
    segmentation: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
  """Assembles the tracking annotations for one branch.

  3-D state comes from the log, unchanged. 2-D image positions come from the
  same camera Blender rendered with, evaluated at the frame the projection
  belongs to. Boxes and visibility come from the rendered mask when one was
  requested, and are omitted rather than guessed when it was not.
  """
  width, height = resolution
  num_frames = len(frame_steps)
  num_objects = len(log.object_ids)

  states = np.stack([log.states[step] for step in frame_steps], axis=0)
  image_positions = np.zeros((num_frames, num_objects, 2), dtype=np.float32)
  in_front = np.zeros((num_frames, num_objects), dtype=bool)
  matrix_world = np.zeros((num_frames, 4, 4), dtype=np.float32)
  intrinsics = np.zeros((num_frames, 3, 3), dtype=np.float32)

  for offset in range(num_frames):
    frame = scene.frame_start + offset
    with scene.camera.at_frame(frame):
      matrix_world[offset] = scene.camera.matrix_world
      intrinsics[offset] = scene.camera.intrinsics
    for row in range(num_objects):
      projected = scene.camera.project_point(states[offset, row, 0:3], frame=frame)
      image_positions[offset, row] = (
          float(projected[0]) * width,
          float(projected[1]) * height,
      )
      in_front[offset, row] = float(projected[2]) > 0.0

  payload = {
      # Fixed-width unicode, not ``dtype=object``. An object array is pickled
      # inside the npz, and a reader then has to pass ``allow_pickle=True`` to
      # get at the numbers next to it -- which is arbitrary code execution on a
      # file downloaded from a dataset host. These are short ASCII ids; they fit.
      "object_ids": np.array(list(log.object_ids), dtype=str),
      "frame_steps": np.asarray(frame_steps, dtype=np.int64),
      "states": states.astype(np.float64),
      "image_positions": image_positions,
      "in_front_of_camera": in_front,
      "camera_matrix_world": matrix_world,
      "camera_intrinsics": intrinsics,
      "resolution": np.asarray(resolution, dtype=np.int32),
      "state_layout": np.array(
          ["position_xyz", "quaternion_wxyz", "linear_velocity", "angular_velocity"],
          dtype=str,
      ),
  }
  if segmentation is not None:
    boxes, counts = _bounding_boxes(segmentation, num_objects)
    payload["bboxes"] = boxes
    payload["visible_pixels"] = counts
    payload["visible"] = counts > 0
  return payload


def _write_graph(path: Path, log: SimulationLog) -> Dict[str, Any]:
  """Writes the branch's temporal contact graph beside its pixels."""
  graph = contact_log_to_temporal_graph(log.contacts, log.step_rate)
  payload = {
      "schema_version": "1.0",
      "branch": log.branch,
      "step_rate": log.step_rate,
      "object_ids": list(log.object_ids),
      "graph": to_jsonable(graph),
      "contact_count": len(log.contacts),
  }
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  return payload


# ---------------------------------------------------------------------------
# Branch rendering
# ---------------------------------------------------------------------------


def render_branch(
    spec: velocity_scenes.VelocityInstanceSpec,
    log: SimulationLog,
    profile: appearance.RenderProfile,
    output_dir: Path,
    *,
    use_gpu: bool = True,
    save_blend: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
  """Renders one branch and writes its video, mask, graph, and tracking."""
  import kubric as kb  # pylint: disable=import-outside-toplevel
  from kubric.renderer.blender import Blender as KubricRenderer  # pylint: disable=import-outside-toplevel

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  layers = tuple(profile.layers)
  if "rgba" not in layers:
    raise ValueError("render.layers must include rgba; it is the published video")

  started = time.perf_counter()
  with tempfile.TemporaryDirectory(prefix="kubric-velocity-render-") as scratch:
    scene, assets, rendered_ids, frame_steps = _build_scene(kb, spec, log, profile)
    renderer = KubricRenderer(
        scene,
        scratch_dir=Path(scratch),
        adaptive_sampling=profile.adaptive_sampling,
        use_denoising=profile.use_denoising,
        samples_per_pixel=profile.samples_per_pixel,
        background_transparency=profile.background_transparency,
        verbose=verbose,
    )

    device_info = (
        enable_gpu(renderer)
        if use_gpu and profile.device == "GPU"
        else {"device": "CPU", "backend": None, "devices": []}
    )
    if device_info["device"] == "CPU":
      renderer.blender_scene.cycles.device = "CPU"

    # Renderer exists, so keyframes now reach Blender's animation curves.
    near, far = _configure_camera(scene, spec.visual_scene)
    blender_camera = scene.camera.linked_objects[renderer]
    blender_camera.data.clip_start = near
    blender_camera.data.clip_end = far

    _keyframe(scene, assets, rendered_ids, log, frame_steps)

    materials_by_id = {item.object_id: item.material for item in spec.visual_scene.objects}
    import bpy  # pylint: disable=import-outside-toplevel
    for object_id, asset in assets.items():
      asset.linked_objects[renderer].active_material = build_material(
          bpy, "{}_material".format(object_id), materials_by_id[object_id]
      )

    if save_blend:
      renderer.save_state(str(output_dir / "scene.blend"))

    render_started = time.perf_counter()
    frames = renderer.render(
        frames=range(scene.frame_start, scene.frame_end + 1),
        return_layers=layers,
    )
    render_seconds = time.perf_counter() - render_started

    frame_rate = spec.scene_config.frame_rate
    _encode_mp4(output_dir / "video.mp4", frames["rgba"], frame_rate)

    segmentation = None
    if "segmentation" in frames:
      segmentation = np.asarray(frames["segmentation"]).astype(np.int32)
      np.savez_compressed(
          output_dir / "segmentation.npz",
          segmentation=segmentation.astype(np.uint16),
          object_ids=np.array(list(rendered_ids), dtype=str),
          # label == row index + 1; 0 is background.
          label_offset=np.int32(1),
      )
      _encode_mp4(
          output_dir / "segmentation_preview.mp4",
          _colorize_mask(segmentation),
          frame_rate,
      )

    if "depth" in frames:
      np.savez_compressed(
          output_dir / "depth.npz", depth=np.asarray(frames["depth"], dtype=np.float32)
      )

    tracking = _tracking(
        scene, log, frame_steps, tuple(profile.resolution), segmentation
    )
    np.savez_compressed(output_dir / "tracking.npz", **tracking)

    graph_payload = _write_graph(output_dir / "graph.json", log)

  return {
      "branch": log.branch,
      "frames": len(frame_steps),
      "frame_rate": spec.scene_config.frame_rate,
      "resolution": list(profile.resolution),
      "samples_per_pixel": profile.samples_per_pixel,
      "layers": list(layers),
      "device": device_info,
      "render_seconds": round(render_seconds, 3),
      "wall_seconds": round(time.perf_counter() - started, 3),
      "contact_edges": len(graph_payload["graph"].get("edges", ())),
      "outputs": sorted(
          str(path.relative_to(output_dir)) for path in output_dir.iterdir()
      ),
  }


def render_instance(
    spec: velocity_scenes.VelocityInstanceSpec,
    logs: Mapping[str, SimulationLog],
    profile: appearance.RenderProfile,
    output_dir: Path,
    *,
    branches: Sequence[str] = velocity_scenes.BRANCHES,
    use_gpu: bool = True,
    save_blend: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
  """Renders every requested branch of one instance from its shared appearance."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  missing = [branch for branch in branches if branch not in logs]
  if missing:
    raise ValueError("no simulation log for branches {}".format(sorted(missing)))

  results = {}
  for branch in branches:
    results[branch] = render_branch(
        spec,
        logs[branch],
        profile,
        output_dir / branch,
        use_gpu=use_gpu,
        save_blend=save_blend,
        verbose=verbose,
    )

  record = {
      "schema_version": "1.0",
      "trust_model": velocity_scenes.VELOCITY_TRUST_MODEL,
      "instance_id": spec.instance_id,
      # One appearance record, three branches. The hash is a claim about what
      # was handed to Blender, not about the encoded pixels.
      "visual_scene_hash": spec.visual_scene_hash,
      "render_profile_hash": appearance.render_profile_hash(profile),
      "render_profile": to_jsonable(profile),
      "branches": {branch: results[branch] for branch in branches},
  }
  (output_dir / "render.json").write_text(
      json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
  )
  return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Render the three branches of one velocity-intervention instance."
  )
  parser.add_argument(
      "--config", type=Path, default=Path("configs/scene_ranges_velocity.yaml")
  )
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--branches", nargs="+", default=list(velocity_scenes.BRANCHES)
  )
  parser.add_argument(
      "--device", choices=("GPU", "CPU"), default=None,
      help="Overrides render.device from the config.",
  )
  parser.add_argument("--samples-per-pixel", type=int, default=None)
  parser.add_argument(
      "--resolution", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT")
  )
  parser.add_argument("--save-blend", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  parser.add_argument(
      "--allow-rejected", action="store_true",
      help="Render even when QC rejects the instance.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Renders one instance's requested branches and returns its CLI exit status."""
  args = _parser().parse_args(argv)

  unknown = [b for b in args.branches if b not in velocity_scenes.BRANCHES]
  if unknown:
    raise SystemExit("unknown branches: {}".format(sorted(unknown)))

  ranges = velocity_scenes.load_ranges(args.config)
  profile = render_profile_from_ranges(ranges)
  if args.device is not None:
    profile = dataclasses.replace(profile, device=args.device)
  if args.samples_per_pixel is not None:
    profile = dataclasses.replace(profile, samples_per_pixel=args.samples_per_pixel)
  if args.resolution is not None:
    profile = dataclasses.replace(profile, resolution=tuple(args.resolution))

  spec = velocity_scenes.sample_instance(ranges, args.seed, args.index)
  logs = velocity_scenes.generate_triplet(spec)
  # The camera is only fixable on the rollout once the rollout exists.
  spec = velocity_scenes.refit_camera(ranges, spec, logs)
  qc = ranges.get("qc", {})
  pair_truth = velocity_scenes.pair_ground_truth(
      logs["factual"], logs["counterfactual"], spec.subject_id, qc
  )
  removal_truth = velocity_scenes.removal_ground_truth(
      logs["factual"], logs["subject_removed"], spec.subject_id, qc
  )
  qc_result = velocity_scenes.evaluate_qc(spec, logs, pair_truth, qc)
  if not qc_result.accepted and not args.allow_rejected:
    print(
        "instance {} rejected by QC: {}".format(
            spec.instance_id, ", ".join(qc_result.reasons)
        ),
        file=sys.stderr,
    )
    return 2

  instance_dir = Path(args.output) / spec.instance_id
  instance_dir.mkdir(parents=True, exist_ok=True)
  summary = velocity_scenes.instance_summary(
      spec, logs, pair_truth, removal_truth, qc_result
  )
  (instance_dir / "instance.json").write_text(
      json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
  )

  record = render_instance(
      spec,
      logs,
      profile,
      instance_dir,
      branches=args.branches,
      use_gpu=profile.device == "GPU",
      save_blend=args.save_blend,
      verbose=args.verbose,
  )
  print(json.dumps(record["branches"], indent=2, sort_keys=True))
  return 0


__all__ = [
    "GPU_BACKENDS",
    "PUBLISHABLE_LAYERS",
    "build_material",
    "enable_gpu",
    "main",
    "render_branch",
    "render_instance",
    "render_profile_from_ranges",
]


if __name__ == "__main__":
  raise SystemExit(main())
