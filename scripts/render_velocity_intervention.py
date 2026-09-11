"""Renders the three branches of one velocity-intervention instance with Blender.

Purpose:
  Turns the physics artifacts written by
  :func:`interventions.velocity_intervention.write_instance` into the dataset's
  visual outputs. For each branch the same visual scene and the same static
  camera are rebuilt with :mod:`kubric` primitives, object poses are keyframed
  from the logged states, and Blender/Cycles renders RGB, segmentation, depth
  and optical flow. Outputs are ``video.mp4``, ``mask.mp4``, ``depth.mp4``,
  ``flow.mp4`` plus ``segmentation.npz``, ``depth.npz``, ``forward_flow.npz``,
  ``tracking.npz`` (poses, velocities, 2D projections, boxes, visibility, presence)
  and ``render_info.json``. Cycles is
  pointed at a GPU backend (OptiX, then CUDA, HIP, oneAPI, Metal) when one is
  available and falls back to the CPU explicitly, recording the choice.

Public API:
  ``enable_gpu``, ``build_material``, ``render_branch``, ``render_instance``,
  ``main``.

Dependencies:
  ``bpy`` (Blender 4.2 as a Python module or inside Blender), :mod:`kubric`,
  NumPy, ``imageio-ffmpeg`` for H.264 encoding, and
  :mod:`interventions.velocity_intervention` for reading instance artifacts.

Trust boundary:
  Instance directories are trusted only after ``read_instance_spec`` and
  ``read_simulation_log`` validate them; mismatched object sets abort the render.
  Every output is written to a temporary sibling directory and moved into place
  only after all frames rendered, so a partially rendered branch never looks
  complete. The reported ``device`` is the one Cycles actually used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from interventions import appearance  # noqa: E402  pylint: disable=wrong-import-position
from interventions import velocity_intervention as vi  # noqa: E402  pylint: disable=wrong-import-position
from interventions.logging import (  # noqa: E402  pylint: disable=wrong-import-position
    ANGULAR_VELOCITY_SLICE,
    LINEAR_VELOCITY_SLICE,
    POSITION_SLICE,
    QUATERNION_SLICE,
    SimulationLog,
)

GPU_BACKENDS: Tuple[str, ...] = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")
DEFAULT_LAYERS: Tuple[str, ...] = ("rgba", "segmentation", "depth", "forward_flow")
OPTIONAL_LAYERS: Tuple[str, ...] = ("backward_flow", "normal")
_MASK_PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
    [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------


def enable_gpu(renderer, preference: Sequence[str] = GPU_BACKENDS, allow_cpu: bool = True) -> Dict[str, Any]:
  """Selects a Cycles GPU backend, enables its devices and reports what was chosen.

  ``KUBRIC_USE_GPU=true`` alone only flips ``cycles.device``; Cycles additionally
  needs ``compute_device_type`` set and each device's ``use`` flag enabled, which
  is what this does. When no backend exposes a device the renderer stays on the
  CPU (or raises when ``allow_cpu`` is false) and the returned record says so.
  """
  import bpy  # pylint: disable=import-outside-toplevel

  addon = bpy.context.preferences.addons.get("cycles")
  if addon is None:
    if not allow_cpu:
      raise RuntimeError("cycles addon unavailable")
    renderer.blender_scene.cycles.device = "CPU"
    return {"device": "CPU", "backend": None, "devices": [], "reason": "cycles addon unavailable"}

  preferences = addon.preferences
  probed: Dict[str, list] = {}
  chosen = None
  for backend in preference:
    try:
      candidates = [device.name for device in preferences.get_devices_for_type(backend)]
    except (TypeError, RuntimeError, ValueError):
      continue
    probed[backend] = candidates
    if candidates and chosen is None:
      chosen = backend

  if chosen is None:
    if not allow_cpu:
      raise RuntimeError("no Cycles GPU backend reported a device: {}".format(probed))
    renderer.blender_scene.cycles.device = "CPU"
    return {"device": "CPU", "backend": None, "devices": [], "probed": probed,
            "reason": "no backend reported a device"}

  preferences.compute_device_type = chosen
  preferences.get_devices()
  enabled = []
  for device in preferences.devices:
    device.use = device.type == chosen
    if device.use:
      enabled.append(device.name)
  if not enabled:
    if not allow_cpu:
      raise RuntimeError("backend {} selected but no device could be enabled".format(chosen))
    renderer.blender_scene.cycles.device = "CPU"
    return {"device": "CPU", "backend": chosen, "devices": [], "probed": probed,
            "reason": "backend selected but no device could be enabled"}

  renderer.blender_scene.cycles.device = "GPU"
  if chosen == "OPTIX":
    try:
      renderer.blender_scene.cycles.denoiser = "OPTIX"
    except TypeError:  # pragma: no cover - denoiser enum differs between builds
      pass
  return {"device": "GPU", "backend": chosen, "devices": enabled, "probed": probed}


# ---------------------------------------------------------------------------
# Materials from MaterialSpec / TextureSpec
# ---------------------------------------------------------------------------


def _socket(node, *names):
  for name in names:
    if name in node.inputs:
      return node.inputs[name]
  return None


def _set(node, value, *names) -> bool:
  socket = _socket(node, *names)
  if socket is None:
    return False
  socket.default_value = value
  return True


def _rgba(color: Sequence[float]) -> Tuple[float, float, float, float]:
  values = tuple(float(c) for c in color)
  return values if len(values) == 4 else (values + (1.0,))[:4]


def _texture_pattern(tree, texture: appearance.TextureSpec, base_color):
  """Builds a colour-producing node subgraph for ``texture``; returns its output socket."""
  nodes, links = tree.nodes, tree.links
  colors = [_rgba(c) for c in texture.colors] or [base_color]
  if texture.kind in ("solid", "image") or len(colors) < 1:
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = colors[0] if texture.colors else base_color
    return rgb.outputs[0]

  coords = nodes.new("ShaderNodeTexCoord")
  mapping = nodes.new("ShaderNodeMapping")
  mapping.inputs["Scale"].default_value = (texture.scale, texture.scale, texture.scale)
  mapping.inputs["Rotation"].default_value = (0.0, 0.0, texture.rotation * math.tau)
  links.new(coords.outputs["Object"], mapping.inputs["Vector"])

  ramp = nodes.new("ShaderNodeValToRGB")
  ramp.color_ramp.interpolation = "LINEAR"
  stops = colors if len(colors) > 1 else [colors[0], base_color]
  while len(ramp.color_ramp.elements) < len(stops):
    ramp.color_ramp.elements.new(0.5)
  for index, (element, color) in enumerate(zip(ramp.color_ramp.elements, stops)):
    element.position = index / max(len(stops) - 1, 1)
    element.color = color

  if texture.kind == "checker":
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = stops[0]
    checker.inputs["Color2"].default_value = stops[-1]
    checker.inputs["Scale"].default_value = 1.0
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    return checker.outputs["Color"]
  if texture.kind == "wood":
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "RINGS"
    wave.rings_direction = "Z"
    wave.inputs["Scale"].default_value = 1.0
    wave.inputs["Distortion"].default_value = 4.0 * texture.distortion + 1.0
    wave.inputs["Detail"].default_value = texture.detail
    wave.inputs["Detail Roughness"].default_value = texture.roughness
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    return ramp.outputs["Color"]
  if texture.kind == "marble":
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = texture.detail
    noise.inputs["Roughness"].default_value = texture.roughness
    noise.inputs["Distortion"].default_value = texture.distortion
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.inputs["Scale"].default_value = 0.6
    wave.inputs["Distortion"].default_value = 6.0
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Color"], wave.inputs["Vector"])
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    return ramp.outputs["Color"]
  if texture.kind == "speckle":
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "F1"
    voronoi.inputs["Scale"].default_value = 4.0
    voronoi.inputs["Randomness"].default_value = 1.0
    links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
    links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])
    return ramp.outputs["Color"]
  # "noise" and anything else.
  noise = nodes.new("ShaderNodeTexNoise")
  noise.inputs["Scale"].default_value = 1.0
  noise.inputs["Detail"].default_value = texture.detail
  noise.inputs["Roughness"].default_value = texture.roughness
  noise.inputs["Distortion"].default_value = texture.distortion
  links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
  links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
  return ramp.outputs["Color"]


def build_material(blender_material, spec: appearance.MaterialSpec) -> None:
  """Rewrites ``blender_material`` (created by Kubric) to realize ``spec``.

  Kubric's :class:`PrincipledBSDFMaterial` already owns a Principled BSDF node;
  this keeps that node (so Kubric's observers stay valid) and feeds its base
  colour from the sampled procedural texture.
  """
  tree = blender_material.node_tree
  bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
  if bsdf is None:
    raise RuntimeError("material has no Principled BSDF node")
  base = _rgba(spec.base_color)
  _set(bsdf, base, "Base Color")
  _set(bsdf, float(spec.metallic), "Metallic")
  _set(bsdf, float(spec.roughness), "Roughness")
  _set(bsdf, float(spec.specular), "Specular IOR Level", "Specular")
  _set(bsdf, float(spec.ior), "IOR")
  _set(bsdf, float(spec.transmission), "Transmission Weight", "Transmission")
  emission = _rgba(spec.emission)
  _set(bsdf, emission, "Emission Color", "Emission")
  _set(bsdf, 1.0 if any(emission[:3]) else 0.0, "Emission Strength")
  if spec.texture.kind != "solid" or spec.texture.colors:
    color_output = _texture_pattern(tree, spec.texture, base)
    tree.links.new(color_output, bsdf.inputs["Base Color"])


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def _add_light(kb, light: appearance.LightSpec):
  color = _rgba(light.color)[:3]
  common = dict(position=light.position, look_at=light.look_at, color=color,
                intensity=light.intensity)
  if light.kind == "directional":
    return kb.DirectionalLight(**common)
  if light.kind == "point":
    return kb.PointLight(**common)
  if light.kind == "spot":
    return kb.SpotLight(spot_size=light.spot_size or math.pi / 4,
                        spot_blend=light.spot_blend or 1.0, **common)
  return kb.RectAreaLight(width=light.width or 1.0, height=light.height or 1.0, **common)


def _build_scene(
    kb, spec: vi.VelocityInstanceSpec, branch: str, log: SimulationLog,
    resolution: int, samples: int, scratch: Path, denoise: bool,
):
  from kubric.renderer import Blender  # pylint: disable=import-outside-toplevel

  scene_config = spec.scene_for(branch)
  frame_steps = spec.visual.frame_steps
  frames = len(frame_steps)
  kscene = kb.Scene(
      frame_start=0, frame_end=frames - 1, frame_rate=scene_config.frame_rate,
      step_rate=scene_config.step_rate, resolution=(resolution, resolution),
  )
  renderer = Blender(
      kscene, scratch_dir=str(scratch), samples_per_pixel=samples,
      use_denoising=denoise, adaptive_sampling=True, background_transparency=False,
  )
  device_record = enable_gpu(renderer)
  renderer.blender_scene.cycles.seed = int(spec.visual.render_seed)

  background = spec.visual.background
  bg_color = _rgba(background.color or (0.2, 0.2, 0.2, 1.0))
  kscene.background = kb.Color(*bg_color)
  kscene.ambient_illumination = kb.Color(*(c * 0.35 * background.strength for c in bg_color[:3]), 1.0)

  camera_spec = spec.visual.camera
  camera = kb.PerspectiveCamera(
      focal_length=camera_spec.focal_length, sensor_width=camera_spec.sensor_width,
      position=camera_spec.positions[0], look_at=camera_spec.look_ats[0],
      min_render_distance=camera_spec.clipping_range[0],
      max_render_distance=camera_spec.clipping_range[1],
  )
  kscene += camera
  for light in spec.visual.lights:
    kscene += _add_light(kb, light)

  visual_by_id = {item.object_id: item for item in spec.visual.objects}
  kinds = {"cube": kb.Cube, "sphere": kb.Sphere, "cylinder": kb.Cylinder, "capsule": kb.Capsule}
  segmentation_ids = {oid: index for index, oid in enumerate(spec.object_ids)}  # floor -> 0
  assets = {}
  for item in scene_config.objects:
    asset = kinds[item.shape](
        scale=item.size, position=item.position, quaternion=item.quaternion,
        static=item.static, mass=item.mass or 1.0, friction=item.friction,
        restitution=item.restitution, segmentation_id=segmentation_ids[item.object_id],
    )
    material_spec = visual_by_id[item.object_id].material
    asset.material = kb.PrincipledBSDFMaterial(
        color=kb.Color(*_rgba(material_spec.base_color)),
        metallic=material_spec.metallic, roughness=material_spec.roughness,
        ior=material_spec.ior, transmission=material_spec.transmission,
        specular=material_spec.specular,
    )
    asset.metadata["object_id"] = item.object_id
    asset.metadata["role"] = spec.roles[item.object_id]
    kscene += asset
    build_material(asset.material.linked_objects[renderer], material_spec)
    assets[item.object_id] = asset

  # Keyframe poses from the simulation log (frame f <- physics step frame_steps[f]).
  columns = {oid: index for index, oid in enumerate(log.object_ids)}
  for object_id, asset in assets.items():
    column = columns[object_id]
    blender_obj = asset.linked_objects.get(renderer)
    for frame, step in enumerate(frame_steps):
      state = log.states[step, column]
      asset.position = state[POSITION_SLICE]
      asset.quaternion = state[QUATERNION_SLICE]
      asset.velocity = state[LINEAR_VELOCITY_SLICE]
      asset.angular_velocity = state[ANGULAR_VELOCITY_SLICE]
      for member in ("position", "quaternion", "velocity", "angular_velocity"):
        asset.keyframe_insert(member, frame)
      if blender_obj is not None:
        is_hidden = bool(state[POSITION_SLICE][2] < -500.0)
        blender_obj.hide_render = is_hidden
        blender_obj.hide_viewport = is_hidden
        blender_obj.keyframe_insert(data_path="hide_render", frame=frame)
        blender_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
  return kscene, renderer, assets, device_record


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _to_uint8(frames: np.ndarray) -> np.ndarray:
  if frames.dtype == np.uint8:
    return frames
  return np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)


def _encode_mp4(output: Path, frames: np.ndarray, frame_rate: int) -> None:
  import imageio_ffmpeg  # pylint: disable=import-outside-toplevel

  frames = np.ascontiguousarray(_to_uint8(frames)[..., :3])
  height, width = frames.shape[1:3]
  writer = imageio_ffmpeg.write_frames(
      str(output), (width, height), fps=frame_rate, codec="libx264", pix_fmt_in="rgb24",
      pix_fmt_out="yuv420p", quality=8, macro_block_size=1,
  )
  writer.send(None)
  for frame in frames:
    writer.send(frame.tobytes())
  writer.close()


def _colorize_mask(segmentation: np.ndarray) -> np.ndarray:
  ids = segmentation[..., 0] if segmentation.ndim == 4 else segmentation
  return _MASK_PALETTE[np.clip(ids, 0, len(_MASK_PALETTE) - 1)]


def _viridis_lut() -> np.ndarray:
  """Builds a smooth 256x3 Viridis colour lookup table in pure NumPy."""
  anchors = np.array([
      [68, 1, 84],
      [72, 40, 120],
      [62, 74, 137],
      [49, 104, 142],
      [38, 130, 142],
      [31, 158, 137],
      [53, 183, 121],
      [109, 205, 89],
      [253, 231, 37],
  ], dtype=np.float32)
  x = np.linspace(0.0, 1.0, len(anchors))
  xi = np.linspace(0.0, 1.0, 256)
  lut = np.zeros((256, 3), dtype=np.uint8)
  for c in range(3):
    lut[:, c] = np.clip(np.interp(xi, x, anchors[:, c]), 0, 255).astype(np.uint8)
  return lut


_VIRIDIS_LUT = _viridis_lut()


def _colorize_depth(depth: np.ndarray, far_clip: float) -> np.ndarray:
  """Converts depth frames [T, H, W] or [T, H, W, 1] into colourised viridis RGB [T, H, W, 3]."""
  d = depth[..., 0] if depth.ndim == 4 else depth
  valid = d < (float(far_clip) * 0.999)
  if valid.any():
    d_min = float(d[valid].min())
    d_max = float(d[valid].max())
  else:
    d_min, d_max = 0.0, float(far_clip)
  if d_max <= d_min:
    d_max = d_min + 1.0

  # Closer objects (smaller distance) appear brighter/yellow; far objects appear darker/purple.
  norm = np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)
  indices = np.clip(np.rint((1.0 - norm) * 255.0), 0, 255).astype(np.uint8)
  rgb = _VIRIDIS_LUT[indices].copy()
  rgb[~valid] = [0, 0, 0]
  return rgb


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
  """Vectorized HSV to RGB conversion in pure NumPy for arrays of shape [..., 3]."""
  h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
  c = v * s
  x = c * (1.0 - np.abs((h * 6.0) % 2.0 - 1.0))
  m = v - c
  i = (h * 6.0).astype(int) % 6
  r = np.zeros_like(h)
  g = np.zeros_like(h)
  b = np.zeros_like(h)
  for idx, (rc, gc, bc) in enumerate([
      (c, x, 0.0), (x, c, 0.0), (0.0, c, x), (0.0, x, c), (x, 0.0, c), (c, 0.0, x)
  ]):
    mask = (i == idx)
    r[mask] = rc[mask] if isinstance(rc, np.ndarray) else rc
    g[mask] = gc[mask] if isinstance(gc, np.ndarray) else gc
    b[mask] = bc[mask] if isinstance(bc, np.ndarray) else bc
  return np.stack([r + m, g + m, b + m], axis=-1)


def _colorize_flow(flow: np.ndarray) -> np.ndarray:
  """Converts optical flow vectors [T, H, W, 2] to colourised RGB [T, H, W, 3] (Middlebury wheel)."""
  u = flow[..., 0]
  v = flow[..., 1]
  angle = np.arctan2(v, u)
  hue = (angle + np.pi) / (2.0 * np.pi)
  mag = np.sqrt(u**2 + v**2)
  max_mag = float(np.percentile(mag, 99.0)) if mag.size else 1.0
  if max_mag < 1.0:
    max_mag = 1.0
  val = np.clip(mag / max_mag, 0.0, 1.0)
  sat = np.ones_like(val)
  hsv = np.stack([hue, sat, val], axis=-1)
  rgb = np.clip(_hsv_to_rgb(hsv) * 255.0, 0, 255).astype(np.uint8)
  return rgb


def _tracking(
    kscene, spec: vi.VelocityInstanceSpec, branch: str, log: SimulationLog,
    segmentation: np.ndarray, rendered_frames: Sequence[int],
) -> Dict[str, np.ndarray]:
  frame_steps = [spec.visual.frame_steps[frame] for frame in rendered_frames]
  frames = len(frame_steps)
  object_ids = list(spec.object_ids)  # dataset-wide order (floor first)
  n = len(object_ids)
  columns = {oid: index for index, oid in enumerate(log.object_ids)}
  states = np.full((frames, n, 13), np.nan, dtype=np.float32)
  presence = np.zeros((n,), dtype=bool)
  frame_presence = np.zeros((frames, n), dtype=bool)
  image_positions = np.full((frames, n, 2), np.nan, dtype=np.float32)
  in_front = np.zeros((frames, n), dtype=bool)
  bboxes = np.full((frames, n, 4), np.nan, dtype=np.float32)
  visibility = np.zeros((frames, n), dtype=np.int32)
  seg = segmentation[..., 0] if segmentation.ndim == 4 else segmentation
  height, width = seg.shape[1:3]
  for index, object_id in enumerate(object_ids):
    if object_id not in columns:
      continue
    presence[index] = True
    column = columns[object_id]
    for frame, step in enumerate(frame_steps):
      state = log.states[step, column]
      if state[POSITION_SLICE][2] < -500.0:
        # Object is removed from scene at this frame
        frame_presence[frame, index] = False
        states[frame, index] = np.nan
        image_positions[frame, index] = np.nan
        in_front[frame, index] = False
        bboxes[frame, index] = np.nan
        visibility[frame, index] = 0
        continue
      frame_presence[frame, index] = True
      states[frame, index] = state
      projected = kscene.camera.project_point(state[POSITION_SLICE], frame=rendered_frames[frame])
      image_positions[frame, index] = projected[:2]
      in_front[frame, index] = projected[2] > 0
      pixels = seg[frame] == index
      count = int(pixels.sum())
      visibility[frame, index] = count
      if count:
        rows, cols = np.where(pixels)
        bboxes[frame, index] = (
            rows.min() / height, cols.min() / width,
            (rows.max() + 1) / height, (cols.max() + 1) / width,
        )
  return {
      "object_ids": np.array(object_ids),
      "roles": np.array([spec.roles[oid] for oid in object_ids]),
      "segmentation_ids": np.arange(n, dtype=np.int32),
      "present": presence,
      "frame_presence": frame_presence,
      "frames": np.asarray(rendered_frames, dtype=np.int32),
      "physics_steps": np.asarray(frame_steps, dtype=np.int32),
      "positions": states[..., POSITION_SLICE],
      "quaternions_wxyz": states[..., QUATERNION_SLICE],
      "linear_velocities": states[..., LINEAR_VELOCITY_SLICE],
      "angular_velocities": states[..., ANGULAR_VELOCITY_SLICE],
      "image_positions": image_positions,
      "in_front_of_camera": in_front,
      "bboxes_yxyx": bboxes,
      "visible_pixels": visibility,
      "subject_id": np.array(spec.subject_id),
      "branch": np.array(branch),
      "initial_velocity": np.asarray(
          next((o.linear_velocity for o in spec.scene_for(branch).objects
                if o.object_id == spec.subject_id), (0.0, 0.0, 0.0)),
          dtype=np.float32,
      ),
  }


def render_branch(
    instance_dir: Path, branch: str, *, resolution: int = 256, samples: int = 64,
    layers: Sequence[str] = DEFAULT_LAYERS, denoise: bool = True, save_frames: bool = False,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
  """Renders one branch into ``<instance_dir>/<branch>/`` and returns render info."""
  import kubric as kb  # pylint: disable=import-outside-toplevel

  spec = vi.read_instance_spec(instance_dir)
  log = vi.read_branch_log(instance_dir, branch)
  expected = tuple(item.object_id for item in spec.scene_for(branch).objects)
  if tuple(log.object_ids) != expected:
    raise ValueError("branch {} log objects {} do not match scene {}".format(
        branch, log.object_ids, expected))
  branch_dir = Path(instance_dir) / branch
  staging = Path(tempfile.mkdtemp(prefix=".render-{}-".format(branch), dir=str(Path(instance_dir))))
  scratch = Path(tempfile.mkdtemp(prefix="kubric_render_"))
  started = time.time()
  try:
    kscene, renderer, assets, device = _build_scene(
        kb, spec, branch, log, resolution, samples, scratch, denoise,
    )
    frames = list(range(kscene.frame_start, kscene.frame_end + 1))
    if max_frames is not None:
      frames = frames[:max_frames]
    wanted = list(dict.fromkeys(("rgba", "segmentation", "depth", "forward_flow", *layers)))
    data = renderer.render(frames=frames, return_layers=wanted)
    rgba = data["rgba"]
    # Kubric already maps Cryptomatte hashes to each asset's ``segmentation_id``
    # (floor -> 0, dynamic bodies -> their index in spec.object_ids).
    segmentation = data["segmentation"].astype(np.uint8)
    frame_rate = spec.scene.frame_rate

    _encode_mp4(staging / "video.mp4", rgba, frame_rate)
    _encode_mp4(staging / "mask.mp4", _colorize_mask(segmentation), frame_rate)
    np.savez_compressed(staging / "segmentation.npz", segmentation=segmentation[..., 0],
                        object_ids=np.array(spec.object_ids))
    if "depth" in data:
      far = float(spec.visual.camera.clipping_range[1])
      depth = np.minimum(np.nan_to_num(data["depth"][..., 0], nan=far, posinf=far), far)
      np.savez_compressed(staging / "depth.npz", depth=depth.astype(np.float16),
                          far_clip=np.float32(far))
      _encode_mp4(staging / "depth.mp4", _colorize_depth(depth, far), frame_rate)
      shutil.copyfile(staging / "depth.mp4", staging / "depth_map.mp4")
    if "forward_flow" in data:
      flow = data["forward_flow"]
      np.savez_compressed(staging / "forward_flow.npz", forward_flow=flow.astype(np.float16))
      np.savez_compressed(staging / "flow.npz", flow=flow.astype(np.float16))
      _encode_mp4(staging / "flow.mp4", _colorize_flow(flow), frame_rate)
      shutil.copyfile(staging / "flow.mp4", staging / "optical_flow.mp4")
    for layer in ("backward_flow", "normal"):
      if layer in data:
        np.savez_compressed(staging / "{}.npz".format(layer), **{layer: data[layer]})
    if save_frames:
      from kubric import file_io  # pylint: disable=import-outside-toplevel
      file_io.write_rgba_batch(rgba, staging / "frames")
    tracking = _tracking(kscene, spec, branch, log, segmentation, frames)
    np.savez_compressed(staging / "tracking.npz", **tracking)

    info = {
        "instance_id": spec.instance_id,
        "branch": branch,
        "device": device,
        "resolution": [resolution, resolution],
        "samples_per_pixel": samples,
        "denoise": denoise,
        "frames": len(frames),
        "frame_rate": frame_rate,
        "layers": wanted,
        "render_seconds": round(time.time() - started, 2),
        "blender_version": _blender_version(),
        "visual_scene_hash": appearance.visual_scene_hash(spec.visual),
        "camera": {
            "position": list(spec.visual.camera.positions[0]),
            "look_at": list(spec.visual.camera.look_ats[0]),
            "focal_length": spec.visual.camera.focal_length,
            "sensor_width": spec.visual.camera.sensor_width,
        },
    }
    (staging / "render_info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    for produced in staging.iterdir():
      target = branch_dir / produced.name
      if target.is_dir():
        shutil.rmtree(target)
      elif target.exists():
        target.unlink()
      shutil.move(str(produced), str(target))
    return info
  finally:
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    try:
      import bpy  # pylint: disable=import-outside-toplevel
      bpy.ops.wm.read_factory_settings(use_empty=True)
    except Exception:  # pragma: no cover
      pass


def _blender_version() -> str:
  try:
    import bpy  # pylint: disable=import-outside-toplevel
    return bpy.app.version_string
  except ImportError:  # pragma: no cover
    return "unavailable"


def render_instance(
    instance_dir: Path, branches: Sequence[str] = vi.BRANCHES, **kwargs
) -> Dict[str, Any]:
  """Renders every requested branch of one instance and returns the info records."""
  records = {}
  for branch in branches:
    records[branch] = render_branch(Path(instance_dir), branch, **kwargs)
  return records


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("instance_dir", type=Path, nargs="+",
                      help="instance directories written by write_instance()")
  parser.add_argument("--branches", nargs="+", default=list(vi.BRANCHES), choices=vi.BRANCHES)
  parser.add_argument("--resolution", type=int, default=256)
  parser.add_argument("--samples", type=int, default=64)
  parser.add_argument("--layers", nargs="*", default=list(DEFAULT_LAYERS),
                      choices=sorted(set(DEFAULT_LAYERS) | set(OPTIONAL_LAYERS)))
  parser.add_argument("--no-denoise", action="store_true")
  parser.add_argument("--save-frames", action="store_true", help="also write per-frame PNGs")
  parser.add_argument("--max-frames", type=int, default=None, help="debug: render only the first N frames")
  parser.add_argument("--require-gpu", action="store_true",
                      help="fail instead of silently rendering on the CPU")
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Renders the requested branches of each instance directory and returns zero."""
  args = _parser().parse_args(argv)
  for instance_dir in args.instance_dir:
    for branch in args.branches:
      info = render_branch(
          instance_dir, branch, resolution=args.resolution, samples=args.samples,
          layers=args.layers, denoise=not args.no_denoise, save_frames=args.save_frames,
          max_frames=args.max_frames,
      )
      if args.require_gpu and info["device"]["device"] != "GPU":
        raise SystemExit("GPU required but Cycles used {}".format(info["device"]))
      print("[render] {} {} device={} backend={} {:.1f}s".format(
          instance_dir, branch, info["device"]["device"], info["device"].get("backend"),
          info["render_seconds"]))
  return 0


if __name__ == "__main__":
  sys.exit(main())
