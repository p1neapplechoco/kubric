"""Deterministic sampling of one realized visual scene from validated ranges.

Purpose: turn YAML appearance ranges into a single immutable VisualSceneSpec.
Public API: APPEARANCE_DOMAINS, validate_appearance_ranges, sample_visual_scene,
sample_object_geometry, sample_color, sample_texture, sample_camera, sample_lights,
and sample_background.
Dependencies: NumPy, interventions.appearance, interventions.materials, and
interventions.schema; Kubric, Blender, and PyBullet are never imported.
Trust boundary: sampling realizes and records every value it draws; it does not
validate that an external asset exists or that a scene is physically feasible.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from interventions import appearance, dataset, materials, schema
from interventions.schema import to_jsonable

APPEARANCE_DOMAINS = (
    "geometry",
    "physics",
    "appearance",
    "texture",
    "camera",
    "lighting",
    "background",
    "render",
)


def _freeze_ranges(ranges: Mapping[str, Any]) -> Mapping[str, Any]:
  return dataset._freeze(to_jsonable(ranges))


def _appearance_section(ranges: Mapping[str, Any]) -> Mapping[str, Any]:
  try:
    section = ranges["appearance"]
  except (KeyError, TypeError) as error:
    raise ValueError("range config is missing 'appearance'") from error
  if not isinstance(section, Mapping):
    raise ValueError("appearance must be a mapping")
  return section


def _pair(section: Mapping[str, Any], key: str) -> Tuple[float, float]:
  try:
    values = tuple(section[key])
  except (KeyError, TypeError) as error:
    raise ValueError("{} must be a two-value range".format(key)) from error
  if len(values) != 2:
    raise ValueError("{} must be a two-value range".format(key))
  low = float(values[0])
  high = float(values[1])
  if not math.isfinite(low) or not math.isfinite(high):
    raise ValueError("{} range must be finite".format(key))
  if low > high:
    raise ValueError("{} range minimum exceeds maximum".format(key))
  return (low, high)


def _weighted_choice(rng: np.random.Generator, weights: Mapping[str, float]) -> str:
  if not weights:
    raise ValueError("weights must not be empty")
  items = list(weights.items())
  total = sum(float(weight) for _, weight in items)
  if total <= 0.0:
    raise ValueError("weights must sum to a positive number")
  picks = [float(weight) / total for _, weight in items]
  choice = float(rng.random())
  cumulative = 0.0
  for (name, _), weight in zip(items, picks):
    cumulative += weight
    if choice <= cumulative:
      return name
  return items[-1][0]


def _sample_float(rng: np.random.Generator, pair: Sequence[float]) -> float:
  low, high = pair
  if low == high:
    return float(low)
  return float(rng.uniform(float(low), float(high)))


def _sample_vector(rng: np.random.Generator, pair: Sequence[Sequence[float]]) -> Tuple[float, ...]:
  if len(pair) != 2:
    raise ValueError("vector range must contain two endpoints")
  low = tuple(float(component) for component in pair[0])
  high = tuple(float(component) for component in pair[1])
  if len(low) != len(high):
    raise ValueError("vector endpoints must have the same dimension")
  return tuple(_sample_float(rng, (low[index], high[index])) for index in range(len(low)))


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> Tuple[float, float, float]:
  hue = (hue % 1.0) * 6.0
  chroma = value * saturation
  x = chroma * (1.0 - abs(hue % 2.0 - 1.0))
  if 0.0 <= hue < 1.0:
    r, g, b = chroma, x, 0.0
  elif 1.0 <= hue < 2.0:
    r, g, b = x, chroma, 0.0
  elif 2.0 <= hue < 3.0:
    r, g, b = 0.0, chroma, x
  elif 3.0 <= hue < 4.0:
    r, g, b = 0.0, x, chroma
  elif 4.0 <= hue < 5.0:
    r, g, b = x, 0.0, chroma
  else:
    r, g, b = chroma, 0.0, x
  match = value - chroma
  return (r + match, g + match, b + match)


def _material_family_choice(
    ranges: Mapping[str, Any], rng: np.random.Generator, *, objects: Sequence[str] = ()
) -> str:
  appearance_ranges = _appearance_section(ranges)
  families = tuple(appearance_ranges["materials"]["families"])
  weights = tuple(float(item) for item in appearance_ranges["materials"]["weights"])
  if len(families) != len(weights):
    raise ValueError("materials.weights length must equal materials.families")
  if any(weight <= 0.0 for weight in weights):
    raise ValueError("material weights must be positive")
  pool = list(families)
  candidates = []
  for family, weight in zip(families, weights):
    if weight > 0.0:
      candidates.append((family, float(weight)))
  holdouts = []
  coupling = appearance_ranges.get("coupling", {} )
  if isinstance(coupling, Mapping):
    holdouts = list(coupling.get("held_out", ()))
  for index in range(64):
    family = _weighted_choice(rng, {name: weight for name, weight in candidates})
    combination = {"material_family": family}
    if materials.is_held_out(combination, tuple(holdouts)):
      continue
    return family
  raise ValueError("held-out combinations exhaust the family pool")


def validate_appearance_ranges(ranges: Mapping[str, Any]) -> Mapping[str, Any]:
  """Validates and freezes appearance sampling ranges."""
  if not isinstance(ranges, Mapping):
    raise TypeError("ranges must be a mapping")
  payload = _freeze_ranges(ranges)
  appearance_ranges = _appearance_section(payload)

  families = tuple(appearance_ranges.get("materials", {}).get("families", ()))
  weights = tuple(appearance_ranges.get("materials", {}).get("weights", ()))
  for family in families:
    if family not in appearance.MATERIAL_FAMILIES:
      raise ValueError("unknown material family: {!r}".format(family))
  if len(families) != len(weights):
    raise ValueError("materials.weights length must equal materials.families")
  if any(float(value) <= 0.0 for value in weights):
    raise ValueError("material weights must be positive")

  for key in ("radius", "elevation", "azimuth", "focal_length", "linear_azimuth_delta"):
    if key in appearance_ranges.get("camera", {}):
      low, high = _pair(appearance_ranges["camera"], key)
      if low > high:
        raise ValueError("{} range minimum exceeds maximum".format(key))

  geometry = appearance_ranges.get("geometry", {})
  if not isinstance(geometry, Mapping):
    raise ValueError("geometry must be a mapping")
  shape_weights = geometry.get("shapes", {})
  if not isinstance(shape_weights, Mapping):
    raise ValueError("geometry.shapes must be a mapping")
  if any(float(value) < 0.0 for value in shape_weights.values()):
    raise ValueError("geometry shape weights must be nonnegative")
  if sum(float(value) for value in shape_weights.values()) <= 0.0:
    raise ValueError("geometry shape weights must sum to a positive number")
  unknown_shapes = set(shape_weights) - set(schema.SUPPORTED_SHAPES)
  if unknown_shapes:
    raise ValueError("geometry.shapes contains unsupported shapes: {}".format(sorted(unknown_shapes)))
  if "size" in geometry:
    _pair(geometry, "size")
  if "aspect_ratio" in geometry:
    _pair(geometry, "aspect_ratio")

  coupling = appearance_ranges.get("coupling", {})
  if not isinstance(coupling, Mapping):
    raise ValueError("coupling must be a mapping")
  mode = coupling.get("mode")
  if mode not in materials.COUPLING_MODES:
    raise ValueError("coupling.mode must be one of {}".format(sorted(materials.COUPLING_MODES)))

  orientation = geometry.get("orientation")
  if orientation not in {"upright_yaw", "free_so3"}:
    raise ValueError("geometry.orientation must be 'upright_yaw' or 'free_so3'")

  background = appearance_ranges.get("background", {})
  if not isinstance(background, Mapping):
    raise ValueError("background must be a mapping")
  if background.get("kind") not in appearance.BACKGROUND_KINDS:
    raise ValueError("background.kind must be a supported kind")

  camera = appearance_ranges.get("camera", {})
  if not isinstance(camera, Mapping):
    raise ValueError("camera must be a mapping")
  if camera.get("motion") not in {"fixed", "linear"}:
    raise ValueError("camera.motion must be 'fixed' or 'linear'")

  return payload


def sample_color(ranges: Mapping[str, Any], rng: np.random.Generator) -> Tuple[float, float, float, float]:
  """Sample a single RGBA color from the configured appearance color domain."""
  appearance_ranges = _appearance_section(validate_appearance_ranges(ranges))
  colors = appearance_ranges["colors"]
  strategy_weights = colors.get("strategies", {})
  strategy = _weighted_choice(rng, {key: float(value) for key, value in strategy_weights.items()})
  if strategy == "hsv":
    hue = _sample_float(rng, _pair(colors, "hue"))
    saturation = _sample_float(rng, _pair(colors, "saturation"))
    value = _sample_float(rng, _pair(colors, "value"))
    rgb = _hsv_to_rgb(hue, saturation, value)
    return (rgb[0], rgb[1], rgb[2], 1.0)
  if strategy == "palette":
    palette = list(colors.get("palette", ()))
    if not palette:
      raise ValueError("colors.palette must not be empty")
    return tuple(float(component) for component in palette[int(rng.integers(0, len(palette)))])
  neutral_value = _sample_float(rng, _pair(colors, "neutral_value"))
  return (neutral_value, neutral_value, neutral_value, 1.0)


def sample_texture(ranges: Mapping[str, Any], rng: np.random.Generator, family: str) -> appearance.TextureSpec:
  """Sample one texture spec for the given material family and seeded domain."""
  validated = validate_appearance_ranges(ranges)
  appearance_ranges = _appearance_section(validated)
  allowed = tuple(materials.FAMILY_PRIORS[family].texture_kinds)
  kind = _weighted_choice(rng, {name: 1.0 for name in allowed})
  textures = appearance_ranges["textures"]
  scale = _sample_float(rng, _pair(textures, "scale"))
  detail = _sample_float(rng, _pair(textures, "detail"))
  roughness = _sample_float(rng, _pair(textures, "roughness"))
  distortion = _sample_float(rng, _pair(textures, "distortion"))
  rotation = _sample_float(rng, _pair(textures, "rotation"))
  color_count = int(rng.integers(1, int(_pair(textures, "color_count")[1]) + 1))
  base = sample_color(validated, rng)
  accent = sample_color(validated, rng)
  color_pool = [base, accent]
  if kind == "solid":
    colors = (base,)
  else:
    colors = tuple(color_pool[:min(len(color_pool), max(2, color_count))])
  return appearance.TextureSpec(
      kind=kind,
      seed=int(rng.integers(0, 2 ** 31)),
      colors=colors,
      scale=scale,
      detail=detail,
      roughness=roughness,
      distortion=distortion,
      rotation=rotation,
  )


def sample_object_geometry(ranges: Mapping[str, Any], rng: np.random.Generator) -> Mapping[str, Any]:
  """Sample one proxy geometry choice and its deterministic local size/orientation."""
  validated = validate_appearance_ranges(ranges)
  geometry = _appearance_section(validated)["geometry"]
  shape = _weighted_choice(rng, {key: float(value) for key, value in geometry["shapes"].items()})
  size_range = _pair(geometry, "size")
  aspect_range = _pair(geometry, "aspect_ratio")
  if shape == "cube":
    size = tuple(_sample_float(rng, size_range) for _ in range(3))
  elif shape == "sphere":
    radius = _sample_float(rng, size_range)
    size = (radius, radius, radius)
  elif shape in {"cylinder", "capsule"}:
    radius = _sample_float(rng, size_range)
    height = radius * _sample_float(rng, aspect_range)
    size = (radius, radius, height)
  else:
    raise ValueError("unsupported shape: {!r}".format(shape))
  yaw = float(rng.uniform(0.0, 2.0 * math.pi))
  return {"shape": shape, "size": size, "yaw": yaw}


def _camera_target(scene_config: schema.SceneConfig) -> Tuple[float, float, float]:
  lower, upper = scene_config.scene_bounds
  return tuple((low + high) / 2.0 for low, high in zip(lower, upper))


def _camera_fits(scene_config: schema.SceneConfig, position: Sequence[float], look_at: Sequence[float], safety_margin: float) -> bool:
  lower, upper = scene_config.scene_bounds
  center = _camera_target(scene_config)
  camera_vector = tuple(position[i] - look_at[i] for i in range(3))
  if math.isclose(sum(v * v for v in camera_vector), 0.0, abs_tol=1e-12):
    return False
  forward = tuple(look_at[i] - position[i] for i in range(3))
  forward_norm = math.sqrt(sum(value * value for value in forward))
  if forward_norm <= 0.0:
    return False
  forward = tuple(value / forward_norm for value in forward)
  up = (0.0, 0.0, 1.0)
  right = tuple(
      (up[1] * forward[2] - up[2] * forward[1],
       up[2] * forward[0] - up[0] * forward[2],
       up[0] * forward[1] - up[1] * forward[0])
  )
  right_norm = math.sqrt(sum(value * value for value in right))
  if right_norm <= 1e-12:
    right = (1.0, 0.0, 0.0)
  else:
    right = tuple(value / right_norm for value in right)
  up_vec = (
      forward[1] * right[2] - forward[2] * right[1],
      forward[2] * right[0] - forward[0] * right[2],
      forward[0] * right[1] - forward[1] * right[0],
  )
  up_norm = math.sqrt(sum(value * value for value in up_vec))
  if up_norm <= 1e-12:
    up_vec = (0.0, 0.0, 1.0)
  else:
    up_vec = tuple(value / up_norm for value in up_vec)

  half_size = tuple((upper[i] - lower[i]) / 2.0 for i in range(3))
  corners = []
  for x_sign in (-1.0, 1.0):
    for y_sign in (-1.0, 1.0):
      for z_sign in (-1.0, 1.0):
        corner = (
            center[0] + x_sign * half_size[0],
            center[1] + y_sign * half_size[1],
            center[2] + z_sign * half_size[2],
        )
        corners.append(corner)
  for corner in corners:
    rel = tuple(corner[i] - position[i] for i in range(3))
    depth = sum(rel[i] * forward[i] for i in range(3))
    if depth <= 0.0:
      return False
    x = sum(rel[i] * right[i] for i in range(3))
    y = sum(rel[i] * up_vec[i] for i in range(3))
    if abs(x) > (depth * safety_margin) or abs(y) > (depth * safety_margin):
      return False
  return True


def sample_camera(ranges: Mapping[str, Any], rng: np.random.Generator, scene_config: schema.SceneConfig) -> appearance.CameraRenderSpec:
  """Sample a camera trajectory that frames the configured scene bounds."""
  validated = validate_appearance_ranges(ranges)
  camera_ranges = _appearance_section(validated)["camera"]
  center = _camera_target(scene_config)
  radius_range = _pair(camera_ranges, "radius")
  elevation_range = _pair(camera_ranges, "elevation")
  azimuth_range = _pair(camera_ranges, "azimuth")
  focal_length = _sample_float(rng, _pair(camera_ranges, "focal_length"))
  positions = []
  look_ats = []
  for _ in range(len(appearance.frame_steps_for(scene_config))):
    radius = _sample_float(rng, radius_range)
    elevation = _sample_float(rng, elevation_range)
    azimuth = _sample_float(rng, azimuth_range)
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)
    if z <= 0.0:
      z = abs(z) + 0.05
    position = (center[0] + x, center[1] + y, center[2] + z)
    if not (radius >= radius_range[0] and radius <= radius_range[1]):
      raise ValueError("camera radius is outside the configured range")
    positions.append(position)
    look_ats.append(center)
  return appearance.CameraRenderSpec(
      positions=tuple(positions),
      look_ats=tuple(look_ats),
      focal_length=focal_length,
      sensor_width=float(camera_ranges.get("sensor_width", 36.0)),
      clipping_range=tuple(camera_ranges.get("clipping_range", (0.1, 200.0))),
  )


def _kelvin_to_rgb(kelvin: float) -> Tuple[float, float, float]:
  temp = max(1000.0, min(40000.0, float(kelvin)))
  temp = temp / 100.0
  red = 0.0
  green = 0.0
  blue = 0.0
  if temp <= 66.0:
    red = 1.0
    green = temp
    blue = 0.0
  else:
    red = 1.0
    green = 1.0
    blue = 1.0
  if temp > 66.0:
    red = 1.0
    green = 1.0
    blue = 1.0
  if temp > 66.0:
    red = 1.0
  if temp <= 66.0:
    green = 1.0
  if temp <= 66.0:
    blue = 1.0
  red = max(0.0, min(1.0, red))
  green = max(0.0, min(1.0, green))
  blue = max(0.0, min(1.0, blue))
  return (red, green, blue)


def sample_lights(ranges: Mapping[str, Any], rng: np.random.Generator) -> Tuple[appearance.LightSpec, ...]:
  """Sample the rig of key/fill/rim lights for a procedural render."""
  validated = validate_appearance_ranges(ranges)
  lighting = _appearance_section(validated)["lighting"]
  light_ids = ("key", "fill", "rim")
  lights = []
  for light_id in light_ids:
    section = lighting[light_id]
    position = _sample_vector(rng, section["position"])
    intensity = _sample_float(rng, _pair(section, "intensity"))
    kelvin = _sample_float(rng, _pair(section, "color_temperature"))
    red, green, blue = _kelvin_to_rgb(kelvin)
    width = _sample_float(rng, _pair(section, "size"))
    height = _sample_float(rng, _pair(section, "size"))
    lights.append(
        appearance.LightSpec(
            light_id=light_id,
            kind="rect_area",
            position=position,
            look_at=(0.0, 0.0, 0.0),
            color=(red, green, blue, 1.0),
            intensity=intensity,
            width=width,
            height=height,
        )
    )
  return tuple(lights)


def sample_background(ranges: Mapping[str, Any], rng: np.random.Generator) -> appearance.BackgroundSpec:
  """Sample a flat background color from the configured environment palette."""
  validated = validate_appearance_ranges(ranges)
  background = _appearance_section(validated)["background"]
  kind = background.get("kind", "color")
  if kind == "color":
    value = _sample_float(rng, _pair(background, "color_value"))
    saturation = _sample_float(rng, _pair(background, "color_saturation"))
    color = _hsv_to_rgb(0.0, saturation, value)
    return appearance.BackgroundSpec(
        kind="color",
        color=(color[0], color[1], color[2], 1.0),
        hdri=None,
        rotation=_sample_float(rng, _pair(background, "rotation")),
        strength=_sample_float(rng, _pair(background, "strength")),
        exposure=_sample_float(rng, _pair(background, "exposure")),
    )
  raise ValueError("unsupported background kind: {!r}".format(kind))


def sample_visual_scene(
    ranges: Mapping[str, Any], scene_config: schema.SceneConfig, master_seed: int, index: int
) -> appearance.VisualSceneSpec:
  """Samples one immutable visual scene for a given scene config."""
  validated = validate_appearance_ranges(ranges)
  rngs = {
      domain: np.random.default_rng(dataset.derive_seed(master_seed, index, domain))
      for domain in APPEARANCE_DOMAINS
  }
  objects = []
  for object_config in sorted(scene_config.objects, key=lambda item: item.object_id):
    family = _material_family_choice(validated, rngs["appearance"])
    color = sample_color(validated, rngs["appearance"])
    texture = sample_texture(validated, rngs["texture"], family)
    material = materials.sample_material(rngs["appearance"], family, color, texture)
    objects.append(
        appearance.VisualObjectSpec(
            object_id=object_config.object_id,
            source_kind="procedural",
            asset=None,
            collision_proxy_id=object_config.object_id,
            scale=(1.0, 1.0, 1.0),
            origin_offset=(0.0, 0.0, 0.0),
            alignment_quaternion=(1.0, 0.0, 0.0, 0.0),
            material=material,
        )
    )

  camera = sample_camera(validated, rngs["camera"], scene_config)
  lights = sample_lights(validated, rngs["lighting"])
  background = sample_background(validated, rngs["background"])
  render_seed = int(dataset.derive_seed(master_seed, index, "render") % (2 ** 31))
  return appearance.VisualSceneSpec(
      objects=tuple(objects),
      camera=camera,
      lights=lights,
      background=background,
      render_seed=render_seed,
      frame_steps=appearance.frame_steps_for(scene_config),
  )


__all__ = [
    "APPEARANCE_DOMAINS",
    "sample_background",
    "sample_camera",
    "sample_color",
    "sample_lights",
    "sample_object_geometry",
    "sample_texture",
    "sample_visual_scene",
    "validate_appearance_ranges",
]
