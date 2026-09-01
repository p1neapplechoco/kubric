"""Immutable renderer-independent appearance schemas for intervention instances.

Purpose: define frozen, validated visual values and their canonical JSON form.
Public API: APPEARANCE_SCHEMA_VERSION, TEXTURE_KINDS, MATERIAL_FAMILIES, SOURCE_KINDS,
MATERIAL_MODES, COLOR_SPACES, IMAGE_ROLES, LIGHT_KINDS, BACKGROUND_KINDS, RENDER_DEVICES,
RENDER_LAYERS, SMOKE_PROFILE, PRODUCTION_PROFILE, PROFILES_BY_NAME, ImageReference,
TextureSpec, MaterialSpec, AssetReference, VisualObjectSpec, CameraRenderSpec, LightSpec,
BackgroundSpec, RenderProfile, VisualSceneSpec, visual_scene_hash, render_profile_hash,
validate_scene_correspondence, visual_scene_from_payload, and frame_steps_for.
Dependencies: Python's standard library and interventions.schema helpers only, so
appearance never imports Kubric, Blender, or a simulator backend.
Trust boundary: validation enforces value ranges, enum membership, and JSON safety;
it does not verify that a referenced asset or image exists or is authentic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from interventions.schema import (
    SceneConfig,
    _integer,
    _nonempty_string,
    _real,
    _vector,
    _SchemaMixin,
)


APPEARANCE_SCHEMA_VERSION = "1.0"

TEXTURE_KINDS = frozenset(
    ("solid", "noise", "checker", "wood", "marble", "speckle", "image")
)
MATERIAL_FAMILIES = frozenset(
    ("metal", "rubber", "plastic", "ceramic", "glass", "wood", "stone")
)
SOURCE_KINDS = frozenset(("procedural", "kubasic", "gso"))
MATERIAL_MODES = frozenset(("native", "override"))
COLOR_SPACES = frozenset(("sRGB", "Non-Color"))
LIGHT_KINDS = frozenset(("directional", "point", "spot", "rect_area"))
BACKGROUND_KINDS = frozenset(("color", "hdri"))
RENDER_DEVICES = frozenset(("CPU", "GPU"))

RENDER_LAYERS = (
    "rgba",
    "segmentation",
    "depth",
    "normal",
    "forward_flow",
    "backward_flow",
    "object_coordinates",
)

_COLOR_ROLES = frozenset(("base_color", "emission"))
_NON_COLOR_ROLES = frozenset(("roughness", "metallic", "normal", "height"))
IMAGE_ROLES = _COLOR_ROLES | _NON_COLOR_ROLES

_ASSET_SOURCE_KINDS = frozenset(("kubasic", "gso"))
_MAX_TEXTURE_COLORS = 4
_HEX_DIGITS = frozenset("0123456789abcdef")


def _enum(value: Any, allowed: frozenset, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if value not in allowed:
    raise ValueError(
        "{} must be one of {}: {!r}".format(name, sorted(allowed), value)
    )
  return value


def _unit(value: Any, name: str) -> float:
  result = _real(value, name)
  if not 0.0 <= result <= 1.0:
    raise ValueError("{} must lie in [0, 1]".format(name))
  return result


def _positive(value: Any, name: str) -> float:
  result = _real(value, name)
  if result <= 0.0:
    raise ValueError("{} must be positive".format(name))
  return result


def _rgba(value: Any, name: str) -> Tuple[float, float, float, float]:
  components = _vector(value, 4, name)
  for index, component in enumerate(components):
    if not 0.0 <= component <= 1.0:
      raise ValueError("{}[{}] must lie in [0, 1]".format(name, index))
  return components


def _hex_digest(value: Any, name: str) -> str:
  digest = _nonempty_string(value, name)
  if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
    raise ValueError("{} must be a lowercase 64-character hex digest".format(name))
  return digest


def _appearance_version(value: Any) -> str:
  if value != APPEARANCE_SCHEMA_VERSION:
    raise ValueError(
        "schema_version must be {!r}".format(APPEARANCE_SCHEMA_VERSION)
    )
  return APPEARANCE_SCHEMA_VERSION


def _canonical_bytes(payload: Any) -> bytes:
  """Serializes a JSON-safe payload to the repository's canonical byte form."""
  return json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
  ).encode("utf-8")


def _points(value: Any, name: str) -> Tuple[Tuple[float, float, float], ...]:
  if isinstance(value, (str, bytes)):
    raise TypeError("{} must be a sequence of XYZ vectors".format(name))
  points = tuple(
      _vector(point, 3, "{}[{}]".format(name, index))
      for index, point in enumerate(tuple(value))
  )
  if not points:
    raise ValueError("{} must not be empty".format(name))
  return points


def _unit_quaternion(value: Any, name: str) -> Tuple[float, float, float, float]:
  quaternion = _vector(value, 4, name)
  scale = max(abs(component) for component in quaternion)
  if scale == 0.0:
    raise ValueError("{} must be non-zero".format(name))
  scaled = tuple(component / scale for component in quaternion)
  norm = math.hypot(*scaled)
  normalized = tuple(component / norm for component in scaled)
  if abs(math.hypot(*normalized) - 1.0) > 1e-12:
    raise ValueError("{} normalization failed".format(name))
  return normalized


@dataclass(frozen=True)
class ImageReference(_SchemaMixin):
  """A digest-pinned texture image and the color space it must be read in."""

  role: str
  uri: str
  sha256: str
  color_space: str
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    role = _enum(self.role, IMAGE_ROLES, "role")
    color_space = _enum(self.color_space, COLOR_SPACES, "color_space")
    expected = "sRGB" if role in _COLOR_ROLES else "Non-Color"
    if color_space != expected:
      raise ValueError(
          "color_space for role {!r} must be {!r}".format(role, expected)
      )
    object.__setattr__(self, "role", role)
    object.__setattr__(self, "uri", _nonempty_string(self.uri, "uri"))
    object.__setattr__(self, "sha256", _hex_digest(self.sha256, "sha256"))
    object.__setattr__(self, "color_space", color_space)
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class TextureSpec(_SchemaMixin):
  """A procedural or image-backed surface pattern.

  ``rotation`` is a turn fraction in [0, 1] rather than radians, so that the
  canonical JSON form stays renderer-independent.
  """

  kind: str
  seed: int = 0
  colors: Tuple[Tuple[float, ...], ...] = ()
  scale: float = 1.0
  detail: float = 2.0
  roughness: float = 0.5
  distortion: float = 0.0
  rotation: float = 0.0
  images: Tuple[ImageReference, ...] = ()
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    kind = _enum(self.kind, TEXTURE_KINDS, "kind")

    if isinstance(self.colors, (str, bytes)):
      raise TypeError("colors must be a sequence of RGBA values")
    colors = tuple(
        _rgba(color, "colors[{}]".format(index))
        for index, color in enumerate(tuple(self.colors))
    )
    if kind != "image" and not 1 <= len(colors) <= _MAX_TEXTURE_COLORS:
      raise ValueError(
          "colors must contain between 1 and {} entries".format(_MAX_TEXTURE_COLORS)
      )
    if kind == "image" and len(colors) > _MAX_TEXTURE_COLORS:
      raise ValueError(
          "colors must contain at most {} entries".format(_MAX_TEXTURE_COLORS)
      )

    if isinstance(self.images, (str, bytes)):
      raise TypeError("images must be a sequence of ImageReference values")
    images = tuple(self.images)
    for image in images:
      if not isinstance(image, ImageReference):
        raise TypeError("images must contain ImageReference values")
    if (kind == "image") != bool(images):
      raise ValueError("images must be non-empty exactly for image textures")
    roles = [image.role for image in images]
    if len(set(roles)) != len(roles):
      raise ValueError("images must not repeat a role")

    detail = _real(self.detail, "detail")
    if detail < 0.0:
      raise ValueError("detail must be nonnegative")

    object.__setattr__(self, "kind", kind)
    object.__setattr__(self, "seed", _integer(self.seed, "seed"))
    object.__setattr__(self, "colors", colors)
    object.__setattr__(self, "scale", _positive(self.scale, "scale"))
    object.__setattr__(self, "detail", detail)
    object.__setattr__(self, "roughness", _unit(self.roughness, "roughness"))
    object.__setattr__(self, "distortion", _unit(self.distortion, "distortion"))
    object.__setattr__(self, "rotation", _unit(self.rotation, "rotation"))
    object.__setattr__(self, "images", images)
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class MaterialSpec(_SchemaMixin):
  """Principled-BSDF parameters together with the surface pattern to drive them."""

  family: str
  base_color: Tuple[float, float, float, float]
  metallic: float
  roughness: float
  specular: float
  ior: float
  transmission: float
  emission: Tuple[float, float, float, float]
  texture: TextureSpec
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if not isinstance(self.texture, TextureSpec):
      raise TypeError("texture must be a TextureSpec")
    base_color = _rgba(self.base_color, "base_color")
    transmission = _unit(self.transmission, "transmission")
    if base_color[3] < 1.0 and transmission <= 0.0:
      raise ValueError(
          "base_color alpha below one requires positive transmission"
      )
    ior = _real(self.ior, "ior")
    if not 1.0 <= ior <= 3.0:
      raise ValueError("ior must lie in [1, 3]")

    object.__setattr__(self, "family", _enum(self.family, MATERIAL_FAMILIES, "family"))
    object.__setattr__(self, "base_color", base_color)
    object.__setattr__(self, "metallic", _unit(self.metallic, "metallic"))
    object.__setattr__(self, "roughness", _unit(self.roughness, "roughness"))
    object.__setattr__(self, "specular", _unit(self.specular, "specular"))
    object.__setattr__(self, "ior", ior)
    object.__setattr__(self, "transmission", transmission)
    object.__setattr__(self, "emission", _rgba(self.emission, "emission"))
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class AssetReference(_SchemaMixin):
  """A catalog asset pinned by both its manifest digest and its archive digest."""

  source_kind: str
  manifest_uri: str
  manifest_sha256: str
  asset_id: str
  archive_sha256: str
  material_mode: str
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    object.__setattr__(
        self, "source_kind", _enum(self.source_kind, _ASSET_SOURCE_KINDS, "source_kind")
    )
    object.__setattr__(
        self, "manifest_uri", _nonempty_string(self.manifest_uri, "manifest_uri")
    )
    object.__setattr__(
        self, "manifest_sha256", _hex_digest(self.manifest_sha256, "manifest_sha256")
    )
    object.__setattr__(self, "asset_id", _nonempty_string(self.asset_id, "asset_id"))
    object.__setattr__(
        self, "archive_sha256", _hex_digest(self.archive_sha256, "archive_sha256")
    )
    object.__setattr__(
        self, "material_mode", _enum(self.material_mode, MATERIAL_MODES, "material_mode")
    )
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class VisualObjectSpec(_SchemaMixin):
  """The renderable counterpart of one simulated collision proxy.

  ``origin_offset`` and ``alignment_quaternion`` place the visual mesh relative to
  the proxy whose pose the simulator reports, so physics and pixels stay bound.
  """

  object_id: str
  source_kind: str
  asset: Optional[AssetReference]
  collision_proxy_id: str
  scale: Tuple[float, float, float]
  origin_offset: Tuple[float, float, float]
  alignment_quaternion: Tuple[float, float, float, float]
  material: MaterialSpec
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if not isinstance(self.material, MaterialSpec):
      raise TypeError("material must be a MaterialSpec")
    source_kind = _enum(self.source_kind, SOURCE_KINDS, "source_kind")
    if self.asset is not None and not isinstance(self.asset, AssetReference):
      raise TypeError("asset must be an AssetReference or None")
    if (self.asset is None) != (source_kind == "procedural"):
      raise ValueError(
          "asset must be present exactly for non-procedural source kinds"
      )
    if self.asset is not None and self.asset.source_kind != source_kind:
      raise ValueError("asset source_kind must match the visual object source_kind")

    scale = _vector(self.scale, 3, "scale")
    if any(component <= 0.0 for component in scale):
      raise ValueError("scale components must be positive")

    object.__setattr__(self, "object_id", _nonempty_string(self.object_id, "object_id"))
    object.__setattr__(self, "source_kind", source_kind)
    object.__setattr__(
        self,
        "collision_proxy_id",
        _nonempty_string(self.collision_proxy_id, "collision_proxy_id"),
    )
    object.__setattr__(self, "scale", scale)
    object.__setattr__(
        self, "origin_offset", _vector(self.origin_offset, 3, "origin_offset")
    )
    object.__setattr__(
        self,
        "alignment_quaternion",
        _unit_quaternion(self.alignment_quaternion, "alignment_quaternion"),
    )
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class CameraRenderSpec(_SchemaMixin):
  """A per-frame camera path plus the intrinsics needed to reproduce it.

  ``positions`` and ``look_ats`` are indexed by output frame, so a static camera
  is expressed as a repeated entry rather than as a separate mode.
  """

  positions: Tuple[Tuple[float, float, float], ...]
  look_ats: Tuple[Tuple[float, float, float], ...]
  focal_length: float
  sensor_width: float
  clipping_range: Tuple[float, float]
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    positions = _points(self.positions, "positions")
    look_ats = _points(self.look_ats, "look_ats")
    if len(look_ats) != len(positions):
      raise ValueError("look_ats must have the same length as positions")
    for index, (position, look_at) in enumerate(zip(positions, look_ats)):
      if position == look_at:
        raise ValueError(
            "positions[{0}] and look_ats[{0}] must differ".format(index)
        )

    clipping_range = _vector(self.clipping_range, 2, "clipping_range")
    if not 0.0 < clipping_range[0] < clipping_range[1]:
      raise ValueError("clipping_range must satisfy 0 < near < far")

    object.__setattr__(self, "positions", positions)
    object.__setattr__(self, "look_ats", look_ats)
    object.__setattr__(
        self, "focal_length", _positive(self.focal_length, "focal_length")
    )
    object.__setattr__(
        self, "sensor_width", _positive(self.sensor_width, "sensor_width")
    )
    object.__setattr__(self, "clipping_range", clipping_range)
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class LightSpec(_SchemaMixin):
  """One light, carrying only the parameters its kind actually uses.

  Parameters that do not apply to a kind must be ``None`` rather than a default,
  so that two scenes differing only in an unused field cannot hash differently.
  """

  light_id: str
  kind: str
  position: Tuple[float, float, float]
  look_at: Tuple[float, float, float]
  color: Tuple[float, float, float, float]
  intensity: float
  width: Optional[float] = None
  height: Optional[float] = None
  spot_size: Optional[float] = None
  spot_blend: Optional[float] = None
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    kind = _enum(self.kind, LIGHT_KINDS, "kind")

    if kind == "rect_area":
      if self.width is None or self.height is None:
        raise ValueError("width and height are required for rect_area lights")
      width = _positive(self.width, "width")
      height = _positive(self.height, "height")
    else:
      if self.width is not None or self.height is not None:
        raise ValueError("width and height apply only to rect_area lights")
      width = None
      height = None

    if kind == "spot":
      if self.spot_size is None or self.spot_blend is None:
        raise ValueError("spot_size and spot_blend are required for spot lights")
      spot_size = _real(self.spot_size, "spot_size")
      if not 0.0 < spot_size <= math.pi:
        raise ValueError("spot_size must lie in (0, pi]")
      spot_blend = _unit(self.spot_blend, "spot_blend")
    else:
      if self.spot_size is not None or self.spot_blend is not None:
        raise ValueError("spot_size and spot_blend apply only to spot lights")
      spot_size = None
      spot_blend = None

    object.__setattr__(self, "light_id", _nonempty_string(self.light_id, "light_id"))
    object.__setattr__(self, "kind", kind)
    object.__setattr__(self, "position", _vector(self.position, 3, "position"))
    object.__setattr__(self, "look_at", _vector(self.look_at, 3, "look_at"))
    object.__setattr__(self, "color", _rgba(self.color, "color"))
    object.__setattr__(self, "intensity", _positive(self.intensity, "intensity"))
    object.__setattr__(self, "width", width)
    object.__setattr__(self, "height", height)
    object.__setattr__(self, "spot_size", spot_size)
    object.__setattr__(self, "spot_blend", spot_blend)
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class BackgroundSpec(_SchemaMixin):
  """A flat color or an HDRI environment, never both."""

  kind: str
  color: Optional[Tuple[float, float, float, float]] = None
  hdri: Optional[ImageReference] = None
  rotation: float = 0.0
  strength: float = 1.0
  exposure: float = 0.0
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    kind = _enum(self.kind, BACKGROUND_KINDS, "kind")
    if self.hdri is not None and not isinstance(self.hdri, ImageReference):
      raise TypeError("hdri must be an ImageReference or None")

    if kind == "color":
      if self.color is None or self.hdri is not None:
        raise ValueError("color backgrounds require color and no hdri")
      color = _rgba(self.color, "color")
      hdri = None
    else:
      if self.hdri is None or self.color is not None:
        raise ValueError("hdri backgrounds require hdri and no color")
      if self.hdri.role != "base_color":
        raise ValueError("hdri must use the base_color role")
      color = None
      hdri = self.hdri

    exposure = _real(self.exposure, "exposure")
    if not -10.0 <= exposure <= 10.0:
      raise ValueError("exposure must lie in [-10, 10]")

    object.__setattr__(self, "kind", kind)
    object.__setattr__(self, "color", color)
    object.__setattr__(self, "hdri", hdri)
    object.__setattr__(self, "rotation", _unit(self.rotation, "rotation"))
    object.__setattr__(self, "strength", _positive(self.strength, "strength"))
    object.__setattr__(self, "exposure", exposure)
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class RenderProfile(_SchemaMixin):
  """Renderer cost and output settings, independent of any particular scene."""

  name: str
  resolution: Tuple[int, int]
  samples_per_pixel: int
  adaptive_sampling: bool
  use_denoising: bool
  background_transparency: bool
  layers: Tuple[str, ...] = RENDER_LAYERS
  device: str = "CPU"
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    for name in ("adaptive_sampling", "use_denoising", "background_transparency"):
      if not isinstance(getattr(self, name), bool):
        raise TypeError("{} must be a bool".format(name))

    if len(tuple(self.resolution)) != 2:
      raise ValueError("resolution must contain width and height")
    resolution = tuple(
        _integer(value, "resolution[{}]".format(index))
        for index, value in enumerate(self.resolution)
    )
    if any(value <= 0 for value in resolution):
      raise ValueError("resolution components must be positive")

    samples_per_pixel = _integer(self.samples_per_pixel, "samples_per_pixel")
    if samples_per_pixel < 1:
      raise ValueError("samples_per_pixel must be at least 1")

    if isinstance(self.layers, (str, bytes)):
      raise TypeError("layers must be a sequence of layer names")
    layers = tuple(self.layers)
    for layer in layers:
      _enum(layer, frozenset(RENDER_LAYERS), "layers")
    if not layers:
      raise ValueError("layers must not be empty")
    if len(set(layers)) != len(layers):
      raise ValueError("layers must not repeat a name")
    # Order follows RENDER_LAYERS so that two profiles requesting the same set of
    # layers in a different order produce the same hash.
    layers = tuple(layer for layer in RENDER_LAYERS if layer in set(layers))

    object.__setattr__(self, "name", _nonempty_string(self.name, "name"))
    object.__setattr__(self, "resolution", resolution)
    object.__setattr__(self, "samples_per_pixel", samples_per_pixel)
    object.__setattr__(self, "layers", layers)
    object.__setattr__(self, "device", _enum(self.device, RENDER_DEVICES, "device"))
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


@dataclass(frozen=True)
class VisualSceneSpec(_SchemaMixin):
  """Everything needed to render one simulated scene, minus the renderer settings.

  ``frame_steps`` names the physics step each output frame samples, which is what
  binds a rendered frame back to a row of the simulation log.
  """

  objects: Tuple[VisualObjectSpec, ...]
  camera: CameraRenderSpec
  lights: Tuple[LightSpec, ...]
  background: BackgroundSpec
  render_seed: int
  frame_steps: Tuple[int, ...]
  schema_version: str = APPEARANCE_SCHEMA_VERSION

  def __post_init__(self) -> None:
    if not isinstance(self.camera, CameraRenderSpec):
      raise TypeError("camera must be a CameraRenderSpec")
    if not isinstance(self.background, BackgroundSpec):
      raise TypeError("background must be a BackgroundSpec")

    if isinstance(self.objects, (str, bytes)):
      raise TypeError("objects must be a sequence of VisualObjectSpec")
    objects = tuple(self.objects)
    for item in objects:
      if not isinstance(item, VisualObjectSpec):
        raise TypeError("objects must contain only VisualObjectSpec values")
    if not objects:
      raise ValueError("objects must not be empty")
    object_ids = [item.object_id for item in objects]
    if len(set(object_ids)) != len(object_ids):
      raise ValueError("object_id values must be unique")
    objects = tuple(sorted(objects, key=lambda item: item.object_id))

    if isinstance(self.lights, (str, bytes)):
      raise TypeError("lights must be a sequence of LightSpec")
    lights = tuple(self.lights)
    for light in lights:
      if not isinstance(light, LightSpec):
        raise TypeError("lights must contain only LightSpec values")
    if not lights:
      raise ValueError("lights must not be empty")
    light_ids = [light.light_id for light in lights]
    if len(set(light_ids)) != len(light_ids):
      raise ValueError("light_id values must be unique")
    lights = tuple(sorted(lights, key=lambda light: light.light_id))

    if isinstance(self.frame_steps, (str, bytes)):
      raise TypeError("frame_steps must be a sequence of integers")
    frame_steps = tuple(
        _integer(step, "frame_steps[{}]".format(index))
        for index, step in enumerate(tuple(self.frame_steps))
    )
    if not frame_steps:
      raise ValueError("frame_steps must not be empty")
    if frame_steps[0] < 0:
      raise ValueError("frame_steps must be nonnegative")
    if any(later <= earlier
           for earlier, later in zip(frame_steps, frame_steps[1:])):
      raise ValueError("frame_steps must be strictly increasing")
    if len(frame_steps) != len(self.camera.positions):
      raise ValueError("frame_steps must have one entry per camera position")

    render_seed = _integer(self.render_seed, "render_seed")
    if render_seed < 0:
      raise ValueError("render_seed must be nonnegative")

    object.__setattr__(self, "objects", objects)
    object.__setattr__(self, "lights", lights)
    object.__setattr__(self, "frame_steps", frame_steps)
    object.__setattr__(self, "render_seed", render_seed)
    object.__setattr__(
        self, "schema_version", _appearance_version(self.schema_version)
    )


def visual_scene_hash(spec: VisualSceneSpec) -> str:
  """Returns the SHA-256 digest of a visual scene's canonical JSON form."""
  if not isinstance(spec, VisualSceneSpec):
    raise TypeError("spec must be a VisualSceneSpec")
  return hashlib.sha256(_canonical_bytes(spec.to_dict())).hexdigest()


def render_profile_hash(profile: RenderProfile) -> str:
  """Returns the SHA-256 digest of a render profile's canonical JSON form."""
  if not isinstance(profile, RenderProfile):
    raise TypeError("profile must be a RenderProfile")
  return hashlib.sha256(_canonical_bytes(profile.to_dict())).hexdigest()


def _image_from_payload(payload: Any) -> ImageReference:
  return ImageReference(**dict(payload))


def _texture_from_payload(payload: Any) -> TextureSpec:
  values = dict(payload)
  images = tuple(_image_from_payload(item) for item in values.pop("images", ()))
  return TextureSpec(images=images, **values)


def _material_from_payload(payload: Any) -> MaterialSpec:
  values = dict(payload)
  return MaterialSpec(texture=_texture_from_payload(values.pop("texture")), **values)


def _visual_object_from_payload(payload: Any) -> VisualObjectSpec:
  values = dict(payload)
  asset_payload = values.pop("asset")
  asset = None if asset_payload is None else AssetReference(**dict(asset_payload))
  material = _material_from_payload(values.pop("material"))
  return VisualObjectSpec(asset=asset, material=material, **values)


def _background_from_payload(payload: Any) -> BackgroundSpec:
  values = dict(payload)
  hdri_payload = values.pop("hdri", None)
  hdri = None if hdri_payload is None else _image_from_payload(hdri_payload)
  return BackgroundSpec(hdri=hdri, **values)


def visual_scene_from_payload(payload: Any) -> VisualSceneSpec:
  """Rebuilds a :class:`VisualSceneSpec` from its ``to_dict`` JSON form.

  Every nested value is re-validated by the same constructors that produced it,
  so a payload edited after publication fails here instead of reaching a
  renderer.  Round-tripping is exact: ``visual_scene_from_payload(spec.to_dict())
  == spec``.
  """
  if not isinstance(payload, Mapping):
    raise ValueError("visual scene payload is malformed")
  try:
    values = dict(payload)
    objects = tuple(
        _visual_object_from_payload(item) for item in values.pop("objects")
    )
    camera = CameraRenderSpec(**dict(values.pop("camera")))
    lights = tuple(LightSpec(**dict(item)) for item in values.pop("lights"))
    background = _background_from_payload(values.pop("background"))
    return VisualSceneSpec(
        objects=objects,
        camera=camera,
        lights=lights,
        background=background,
        **values,
    )
  except (AttributeError, KeyError, TypeError, ValueError) as error:
    raise ValueError("visual scene payload is malformed") from error


def frame_steps_for(scene: SceneConfig) -> Tuple[int, ...]:
  """Returns the physics step sampled by each output frame of ``scene``."""
  if not isinstance(scene, SceneConfig):
    raise TypeError("scene must be a SceneConfig")
  steps_per_frame = scene.step_rate // scene.frame_rate
  start, end = scene.frame_range
  return tuple((frame - start) * steps_per_frame for frame in range(start, end))


def validate_scene_correspondence(
    visual: VisualSceneSpec, scene: SceneConfig
) -> None:
  """Raises ``ValueError`` unless ``visual`` renders exactly ``scene``.

  This is the guard that keeps rendered pixels attributable to logged physics: a
  visual object with no simulated counterpart would be unexplained in the
  annotations, and a simulated object with no visual counterpart would be invisible.
  """
  if not isinstance(visual, VisualSceneSpec):
    raise TypeError("visual must be a VisualSceneSpec")
  if not isinstance(scene, SceneConfig):
    raise TypeError("scene must be a SceneConfig")

  simulated = {item.object_id for item in scene.objects}
  rendered = {item.object_id for item in visual.objects}
  if simulated != rendered:
    difference = sorted(simulated.symmetric_difference(rendered))
    raise ValueError(
        "visual and simulated object sets differ: {}".format(difference)
    )

  for item in visual.objects:
    if item.collision_proxy_id not in simulated:
      raise ValueError(
          "collision_proxy_id {!r} names no simulated object".format(
              item.collision_proxy_id
          )
      )

  expected_steps = frame_steps_for(scene)
  if visual.frame_steps != expected_steps:
    raise ValueError(
        "frame_steps {} do not match the scene frame range {}".format(
            list(visual.frame_steps), list(expected_steps)
        )
    )


SMOKE_PROFILE = RenderProfile(
    name="smoke",
    resolution=(64, 64),
    samples_per_pixel=1,
    adaptive_sampling=False,
    use_denoising=False,
    background_transparency=False,
    layers=RENDER_LAYERS,
    device="CPU",
)

PRODUCTION_PROFILE = RenderProfile(
    name="production",
    resolution=(256, 256),
    samples_per_pixel=64,
    adaptive_sampling=True,
    use_denoising=True,
    background_transparency=False,
    layers=RENDER_LAYERS,
    device="CPU",
)

PROFILES_BY_NAME: Mapping[str, RenderProfile] = MappingProxyType({
    SMOKE_PROFILE.name: SMOKE_PROFILE,
    PRODUCTION_PROFILE.name: PRODUCTION_PROFILE,
})


__all__ = [
    "APPEARANCE_SCHEMA_VERSION",
    "BACKGROUND_KINDS",
    "COLOR_SPACES",
    "IMAGE_ROLES",
    "LIGHT_KINDS",
    "MATERIAL_FAMILIES",
    "MATERIAL_MODES",
    "PRODUCTION_PROFILE",
    "PROFILES_BY_NAME",
    "RENDER_DEVICES",
    "RENDER_LAYERS",
    "SMOKE_PROFILE",
    "SOURCE_KINDS",
    "TEXTURE_KINDS",
    "AssetReference",
    "BackgroundSpec",
    "CameraRenderSpec",
    "ImageReference",
    "LightSpec",
    "MaterialSpec",
    "RenderProfile",
    "TextureSpec",
    "VisualObjectSpec",
    "VisualSceneSpec",
    "frame_steps_for",
    "render_profile_hash",
    "validate_scene_correspondence",
    "visual_scene_from_payload",
    "visual_scene_hash",
]
