"""Immutable renderer-independent appearance schemas for intervention instances.

Purpose: define frozen, validated visual values and their canonical JSON form.
Public API: APPEARANCE_SCHEMA_VERSION, TEXTURE_KINDS, MATERIAL_FAMILIES, SOURCE_KINDS,
MATERIAL_MODES, COLOR_SPACES, IMAGE_ROLES, ImageReference, TextureSpec, MaterialSpec,
AssetReference, and VisualObjectSpec.
Dependencies: Python's standard library and interventions.schema helpers only, so
appearance never imports Kubric, Blender, or a simulator backend.
Trust boundary: validation enforces value ranges, enum membership, and JSON safety;
it does not verify that a referenced asset or image exists or is authentic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from interventions.schema import (
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


__all__ = [
    "APPEARANCE_SCHEMA_VERSION",
    "COLOR_SPACES",
    "IMAGE_ROLES",
    "MATERIAL_FAMILIES",
    "MATERIAL_MODES",
    "SOURCE_KINDS",
    "TEXTURE_KINDS",
    "AssetReference",
    "ImageReference",
    "MaterialSpec",
    "TextureSpec",
    "VisualObjectSpec",
]
