"""Contract tests for the frozen appearance value schemas."""

import dataclasses
import json

import pytest

from interventions import appearance


def _texture(**overrides):
  values = {
      "kind": "checker",
      "seed": 7,
      "colors": ((0.1, 0.2, 0.3, 1.0), (0.9, 0.8, 0.7, 1.0)),
      "scale": 4.0,
      "detail": 2.0,
      "roughness": 0.5,
      "distortion": 0.0,
      "rotation": 0.25,
  }
  values.update(overrides)
  return appearance.TextureSpec(**values)


def _material(**overrides):
  values = {
      "family": "plastic",
      "base_color": (0.4, 0.5, 0.6, 1.0),
      "metallic": 0.0,
      "roughness": 0.4,
      "specular": 0.5,
      "ior": 1.45,
      "transmission": 0.0,
      "emission": (0.0, 0.0, 0.0, 1.0),
      "texture": _texture(),
  }
  values.update(overrides)
  return appearance.MaterialSpec(**values)


def test_texture_spec_is_frozen_and_canonical():
  texture = _texture()
  assert dataclasses.is_dataclass(texture)
  with pytest.raises(dataclasses.FrozenInstanceError):
    texture.scale = 2.0
  assert isinstance(texture.colors, tuple)
  assert texture.to_dict()["kind"] == "checker"
  assert json.dumps(texture.to_dict(), sort_keys=True)


def test_texture_spec_rejects_unknown_kind():
  with pytest.raises(ValueError, match="kind"):
    _texture(kind="plaid")


def test_texture_spec_rejects_nonfinite_and_out_of_range_colors():
  with pytest.raises(ValueError):
    _texture(colors=((0.0, 0.0, 0.0, float("nan")),))
  with pytest.raises(ValueError, match="colors"):
    _texture(colors=((0.0, 0.0, 0.0, 1.5),))
  with pytest.raises(ValueError, match="colors"):
    _texture(colors=())


def test_image_texture_requires_pinned_images():
  with pytest.raises(ValueError, match="images"):
    _texture(kind="image", images=())


def test_image_reference_requires_hex_digest_and_known_color_space():
  with pytest.raises(ValueError, match="sha256"):
    appearance.ImageReference(
        role="base_color", uri="file:///t.png", sha256="zz", color_space="sRGB")
  with pytest.raises(ValueError, match="color_space"):
    appearance.ImageReference(
        role="roughness", uri="file:///t.png", sha256="a" * 64, color_space="sRGB")
  reference = appearance.ImageReference(
      role="roughness", uri="file:///t.png", sha256="a" * 64, color_space="Non-Color")
  assert reference.color_space == "Non-Color"


def test_material_spec_rejects_unknown_family_and_bad_ranges():
  with pytest.raises(ValueError, match="family"):
    _material(family="unobtanium")
  with pytest.raises(ValueError, match="metallic"):
    _material(metallic=1.5)
  with pytest.raises(ValueError, match="ior"):
    _material(ior=0.5)


def test_material_spec_rejects_alpha_below_one_without_transmission():
  with pytest.raises(ValueError, match="transmission"):
    _material(base_color=(0.4, 0.5, 0.6, 0.5))
  translucent = _material(
      family="glass", base_color=(0.4, 0.5, 0.6, 0.5), transmission=0.9, roughness=0.05)
  assert translucent.base_color[3] == 0.5


def test_asset_reference_pins_manifest_and_archive_digests():
  reference = appearance.AssetReference(
      source_kind="gso",
      manifest_uri="file:///manifest.json",
      manifest_sha256="b" * 64,
      asset_id="Mug_001",
      archive_sha256="c" * 64,
      material_mode="native")
  assert reference.to_dict()["asset_id"] == "Mug_001"
  with pytest.raises(ValueError, match="material_mode"):
    dataclasses.replace(reference, material_mode="inherit")
  with pytest.raises(ValueError, match="source_kind"):
    dataclasses.replace(reference, source_kind="procedural")


def test_visual_object_spec_requires_asset_for_external_sources():
  with pytest.raises(ValueError, match="asset"):
    appearance.VisualObjectSpec(
        object_id="obj_0",
        source_kind="gso",
        asset=None,
        collision_proxy_id="obj_0",
        scale=(1.0, 1.0, 1.0),
        origin_offset=(0.0, 0.0, 0.0),
        alignment_quaternion=(1.0, 0.0, 0.0, 0.0),
        material=_material())


def test_visual_object_spec_normalizes_alignment_quaternion():
  spec = appearance.VisualObjectSpec(
      object_id="obj_0",
      source_kind="procedural",
      asset=None,
      collision_proxy_id="obj_0",
      scale=(1.0, 2.0, 3.0),
      origin_offset=(0.0, 0.0, 0.5),
      alignment_quaternion=(2.0, 0.0, 0.0, 0.0),
      material=_material())
  assert spec.alignment_quaternion == (1.0, 0.0, 0.0, 0.0)
  assert spec.to_dict()["scale"] == [1.0, 2.0, 3.0]
