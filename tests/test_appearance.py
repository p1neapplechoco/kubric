"""Contract tests for the frozen appearance value schemas."""

import dataclasses
import json

import pytest

from interventions import appearance
from interventions import schema


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


def _visual_object(object_id="obj_0"):
  return appearance.VisualObjectSpec(
      object_id=object_id,
      source_kind="procedural",
      asset=None,
      collision_proxy_id=object_id,
      scale=(1.0, 1.0, 1.0),
      origin_offset=(0.0, 0.0, 0.0),
      alignment_quaternion=(1.0, 0.0, 0.0, 0.0),
      material=_material())


def _camera(num_frames=2):
  return appearance.CameraRenderSpec(
      positions=tuple((3.0 + i, 3.0, 2.0) for i in range(num_frames)),
      look_ats=tuple((0.0, 0.0, 0.0) for _ in range(num_frames)),
      focal_length=35.0,
      sensor_width=36.0,
      clipping_range=(0.1, 100.0))


def _light(light_id="key"):
  return appearance.LightSpec(
      light_id=light_id,
      kind="rect_area",
      position=(2.0, 2.0, 4.0),
      look_at=(0.0, 0.0, 0.0),
      color=(1.0, 0.98, 0.95, 1.0),
      intensity=120.0,
      width=1.5,
      height=1.5)


def _background():
  return appearance.BackgroundSpec(
      kind="color",
      color=(0.2, 0.2, 0.22, 1.0),
      hdri=None,
      rotation=0.0,
      strength=1.0,
      exposure=0.0)


def _visual_scene(object_ids=("obj_0", "obj_1"), frame_steps=(0, 10)):
  return appearance.VisualSceneSpec(
      objects=tuple(_visual_object(name) for name in object_ids),
      camera=_camera(len(frame_steps)),
      lights=(_light("key"), _light("fill")),
      background=_background(),
      render_seed=99,
      frame_steps=frame_steps)


def _scene_config(object_ids=("obj_0", "obj_1")):
  return schema.SceneConfig(
      objects=tuple(
          schema.ObjectConfig(object_id=name, shape="cube", size=0.2)
          for name in object_ids),
      camera=schema.CameraConfig(
          position=(3.0, 3.0, 2.0), look_at=(0.0, 0.0, 0.0), focal_length=35.0),
      frame_range=(0, 2),
      frame_rate=24,
      step_rate=240)


def test_camera_render_spec_requires_matching_path_lengths():
  with pytest.raises(ValueError, match="look_ats"):
    appearance.CameraRenderSpec(
        positions=((1.0, 1.0, 1.0),),
        look_ats=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        focal_length=35.0,
        sensor_width=36.0,
        clipping_range=(0.1, 100.0))
  with pytest.raises(ValueError, match="clipping_range"):
    appearance.CameraRenderSpec(
        positions=((1.0, 1.0, 1.0),),
        look_ats=((0.0, 0.0, 0.0),),
        focal_length=35.0,
        sensor_width=36.0,
        clipping_range=(10.0, 1.0))


def test_visual_scene_spec_requires_unique_objects_and_frame_alignment():
  with pytest.raises(ValueError, match="object_id"):
    _visual_scene(object_ids=("obj_0", "obj_0"))
  with pytest.raises(ValueError, match="frame_steps"):
    appearance.VisualSceneSpec(
        objects=(_visual_object(),),
        camera=_camera(2),
        lights=(_light(),),
        background=_background(),
        render_seed=1,
        frame_steps=(10, 0))


def test_frame_steps_for_maps_output_frames_to_physics_steps():
  scene = _scene_config()
  assert appearance.frame_steps_for(scene) == (0, 10)


def test_validate_scene_correspondence_requires_exact_object_match():
  visual = _visual_scene()
  appearance.validate_scene_correspondence(visual, _scene_config())
  with pytest.raises(ValueError, match="object"):
    appearance.validate_scene_correspondence(
        visual, _scene_config(object_ids=("obj_0", "obj_2")))


def test_visual_scene_hash_is_stable_and_sensitive():
  first = appearance.visual_scene_hash(_visual_scene())
  assert len(first) == 64
  assert first == appearance.visual_scene_hash(_visual_scene())
  changed = dataclasses.replace(_visual_scene(), render_seed=100)
  assert appearance.visual_scene_hash(changed) != first


def test_render_profiles_are_distinct_and_cover_all_layers():
  assert appearance.SMOKE_PROFILE.resolution == (64, 64)
  assert appearance.SMOKE_PROFILE.samples_per_pixel == 1
  assert appearance.SMOKE_PROFILE.adaptive_sampling is False
  assert appearance.SMOKE_PROFILE.use_denoising is False
  assert appearance.PRODUCTION_PROFILE.resolution == (256, 256)
  assert appearance.PRODUCTION_PROFILE.samples_per_pixel == 64
  assert appearance.PRODUCTION_PROFILE.adaptive_sampling is True
  assert appearance.PRODUCTION_PROFILE.use_denoising is True
  assert set(appearance.SMOKE_PROFILE.layers) == set(appearance.RENDER_LAYERS)
  assert (appearance.render_profile_hash(appearance.SMOKE_PROFILE)
          != appearance.render_profile_hash(appearance.PRODUCTION_PROFILE))


def test_profiles_by_name_resolves_both_profiles_immutably():
  assert appearance.PROFILES_BY_NAME["smoke"] is appearance.SMOKE_PROFILE
  assert appearance.PROFILES_BY_NAME["production"] is appearance.PRODUCTION_PROFILE
  with pytest.raises(TypeError):
    appearance.PROFILES_BY_NAME["smoke"] = appearance.PRODUCTION_PROFILE
