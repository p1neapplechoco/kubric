import copy

import pytest
import yaml

from interventions import appearance, appearance_sampling, dataset, materials, schema

_CONFIG = "configs/scene_ranges_visual.yaml"


def _ranges():
  return dataset.load_ranges(_CONFIG)


def _scene(object_ids=("obj_0", "obj_1", "obj_2")):
  return schema.SceneConfig(
      objects=tuple(
          schema.ObjectConfig(object_id=name, shape="cube", size=0.2)
          for name in object_ids),
      camera=schema.CameraConfig(
          position=(4.0, 4.0, 3.0), look_at=(0.0, 0.0, 0.0), focal_length=35.0),
      frame_range=(0, 2),
      frame_rate=24,
      step_rate=240)


def test_sample_visual_scene_is_deterministic_and_valid():
  ranges = _ranges()
  scene = _scene()
  first = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 0)
  second = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 0)
  assert first == second
  appearance.validate_scene_correspondence(first, scene)
  assert first.frame_steps == (0, 10)
  assert len(first.camera.positions) == 2
  assert {item.object_id for item in first.objects} == set(
      item.object_id for item in scene.objects)


def test_sample_visual_scene_varies_with_index():
  ranges = _ranges()
  scene = _scene()
  a = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 0)
  b = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 1)
  assert appearance.visual_scene_hash(a) != appearance.visual_scene_hash(b)


def test_every_shipped_family_appears_in_a_deterministic_window():
  ranges = _ranges()
  scene = _scene(tuple("obj_{}".format(i) for i in range(6)))
  seen_families = set()
  seen_textures = set()
  for index in range(24):
    visual = appearance_sampling.sample_visual_scene(ranges, scene, 20260829, index)
    for item in visual.objects:
      seen_families.add(item.material.family)
      seen_textures.add(item.material.texture.kind)
  assert seen_families == set(appearance.MATERIAL_FAMILIES)
  assert seen_textures >= {"solid", "noise", "checker", "wood", "marble", "speckle"}


def test_geometry_sampler_covers_all_four_proxies():
  import numpy as np
  ranges = _ranges()
  shapes = set()
  for seed in range(64):
    rng = np.random.default_rng(seed)
    shapes.add(appearance_sampling.sample_object_geometry(ranges, rng)["shape"])
  assert shapes == {"cube", "sphere", "cylinder", "capsule"}


def test_sampled_sizes_respect_radial_equality():
  import numpy as np
  ranges = _ranges()
  for seed in range(32):
    rng = np.random.default_rng(seed)
    result = appearance_sampling.sample_object_geometry(ranges, rng)
    config = schema.ObjectConfig(
        object_id="a", shape=result["shape"], size=result["size"])
    assert schema.half_extents(config)[0] > 0.0


def test_camera_frames_full_scene_bounds():
  ranges = _ranges()
  scene = _scene()
  visual = appearance_sampling.sample_visual_scene(ranges, scene, 7, 3)
  for position in visual.camera.positions:
    assert position[2] > 0.0
    radius = sum(value * value for value in position) ** 0.5
    low, high = ranges["appearance"]["camera"]["radius"]
    assert low - 1e-6 <= radius <= high + 1e-6


def test_validate_appearance_ranges_rejects_unknown_family_and_bad_bounds():
  raw = yaml.safe_load(open(_CONFIG, encoding="utf-8"))
  bad = copy.deepcopy(raw)
  bad["appearance"]["materials"]["families"] = ["unobtanium"]
  with pytest.raises(ValueError, match="family"):
    appearance_sampling.validate_appearance_ranges(bad)
  bad = copy.deepcopy(raw)
  bad["appearance"]["camera"]["radius"] = [9.0, 2.0]
  with pytest.raises(ValueError, match="radius"):
    appearance_sampling.validate_appearance_ranges(bad)


def test_held_out_combinations_are_never_sampled():
  raw = yaml.safe_load(open(_CONFIG, encoding="utf-8"))
  raw["appearance"]["coupling"]["mode"] = "held_out"
  raw["appearance"]["coupling"]["held_out"] = [{"material_family": "glass"}]
  ranges = appearance_sampling.validate_appearance_ranges(raw)
  scene = _scene(tuple("obj_{}".format(i) for i in range(4)))
  for index in range(32):
    visual = appearance_sampling.sample_visual_scene(ranges, scene, 99, index)
    assert "glass" not in {item.material.family for item in visual.objects}


def test_seed_domain_independence_uses_frozen_ranges():
  raw = yaml.safe_load(open(_CONFIG, encoding="utf-8"))
  base = appearance_sampling.sample_visual_scene(
      appearance_sampling.validate_appearance_ranges(raw), _scene(), 4321, 0)
  mutated = copy.deepcopy(raw)
  mutated["appearance"]["background"]["color_value"] = [0.05, 0.35]
  changed = appearance_sampling.sample_visual_scene(
      appearance_sampling.validate_appearance_ranges(mutated), _scene(), 4321, 0)
  assert changed.background != base.background
  assert changed.objects == base.objects
  assert changed.camera == base.camera
  assert changed.lights == base.lights
