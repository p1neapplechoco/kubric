"""Shared-appearance and environment-object sampling across a branch pair."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from interventions import (
    Intervention,
    ObjectConfig,
    SceneConfig,
    generate_paired_instance,
    read_paired_artifact,
    write_paired_artifact,
)
from interventions import appearance, appearance_sampling, dataset

_ROOT = Path(__file__).resolve().parents[1]
_VISUAL_CONFIG = _ROOT / "configs" / "scene_ranges_visual.yaml"
_PHYSICS_CONFIG = _ROOT / "configs" / "scene_ranges.yaml"


def _visual_ranges():
  return dataset.load_ranges(_VISUAL_CONFIG)


def _object(object_id, *, position, static=False, mass=1.0, metadata=None):
  return ObjectConfig(
      object_id=object_id,
      shape="cube",
      size=0.25,
      position=position,
      static=static,
      mass=mass,
      friction=0.0,
      restitution=0.0,
      metadata={} if metadata is None else metadata,
  )


def _scene(*objects):
  return SceneConfig(
      objects=objects,
      camera=None,
      seed=7,
      scene_bounds=((-20, -20, -20), (20, 20, 20)),
      gravity=(0, 0, 0),
      frame_range=(0, 1),
      frame_rate=24,
      step_rate=240,
  )


def _intervention():
  return Intervention(
      target_id="target",
      recipe="remove_collision",
      magnitude=0.5,
      time_window=(1, 9),
      push_mass=2.0,
  )


def _pair_generation(directory):
  manifest = json.loads((directory / "manifest.json").read_text())
  return directory / "generations" / manifest["generation"]


def _pair_payload(directory):
  return json.loads((_pair_generation(directory) / "pair.json").read_text())


def _replace_pair_payload(directory, payload):
  import interventions.twin_runner as runner

  generation = _pair_generation(directory)
  (generation / "pair.json").write_bytes(runner._canonical_json(payload))
  manifest = runner._pair_manifest(generation)
  replacement = generation.parent / manifest["generation"]
  generation.rename(replacement)
  (directory / "manifest.json").write_bytes(runner._canonical_json(manifest))


@pytest.fixture(name="published_pair")
def _published_pair(tmp_path):
  """Publishes one pair carrying a sampled visual scene and returns its parts."""
  scene = _scene(
      _object("target", position=(0.0, 0.0, 0.0), static=True, mass=0.0),
      _object("free", position=(1.2, 0.0, 0.0)),
  )
  intervention = _intervention()
  path = np.zeros((10, 7), dtype=np.float64)
  path[:, 0] = np.linspace(0.0, 0.9, 10)
  path[:, 3] = 1.0
  factual, counterfactual = generate_paired_instance(
      scene, "target", intervention, 11, factual_path=path
  )
  visual = appearance_sampling.sample_visual_scene(
      _visual_ranges(), scene, 20260831, 3
  )
  directory = tmp_path / "pair"
  write_paired_artifact(
      directory,
      scene,
      intervention,
      11,
      factual,
      counterfactual,
      visual_scene=visual,
  )
  return directory, scene, visual


def test_pair_records_exactly_one_visual_scene_for_both_branches(published_pair):
  directory, scene, visual = published_pair
  payload = _pair_payload(directory)

  assert payload["visual_scene"] == visual.to_dict()
  assert payload["visual_scene_hash"] == appearance.visual_scene_hash(visual)
  # The appearance lives beside scene_config, not inside either branch, so the
  # branches cannot disagree about it.
  generation = _pair_generation(directory)
  for branch in ("factual", "counterfactual"):
    branch_files = {item.name for item in (generation / branch).iterdir()}
    assert not any("visual" in name for name in branch_files)


def test_published_visual_scene_covers_every_simulated_object(published_pair):
  directory, scene, visual = published_pair
  _, _, _, provenance = read_paired_artifact(directory)

  restored = appearance.visual_scene_from_payload(provenance["visual_scene"])
  assert restored == visual
  appearance.validate_scene_correspondence(restored, scene)
  assert {item.object_id for item in restored.objects} == {
      item.object_id for item in scene.objects
  }


def test_trust_model_is_unchanged_by_visual_provenance(published_pair):
  directory, _, _ = published_pair
  import interventions.twin_runner as runner

  payload = _pair_payload(directory)
  assert payload["trust_model"] == runner.PAIR_TRUST_MODEL

  _, _, _, provenance = read_paired_artifact(directory)
  assert provenance["trust_model"] == runner.PAIR_TRUST_MODEL


def test_reader_rejects_tampered_visual_scene(published_pair):
  directory, _, _ = published_pair
  payload = copy.deepcopy(_pair_payload(directory))
  payload["visual_scene"]["render_seed"] += 1
  _replace_pair_payload(directory, payload)

  with pytest.raises(ValueError, match="visual_scene_hash"):
    read_paired_artifact(directory)


def test_reader_rejects_visual_scene_that_omits_a_simulated_object(published_pair):
  directory, _, _ = published_pair
  payload = copy.deepcopy(_pair_payload(directory))
  payload["visual_scene"]["objects"] = payload["visual_scene"]["objects"][:1]
  payload["visual_scene_hash"] = appearance.visual_scene_hash(
      appearance.visual_scene_from_payload(payload["visual_scene"])
  )
  _replace_pair_payload(directory, payload)

  with pytest.raises(ValueError, match="does not describe the simulated scene"):
    read_paired_artifact(directory)


@pytest.mark.parametrize("dropped", ("visual_scene", "visual_scene_hash"))
def test_reader_rejects_half_written_visual_provenance(published_pair, dropped):
  directory, _, _ = published_pair
  payload = copy.deepcopy(_pair_payload(directory))
  del payload[dropped]
  _replace_pair_payload(directory, payload)

  with pytest.raises(ValueError, match="visual scene provenance is incomplete"):
    read_paired_artifact(directory)


def test_physics_only_pairs_stay_free_of_visual_keys(tmp_path):
  scene = _scene(
      _object("target", position=(0.0, 0.0, 0.0), static=True, mass=0.0),
      _object("free", position=(1.2, 0.0, 0.0)),
  )
  intervention = _intervention()
  path = np.zeros((10, 7), dtype=np.float64)
  path[:, 0] = np.linspace(0.0, 0.9, 10)
  path[:, 3] = 1.0
  factual, counterfactual = generate_paired_instance(
      scene, "target", intervention, 11, factual_path=path
  )
  directory = tmp_path / "pair"
  write_paired_artifact(
      directory, scene, intervention, 11, factual, counterfactual
  )

  payload = _pair_payload(directory)
  assert "visual_scene" not in payload
  assert "visual_scene_hash" not in payload
  _, _, _, provenance = read_paired_artifact(directory)
  assert "visual_scene" not in provenance


def test_write_paired_artifact_rejects_a_non_visual_scene(tmp_path):
  scene = _scene(
      _object("target", position=(0.0, 0.0, 0.0), static=True, mass=0.0),
  )
  intervention = _intervention()
  path = np.zeros((10, 7), dtype=np.float64)
  path[:, 3] = 1.0
  factual, counterfactual = generate_paired_instance(
      scene, "target", intervention, 11, factual_path=path
  )

  with pytest.raises(TypeError, match="visual_scene"):
    write_paired_artifact(
        tmp_path / "pair",
        scene,
        intervention,
        11,
        factual,
        counterfactual,
        visual_scene={"objects": []},
    )


def test_instance_appearance_is_derived_from_the_attempt_seed():
  ranges = _visual_ranges()
  spec = dataset.sample_instance_spec(ranges, 20260831, 2)

  first = dataset.sample_instance_appearance(ranges, spec, 20260831, 2)
  second = dataset.sample_instance_appearance(ranges, spec, 20260831, 2)
  other_attempt = dataset.sample_instance_appearance(ranges, spec, 20260831, 3)
  other_run = dataset.sample_instance_appearance(ranges, spec, 20260901, 2)

  assert first == second
  assert appearance.visual_scene_hash(first) != appearance.visual_scene_hash(
      other_attempt
  )
  assert appearance.visual_scene_hash(first) != appearance.visual_scene_hash(
      other_run
  )
  appearance.validate_scene_correspondence(first, spec.scene_config)


def test_instance_appearance_is_absent_without_appearance_ranges():
  ranges = dataset.load_ranges(_PHYSICS_CONFIG)
  spec = dataset.sample_instance_spec(ranges, 20260811, 0)

  assert dataset.sample_instance_appearance(ranges, spec, 20260811, 0) is None


def test_batch_publishes_one_shared_visual_scene_per_accepted_instance(tmp_path):
  ranges = _visual_ranges()
  result = dataset.run_batch(ranges, tmp_path / "first", 20260831, 1, 4)
  assert result["status"] in ("complete", "capacity_exhausted")
  instances = sorted(
      item
      for item in (tmp_path / "first" / "instances").iterdir()
      if item.name.startswith("instance_")
  )
  assert instances, "the batch published no instance to inspect"

  for instance in instances:
    payload = _pair_payload(instance)
    restored = appearance.visual_scene_from_payload(payload["visual_scene"])
    assert appearance.visual_scene_hash(restored) == payload["visual_scene_hash"]
    # read_paired_artifact re-derives both branches from this single record.
    factual, counterfactual, _, provenance = read_paired_artifact(instance)
    assert factual.object_ids == counterfactual.object_ids
    assert set(factual.object_ids) == {
        item.object_id for item in restored.objects
    }
    assert provenance["visual_scene_hash"] == payload["visual_scene_hash"]


def test_batch_visual_scenes_are_reproducible(tmp_path):
  ranges = _visual_ranges()

  def _hashes(name):
    root = tmp_path / name
    dataset.run_batch(ranges, root, 20260831, 1, 4)
    return {
        item.name: _pair_payload(item)["visual_scene_hash"]
        for item in sorted((root / "instances").iterdir())
        if item.name.startswith("instance_")
    }

  first = _hashes("first")
  assert first
  assert first == _hashes("second")


def test_static_fraction_produces_static_environment_obstacles():
  ranges = _visual_ranges()
  assert ranges["objects"]["static_fraction"] == (0.0, 0.5)

  environment = []
  for index in range(24):
    spec = dataset.sample_instance_spec(ranges, 20260831, index)
    environment.extend(
        item
        for item in spec.scene_config.objects
        if item.object_id.startswith("object_")
        and item.metadata.get("role") == "environment"
    )
  assert environment, "static_fraction never designated an environment object"
  for item in environment:
    assert item.static is True
    assert item.mass == 0.0
    assert item.metadata["role"] == "environment"


def test_free_objects_are_dynamic_without_static_fraction():
  ranges = dict(dataset.load_ranges(_VISUAL_CONFIG))
  objects = dict(ranges["objects"])
  del objects["static_fraction"]
  ranges["objects"] = objects

  for index in range(24):
    spec = dataset.sample_instance_spec(ranges, 20260831, index)
    for item in spec.scene_config.objects:
      if item.object_id.startswith("object_"):
        assert item.static is False
        assert item.metadata["role"] == "dynamic"


def test_static_fraction_leaves_the_physics_sampling_stream_untouched():
  without = dict(dataset.load_ranges(_VISUAL_CONFIG))
  objects = dict(without["objects"])
  del objects["static_fraction"]
  without["objects"] = objects
  zero = dict(dataset.load_ranges(_VISUAL_CONFIG))
  zero_objects = dict(zero["objects"])
  zero_objects["static_fraction"] = [0.0, 0.0]
  zero["objects"] = zero_objects

  for index in range(8):
    baseline = dataset.sample_instance_spec(without, 20260831, index)
    disabled = dataset.sample_instance_spec(zero, 20260831, index)
    assert baseline.instance_id == disabled.instance_id
    assert baseline.to_dict() == disabled.to_dict()


def test_environment_objects_stay_clear_of_the_target_corridor():
  import interventions.twin_runner as runner

  ranges = _visual_ranges()
  checked = 0
  for index in range(24):
    spec = dataset.sample_instance_spec(ranges, 20260831, index)
    obstacles = [
        item
        for item in spec.scene_config.objects
        if item.static and item.object_id != spec.target_id
    ]
    if not any(item.object_id.startswith("object_") for item in obstacles):
      continue
    checked += 1
    # The same AABB set twin_runner feeds to swept-volume validation and to
    # perturb_path's obstacle avoidance; sampling must never place one in the way.
    aabbs = runner._static_aabbs(spec.scene_config, spec.target_id)
    assert len(aabbs) > 1
    target = next(
        item
        for item in spec.scene_config.objects
        if item.object_id == spec.target_id
    )
    runner._validate_target_sweep(
        np.asarray(spec.factual_path, dtype=np.float64),
        target,
        np.asarray(spec.scene_config.scene_bounds, dtype=np.float64),
        aabbs,
    )
  assert checked, "no sampled scene contained a static environment object"


def test_environment_objects_survive_candidate_generation():
  ranges = _visual_ranges()
  for index in range(24):
    spec = dataset.sample_instance_spec(ranges, 20260831, index)
    if not any(
        item.metadata.get("role") == "environment"
        and item.object_id.startswith("object_")
        for item in spec.scene_config.objects
    ):
      continue
    factual, counterfactual, _ = dataset.generate_candidate(spec)
    assert factual.object_ids == counterfactual.object_ids
    return
  pytest.fail("no sampled scene contained a static environment object")
