"""Deterministic paired-runner integration and artifact tests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pybullet as pb
import pytest

import kubric as kb

from interventions import (
    CameraConfig,
    ContactRecord,
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    KinematicDragSimulator,
    ObjectConfig,
    SceneConfig,
    SimulationLog,
    extract_pair_ground_truth,
    generate_paired_instance,
    read_paired_artifact,
    read_simulation_log,
    write_paired_artifact,
)


def _object(
    object_id,
    *,
    shape="cube",
    size=0.25,
    position=(0.0, 0.0, 0.0),
    linear_velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
    static=False,
    mass=1.0,
    friction=0.0,
    restitution=0.0,
    metadata=None,
):
  return ObjectConfig(
      object_id=object_id,
      shape=shape,
      size=size,
      position=position,
      linear_velocity=linear_velocity,
      angular_velocity=angular_velocity,
      static=static,
      mass=mass,
      friction=friction,
      restitution=restitution,
      metadata={} if metadata is None else metadata,
  )


def _scene(*objects, seed=7, camera=None, frame_range=(0, 1), gravity=(0, 0, 0)):
  return SceneConfig(
      objects=objects,
      camera=camera,
      seed=seed,
      scene_bounds=((-20, -20, -20), (20, 20, 20)),
      gravity=gravity,
      frame_range=frame_range,
      frame_rate=24,
      step_rate=240,
  )


def _intervention(
    *, target_id="target", magnitude=0.5, time_window=(1, 9), recipe="remove_collision"
):
  return Intervention(
      target_id=target_id,
      recipe=recipe,
      magnitude=magnitude,
      time_window=time_window,
      push_mass=2.0,
  )


def _connected_count(limit=32):
  return sum(bool(pb.isConnected(client_id)) for client_id in range(limit))


def _assert_physics_equal(left, right):
  assert left.object_ids == right.object_ids
  assert left.steps == right.steps
  np.testing.assert_array_equal(left.states, right.states)
  assert left.contacts == right.contacts
  np.testing.assert_array_equal(left.commanded_path, right.commanded_path)


def _replace_log(log, **changes):
  return dataclasses.replace(log, **changes)


def _pair_generation(directory):
  manifest = json.loads((directory / "manifest.json").read_text())
  return directory / "generations" / manifest["generation"], manifest


def test_default_path_uses_schema_exclusive_duration_and_velocity_formula():
  target = _object(
      "target",
      static=True,
      position=(-1.0, 2.0, 3.0),
      linear_velocity=(2.4, -1.2, 0.6),
  )
  config = _scene(target)
  intervention = _intervention(magnitude=0.0)

  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 3
  )

  expected = np.asarray(target.position)[None, :] + (
      np.arange(10)[:, None] * np.asarray(target.linear_velocity)[None, :] / 240.0
  )
  assert factual.commanded_path.shape == (10, 7)
  np.testing.assert_allclose(factual.commanded_path[:, :3], expected, atol=0, rtol=0)
  np.testing.assert_array_equal(
      factual.commanded_path[:, 3:], np.tile(target.quaternion, (10, 1))
  )
  _assert_physics_equal(factual, counterfactual)


def test_explicit_path_is_copied_and_counterfactual_changes_only_inside_window():
  target = _object("target", static=True, position=(-1, 0, 0))
  config = _scene(target)
  path = np.zeros((10, 7), dtype=np.float64)
  path[:, 0] = np.linspace(-1, 1, 10)
  path[:, 3] = 1.0
  original = path.copy()

  factual, counterfactual = generate_paired_instance(
      config,
      "target",
      _intervention(magnitude=0.7, time_window=(1, 9)),
      0,
      factual_path=path,
  )

  np.testing.assert_array_equal(path, original)
  np.testing.assert_array_equal(factual.commanded_path, original)
  np.testing.assert_array_equal(counterfactual.commanded_path[:1], original[:1])
  np.testing.assert_array_equal(counterfactual.commanded_path[9:], original[9:])
  np.testing.assert_array_equal(counterfactual.commanded_path[1], original[1])
  np.testing.assert_array_equal(counterfactual.commanded_path[8], original[8])
  assert np.any(counterfactual.commanded_path[2:8, :3] != original[2:8, :3])
  deviation = np.linalg.norm(
      counterfactual.commanded_path[:, :3] - original[:, :3], axis=1
  )
  assert deviation.max() <= 0.7 + 1e-12


def test_repeated_runs_are_exact_and_factual_does_not_depend_on_rng_seed():
  config = _scene(
      _object("target", static=True, position=(-1, 0, 0), linear_velocity=(24, 0, 0)),
      _object("ball", shape="sphere", position=(5, 0, 0)),
  )
  intervention = _intervention(magnitude=0.5)

  first = generate_paired_instance(config, "target", intervention, 11)
  second = generate_paired_instance(config, "target", intervention, 11)
  different_seed = generate_paired_instance(config, "target", intervention, 12)

  assert first == second
  assert extract_pair_ground_truth(config, intervention, *first) == (
      extract_pair_ground_truth(config, intervention, *second)
  )
  assert first[0] == different_seed[0]


def test_prefix_states_contacts_and_first_non_target_state_are_exact_twins():
  config = _scene(
      _object("target", static=True, position=(-2, 0, 0), linear_velocity=(30, 0, 0)),
      _object("ball", shape="sphere", position=(10, 0, 0)),
  )
  factual, counterfactual = generate_paired_instance(
      config, "target", _intervention(time_window=(4, 9)), 4
  )

  np.testing.assert_array_equal(factual.states[:4], counterfactual.states[:4])
  assert tuple(record for record in factual.contacts if record.step < 4) == tuple(
      record for record in counterfactual.contacts if record.step < 4
  )
  ball = factual.object_ids.index("ball")
  np.testing.assert_array_equal(factual.states[0, ball], counterfactual.states[0, ball])


def test_zero_magnitude_has_identical_physics_and_empty_ground_truth():
  config = _scene(
      _object("target", static=True, position=(-1, 0, 0), linear_velocity=(24, 0, 0)),
      _object("ball", shape="sphere", position=(0, 0, 0)),
  )
  intervention = _intervention(magnitude=0.0)

  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 99
  )
  truth = extract_pair_ground_truth(config, intervention, factual, counterfactual)

  _assert_physics_equal(factual, counterfactual)
  assert truth == GroundTruth(graph_delta=GraphEdgeDelta())


def test_every_branch_gets_a_fresh_simulator_and_closes_on_success(
    monkeypatch, tmp_path
):
  import interventions.twin_runner as runner

  real_class = KinematicDragSimulator
  constructed = []
  closed = []

  class SpySimulator(real_class):
    def __init__(self, scene, scratch_dir=None):
      constructed.append((scene, Path(scratch_dir)))
      super().__init__(scene, scratch_dir=scratch_dir)

    def close(self):
      was_connected = self.is_connected
      super().close()
      if was_connected:
        closed.append(self)

  monkeypatch.setattr(runner, "KinematicDragSimulator", SpySimulator)
  config = _scene(_object("target", static=True))

  generate_paired_instance(config, "target", _intervention(magnitude=0), 0)

  assert len(constructed) == 2
  assert constructed[0][0] is not constructed[1][0]
  assert constructed[0][1] != constructed[1][1]
  assert len(closed) == 2
  assert all(not simulator.is_connected for simulator in closed)


def test_counterfactual_failure_still_closes_both_clients(monkeypatch):
  import interventions.twin_runner as runner

  real_class = KinematicDragSimulator
  closed = []
  run_count = 0

  class FailingSecondSimulator(real_class):
    def run_with_intervention(self, *args, **kwargs):
      nonlocal run_count
      run_count += 1
      if run_count == 2:
        raise RuntimeError("counterfactual exploded")
      return super().run_with_intervention(*args, **kwargs)

    def close(self):
      was_connected = self.is_connected
      super().close()
      if was_connected:
        closed.append(self)

  monkeypatch.setattr(runner, "KinematicDragSimulator", FailingSecondSimulator)
  config = _scene(_object("target", static=True))

  with pytest.raises(RuntimeError, match="counterfactual exploded"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)

  assert len(closed) == 2
  assert all(not simulator.is_connected for simulator in closed)


def test_scene_primitive_camera_and_physics_properties_are_mapped(monkeypatch):
  import interventions.twin_runner as runner

  real_class = KinematicDragSimulator
  scenes = []

  class InspectingSimulator(real_class):
    def __init__(self, scene, scratch_dir=None):
      scenes.append(scene)
      super().__init__(scene, scratch_dir=scratch_dir)

    def run_with_intervention(self, target, *args, **kwargs):
      by_id = {
          asset.metadata["logical_id"]: asset
          for asset in self.scene.assets
          if isinstance(asset, kb.PhysicalObject)
      }
      cube = by_id["target"]
      sphere = by_id["ball"]
      assert isinstance(cube, kb.Cube)
      assert isinstance(sphere, kb.Sphere)
      np.testing.assert_allclose(cube.scale, (0.2, 0.3, 0.4), atol=2e-8, rtol=0)
      np.testing.assert_allclose(sphere.scale, (0.6, 0.6, 0.6), atol=3e-8, rtol=0)
      assert cube.metadata == {"kind": "mover", "logical_id": "target"}
      assert sphere.mass == 3.0
      assert sphere.friction == 0.4
      assert sphere.restitution == 0.7
      body = sphere.linked_objects[self]
      dynamics = self.bullet_client.getDynamicsInfo(body, -1)
      assert dynamics[0] == pytest.approx(3.0)
      assert dynamics[1] == pytest.approx(0.4)
      velocity, angular = self.bullet_client.getBaseVelocity(body)
      np.testing.assert_allclose(velocity, (1, 2, 3), atol=0, rtol=0)
      np.testing.assert_allclose(angular, (4, 5, 6), atol=0, rtol=0)
      assert isinstance(self.scene.camera, kb.PerspectiveCamera)
      np.testing.assert_array_equal(self.scene.camera.position, (3, -4, 5))
      assert self.scene.camera.focal_length == 35
      return super().run_with_intervention(target, *args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", InspectingSimulator)
  camera = CameraConfig(position=(3, -4, 5), look_at=(0, 0, 0), focal_length=35)
  config = _scene(
      _object("target", static=True, size=(0.2, 0.3, 0.4), metadata={"kind": "mover"}),
      _object(
          "ball",
          shape="sphere",
          size=0.6,
          position=(5, 0, 0),
          linear_velocity=(1, 2, 3),
          angular_velocity=(4, 5, 6),
          mass=3,
          friction=0.4,
          restitution=0.7,
      ),
      camera=camera,
  )

  generate_paired_instance(config, "target", _intervention(magnitude=0), 0)

  assert len(scenes) == 2
  for scene in scenes:
    physical_ids = [
        asset.metadata["logical_id"]
        for asset in scene.assets
        if isinstance(asset, kb.PhysicalObject)
    ]
    assert physical_ids == ["target", "ball"]
    assert scene.frame_start == 0
    assert scene.frame_end == 0  # SceneConfig end is exclusive; Kubric end is inclusive.


@pytest.mark.parametrize(
    "case, message",
    [
        ("target_mismatch", "target"),
        ("missing_target", "target"),
        ("dynamic_target", "static"),
        ("fractional_window", "integer"),
        ("window_outside", "window"),
        ("nonuniform_sphere", "uniform"),
        ("conflicting_logical_id", "logical_id"),
        ("angular_target", "angular"),
    ],
)
def test_validation_rejects_unsupported_or_inconsistent_inputs(case, message):
  target = _object("target", static=True)
  config = _scene(target)
  intervention = _intervention(magnitude=0)
  target_id = "target"

  if case == "target_mismatch":
    target_id = "different"
  elif case == "missing_target":
    target_id = "missing"
    intervention = _intervention(target_id="missing", magnitude=0)
  elif case == "dynamic_target":
    config = _scene(_object("target", static=False))
  elif case == "fractional_window":
    intervention = _intervention(magnitude=0, time_window=(1.5, 9))
  elif case == "window_outside":
    intervention = _intervention(magnitude=0, time_window=(1, 11))
  elif case == "nonuniform_sphere":
    config = _scene(target, _object("ball", shape="sphere", size=(1, 2, 1)))
  elif case == "conflicting_logical_id":
    config = _scene(_object("target", static=True, metadata={"logical_id": "wrong"}))
  elif case == "angular_target":
    config = _scene(_object("target", static=True, angular_velocity=(0, 0, 1)))

  with pytest.raises((TypeError, ValueError), match=message):
    generate_paired_instance(config, target_id, intervention, 0)


@pytest.mark.parametrize(
    "path, message",
    [
        (np.zeros((9, 7)), "shape|duration"),
        (np.column_stack((np.ones((10, 3)), np.tile((1, 0, 0, 0), (10, 1)))), "initial"),
        (np.column_stack((np.zeros((10, 3)), np.tile((2, 0, 0, 0), (10, 1)))), "quaternion"),
    ],
)
def test_explicit_path_validation(path, message):
  config = _scene(_object("target", static=True))
  with pytest.raises(ValueError, match=message):
    generate_paired_instance(
        config,
        "target",
        _intervention(magnitude=0),
        0,
        factual_path=path,
    )


def test_repeated_calls_do_not_leak_pybullet_connections():
  before = _connected_count()
  config = _scene(_object("target", static=True))

  for seed in range(4):
    generate_paired_instance(config, "target", _intervention(magnitude=0), seed)

  assert _connected_count() == before


@pytest.mark.parametrize("time_window", [(3, 4), (3, 5)])
def test_nonzero_intervention_requires_three_anchored_samples(time_window):
  config = _scene(_object("target", static=True))
  with pytest.raises(ValueError, match="three|3|samples"):
    generate_paired_instance(
        config,
        "target",
        _intervention(magnitude=0.5, time_window=time_window),
        0,
    )


def test_extraction_and_writer_reject_swapped_branches(tmp_path):
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 2
  )

  with pytest.raises(ValueError, match="branch|factual|counterfactual"):
    extract_pair_ground_truth(config, intervention, counterfactual, factual)
  with pytest.raises(ValueError, match="branch|factual|counterfactual"):
    write_paired_artifact(
        tmp_path / "pair",
        config,
        intervention,
        2,
        counterfactual,
        factual,
    )
  assert not (tmp_path / "pair").exists()


def test_pair_validation_rejects_unrelated_schema_and_intervention():
  config = _scene(
      _object("target", static=True),
      _object("ball", shape="sphere", position=(5, 0, 0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 2
  )
  unrelated_scene = _scene(
      _object("target", static=True, mass=4),
      _object("ball", shape="sphere", position=(5, 0, 0)),
  )
  unrelated_intervention = _intervention(
      magnitude=0.25, recipe="break_contact"
  )

  with pytest.raises(ValueError, match="scene|provenance|config"):
    extract_pair_ground_truth(
        unrelated_scene, intervention, factual, counterfactual
    )
  with pytest.raises(ValueError, match="intervention|provenance"):
    extract_pair_ground_truth(
        config, unrelated_intervention, factual, counterfactual
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("branch", "branch"),
        ("object_order", "object_ids|order"),
        ("steps", "steps|duration"),
        ("step_rate", "step_rate"),
        ("target_metadata", "target_id|metadata"),
        ("push_mass_metadata", "push_mass|metadata"),
        ("dt_metadata", "dt|metadata"),
        ("path_width", "commanded_path|shape"),
        ("outside_window", "window|commanded_path"),
        ("excess_magnitude", "magnitude|commanded_path"),
    ],
)
def test_pair_validation_rejects_malformed_log_provenance(mutation, message):
  config = _scene(
      _object("target", static=True),
      _object("ball", shape="sphere", position=(5, 0, 0)),
  )
  intervention = _intervention(magnitude=0.5)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 3
  )
  bad = counterfactual
  if mutation == "branch":
    bad = _replace_log(bad, branch="wrong")
  elif mutation == "object_order":
    bad = _replace_log(bad, object_ids=tuple(reversed(bad.object_ids)))
  elif mutation == "steps":
    bad = _replace_log(bad, steps=tuple(step + 1 for step in bad.steps))
  elif mutation == "step_rate":
    bad = _replace_log(bad, step_rate=120)
  elif mutation in {"target_metadata", "push_mass_metadata", "dt_metadata"}:
    metadata = dict(bad.metadata)
    key, value = {
        "target_metadata": ("target_id", "ball"),
        "push_mass_metadata": ("push_mass", 9.0),
        "dt_metadata": ("dt", 0.25),
    }[mutation]
    metadata[key] = value
    bad = _replace_log(bad, metadata=metadata)
  else:
    if mutation == "path_width":
      path = bad.commanded_path[:, :3]
    else:
      path = bad.commanded_path.copy()
    if mutation == "outside_window":
      path[9, 1] += 0.1
    elif mutation == "excess_magnitude":
      path[4, 1] += 1.0
    bad = _replace_log(bad, commanded_path=path)

  with pytest.raises(ValueError, match=message):
    extract_pair_ground_truth(config, intervention, factual, bad)


def test_pair_validation_binds_counterfactual_path_to_recorded_rng_seed():
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0.5)
  factual, counterfactual_seed_3 = generate_paired_instance(
      config, "target", intervention, 3
  )
  _, counterfactual_seed_4 = generate_paired_instance(
      config, "target", intervention, 4
  )
  assert not np.array_equal(
      counterfactual_seed_3.commanded_path,
      counterfactual_seed_4.commanded_path,
  )
  mismatched = _replace_log(
      counterfactual_seed_3,
      commanded_path=counterfactual_seed_4.commanded_path,
  )

  with pytest.raises(ValueError, match="rng_seed|recipe|commanded_path|provenance"):
    extract_pair_ground_truth(config, intervention, factual, mismatched)


@pytest.mark.parametrize("mutation", ["position", "quaternion"])
def test_pair_validation_rejects_target_pose_unbound_from_commanded_path(
    mutation, tmp_path
):
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 3
  )
  target_index = factual.object_ids.index("target")

  def corrupt(log):
    states = log.states.copy()
    if mutation == "position":
      states[:, target_index, 0] += 7.0
    else:
      states[:, target_index, 3:7] = np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)
    return _replace_log(log, states=states)

  bad_factual = corrupt(factual)
  bad_counterfactual = corrupt(counterfactual)
  with pytest.raises(ValueError, match="target|pose|position|quaternion|commanded"):
    extract_pair_ground_truth(
        config, intervention, bad_factual, bad_counterfactual
    )
  with pytest.raises(ValueError, match="target|pose|position|quaternion|commanded"):
    write_paired_artifact(
        tmp_path / "pair",
        config,
        intervention,
        3,
        bad_factual,
        bad_counterfactual,
    )
  assert not (tmp_path / "pair").exists()


def test_pair_validation_accepts_quaternion_sign_equivalent_target_state():
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 3
  )
  target_index = factual.object_ids.index("target")

  def flip(log):
    states = log.states.copy()
    states[:, target_index, 3:7] *= -1.0
    return _replace_log(log, states=states)

  assert extract_pair_ground_truth(
      config, intervention, flip(factual), flip(counterfactual)
  ) == GroundTruth(graph_delta=GraphEdgeDelta())


def test_pair_validation_rejects_scaled_target_quaternion():
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 3
  )
  target_index = factual.object_ids.index("target")

  def scale(log):
    states = log.states.copy()
    states[:, target_index, 3:7] *= 1.0 + 8e-7
    return _replace_log(log, states=states)

  with pytest.raises(ValueError, match="target|quaternion|unit|norm"):
    extract_pair_ground_truth(
        config, intervention, scale(factual), scale(counterfactual)
    )


@pytest.mark.parametrize("operation", ["extract", "write"])
def test_pair_validation_rejects_unknown_contact_endpoint(
    operation, tmp_path
):
  config = _scene(_object("target", shape="sphere", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  target_index = counterfactual.object_ids.index("target")
  states = counterfactual.states.copy()
  states[1, target_index, 0] += 7.0
  fake_contact = ContactRecord(
      step=1,
      object_a="target",
      object_b="ghost",
      position=(7.0, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=0.0,
      contact_distance=-8.0,
  )
  tampered = _replace_log(
      counterfactual,
      states=states,
      contacts=counterfactual.contacts + (fake_contact,),
  )

  with pytest.raises(ValueError, match="contact|endpoint|object_ids|unknown"):
    if operation == "extract":
      extract_pair_ground_truth(
          config, intervention, factual, tampered
      )
    else:
      write_paired_artifact(
          tmp_path / "pair",
          config,
          intervention,
          0,
          factual,
          tampered,
      )
  assert not (tmp_path / "pair").exists()


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("step", 1.0, "contact step"),
        ("step", True, "contact step"),
        ("step", 99, "contact step"),
        ("object_b", "ball", "contact endpoints must be distinct"),
    ],
)
def test_pair_validation_rechecks_forged_contact_identity(
    field, invalid_value, message
):
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(1.0, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(0.5, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=0.0,
  )
  object.__setattr__(contact, field, invalid_value)
  tampered = _replace_log(counterfactual)
  object.__setattr__(tampered, "contacts", (contact,))

  with pytest.raises(ValueError, match=message):
    extract_pair_ground_truth(config, intervention, factual, tampered)


def test_pair_validation_rejects_contact_geometry_far_from_both_bodies():
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(1.0, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  fake_contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(15.0, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=-0.1,
  )
  tampered = _replace_log(counterfactual, contacts=(fake_contact,))

  with pytest.raises(ValueError, match="contact|geometry|position|distance"):
    extract_pair_ground_truth(config, intervention, factual, tampered)
  assert tampered.contacts == (fake_contact,)
  assert tampered.contacts[0].contact_distance == -0.1


@pytest.mark.parametrize("contact_x", [1e-16, 1e-7])
def test_contact_geometry_tolerance_preserves_micro_scale(contact_x):
  scale = 1e-16
  config = _scene(
      _object("target", shape="sphere", size=scale, static=True),
      _object("ball", shape="sphere", size=scale, position=(2e-16, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(contact_x, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=0.0,
  )
  clean_factual = _replace_log(factual, contacts=())
  tampered = _replace_log(counterfactual, contacts=(contact,))

  if contact_x == 1e-16:
    extract_pair_ground_truth(
        config, intervention, clean_factual, tampered
    )
  else:
    with pytest.raises(ValueError, match="contact|geometry|position"):
      extract_pair_ground_truth(
          config, intervention, clean_factual, tampered
      )


def test_pair_validation_rejects_unbounded_contact_penetration():
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(8.5, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  fake_contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(0.25, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=-8.0,
  )
  tampered = _replace_log(counterfactual, contacts=(fake_contact,))

  with pytest.raises(ValueError, match="contact|distance|depth|penetration"):
    extract_pair_ground_truth(config, intervention, factual, tampered)


def test_zero_force_contact_cannot_relax_target_pose_binding():
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(0.6, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  target_index = counterfactual.object_ids.index("target")
  states = counterfactual.states.copy()
  states[1, target_index, 0] += 0.1
  fake_contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(0.35, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=0.0,
      contact_distance=-0.1,
  )
  tampered = _replace_log(
      counterfactual, states=states, contacts=(fake_contact,)
  )

  with pytest.raises(ValueError, match="contact|force|positive"):
    extract_pair_ground_truth(config, intervention, factual, tampered)


@pytest.mark.parametrize("mutation", ["position", "quaternion"])
def test_pair_validation_caps_fabricated_contact_pose_envelope(mutation):
  ball_x = 7.4 if mutation == "position" else 0.6
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(ball_x, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  target_index = counterfactual.object_ids.index("target")
  states = counterfactual.states.copy()
  if mutation == "position":
    states[1, target_index, 0] += 7.0
    contact_position = (7.25, 0.0, 0.0)
  else:
    states[1, target_index, 3:7] = (0.0, 1.0, 0.0, 0.0)
    contact_position = (0.25, 0.0, 0.0)
  fake_contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=contact_position,
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=-0.1,
  )
  tampered = _replace_log(
      counterfactual, states=states, contacts=(fake_contact,)
  )

  with pytest.raises(
      ValueError, match="target|pose|position|quaternion|envelope"
  ):
    extract_pair_ground_truth(config, intervention, factual, tampered)


@pytest.mark.parametrize("mutation", ["linear", "angular"])
def test_logged_target_velocity_cannot_expand_contact_pose_envelope(mutation):
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(0.56, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  target_index = counterfactual.object_ids.index("target")
  states = counterfactual.states.copy()
  if mutation == "linear":
    states[1, target_index, 0] += 0.2
    states[1, target_index, 7] = 0.2 * config.step_rate
  else:
    angle = 0.5
    states[1, target_index, 3:7] = (
        np.cos(angle / 2.0),
        np.sin(angle / 2.0),
        0.0,
        0.0,
    )
    states[1, target_index, 10] = angle * config.step_rate
  fake_contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(0.28, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=-0.03,
  )
  tampered = _replace_log(
      counterfactual, states=states, contacts=(fake_contact,)
  )

  with pytest.raises(
      ValueError, match="target|position|quaternion|velocity|envelope"
  ):
    extract_pair_ground_truth(config, intervention, factual, tampered)


def test_huge_finite_logged_velocity_is_rejected_without_overflow_warning():
  config = _scene(
      _object("target", shape="sphere", static=True),
      _object("ball", shape="sphere", position=(0.56, 0.0, 0.0)),
  )
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  target_index = counterfactual.object_ids.index("target")
  states = counterfactual.states.copy()
  states[1, target_index, 0] += 0.2
  states[1, target_index, 7:10] = 1e308
  fake_contact = ContactRecord(
      step=1,
      object_a="ball",
      object_b="target",
      position=(0.28, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=1.0,
      contact_distance=-0.03,
  )
  tampered = _replace_log(
      counterfactual, states=states, contacts=(fake_contact,)
  )

  with pytest.raises(ValueError, match="target|position|envelope"):
    extract_pair_ground_truth(config, intervention, factual, tampered)


def test_writer_rejects_rng_seed_that_did_not_generate_counterfactual(tmp_path):
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 4
  )

  with pytest.raises(ValueError, match="rng_seed|provenance"):
    write_paired_artifact(
        tmp_path / "pair",
        config,
        intervention,
        5,
        factual,
        counterfactual,
    )


def test_swept_sphere_volume_rejects_obstacle_clipping_before_client(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  config = _scene(
      _object("target", shape="sphere", size=0.5, static=True),
      _object("wall", size=(0.6, 2, 2), position=(1.0, 0, 0), static=True),
  )

  with pytest.raises(ValueError, match="volume|obstacle|AABB|intersect"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_swept_cube_uses_quaternion_dependent_extents(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  half_angle = np.pi / 8
  config = _scene(
      ObjectConfig(
          "target",
          "cube",
          size=(0.5, 0.1, 0.1),
          quaternion=(np.cos(half_angle), 0, 0, np.sin(half_angle)),
          static=True,
          friction=0,
          restitution=0,
      ),
      _object("wall", size=(2, 0.15, 2), position=(0, 0.55, 0), static=True),
  )

  with pytest.raises(ValueError, match="volume|obstacle|AABB|intersect"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_swept_cube_covers_intermediate_shortest_arc_rotation(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  config = _scene(
      _object("target", size=1.0, static=True),
      _object(
          "wall",
          size=(0.1, 2.0, 2.0),
          position=(1.3, 0.0, 0.0),
          static=True,
      ),
  )
  path = np.zeros((10, 7), dtype=np.float64)
  path[0, 3] = 1.0
  path[1:, 3] = np.sqrt(0.5)
  path[1:, 6] = np.sqrt(0.5)

  with pytest.raises(ValueError, match="volume|obstacle|AABB|rotation"):
    generate_paired_instance(
        config,
        "target",
        _intervention(magnitude=0),
        0,
        factual_path=path,
    )
  assert constructed == []


def test_geometry_tolerance_does_not_erase_micro_scale_overlap(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  config = _scene(
      _object("target", shape="sphere", size=1e-16, static=True),
      _object("wall", size=1e-16, static=True),
  )

  with pytest.raises(ValueError, match="volume|obstacle|AABB|intersect"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_geometry_tolerance_is_local_to_each_axis(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  config = _scene(
      _object("target", shape="sphere", size=1e-16, static=True),
      _object(
          "wall",
          size=(1e-16, 2.0, 2.0),
          position=(1.9e-16, 0.0, 0.0),
          static=True,
      ),
  )

  with pytest.raises(ValueError, match="volume|obstacle|AABB|intersect"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_sweep_uses_float32_realized_geometry(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  target_size = 0.0940377154715784
  wall_size = 0.14545904936907372
  wall_x = 0.23949676988962068
  assert wall_x > target_size + wall_size
  assert float(np.float32(wall_x)) < (
      float(np.float32(target_size)) + float(np.float32(wall_size))
  )
  config = _scene(
      _object("target", shape="sphere", size=target_size, static=True),
      _object(
          "wall",
          size=(wall_size, 1.0, 1.0),
          position=(wall_x, 0.0, 0.0),
          static=True,
      ),
  )

  with pytest.raises(ValueError, match="volume|obstacle|AABB|intersect"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_swept_volume_rejects_scene_bound_clipping_before_client(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  config = SceneConfig(
      objects=(_object("target", shape="sphere", size=0.5, static=True),),
      scene_bounds=((-1, -1, -1), (0.25, 1, 1)),
      gravity=(0, 0, 0),
      frame_range=(0, 1),
      frame_rate=24,
      step_rate=240,
  )

  with pytest.raises(ValueError, match="volume|bounds"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_swept_volume_allows_exact_static_tangency():
  config = _scene(
      _object("target", shape="sphere", size=0.5, static=True),
      _object("wall", size=(0.6, 2, 2), position=(1.1, 0, 0), static=True),
  )
  factual, counterfactual = generate_paired_instance(
      config, "target", _intervention(magnitude=0), 0
  )
  _assert_physics_equal(factual, counterfactual)


@pytest.mark.parametrize("target_z, accepted", [(0.5, True), (0.49, False)])
def test_floor_support_contact_is_allowed_but_penetration_is_rejected(
    target_z, accepted
):
  config = _scene(
      _object(
          "target",
          shape="sphere",
          size=0.5,
          position=(0, 0, target_z),
          static=True,
      ),
      _object(
          "floor",
          size=(4, 4, 0.25),
          position=(0, 0, -0.25),
          static=True,
          metadata={"qc_clip_exempt": True},
      ),
  )
  if accepted:
    factual, counterfactual = generate_paired_instance(
        config, "target", _intervention(magnitude=0), 0
    )
    _assert_physics_equal(factual, counterfactual)
  else:
    with pytest.raises(ValueError, match="volume|obstacle|AABB|intersect"):
      generate_paired_instance(
          config, "target", _intervention(magnitude=0), 0
      )


def test_floor_tangency_tolerates_sub_ulp_interpolation_drift():
  target_size = 0.3
  config = _scene(
      _object(
          "target",
          shape="sphere",
          size=target_size,
          position=(0, 0, target_size),
          static=True,
      ),
      _object(
          "floor",
          size=(4, 4, 0.25),
          position=(0, 0, -0.25),
          static=True,
          metadata={"qc_clip_exempt": True},
      ),
  )
  path = np.zeros((10, 7), dtype=np.float64)
  path[:, 2] = target_size
  path[:, 3] = 1.0
  path[1:, 2] = np.nextafter(target_size, -np.inf)
  assert np.max(target_size - path[:, 2]) < 1e-15

  factual, counterfactual = generate_paired_instance(
      config,
      "target",
      _intervention(magnitude=0),
      0,
      factual_path=path,
  )
  _assert_physics_equal(factual, counterfactual)


@pytest.mark.parametrize("field", ["position", "size", "linear_velocity", "path"])
def test_float32_preflight_rejects_unrepresentable_values_before_client(
    field, monkeypatch
):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  huge = 1e100
  kwargs = {"static": True}
  bounds = ((-1e101, -1e101, -1e101), (1e101, 1e101, 1e101))
  if field != "path":
    kwargs[field] = (huge, huge, huge) if field != "size" else huge
  config = SceneConfig(
      objects=(_object("target", **kwargs),),
      scene_bounds=bounds,
      gravity=(0, 0, 0),
      frame_range=(0, 1),
      frame_rate=24,
      step_rate=240,
  )
  factual_path = None
  if field == "path":
    factual_path = np.zeros((10, 7), dtype=np.float64)
    factual_path[:, 3] = 1
    factual_path[1:, 0] = huge

  with pytest.raises(ValueError, match="float32|represent"):
    generate_paired_instance(
        config,
        "target",
        _intervention(magnitude=0),
        0,
        factual_path=factual_path,
    )
  assert constructed == []


@pytest.mark.parametrize(
    "case", ["collapsed_positions", "underflow_size", "lossy_velocity", "lossy_path"]
)
def test_float32_preflight_rejects_lossy_geometry_before_client(
    case, monkeypatch
):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  bounds = ((-2e9, -2e9, -2e9), (2e9, 2e9, 2e9))
  factual_path = None
  if case == "collapsed_positions":
    config = SceneConfig(
        objects=(
            _object("target", position=(1e9, 0, 0), static=True),
            _object("marker", position=(1e9 + 20, 0, 0)),
        ),
        scene_bounds=bounds,
        gravity=(0, 0, 0),
        frame_range=(0, 1),
        frame_rate=24,
        step_rate=240,
    )
    assert np.float32(1e9) == np.float32(1e9 + 20)
  elif case == "underflow_size":
    config = _scene(_object("target", size=1e-100, static=True))
    assert np.float32(1e-100) == 0.0
  elif case == "lossy_velocity":
    config = SceneConfig(
        objects=(
            _object(
                "target",
                position=(0, 0, 0),
                linear_velocity=(1e9 + 20, 0, 0),
                static=True,
            ),
        ),
        scene_bounds=bounds,
        gravity=(0, 0, 0),
        frame_range=(0, 1),
        frame_rate=24,
        step_rate=240,
    )
  else:
    config = SceneConfig(
        objects=(_object("target", position=(1e9, 0, 0), static=True),),
        scene_bounds=bounds,
        gravity=(0, 0, 0),
        frame_range=(0, 1),
        frame_rate=24,
        step_rate=240,
    )
    factual_path = np.zeros((10, 7), dtype=np.float64)
    factual_path[:, 0] = 1e9 + 20
    factual_path[0, 0] = 1e9
    factual_path[:, 3] = 1.0

  with pytest.raises(
      ValueError, match="float32|precision|roundtrip|collapse|zero"
  ):
    generate_paired_instance(
        config,
        "target",
        _intervention(magnitude=0),
        0,
        factual_path=factual_path,
    )
  assert constructed == []


def test_kubric_trait_domain_preflight_happens_before_client(monkeypatch):
  import interventions.twin_runner as runner

  constructed = []

  class CountingSimulator(KinematicDragSimulator):
    def __init__(self, *args, **kwargs):
      constructed.append(True)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(runner, "KinematicDragSimulator", CountingSimulator)
  config = _scene(_object("target", static=True, friction=1.5))

  with pytest.raises(ValueError, match="friction|Kubric|domain"):
    generate_paired_instance(config, "target", _intervention(magnitude=0), 0)
  assert constructed == []


def test_wide_margin_contact_versus_miss_extracts_removed_edge_and_path():
  config = SceneConfig(
      objects=(
          _object(
              "target",
              static=True,
              size=0.2,
              position=(-2, 0, 0),
              linear_velocity=(4, 0, 0),
          ),
          _object("ball", shape="sphere", size=0.2, position=(0, 0, 0)),
      ),
      seed=7,
      scene_bounds=((-10, -10, -10), (10, 10, 10)),
      gravity=(0, 0, 0),
      frame_range=(0, 24),
      frame_rate=24,
      step_rate=48,
  )
  intervention = _intervention(magnitude=2.0, time_window=(4, 44))

  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 0
  )
  target_index = factual.object_ids.index("target")
  contact_pose_drift = np.linalg.norm(
      factual.states[:, target_index, :3] - factual.commanded_path[:, :3],
      axis=1,
  )
  assert 0.002 < contact_pose_drift.max() < 0.003
  target_penetration = max(
      -record.contact_distance
      for record in factual.contacts
      if (
          "target" in (record.object_a, record.object_b)
          and record.contact_distance is not None
      )
  )
  assert 0.09 < target_penetration < 0.10
  truth = extract_pair_ground_truth(
      config,
      intervention,
      factual,
      counterfactual,
      force_threshold=1e-6,
  )

  assert any(
      {edge["object_a"], edge["object_b"]} == {"target", "ball"}
      for edge in truth.graph_delta.removed
  )
  assert truth.graph_delta.added == ()
  assert truth.hard_affected == ("ball",)
  assert truth.propagation_path["ball"] == ("target", "ball")

  contact_step = next(
      record.step
      for record in factual.contacts
      if "target" in (record.object_a, record.object_b)
  )
  corrupted_states = factual.states.copy()
  corrupted_states[contact_step, target_index, 0] += 7.0
  with pytest.raises(ValueError, match="target|pose|position|commanded"):
    extract_pair_ground_truth(
        config,
        intervention,
        _replace_log(factual, states=corrupted_states),
        counterfactual,
    )


def test_pair_artifact_roundtrip_and_canonical_provenance(tmp_path):
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 8
  )
  destination = tmp_path / "pair"

  truth = write_paired_artifact(
      destination,
      config,
      intervention,
      8,
      factual,
      counterfactual,
  )

  generation, manifest = _pair_generation(destination)
  assert read_simulation_log(generation / "factual") == factual
  assert read_simulation_log(generation / "counterfactual") == counterfactual
  assert json.loads((generation / "ground_truth.json").read_text()) == truth.to_dict()
  pair = json.loads((generation / "pair.json").read_text())
  assert pair["scene_config"] == config.to_dict()
  assert pair["intervention"] == intervention.to_dict()
  assert pair["target_id"] == "target"
  assert pair["rng_seed"] == 8
  assert pair["schema_version"] == config.schema_version
  assert pair["tags"] == ["null_effect", "target_only"]
  assert pair["extraction_thresholds"] == {
      "force_threshold": 0.0,
      "force_tolerance": 1e-6,
      "min_episode_impulse": 0.0,
      "position_epsilon": 1e-3,
      "quaternion_epsilon": 1e-3,
      "velocity_epsilon": 1e-3,
  }
  assert (generation / "pair.json").read_bytes().endswith(b"\n")
  assert manifest["schema_version"] == config.schema_version
  assert len(manifest["generation"]) == 64
  for relative, record in manifest["files"].items():
    payload = (destination / record["path"]).read_bytes()
    assert record["path"] == "generations/{}/{}".format(
        manifest["generation"], relative
    )
    assert record["size"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
  read_factual, read_counterfactual, read_truth, provenance = (
      read_paired_artifact(destination)
  )
  assert read_factual == factual
  assert read_counterfactual == counterfactual
  assert read_truth == truth
  assert provenance == pair
  with pytest.raises(FileExistsError):
    write_paired_artifact(
        destination, config, intervention, 8, factual, counterfactual
    )


def test_pair_artifact_failure_leaves_no_destination_or_staging(monkeypatch, tmp_path):
  import interventions.twin_runner as runner

  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 1
  )
  real_writer = runner.write_simulation_log
  calls = 0

  def fail_second(log, directory, **kwargs):
    nonlocal calls
    calls += 1
    if calls == 2:
      raise RuntimeError("writer exploded")
    return real_writer(log, directory, **kwargs)

  monkeypatch.setattr(runner, "write_simulation_log", fail_second)
  destination = tmp_path / "pair"

  with pytest.raises(RuntimeError, match="writer exploded"):
    write_paired_artifact(
        destination, config, intervention, 1, factual, counterfactual
    )

  assert not destination.exists()
  assert all("tmp" not in path.name for path in tmp_path.iterdir())


def test_artifact_persists_complete_normalized_thresholds_that_change_labels(
    tmp_path,
):
  config = _scene(
      _object("target", static=True),
      _object("ball", shape="sphere", position=(5, 0, 0)),
  )
  intervention = _intervention(magnitude=0.5)
  factual, counterfactual = generate_paired_instance(
      config, "target", intervention, 7
  )
  ball_index = counterfactual.object_ids.index("ball")
  changed_states = counterfactual.states.copy()
  changed_states[2:, ball_index, 0] += 0.01
  counterfactual = _replace_log(counterfactual, states=changed_states)

  low_truth = write_paired_artifact(
      tmp_path / "low",
      config,
      intervention,
      7,
      factual,
      counterfactual,
      position_epsilon=np.float32(0.001),
  )
  high_truth = write_paired_artifact(
      tmp_path / "high",
      config,
      intervention,
      7,
      factual,
      counterfactual,
      position_epsilon=0.1,
  )

  assert low_truth.soft_affected == ("ball",)
  assert high_truth.soft_affected == ()
  low_generation, _ = _pair_generation(tmp_path / "low")
  high_generation, _ = _pair_generation(tmp_path / "high")
  low_pair = json.loads((low_generation / "pair.json").read_text())
  high_pair = json.loads((high_generation / "pair.json").read_text())
  assert low_pair["extraction_thresholds"]["position_epsilon"] == float(
      np.float32(0.001)
  )
  assert high_pair["extraction_thresholds"]["position_epsilon"] == 0.1
  assert set(low_pair["extraction_thresholds"]) == {
      "force_threshold",
      "min_episode_impulse",
      "force_tolerance",
      "position_epsilon",
      "velocity_epsilon",
      "quaternion_epsilon",
  }


def test_overwrite_failure_keeps_previous_manifest_reader_visible(
    monkeypatch, tmp_path
):
  import interventions.twin_runner as runner

  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  first = generate_paired_instance(config, "target", intervention, 1)
  second = generate_paired_instance(config, "target", intervention, 2)
  destination = tmp_path / "pair"
  write_paired_artifact(destination, config, intervention, 1, *first)
  old_manifest = (destination / "manifest.json").read_bytes()
  old_generation, _ = _pair_generation(destination)
  real_replace = runner.os.replace

  def fail_pointer(source, target):
    if Path(target) == destination / "manifest.json":
      raise OSError("pointer publish exploded")
    return real_replace(source, target)

  monkeypatch.setattr(runner.os, "replace", fail_pointer)
  with pytest.raises(OSError, match="pointer publish exploded"):
    write_paired_artifact(
        destination, config, intervention, 2, *second, overwrite=True
    )

  assert (destination / "manifest.json").read_bytes() == old_manifest
  assert read_simulation_log(old_generation / "factual") == first[0]
  assert read_simulation_log(old_generation / "counterfactual") == first[1]


def test_concurrent_overwrites_publish_one_complete_generation(tmp_path):
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  pairs = {
      seed: generate_paired_instance(config, "target", intervention, seed)
      for seed in (3, 4)
  }
  destination = tmp_path / "pair"

  def publish(seed):
    return write_paired_artifact(
        destination,
        config,
        intervention,
        seed,
        *pairs[seed],
        overwrite=True,
    )

  with ThreadPoolExecutor(max_workers=2) as executor:
    results = tuple(executor.map(publish, (3, 4)))

  assert results[0] == results[1]
  generation, manifest = _pair_generation(destination)
  pair = json.loads((generation / "pair.json").read_text())
  assert pair["rng_seed"] in (3, 4)
  chosen = pairs[pair["rng_seed"]]
  assert read_simulation_log(generation / "factual") == chosen[0]
  assert read_simulation_log(generation / "counterfactual") == chosen[1]
  for record in manifest["files"].values():
    payload = (destination / record["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_pair_publication_fsyncs_generation_and_root(monkeypatch, tmp_path):
  import interventions.twin_runner as runner

  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  pair = generate_paired_instance(config, "target", intervention, 1)
  calls = []
  real_fsync = runner._fsync_directory

  def record(path):
    calls.append(Path(path))
    return real_fsync(path)

  monkeypatch.setattr(runner, "_fsync_directory", record)
  destination = tmp_path / "pair"
  write_paired_artifact(destination, config, intervention, 1, *pair)

  assert any(path.name == "generations" for path in calls)
  assert destination in calls


@pytest.mark.parametrize("symlink_component", ["generations", "generation"])
def test_pair_reader_rejects_symlinked_generation_path_component(
    symlink_component, tmp_path
):
  config = _scene(_object("target", static=True))
  intervention = _intervention(magnitude=0)
  pair = generate_paired_instance(config, "target", intervention, 5)
  destination = tmp_path / "pair"
  write_paired_artifact(destination, config, intervention, 5, *pair)
  generation, _ = _pair_generation(destination)
  outside = tmp_path / "outside"
  outside.mkdir()

  if symlink_component == "generations":
    component = destination / "generations"
    relocated = outside / "generations"
  else:
    component = generation
    relocated = outside / generation.name
  shutil.move(str(component), relocated)
  component.symlink_to(relocated, target_is_directory=True)

  with pytest.raises(ValueError, match="symbolic|symlink|artifact path"):
    read_paired_artifact(destination)
