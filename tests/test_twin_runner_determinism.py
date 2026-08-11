"""Deterministic paired-runner integration and artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pybullet as pb
import pytest

import kubric as kb

from interventions import (
    CameraConfig,
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    KinematicDragSimulator,
    ObjectConfig,
    SceneConfig,
    extract_pair_ground_truth,
    generate_paired_instance,
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

  assert read_simulation_log(destination / "factual") == factual
  assert read_simulation_log(destination / "counterfactual") == counterfactual
  assert json.loads((destination / "ground_truth.json").read_text()) == truth.to_dict()
  pair = json.loads((destination / "pair.json").read_text())
  assert pair["scene_config"] == config.to_dict()
  assert pair["intervention"] == intervention.to_dict()
  assert pair["target_id"] == "target"
  assert pair["rng_seed"] == 8
  assert pair["schema_version"] == config.schema_version
  assert pair["tags"] == ["null_effect", "target_only"]
  assert (destination / "pair.json").read_bytes().endswith(b"\n")
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
  assert list(tmp_path.iterdir()) == []
