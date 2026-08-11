"""Contract and regression tests for the kinematic-emulation simulator."""

from __future__ import annotations

import gc

import numpy as np
import pybullet as pb
import pytest

import kubric as kb

from interventions import ContactLogger, KinematicSimulator, SimulationLog


def _make_scene(*, step_rate=240, frame_rate=24, gravity=(0, 0, -9.81)):
  return kb.Scene(
      resolution=(64, 64),
      frame_start=1,
      frame_end=48,
      frame_rate=frame_rate,
      step_rate=step_rate,
      gravity=gravity,
  )


def _cube(*, name="cube", scale=1.0, static=False, position=(0, 0, 0)):
  asset = kb.Cube(
      name=name,
      scale=scale,
      static=static,
      position=position,
      friction=0.0,
      restitution=0.0,
  )
  asset.metadata["logical_id"] = name
  return asset


def _sphere(*, name="sphere", scale=1.0, static=False, position=(0, 0, 0)):
  asset = kb.Sphere(
      name=name,
      scale=scale,
      static=static,
      position=position,
      friction=0.0,
      restitution=0.0,
  )
  asset.metadata["logical_id"] = name
  return asset


def _path(*rows):
  return np.asarray(rows, dtype=np.float64)


def _body_id(simulator, asset):
  return asset.linked_objects[simulator]


def _body_mass(simulator, asset):
  body = _body_id(simulator, asset)
  return simulator.bullet_client.getDynamicsInfo(body, -1)[0]


@pytest.fixture
def scene_and_assets():
  scene = _make_scene(gravity=(0, 0, 0))
  target = _cube(name="target", static=True)
  target.metadata["logical_id"] = "mover"
  other = _sphere(name="other", position=(3, 0, 0))
  other.mass = 1.0
  other.metadata["logical_id"] = "ball"
  scene += target
  scene += other
  return scene, target, other


def test_public_api_and_private_wrapper_timestep(scene_and_assets):
  scene, target, _ = scene_and_assets
  with KinematicSimulator(scene) as simulator:
    assert simulator.is_connected
    assert simulator.bullet_client is not None
    assert simulator.bullet_client.getPhysicsEngineParameters()[
        "fixedTimeStep"
    ] == pytest.approx(1.0 / scene.step_rate, abs=0.0)
    assert target.static is True
  assert simulator.is_connected is False


def test_planned_public_name_is_exported_as_compatibility_identity():
  from interventions import KinematicDragSimulator
  from interventions.kinematic_simulator import (
      KinematicDragSimulator as DirectDragSimulator,
      KinematicSimulator as CompatibilitySimulator,
  )

  assert KinematicDragSimulator is DirectDragSimulator
  assert KinematicDragSimulator is CompatibilitySimulator
  documentation = KinematicDragSimulator.__doc__.lower()
  assert "gravity" in documentation
  assert "next command" in documentation


def test_missing_logical_id_is_rejected_instead_of_using_process_global_uid():
  scene = _make_scene(gravity=(0, 0, 0))
  asset = kb.Cube(name="unstable", static=True)
  scene += asset
  with pytest.raises(ValueError, match="metadata.*logical_id|required"):
    with KinematicSimulator(scene):
      pass


def test_explicit_logical_ids_are_stable_across_scene_reconstruction():
  def run_reconstruction():
    scene = _make_scene(gravity=(0, 0, 0))
    target = _cube(name="reconstructed_target", static=True)
    other = _sphere(name="reconstructed_other", position=(10, 0, 0))
    scene += target
    scene += other
    with KinematicSimulator(scene) as simulator:
      log = simulator.run_with_intervention(
          target,
          _path((0, 0, 0, 1, 0, 0, 0), (0.1, 0, 0, 1, 0, 0, 0)),
          push_mass=1,
      )
    return (target.uid, other.uid), log.object_ids

  first_uids, first_ids = run_reconstruction()
  second_uids, second_ids = run_reconstruction()
  assert first_uids != second_uids
  assert first_ids == second_ids == (
      "reconstructed_other",
      "reconstructed_target",
  )


def test_target_must_be_linked_static_and_logical_ids_unique():
  scene = _make_scene(gravity=(0, 0, 0))
  dynamic = _cube(name="dynamic", static=False)
  scene += dynamic
  with KinematicSimulator(scene) as simulator:
    with pytest.raises(ValueError, match="static"):
      simulator.run_with_intervention(
          dynamic,
          _path((0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0)),
          push_mass=1,
      )

  unlinked = _cube(name="unlinked", static=True)
  with KinematicSimulator(scene) as simulator:
    with pytest.raises(ValueError, match="linked|scene"):
      simulator.run_with_intervention(
          unlinked,
          _path((0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0)),
          push_mass=1,
      )

  duplicate = _cube(name="duplicate", static=True)
  dynamic.metadata["logical_id"] = "same"
  duplicate.metadata["logical_id"] = "same"
  scene += duplicate
  with pytest.raises(ValueError, match="logical.*unique|duplicate"):
    KinematicSimulator(scene)


def test_removed_asset_callbacks_cannot_move_a_reused_body_id():
  scene = _make_scene(gravity=(0, 0, 0))
  removed = _cube(name="removed", static=True)
  scene += removed
  with KinematicSimulator(scene) as simulator:
    removed_body = _body_id(simulator, removed)
    scene.remove(removed)
    replacement = _cube(name="replacement", static=True, position=(1, 2, 3))
    scene += replacement
    replacement_body = _body_id(simulator, replacement)
    assert replacement_body == removed_body
    before = simulator.bullet_client.getBasePositionAndOrientation(
        replacement_body
    )[0]

    removed.position = (90, 80, 70)

    after = simulator.bullet_client.getBasePositionAndOrientation(
        replacement_body
    )[0]
    np.testing.assert_array_equal(after, before)


def test_stale_simulator_close_cannot_touch_reused_client_connection():
  old_scene = _make_scene(gravity=(0, 0, 0))
  old_asset = _cube(name="old_client_asset", static=True)
  old_scene += old_asset
  old_simulator = KinematicSimulator(old_scene)
  old_client_id = old_simulator.bullet_client.client
  old_simulator.bullet_client.disconnect()

  new_scene = _make_scene(gravity=(0, 0, 0))
  new_asset = _cube(name="new_client_asset", static=True, position=(4, 5, 6))
  new_scene += new_asset
  new_simulator = KinematicSimulator(new_scene)
  try:
    assert new_simulator.bullet_client.client == old_client_id
    new_body = _body_id(new_simulator, new_asset)

    old_simulator.close()

    assert new_simulator.is_connected
    assert new_simulator.bullet_client.getNumBodies() >= 1
    position = new_simulator.bullet_client.getBasePositionAndOrientation(new_body)[0]
    np.testing.assert_allclose(position, (4, 5, 6), atol=0, rtol=0)
    new_simulator.step_passive()
    with pytest.raises(RuntimeError, match="ownership|stale|connection"):
      _ = old_simulator.bullet_client
    with pytest.raises(RuntimeError, match="ownership|stale|connection"):
      _ = old_simulator.physics_client
  finally:
    old_simulator.close()
    new_simulator.close()


def test_connection_sentinel_rejects_raw_pybullet_id_reuse():
  scene = _make_scene(gravity=(0, 0, 0))
  asset = _cube(name="raw_reuse_asset", static=True)
  scene += asset
  simulator = KinematicSimulator(scene)
  client_id = simulator.bullet_client.client
  simulator.bullet_client.disconnect()
  replacement_client = pb.connect(pb.DIRECT)
  try:
    assert replacement_client == client_id
    shape = pb.createCollisionShape(
        pb.GEOM_SPHERE, radius=0.25, physicsClientId=replacement_client
    )
    body = pb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shape,
        physicsClientId=replacement_client,
    )
    with pytest.raises(RuntimeError, match="ownership|stale|connection"):
      _ = simulator.bullet_client

    simulator.close()

    assert pb.isConnected(replacement_client)
    assert pb.getBodyUniqueId(0, physicsClientId=replacement_client) == body
  finally:
    simulator.close()
    if pb.isConnected(replacement_client):
      pb.disconnect(replacement_client)


def test_failed_base_constructor_disconnects_partial_client():
  reusable_client = pb.connect(pb.DIRECT)
  pb.disconnect(reusable_client)
  scene = _make_scene(gravity=(0, 0, 0))
  missing = kb.FileBasedObject(
      asset_id="missing",
      simulation_filename="/definitely/missing/kinematic-object.urdf",
  )
  missing.metadata["logical_id"] = "missing"
  scene += missing

  with pytest.raises(OSError, match="does not exist"):
    KinematicSimulator(scene)
  gc.collect()

  replacement = pb.connect(pb.DIRECT)
  try:
    assert replacement == reusable_client
  finally:
    pb.disconnect(replacement)


def test_public_reset_simulation_is_rejected_without_losing_ownership():
  scene = _make_scene(gravity=(0, 0, 0))
  asset = _cube(name="reset_guard", static=True)
  scene += asset
  with KinematicSimulator(scene) as simulator:
    body = _body_id(simulator, asset)
    with pytest.raises(RuntimeError, match="resetSimulation|reset"):
      simulator.bullet_client.resetSimulation()
    assert simulator.is_connected
    assert simulator.bullet_client.getBodyInfo(body)


@pytest.mark.parametrize("push_mass", [0, -1, np.nan, np.inf])
def test_push_mass_must_be_positive_and_finite(scene_and_assets, push_mass):
  scene, target, _ = scene_and_assets
  with KinematicSimulator(scene) as simulator:
    with pytest.raises(ValueError, match="push_mass"):
      simulator.run_with_intervention(
          target,
          _path((0, 0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0)),
          push_mass=push_mass,
      )


@pytest.mark.parametrize(
    "bad_path",
    [
        np.zeros((1, 7)),
        np.zeros((2, 6)),
        np.zeros((2, 8)),
        np.zeros(7),
        np.asarray([[0, 0, 0, 1, 0, 0, 0], [np.nan, 0, 0, 1, 0, 0, 0]]),
        np.asarray([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0]]),
    ],
)
def test_path_validation(scene_and_assets, bad_path):
  scene, target, _ = scene_and_assets
  with KinematicSimulator(scene) as simulator:
    with pytest.raises(ValueError, match="path|quaternion"):
      simulator.run_with_intervention(target, bad_path, push_mass=1)


@pytest.mark.parametrize("large_value", [1e100, -1e100])
def test_unrepresentable_path_rejected_before_any_mutation(
    scene_and_assets, large_value
):
  scene, target, _ = scene_and_assets
  path = _path(
      (large_value, 0, 0, 1, 0, 0, 0),
      (-large_value, 0, 0, 1, 0, 0, 0),
  )
  original_asset_position = target.position.copy()
  original_metadata = dict(target.metadata)
  with KinematicSimulator(scene) as simulator:
    body = _body_id(simulator, target)
    original_body_pose = simulator.bullet_client.getBasePositionAndOrientation(body)
    with pytest.raises(ValueError, match="float32|represent|path|velocity"):
      simulator.run_with_intervention(target, path, push_mass=2)
    np.testing.assert_array_equal(target.position, original_asset_position)
    assert dict(target.metadata) == original_metadata
    assert simulator.bullet_client.getBasePositionAndOrientation(
        body
    ) == original_body_pose
    assert _body_mass(simulator, target) == 0


def test_path_is_not_mutated_and_metadata_is_deterministic(scene_and_assets):
  scene, target, _ = scene_and_assets
  path = _path(
      (0, 0, 0, 1, 0, 0, 0),
      (0.1, 0, 0, 1, 0, 0, 0),
      (0.2, 0, 0, 1, 0, 0, 0),
  )
  original = path.copy()
  with KinematicSimulator(scene) as simulator:
    result = simulator.run_with_intervention(
        target, path, push_mass=2.5, branch="counterfactual", start_step=7
    )
  np.testing.assert_array_equal(path, original)
  assert isinstance(result, SimulationLog)
  assert result.states.shape == (3, 2, 13)
  assert result.metadata == {
      "target_id": "mover",
      "kinematic_emulation": True,
      "push_mass": 2.5,
      "dt": 1.0 / scene.step_rate,
      "velocity_estimator": "backward_difference",
  }
  assert target.metadata["kinematic_emulation"] is True
  assert result.branch == "counterfactual"
  assert result.steps == (7, 8, 9)


def test_velocity_is_backward_difference_times_step_rate_and_call_order(
    scene_and_assets, monkeypatch
):
  scene, target, _ = scene_and_assets
  path = _path(
      (0, 0, 0, 1, 0, 0, 0),
      (0.25, 0, 0, 1, 0, 0, 0),
      (0.50, 0, 0, 1, 0, 0, 0),
  )
  with KinematicSimulator(scene) as simulator:
    body = _body_id(simulator, target)
    client = simulator.bullet_client
    calls = []
    original_change = client.changeDynamics
    original_velocity = client.resetBaseVelocity
    original_pose = client.resetBasePositionAndOrientation
    original_step = client.stepSimulation
    original_contacts = client.getContactPoints
    original_snapshot = simulator._snapshot

    def change(*args, **kwargs):
      calls.append(("mass", kwargs.get("mass")))
      return original_change(*args, **kwargs)

    def velocity(*args, **kwargs):
      calls.append(
          (
              "velocity",
              tuple(kwargs.get("linearVelocity", ())),
              tuple(kwargs.get("angularVelocity", ())),
          )
      )
      return original_velocity(*args, **kwargs)

    def pose(*args, **kwargs):
      calls.append(("pose",))
      return original_pose(*args, **kwargs)

    def step(*args, **kwargs):
      measured = client.getBaseVelocity(body)
      position = client.getBasePositionAndOrientation(body)[0]
      calls.append(
          ("pre-step", tuple(measured[0]), tuple(measured[1]), tuple(position))
      )
      result = original_step(*args, **kwargs)
      # Emulate within-step solver drift so the snapshot/correction contract is
      # observable even though Bullet does not integrate a mass-toggled fixed base.
      drifted, quaternion = client.getBasePositionAndOrientation(body)
      original_pose(
          body,
          (drifted[0] + 0.01, drifted[1], drifted[2]),
          quaternion,
      )
      return result

    def contacts(*args, **kwargs):
      calls.append(("contacts",))
      return original_contacts(*args, **kwargs)

    def snapshot(*args, **kwargs):
      calls.append(("state",))
      return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(client, "changeDynamics", change)
    monkeypatch.setattr(client, "resetBaseVelocity", velocity)
    monkeypatch.setattr(client, "resetBasePositionAndOrientation", pose)
    monkeypatch.setattr(client, "stepSimulation", step)
    monkeypatch.setattr(client, "getContactPoints", contacts)
    monkeypatch.setattr(simulator, "_snapshot", snapshot)
    result = simulator.run_with_intervention(target, path, push_mass=3)

  expected = [0.0, 0.25 * scene.step_rate, 0.25 * scene.step_rate]
  step_indices = [index for index, entry in enumerate(calls) if entry[0] == "pre-step"]
  push_indices = [
      index for index, entry in enumerate(calls) if entry == ("mass", 3.0)
  ]
  zero_indices = [
      index for index, entry in enumerate(calls) if entry == ("mass", 0.0)
  ]
  assert len(step_indices) == 3
  assert len(push_indices) == len(zero_indices) == len(path)
  assert [entry for entry in calls if entry[0] == "mass"] == [
      ("mass", 3.0),
      ("mass", 0.0),
  ] * len(path)
  for command, push_index, step_index, zero_index, vx in zip(
      path, push_indices, step_indices, zero_indices, expected
  ):
    assert push_index < step_index < zero_index
    segment = calls[push_index:step_index + 1]
    assert segment[0] == ("mass", 3.0)
    command_index = next(index for index, entry in enumerate(segment) if entry[0] == "velocity")
    pose_index = next(index for index, entry in enumerate(segment) if entry[0] == "pose")
    assert command_index < pose_index < len(segment) - 1
    measured = segment[-1]
    np.testing.assert_allclose(measured[1], (vx, 0, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured[2], (0, 0, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured[3], command[:3], atol=1e-7, rtol=0)
    assert calls[step_index + 1:zero_index] == [("contacts",), ("state",)]
  target_index = result.object_ids.index("mover")
  np.testing.assert_allclose(
      result.states[:, target_index, 0], path[:, 0] + 0.01, atol=1e-7, rtol=0
  )
  assert target.static is True


def test_mass_restored_on_success_and_exception(scene_and_assets, monkeypatch):
  scene, target, _ = scene_and_assets
  path = _path((0, 0, 0, 1, 0, 0, 0), (0.1, 0, 0, 1, 0, 0, 0))
  with KinematicSimulator(scene) as simulator:
    simulator.run_with_intervention(target, path, push_mass=4)
    assert _body_mass(simulator, target) == 0
    assert target.static is True

    client = simulator.bullet_client
    original_step = client.stepSimulation

    def fail_step(*args, **kwargs):
      original_step(*args, **kwargs)
      raise RuntimeError("step exploded")

    monkeypatch.setattr(client, "stepSimulation", fail_step)
    with pytest.raises(RuntimeError, match="step exploded"):
      simulator.run_with_intervention(target, path, push_mass=5)
    assert _body_mass(simulator, target) == 0
    assert target.static is True


def test_contacts_are_current_run_only_and_states_are_after_step(scene_and_assets):
  scene, target, other = scene_and_assets
  target.position = (-2, 0, 0)
  other.position = (0, 0, 0)
  path = _path(
      (-2, 0, 0, 1, 0, 0, 0),
      (-1, 0, 0, 1, 0, 0, 0),
      (0, 0, 0, 1, 0, 0, 0),
  )
  with KinematicSimulator(scene) as simulator:
    names = {
        _body_id(simulator, target): "mover",
        _body_id(simulator, other): "ball",
    }
    logger = ContactLogger(names, scene.step_rate)
    first = simulator.run_with_intervention(
        target, path, logger, push_mass=1, branch="a", start_step=10
    )
    second = simulator.run_with_intervention(
        target,
        _path(
            (100, 0, 0, 1, 0, 0, 0),
            (100.1, 0, 0, 1, 0, 0, 0),
        ),
        logger,
        push_mass=1,
        branch="b",
        start_step=20,
    )
  assert first.states.shape == (3, 2, 13)
  assert second.states.shape == (2, 2, 13)
  assert first.contacts is not second.contacts
  assert first.contacts
  assert second.contacts == ()
  assert logger.records == second.contacts
  # Snapshot is after integration: at collision, a dynamic object has moved.
  other_id = str(other.metadata["logical_id"])
  assert other_id in first.object_ids
  other_state = first.states[:, first.object_ids.index(other_id)]
  assert other_state[-1, 0] != pytest.approx(0.0)


def test_keyframes_are_optional_and_use_absolute_step_cadence(scene_and_assets):
  scene, target, _ = scene_and_assets
  scene.step_rate = 120
  scene.frame_rate = 24  # cadence == 5
  path = _path(*[(i / 10, 0, 0, 1, 0, 0, 0) for i in range(8)])
  with KinematicSimulator(scene) as simulator:
    simulator.run_with_intervention(
        target, path, push_mass=1, start_step=3, write_keyframes=False
    )
    assert dict(target.keyframes) == {}
    simulator.run_with_intervention(
        target, path, push_mass=1, start_step=3, write_keyframes=True
    )
  expected_frames = {scene.frame_start + 5 // 5, scene.frame_start + 10 // 5}
  assert set(target.keyframes) == {
      "position", "quaternion", "velocity", "angular_velocity"
  }
  assert all(set(frames) == expected_frames for frames in target.keyframes.values())


def test_step_rate_changes_update_timestep_and_run_uses_one_snapshot(monkeypatch):
  scene = _make_scene(step_rate=240, gravity=(0, 0, 0))
  target = _cube(name="rate_target", static=True)
  scene += target
  with KinematicSimulator(scene) as simulator:
    scene.step_rate = 120
    assert simulator.bullet_client.getPhysicsEngineParameters()[
        "fixedTimeStep"
    ] == pytest.approx(1 / 120, abs=0, rel=0)
    body = _body_id(simulator, target)
    measured = []
    original_step = simulator.bullet_client.stepSimulation

    def capture_velocity(*args, **kwargs):
      measured.append(simulator.bullet_client.getBaseVelocity(body)[0])
      return original_step(*args, **kwargs)

    monkeypatch.setattr(simulator.bullet_client, "stepSimulation", capture_velocity)
    log = simulator.run_with_intervention(
        target,
        _path((0, 0, 0, 1, 0, 0, 0), (0.1, 0, 0, 1, 0, 0, 0)),
        push_mass=1,
    )
  np.testing.assert_allclose(measured, ((0, 0, 0), (12, 0, 0)), atol=1e-12)
  assert log.step_rate == 120
  assert log.metadata["dt"] == pytest.approx(1 / 120, abs=0, rel=0)


def test_mid_run_step_rate_change_is_rejected_and_restores_mass(monkeypatch):
  scene = _make_scene(step_rate=240, gravity=(0, 0, 0))
  target = _cube(name="changing_rate_target", static=True)
  scene += target
  with KinematicSimulator(scene) as simulator:
    original_step = simulator.bullet_client.stepSimulation
    changed = False

    def change_rate(*args, **kwargs):
      nonlocal changed
      result = original_step(*args, **kwargs)
      if not changed:
        changed = True
        scene.step_rate = 120
      return result

    monkeypatch.setattr(simulator.bullet_client, "stepSimulation", change_rate)
    with pytest.raises(RuntimeError, match="step_rate.*changed|changed.*step_rate"):
      simulator.run_with_intervention(
          target,
          _path(
              (0, 0, 0, 1, 0, 0, 0),
              (1 / 240, 0, 0, 1, 0, 0, 0),
              (2 / 240, 0, 0, 1, 0, 0, 0),
          ),
          push_mass=2,
      )
    assert _body_mass(simulator, target) == 0


def test_keyframe_publication_is_observational_for_log_and_bullet_state():
  def simulate(write_keyframes):
    scene = _make_scene(step_rate=120, gravity=(0, 0, 0))
    target = _cube(
        name="keyframe_target", static=True, position=(-100, 0, 0)
    )
    moving = _sphere(name="keyframe_moving", position=(0.1234567, 0, 0))
    moving.mass = 1
    moving.velocity = (0.9876543, 0.1234567, 0)
    scene += target
    scene += moving
    path = _path(
        *[(-100 + index / 100, 0, 0, 1, 0, 0, 0) for index in range(13)]
    )
    with KinematicSimulator(scene) as simulator:
      log = simulator.run_with_intervention(
          target,
          path,
          push_mass=1,
          start_step=1,
          write_keyframes=write_keyframes,
      )
      final_state = simulator._snapshot(simulator._physical_assets())
    return log, final_state, target, moving

  plain_log, plain_final, plain_target, plain_moving = simulate(False)
  keyed_log, keyed_final, keyed_target, keyed_moving = simulate(True)
  np.testing.assert_array_equal(keyed_log.states, plain_log.states)
  assert keyed_log.contacts == plain_log.contacts
  np.testing.assert_array_equal(keyed_final, plain_final)
  assert dict(plain_target.keyframes) == {}
  assert dict(plain_moving.keyframes) == {}
  expected_frames = {
      1 + 5 // 5,
      1 + 10 // 5,
  }
  for asset in (keyed_target, keyed_moving):
    assert set(asset.keyframes) == {
        "position", "quaternion", "velocity", "angular_velocity"
    }
    assert all(set(values) == expected_frames for values in asset.keyframes.values())


def test_remove_during_keyframe_publication_does_not_restore_stale_callbacks():
  scene = _make_scene(step_rate=120, gravity=(0, 0, 0))
  target = _cube(name="keyframe_remove_target", static=True, position=(-10, 0, 0))
  removed = _sphere(name="keyframe_removed", position=(0, 0, 0))
  scene += target
  scene += removed
  path = _path(*[(-10 + index / 100, 0, 0, 1, 0, 0, 0) for index in range(7)])
  removed_once = False

  def remove_on_keyframe(change):
    nonlocal removed_once
    if not removed_once:
      removed_once = True
      scene.remove(change.owner)

  removed.observe(remove_on_keyframe, names="position", type="keyframe")
  with KinematicSimulator(scene) as simulator:
    removed_body = _body_id(simulator, removed)
    simulator.run_with_intervention(
        target, path, push_mass=1, write_keyframes=True
    )
    assert removed not in scene.assets
    replacement = _sphere(
        name="keyframe_replacement", position=(1, 2, 3), static=True
    )
    scene += replacement
    replacement_body = _body_id(simulator, replacement)
    assert replacement_body == removed_body
    before = simulator.bullet_client.getBasePositionAndOrientation(
        replacement_body
    )[0]

    removed.position = (9, 8, 7)

    after = simulator.bullet_client.getBasePositionAndOrientation(
        replacement_body
    )[0]
    np.testing.assert_array_equal(after, before)


def test_close_during_keyframe_publication_does_not_restore_callbacks():
  scene = _make_scene(step_rate=120, gravity=(0, 0, 0))
  target = _cube(name="keyframe_close_target", static=True)
  scene += target
  simulator = KinematicSimulator(scene)

  def close_on_keyframe(change):
    del change
    simulator.close()

  target.observe(close_on_keyframe, names="position", type="keyframe")
  simulator.run_with_intervention(
      target,
      _path(*[(index / 100, 0, 0, 1, 0, 0, 0) for index in range(7)]),
      push_mass=1,
      write_keyframes=True,
  )

  assert not simulator.is_connected
  assert simulator not in target.linked_objects
  before = tuple(target.position)
  target.position = (7, 8, 9)
  assert tuple(target.position) != before


def test_checkpoint_restore_remove_and_context_cleanup(scene_and_assets):
  scene, target, _ = scene_and_assets
  simulator = KinematicSimulator(scene)
  checkpoint = simulator.save_checkpoint()
  target.position = (3, 2, 1)
  simulator.bullet_client.resetBasePositionAndOrientation(
      _body_id(simulator, target), (3, 2, 1), (0, 0, 0, 1)
  )
  simulator.restore_checkpoint(checkpoint)
  restored, _ = simulator.bullet_client.getBasePositionAndOrientation(
      _body_id(simulator, target)
  )
  np.testing.assert_allclose(restored, (0, 0, 0), atol=0, rtol=0)
  simulator.remove_checkpoint(checkpoint)
  with pytest.raises((ValueError, RuntimeError, pb.error)):
    simulator.restore_checkpoint(checkpoint)

  with simulator.checkpoint() as scoped_checkpoint:
    simulator.bullet_client.resetBasePositionAndOrientation(
        _body_id(simulator, target), (9, 8, 7), (0, 0, 0, 1)
    )
  restored, _ = simulator.bullet_client.getBasePositionAndOrientation(
      _body_id(simulator, target)
  )
  np.testing.assert_allclose(restored, (0, 0, 0), atol=0, rtol=0)
  with pytest.raises(ValueError, match="checkpoint"):
    simulator.restore_checkpoint(scoped_checkpoint)
  simulator.close()
  simulator.close()
  assert simulator.is_connected is False
  with pytest.raises(RuntimeError, match="closed|connect"):
    simulator.step_passive()
  with pytest.raises(RuntimeError, match="closed|connect"):
    simulator.save_checkpoint()

  for _ in range(3):
    with KinematicSimulator(scene) as scoped:
      assert scoped.is_connected
    assert scoped.is_connected is False


def test_step_passive_advances_one_physics_step():
  scene = _make_scene(gravity=(0, 0, 0))
  sphere = _sphere(position=(0, 0, 0))
  sphere.mass = 1
  sphere.velocity = (1, 0, 0)
  scene += sphere
  with KinematicSimulator(scene) as simulator:
    before = simulator.bullet_client.getBasePositionAndOrientation(
        _body_id(simulator, sphere)
    )[0]
    simulator.step_passive()
    after = simulator.bullet_client.getBasePositionAndOrientation(
        _body_id(simulator, sphere)
    )[0]
  assert after[0] - before[0] == pytest.approx(1 / scene.step_rate, rel=1e-3)


def _configure_raw_world(*, gravity):
  client = pb.connect(pb.DIRECT)
  pb.setPhysicsEngineParameter(
      restitutionVelocityThreshold=0.0,
      warmStartingFactor=0.0,
      useSplitImpulse=True,
      contactSlop=0.0,
      enableConeFriction=False,
      deterministicOverlappingPairs=True,
      physicsClientId=client,
  )
  pb.setTimeStep(1 / 240, physicsClientId=client)
  pb.setGravity(*gravity, physicsClientId=client)
  return client


def _raw_body(client, shape, mass, position):
  body = pb.createMultiBody(
      baseMass=mass,
      baseCollisionShapeIndex=shape,
      basePosition=position,
      useMaximalCoordinates=True,
      physicsClientId=client,
  )
  pb.changeDynamics(
      body,
      -1,
      lateralFriction=0.0,
      restitution=0.0,
      contactProcessingThreshold=0.0,
      physicsClientId=client,
  )
  return body


def _raw_analytic_collision():
  client = _configure_raw_world(gravity=(0, 0, 0))
  try:
    cube_shape = pb.createCollisionShape(
        pb.GEOM_BOX, halfExtents=(0.5, 0.5, 0.5), physicsClientId=client
    )
    sphere_shape = pb.createCollisionShape(
        pb.GEOM_SPHERE, radius=0.5, physicsClientId=client
    )
    cube = _raw_body(client, cube_shape, 0, (-1 / 240, 0, 0))
    sphere = _raw_body(client, sphere_shape, 1, (1, 0, 0))
    mass_trace = []
    force = 0.0
    for index, position in enumerate(((-1 / 240, 0, 0), (0, 0, 0))):
      velocity = (0, 0, 0) if index == 0 else (1, 0, 0)
      pb.changeDynamics(cube, -1, mass=2, physicsClientId=client)
      mass_trace.append(2.0)
      pb.resetBaseVelocity(cube, velocity, (0, 0, 0), physicsClientId=client)
      pb.resetBasePositionAndOrientation(
          cube, position, (0, 0, 0, 1), physicsClientId=client
      )
      # Raw pose resets zero velocity; this is the behavior Kubric preserves.
      pb.resetBaseVelocity(cube, velocity, (0, 0, 0), physicsClientId=client)
      pb.stepSimulation(physicsClientId=client)
      contacts = pb.getContactPoints(cube, sphere, physicsClientId=client)
      if contacts:
        force = contacts[0][9]
      pb.changeDynamics(cube, -1, mass=0, physicsClientId=client)
      mass_trace.append(0.0)
    velocity = pb.getBaseVelocity(sphere, physicsClientId=client)[0][0]
    return velocity, force, mass_trace
  finally:
    pb.disconnect(client)


def test_analytic_collision_momentum_depends_on_temporary_push_mass():
  """Mass ratio 2:1 produces the analytic 2/3 momentum transfer."""
  raw_velocity, raw_force, raw_mass_trace = _raw_analytic_collision()
  assert raw_mass_trace == [2.0, 0.0, 2.0, 0.0]
  assert raw_velocity == pytest.approx(2 / 3, abs=1e-15, rel=0)
  assert raw_force == pytest.approx(160.0, abs=1e-12, rel=0)

  scene = _make_scene(step_rate=240, gravity=(0, 0, 0))
  cube = _cube(
      name="pusher", scale=0.5, static=True, position=(-1 / 240, 0, 0)
  )
  sphere = _sphere(name="sphere", scale=0.5, position=(1, 0, 0))
  sphere.mass = 1.0
  cube.metadata["logical_id"] = "pusher"
  sphere.metadata["logical_id"] = "sphere"
  scene += cube
  scene += sphere
  path = _path(
      (-1 / 240, 0, 0, 1, 0, 0, 0),
      (0, 0, 0, 1, 0, 0, 0),
  )
  with KinematicSimulator(scene) as simulator:
    log = simulator.run_with_intervention(cube, path, push_mass=2)
    velocity = simulator.bullet_client.getBaseVelocity(_body_id(simulator, sphere))[0][0]
    assert _body_mass(simulator, cube) == 0
  assert velocity == pytest.approx(raw_velocity, abs=1e-6, rel=0)
  assert sphere.mass * velocity == pytest.approx(2 / 3, abs=1e-6, rel=0)
  assert log.contacts[-1].normal_force == pytest.approx(raw_force, abs=1e-4, rel=0)


def _run_raw_floor_sweep(*, reapply_velocity):
  client = _configure_raw_world(gravity=(0, 0, -9.81))
  try:
    floor_shape = pb.createCollisionShape(
        pb.GEOM_BOX, halfExtents=(4, 4, 0.25), physicsClientId=client
    )
    sphere_shape = pb.createCollisionShape(
        pb.GEOM_SPHERE, radius=0.5, physicsClientId=client
    )
    cube_shape = pb.createCollisionShape(
        pb.GEOM_BOX, halfExtents=(0.5, 0.5, 0.5), physicsClientId=client
    )
    floor = _raw_body(client, floor_shape, 0, (0, 0, -0.25))
    sphere = _raw_body(client, sphere_shape, 1, (0, 0, 0.5))
    cube = _raw_body(client, cube_shape, 0, (-1 - 1 / 240, 0, 0.5))
    del floor
    for _ in range(240):
      pb.stepSimulation(physicsClientId=client)
    force = 0.0
    after_contact = None
    for index, position in enumerate(((-1 - 1 / 240, 0, 0.5), (-1, 0, 0.5))):
      velocity = (0, 0, 0) if index == 0 else (1, 0, 0)
      pb.changeDynamics(cube, -1, mass=1, physicsClientId=client)
      pb.resetBaseVelocity(cube, velocity, (0, 0, 0), physicsClientId=client)
      pb.resetBasePositionAndOrientation(
          cube, position, (0, 0, 0, 1), physicsClientId=client
      )
      if reapply_velocity:
        pb.resetBaseVelocity(cube, velocity, (0, 0, 0), physicsClientId=client)
      pb.stepSimulation(physicsClientId=client)
      if index == 1:
        contacts = pb.getContactPoints(cube, sphere, physicsClientId=client)
        force = contacts[0][9] if contacts else 0.0
        after_contact = (
            pb.getBasePositionAndOrientation(sphere, physicsClientId=client)[0],
            pb.getBaseVelocity(sphere, physicsClientId=client)[0],
        )
      pb.changeDynamics(cube, -1, mass=0, physicsClientId=client)
    pb.resetBasePositionAndOrientation(
        cube, (-3, 0, 0.5), (0, 0, 0, 1), physicsClientId=client
    )
    for _ in range(23):
      pb.stepSimulation(physicsClientId=client)
    final = (
        pb.getBasePositionAndOrientation(sphere, physicsClientId=client)[0],
        pb.getBaseVelocity(sphere, physicsClientId=client)[0],
    )
    return tuple(np.asarray(value) for value in after_contact), tuple(
        np.asarray(value) for value in final
    ), force
  finally:
    pb.disconnect(client)


def test_floor_sweep_matches_raw_bullet_and_velocity_control_differs():
  raw_contact, raw_final, raw_force = _run_raw_floor_sweep(reapply_velocity=True)
  control_contact, control_final, _ = _run_raw_floor_sweep(reapply_velocity=False)
  assert raw_contact[0][0] == pytest.approx(1 / 480, abs=1e-12, rel=0)
  assert raw_contact[1][0] == pytest.approx(0.5, abs=1e-12, rel=0)
  assert raw_force == pytest.approx(120.0, abs=1e-10, rel=0)
  assert raw_final[0][0] == pytest.approx(0.05, abs=1e-12, rel=0)
  assert raw_final[1][0] == pytest.approx(0.5, abs=1e-12, rel=0)
  assert raw_final[0][0] - control_final[0][0] > 0.049
  assert raw_final[1][0] - control_final[1][0] > 0.49
  assert control_contact[1][0] == pytest.approx(0.0, abs=1e-15, rel=0)

  scene = _make_scene(step_rate=240)
  floor = kb.Cube(
      name="floor",
      scale=(4, 4, 0.25),
      static=True,
      position=(0, 0, -0.25),
      friction=0,
      restitution=0,
  )
  sphere = kb.Sphere(
      name="sphere",
      scale=0.5,
      position=(0, 0, 0.5),
      mass=1,
      friction=0,
      restitution=0,
  )
  cube = kb.Cube(
      name="pusher",
      scale=0.5,
      static=True,
      position=(-1 - 1 / 240, 0, 0.5),
      friction=0,
      restitution=0,
  )
  floor.metadata["logical_id"] = "floor"
  sphere.metadata["logical_id"] = "sphere"
  cube.metadata["logical_id"] = "pusher"
  scene += floor
  scene += sphere
  scene += cube
  path = _path(
      (-1 - 1 / 240, 0, 0.5, 1, 0, 0, 0),
      (-1, 0, 0.5, 1, 0, 0, 0),
  )
  with KinematicSimulator(scene) as simulator:
    for _ in range(240):
      simulator.step_passive()
    log = simulator.run_with_intervention(cube, path, push_mass=1)
    sphere_index = log.object_ids.index("sphere")
    wrapper_contact = (
        log.states[-1, sphere_index, 0:3],
        log.states[-1, sphere_index, 7:10],
    )
    cube.position = (-3, 0, 0.5)
    for _ in range(23):
      simulator.step_passive()
    wrapper_final = (
        np.asarray(
            simulator.bullet_client.getBasePositionAndOrientation(
                _body_id(simulator, sphere)
            )[0]
        ),
        np.asarray(
            simulator.bullet_client.getBaseVelocity(_body_id(simulator, sphere))[0]
        ),
    )
  np.testing.assert_allclose(wrapper_contact[0][[0]], raw_contact[0][[0]], atol=1e-6, rtol=0)
  np.testing.assert_allclose(wrapper_contact[1][[0]], raw_contact[1][[0]], atol=1e-6, rtol=0)
  np.testing.assert_allclose(wrapper_final[0][[0]], raw_final[0][[0]], atol=1e-6, rtol=0)
  np.testing.assert_allclose(wrapper_final[1][[0]], raw_final[1][[0]], atol=1e-6, rtol=0)
  assert log.contacts[-1].normal_force == pytest.approx(raw_force, abs=1e-4, rel=0)
