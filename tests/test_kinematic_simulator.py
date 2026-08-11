"""Contract and regression tests for the kinematic-emulation simulator."""

from __future__ import annotations

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
  return kb.Cube(
      name=name,
      scale=scale,
      static=static,
      position=position,
      friction=0.0,
      restitution=0.0,
  )


def _sphere(*, name="sphere", scale=1.0, static=False, position=(0, 0, 0)):
  return kb.Sphere(
      name=name,
      scale=scale,
      static=static,
      position=position,
      friction=0.0,
      restitution=0.0,
  )


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

    monkeypatch.setattr(client, "changeDynamics", change)
    monkeypatch.setattr(client, "resetBaseVelocity", velocity)
    monkeypatch.setattr(client, "resetBasePositionAndOrientation", pose)
    monkeypatch.setattr(client, "stepSimulation", step)
    result = simulator.run_with_intervention(target, path, push_mass=3)

  expected = [0.0, 0.25 * scene.step_rate, 0.25 * scene.step_rate]
  step_indices = [index for index, entry in enumerate(calls) if entry[0] == "pre-step"]
  assert len(step_indices) == 3
  start = 0
  for command, step_index, vx in zip(path, step_indices, expected):
    segment = calls[start:step_index + 1]
    assert segment[0] == ("mass", 3.0)
    command_index = next(index for index, entry in enumerate(segment) if entry[0] == "velocity")
    pose_index = next(index for index, entry in enumerate(segment) if entry[0] == "pose")
    assert command_index < pose_index < len(segment) - 1
    measured = segment[-1]
    np.testing.assert_allclose(measured[1], (vx, 0, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured[2], (0, 0, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(measured[3], command[:3], atol=1e-7, rtol=0)
    start = step_index + 1
  assert calls[-1] == ("mass", 0.0)
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
    for index, position in enumerate(((-1 / 240, 0, 0), (0, 0, 0))):
      velocity = (0, 0, 0) if index == 0 else (1, 0, 0)
      pb.changeDynamics(cube, -1, mass=2, physicsClientId=client)
      pb.resetBaseVelocity(cube, velocity, (0, 0, 0), physicsClientId=client)
      pb.resetBasePositionAndOrientation(
          cube, position, (0, 0, 0, 1), physicsClientId=client
      )
      # Raw pose resets zero velocity; this is the behavior Kubric preserves.
      pb.resetBaseVelocity(cube, velocity, (0, 0, 0), physicsClientId=client)
      pb.stepSimulation(physicsClientId=client)
    contacts = pb.getContactPoints(cube, sphere, physicsClientId=client)
    velocity = pb.getBaseVelocity(sphere, physicsClientId=client)[0][0]
    return velocity, contacts[0][9]
  finally:
    pb.disconnect(client)


def test_analytic_collision_momentum_depends_on_temporary_push_mass():
  """Mass ratio 2:1 produces the analytic 2/3 momentum transfer."""
  raw_velocity, raw_force = _raw_analytic_collision()
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
