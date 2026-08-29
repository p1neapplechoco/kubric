# Copyright 2026 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing for `kubric.simulator.pybullet` module."""

import kubric as kb
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import numpy as np
import pytest


def test_basic_simulator():
  scene = kb.Scene(
      gravity=(0, -10, 0),  # A planet slightly larger than Earth.
      frame_end=24,  # One second.
  )
  simulator = KubricSimulator(scene)
  cube = kb.Cube(
      name='box',
      position=[0, 0, 0],
  )
  scene.add(cube)
  simulator.run()
  np.testing.assert_allclose(cube.position[1], -0.5 * 10, atol=0.1)


def test_simulator_in_loop():
  # https://github.com/google-research/kubric/issues/208
  # https://github.com/google-research/kubric/issues/234
  for _ in range(10):
    scene = kb.Scene(
        gravity=(0, -10, 0),  # A planet slightly larger than Earth.
        frame_end=24,  # One second.
    )
    simulator = KubricSimulator(scene)
    cube = kb.Cube(
        name='box',
        position=[0, 0, 0],
    )
    scene.add(cube)
    simulator.run()
    np.testing.assert_allclose(cube.position[1], -0.5 * 10, atol=0.1)


def test_simulator_adds_cylinder_with_expected_extents():
  scene = kb.Scene(frame_start=0, frame_end=1)
  simulator = KubricSimulator(scene)
  cylinder = kb.Cylinder(scale=(0.25, 0.25, 0.75), position=(0, 0, 5),
                         static=False)
  scene.add(cylinder)

  body_id = cylinder.linked_objects[simulator]
  aabb_min, aabb_max = simulator._physics_client.getAABB(body_id)
  assert aabb_max[2] - aabb_min[2] == pytest.approx(1.5, abs=1e-2)
  assert aabb_max[0] - aabb_min[0] == pytest.approx(0.5, abs=1e-2)


def test_simulator_adds_capsule_with_cap_extents():
  scene = kb.Scene(frame_start=0, frame_end=1)
  simulator = KubricSimulator(scene)
  # height 2 * 0.5 for the cylindrical section, plus a 0.25 radius cap at each end.
  capsule = kb.Capsule(scale=(0.25, 0.25, 0.5), position=(0, 0, 5), static=False)
  scene.add(capsule)

  body_id = capsule.linked_objects[simulator]
  aabb_min, aabb_max = simulator._physics_client.getAABB(body_id)
  assert aabb_max[2] - aabb_min[2] == pytest.approx(1.5, abs=1e-2)
  assert aabb_max[0] - aabb_min[0] == pytest.approx(0.5, abs=1e-2)


def test_simulator_cylinder_falls_under_gravity():
  scene = kb.Scene(gravity=(0, 0, -10), frame_start=0, frame_end=24)
  simulator = KubricSimulator(scene)
  cylinder = kb.Cylinder(scale=(0.25, 0.25, 0.25), position=(0, 0, 0), mass=1.0)
  scene.add(cylinder)

  simulator.run()
  np.testing.assert_allclose(cylinder.position[2], -0.5 * 10, atol=0.1)


def test_simulator_capsule_falls_under_gravity():
  scene = kb.Scene(gravity=(0, 0, -10), frame_start=0, frame_end=24)
  simulator = KubricSimulator(scene)
  capsule = kb.Capsule(scale=(0.25, 0.25, 0.25), position=(0, 0, 0), mass=1.0)
  scene.add(capsule)

  simulator.run()
  np.testing.assert_allclose(capsule.position[2], -0.5 * 10, atol=0.1)
