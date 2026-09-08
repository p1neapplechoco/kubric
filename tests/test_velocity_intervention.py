"""Tests for the free-body velocity-intervention pipeline (physics only, no Blender)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from interventions import velocity_intervention as vi
from interventions.logging import LINEAR_VELOCITY_SLICE, POSITION_SLICE

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "velocity_intervention.yaml"


@pytest.fixture(scope="module")
def ranges():
  return vi.load_ranges(_CONFIG)


@pytest.fixture(scope="module")
def spec(ranges):
  return vi.sample_instance(ranges, master_seed=7, index=0, attempt=0)


@pytest.fixture(scope="module")
def generated(ranges):
  return vi.generate_instance(ranges, master_seed=7, index=0)


def test_sampling_is_deterministic(ranges, spec):
  again = vi.sample_instance(ranges, master_seed=7, index=0, attempt=0)
  assert again.to_dict() == spec.to_dict()
  other = vi.sample_instance(ranges, master_seed=7, index=1, attempt=0)
  assert other.instance_id != spec.instance_id


def test_scene_matches_dataset_configuration(ranges, spec):
  dynamic = [item for item in spec.scene.objects if not item.static]
  low, high = ranges["objects"]["count"]
  assert low <= len(dynamic) <= high
  # Fixed mass for every dynamic body.
  assert {item.mass for item in dynamic} == {float(ranges["objects"]["mass"])}
  # Exactly one subject with a planar initial velocity; nothing else moves at t=0.
  subjects = [item for item in dynamic if spec.roles[item.object_id] == "subject"]
  assert len(subjects) == 1 and subjects[0].object_id == spec.subject_id
  assert subjects[0].linear_velocity[2] == 0.0
  assert math.hypot(*subjects[0].linear_velocity[:2]) > 0.0
  for item in dynamic:
    if item.object_id != spec.subject_id:
      assert item.linear_velocity == (0.0, 0.0, 0.0)
  # Interactive and bystander roles both present; floor is static.
  roles = set(spec.roles.values())
  assert {"subject", "interactive", "bystander", "floor"} <= roles
  assert spec.object(spec.floor_id).static
  # Rolling and sliding shapes coexist.
  shapes = {item.shape for item in dynamic}
  assert "sphere" in shapes and shapes & {"cube", "cylinder"}


def test_material_sampling_couples_physics(spec):
  families = {spec.physics[oid]["material_family"] for oid in spec.object_ids}
  assert families  # sampled
  visual_families = {item.object_id: item.material.family for item in spec.visual.objects}
  for object_id in spec.object_ids:
    assert visual_families[object_id] == spec.physics[object_id]["material_family"]


def test_camera_is_static_within_the_clip(spec):
  camera = spec.visual.camera
  assert len(set(camera.positions)) == 1
  assert len(set(camera.look_ats)) == 1
  assert len(camera.positions) == len(spec.visual.frame_steps)
  assert spec.scene.camera is not None
  assert tuple(spec.scene.camera.position) == camera.positions[0]


def test_camera_varies_between_instances(ranges):
  a = vi.sample_instance(ranges, 7, 0).visual.camera.positions[0]
  b = vi.sample_instance(ranges, 7, 1).visual.camera.positions[0]
  assert a != b


def test_branches_differ_only_as_specified(spec):
  factual = spec.scene
  counterfactual = spec.counterfactual_scene
  removed = spec.removed_scene
  # All branches share the identical initial scene at t=0.
  assert counterfactual == factual
  assert removed == factual
  # Simulation branches diverge only at intervention_step.
  logs = vi.simulate_branches(spec)
  f_log = logs["factual"]
  c_log = logs["counterfactual"]
  r_log = logs["subject_removed"]
  intervention_step = int(c_log.metadata["intervention_step"])
  assert intervention_step > 0
  # Prior to intervention_step, states are identical.
  np.testing.assert_allclose(
      f_log.states[:intervention_step],
      c_log.states[:intervention_step],
      atol=1e-6,
  )
  np.testing.assert_allclose(
      f_log.states[:intervention_step],
      r_log.states[:intervention_step],
      atol=1e-6,
  )
  # In subject_removed, after intervention_step, subject is removed (z < -500).
  sub_col = f_log.object_ids.index(spec.subject_id)
  assert (r_log.states[intervention_step:, sub_col, POSITION_SLICE][:, 2] < -500.0).all()
  # In counterfactual, subject velocity changes at intervention_step.
  assert not np.allclose(
      f_log.states[intervention_step, sub_col, LINEAR_VELOCITY_SLICE],
      c_log.states[intervention_step, sub_col, LINEAR_VELOCITY_SLICE],
  )


def test_simulation_log_shape_and_initial_velocity(spec):
  log = vi.simulate_scene(spec.scene, spec.physics, "factual")
  steps = (spec.scene.frame_range[1] - spec.scene.frame_range[0]) * (
      spec.scene.step_rate // spec.scene.frame_rate)
  assert log.states.shape == (steps + 1, len(spec.scene.objects), 13)
  column = log.object_ids.index(spec.subject_id)
  np.testing.assert_allclose(
      log.states[0, column, LINEAR_VELOCITY_SLICE], spec.factual_velocity, atol=1e-9)
  # Subject moved along its heading (positive projection onto the initial direction).
  displacement = log.states[-1, column, POSITION_SLICE] - log.states[0, column, POSITION_SLICE]
  heading = np.asarray(spec.factual_velocity) / np.linalg.norm(spec.factual_velocity)
  assert float(displacement @ heading) > 0.3


def test_generated_instance_passes_qc_with_mixed_motion(generated, ranges):
  assert generated.qc.passed, generated.qc.reasons
  motion = generated.qc.metrics["motion"]["factual"]
  assert any(info["rolling_fraction"] > 0.2 for info in motion.values())
  assert any(info["sliding_fraction"] > 0.2 for info in motion.values())
  assert len(generated.qc.metrics["factual_struck"]) >= ranges["qc"]["min_struck"]
  assert len(generated.qc.metrics["factual_untouched"]) >= ranges["qc"]["min_untouched"]
  # Nothing moves when the subject is removed.
  assert generated.qc.metrics["removed_max_travel"] < 0.05
  truth = generated.ground_truth
  delta = truth.graph_delta
  assert delta.added or delta.removed or delta.changed
  assert generated.spec.floor_id not in truth.hard_affected


def test_write_and_read_round_trip(generated, tmp_path):
  target = vi.write_instance(tmp_path / "inst", generated)
  spec = vi.read_instance_spec(target)
  assert spec.to_dict() == generated.spec.to_dict()
  for branch in vi.BRANCHES:
    log = vi.read_branch_log(target, branch)
    assert log == generated.logs[branch]
    graph = json.loads((target / branch / "graph.json").read_text())
    assert graph["branch"] == branch
    assert all(spec.floor_id not in (e["object_a"], e["object_b"]) for e in graph["edges"])
  removed = vi.read_branch_log(target, "subject_removed")
  assert spec.subject_id in removed.object_ids
  assert int(removed.metadata["intervention_step"]) > 0
  payload = json.loads((target / "instance.json").read_text())
  assert payload["instance_id"] == spec.instance_id


def test_motion_summary_labels_static_bodies(spec):
  logs = vi.simulate_branches(spec)
  summary = vi.motion_summary(logs["subject_removed"], spec.scene)
  non_subject_ids = [oid for oid in spec.object_ids if oid not in (spec.floor_id, spec.subject_id)]
  assert all(summary[oid]["label"] == "static" for oid in non_subject_ids)
