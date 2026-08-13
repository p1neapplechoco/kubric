"""Dataset sampling, QC, balancing, split, and journal tests."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from interventions import read_paired_artifact, validate_path
from interventions.dataset import (
    CandidateSummary,
    InstanceSpec,
    QCResult,
    assign_grouped_splits,
    derive_seed,
    evaluate_qc,
    generate_candidate,
    load_ranges,
    primary_category,
    propagation_hop_depth,
    run_batch,
    sample_instance_spec,
    select_balanced,
    topology_signature,
)
from interventions.logging import ContactRecord, SimulationLog
from interventions.schema import (
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    ObjectConfig,
    SceneConfig,
)


_LOG_IDS = ("ball", "floor", "target")


def _ranges(expected="null", magnitude=(0.0, 0.0), object_count=(1, 1)):
  return {
      "scene": {
          "bounds": [[-4.0, -4.0, -1.0], [4.0, 4.0, 4.0]],
          "gravity": [0.0, 0.0, -9.81],
          "frame_range": [0, 1],
          "frame_rate": 24,
          "step_rate": 240,
      },
      "objects": {
          "count": list(object_count),
          "shapes": ["sphere", "cube"],
          "size": [0.12, 0.18],
          "mass": [0.5, 1.5],
          "friction": [0.0, 0.2],
          "restitution": [0.0, 0.1],
          "x": [-2.0, 2.0],
          "y": [-2.0, 2.0],
          "floor_size": [3.0, 3.0, 0.25],
      },
      "target": {
          "shape": "cube",
          "size": [0.15, 0.15],
          "x": [-1.0, -1.0],
          "y": [0.0, 0.0],
          "displacement_x": [0.0, 0.0],
          "displacement_y": [0.0, 0.0],
          "friction": [0.0, 0.0],
          "restitution": [0.0, 0.0],
      },
      "trajectory": {"waypoint_count": [2, 2], "method": "linear"},
      "intervention": {
          "recipes": ["remove_collision"],
          "magnitude": list(magnitude),
          "start_step": [1, 1],
          "duration_steps": [8, 8],
          "push_mass": [1.0, 1.0],
          "expected_effects": [expected],
      },
      "qc": {
          "linear_velocity_ceiling": 1000.0,
          "angular_velocity_ceiling": 1000.0,
          "clip_epsilon": 1e-9,
      },
      "balance": {"seed": 17},
      "split": {
          "seed": 23,
          "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
      },
  }


def _scene(*, support_exempt=True):
  return SceneConfig(
      objects=(
          ObjectConfig(
              "floor", "cube", size=(4, 4, 0.25), mass=0, static=True,
              position=(0, 0, -0.25),
              metadata={"qc_clip_exempt": support_exempt},
          ),
          ObjectConfig(
              "target", "cube", size=0.2, mass=0, static=True,
              position=(0, 0, 0.2),
          ),
          ObjectConfig(
              "ball", "sphere", size=0.2, mass=1, position=(1, 0, 0.2)
          ),
      ),
      seed=3,
      scene_bounds=((-5, -5, -1), (5, 5, 5)),
      gravity=(0, 0, 0),
      frame_range=(0, 1),
      frame_rate=24,
      step_rate=240,
  )


def _spec(expected="null", *, scene=None, start=1):
  scene = _scene() if scene is None else scene
  path = np.zeros((10, 7), dtype=np.float64)
  path[:, 2] = 0.2
  path[:, 3] = 1.0
  return InstanceSpec(
      attempt_index=0,
      instance_seed=derive_seed(7, 0, "instance"),
      instance_id="instance_test",
      scene_config=scene,
      target_id="target",
      factual_path=path,
      intervention=Intervention(
          "target", "remove_collision", 0.0, (start, 3), push_mass=1.0
      ),
      expected_effect=expected,
      intervention_start_step=start,
  )


def _commanded_path(steps=10):
  path = np.zeros((steps, 7), dtype=np.float64)
  path[:, 2] = 0.2
  path[:, 3] = 1.0
  return path


def _states(ids=_LOG_IDS, steps=10):
  states = np.zeros((steps, len(ids), 13), dtype=np.float64)
  states[:, :, 3] = 1.0
  states[:, ids.index("floor"), :3] = (0, 0, -0.25)
  states[:, ids.index("target"), :3] = (0, 0, 0.2)
  if "ball" in ids:
    states[:, ids.index("ball"), :3] = (1, 0, 0.2)
  return states


def _log(
    branch,
    *,
    states=None,
    ids=_LOG_IDS,
    steps=tuple(range(10)),
    contacts=(),
    commanded_path=None,
    step_rate=240,
):
  if states is None:
    states = _states(ids, len(steps))
  if commanded_path is None:
    commanded_path = _commanded_path(len(steps))
  return SimulationLog(
      branch=branch,
      object_ids=ids,
      steps=steps,
      states=states,
      contacts=contacts,
      step_rate=step_rate,
      commanded_path=commanded_path,
  )


def _truth(*, added=(), removed=(), changed=(), hard=(), soft=(), paths=None):
  return GroundTruth(
      GraphEdgeDelta(added=added, removed=removed, changed=changed),
      hard_affected=hard,
      soft_affected=soft,
      propagation_path={} if paths is None else paths,
  )


def _edge(left="target", right="ball"):
  return {
      "object_a": left,
      "object_b": right,
      "start_step": 1,
      "end_step": 2,
  }


def _qc(spec=None, factual=None, counterfactual=None, truth=None, **config):
  return evaluate_qc(
      _spec() if spec is None else spec,
      _log("factual") if factual is None else factual,
      _log("counterfactual") if counterfactual is None else counterfactual,
      _truth() if truth is None else truth,
      {
          "linear_velocity_ceiling": config.pop("linear_velocity_ceiling", 100),
          "angular_velocity_ceiling": config.pop("angular_velocity_ceiling", 100),
          "clip_epsilon": config.pop("clip_epsilon", 1e-9),
          **config,
      },
  )


def test_dataset_public_api_is_importable():
  assert InstanceSpec is not None
  assert QCResult is not None
  assert CandidateSummary is not None


def test_seed_and_sampling_are_domain_separated_deterministic_and_side_effect_free():
  ranges = _ranges()
  original = copy.deepcopy(ranges)
  global_state = np.random.get_state()

  seeds = [derive_seed(123, 4, domain) for domain in ("instance", "scene", "path")]
  first = sample_instance_spec(ranges, 123, 4)
  second = sample_instance_spec(ranges, 123, 4)

  assert len(set(seeds)) == 3
  assert first.to_dict() == second.to_dict()
  assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
      second.to_dict(), sort_keys=True
  )
  assert ranges == original
  after = np.random.get_state()
  assert global_state[0] == after[0]
  np.testing.assert_array_equal(global_state[1], after[1])
  assert global_state[2:] == after[2:]
  assert not first.factual_path.flags.writeable
  with pytest.raises(ValueError):
    first.factual_path[0, 0] = 99


def test_instance_spec_path_length_must_match_scene_duration():
  with pytest.raises(ValueError, match="path|duration|steps"):
    replace(_spec(), factual_path=_commanded_path(9))


@pytest.mark.parametrize("time_window", [(1, 20), (1, 3.5)])
def test_instance_spec_requires_bounded_integer_intervention_window(time_window):
  intervention = Intervention(
      "target", "remove_collision", 0.0, time_window, push_mass=1.0
  )

  with pytest.raises((TypeError, ValueError), match="time_window|window"):
    replace(
        _spec(), intervention=intervention, intervention_start_step=1
    )


def test_load_ranges_copies_and_freezes_yaml(tmp_path):
  config = tmp_path / "ranges.yaml"
  import yaml

  config.write_text(yaml.safe_dump(_ranges(), sort_keys=True), encoding="utf-8")
  loaded = load_ranges(config)

  assert loaded["scene"]["step_rate"] == 240
  with pytest.raises(TypeError):
    loaded["scene"]["step_rate"] = 10


def test_repository_ranges_preserve_null_effect_as_a_string_and_sample():
  config = Path(__file__).resolve().parents[1] / "configs" / "scene_ranges.yaml"
  loaded = load_ranges(config)

  assert loaded["intervention"]["expected_effects"] == ("non_null", "null")
  specs = tuple(sample_instance_spec(loaded, 20260811, index) for index in range(64))
  assert {spec.expected_effect for spec in specs} == {"non_null", "null"}


def _factual_sweep_reaches_initial_dynamic_volume(spec):
  """Approximates whether the intervention can alter an initial contact corridor."""
  target = next(
      item for item in spec.scene_config.objects if item.object_id == spec.target_id
  )
  start, end = (int(value) for value in spec.intervention.time_window)
  window_path = spec.factual_path[start:end]
  target_clearance = max(target.size)
  for item in spec.scene_config.objects:
    if item.metadata.get("role") != "dynamic":
      continue
    center = np.asarray(item.position)
    extent = np.asarray(item.size)
    dynamic_aabb = ((center - extent, center + extent),)
    try:
      validate_path(
          window_path,
          static_aabbs=dynamic_aabb,
          clearance=target_clearance,
      )
    except ValueError as error:
      if "intersects a static AABB" in str(error):
        return True
      raise
  return False


def test_repository_ranges_sample_a_diverse_interaction_corridor():
  config = Path(__file__).resolve().parents[1] / "configs" / "scene_ranges.yaml"
  loaded = load_ranges(config)

  specs = tuple(
      sample_instance_spec(loaded, 20260811, index) for index in range(512)
  )
  interaction_fraction = sum(
      _factual_sweep_reaches_initial_dynamic_volume(spec) for spec in specs
  ) / len(specs)

  assert {spec.intervention.recipe for spec in specs} == {
      "remove_collision",
      "create_collision",
      "retime",
      "break_contact",
      "maintain_contact",
  }
  assert {spec.expected_effect for spec in specs} == {"non_null", "null"}
  assert {len(spec.scene_config.objects) - 2 for spec in specs} == {2, 3, 4, 5}
  assert 0.45 <= interaction_fraction <= 0.75


def test_sampled_scene_has_explicit_ids_valid_path_and_nonoverlap():
  spec = sample_instance_spec(_ranges(object_count=(3, 3)), 987, 2)
  objects = spec.scene_config.objects

  assert [item.object_id for item in objects] == [
      "floor", "target", "object_0", "object_1", "object_2"
  ]
  assert objects[0].static and objects[0].metadata["qc_clip_exempt"]
  assert objects[1].static and spec.target_id == "target"
  assert spec.factual_path.shape == (10, 7)
  np.testing.assert_allclose(np.linalg.norm(spec.factual_path[:, 3:], axis=1), 1)
  lower, upper = np.asarray(spec.scene_config.scene_bounds)
  assert np.all(spec.factual_path[:, :3] >= lower)
  assert np.all(spec.factual_path[:, :3] <= upper)
  for left_index, left in enumerate(objects[1:], start=1):
    for right in objects[left_index + 1:]:
      left_extent = np.asarray(left.size)
      right_extent = np.asarray(right.size)
      separated = np.any(
          np.abs(np.asarray(left.position) - np.asarray(right.position))
          >= left_extent + right_extent
      )
      assert separated, (left.object_id, right.object_id)


def test_invalid_ranges_are_rejected():
  ranges = _ranges()
  ranges["objects"]["mass"] = [2.0, 1.0]
  with pytest.raises(ValueError, match="mass"):
    sample_instance_spec(ranges, 0, 0)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("branch", ("branch_misaligned",)),
        ("prefix_state", ("twin_prefix_mismatch",)),
        ("prefix_contact", ("twin_prefix_mismatch",)),
        ("nonfinite", ("nonfinite_state",)),
        ("linear", ("linear_velocity_ceiling",)),
        ("angular", ("angular_velocity_ceiling",)),
        ("bounds", ("target_out_of_bounds",)),
        ("clip", ("target_static_clip",)),
        ("empty", ("empty_affected",)),
        ("null", ("expected_null_mismatch",)),
    ],
)
def test_each_qc_reason_is_reported_independently(case, expected):
  spec = _spec()
  factual = _log("factual")
  counterfactual = _log("counterfactual")
  truth = _truth()
  if case == "branch":
    counterfactual = _log(
        "counterfactual", ids=("floor", "target"),
        states=_states(("floor", "target")),
    )
  elif case == "prefix_state":
    changed = np.array(counterfactual.states, copy=True)
    changed[0, _LOG_IDS.index("target"), 0] += 0.01
    counterfactual = _log("counterfactual", states=changed)
  elif case == "prefix_contact":
    contact = ContactRecord(0, "target", "ball", (0, 0, 0), (1, 0, 0), 1)
    factual = _log("factual", contacts=(contact,))
  elif case == "nonfinite":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, _LOG_IDS.index("ball"), 0] = np.nan
    counterfactual = SimpleNamespace(
        **{**counterfactual.__dict__, "states": changed}
    )
  elif case == "linear":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, _LOG_IDS.index("ball"), 7] = 101
    counterfactual = _log("counterfactual", states=changed)
  elif case == "angular":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, _LOG_IDS.index("ball"), 10] = 101
    counterfactual = _log("counterfactual", states=changed)
  elif case == "bounds":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, _LOG_IDS.index("target"), 0] = 6
    counterfactual = _log("counterfactual", states=changed)
  elif case == "clip":
    scene = _scene()
    support = replace(
        scene.objects[0], object_id="wall", position=(2, 0, 0.2),
        size=(0.2, 0.2, 0.2), metadata={},
    )
    scene = replace(scene, objects=(support,) + scene.objects[1:])
    spec = _spec(scene=scene)
    ids = ("ball", "target", "wall")
    base = np.zeros((10, 3, 13), dtype=np.float64)
    base[:, :, 3] = 1.0
    base[:, 0, :3] = (1, 0, 0.2)
    base[:, 1, :3] = (0, 0, 0.2)
    base[:, 2, :3] = (2, 0, 0.2)
    factual = _log("factual", ids=ids, states=base)
    changed = np.array(base, copy=True)
    changed[2, 1, :3] = (2, 0, 0.2)
    counterfactual = _log("counterfactual", ids=ids, states=changed)
  elif case == "empty":
    spec = _spec("non_null")
  elif case == "null":
    truth = _truth(hard=("ball",), paths={"ball": ("target", "ball")})

  result = _qc(spec, factual, counterfactual, truth)

  assert result.reason_codes == expected
  assert not result.accepted


def test_qc_checks_target_and_contacts_in_full_twin_prefix():
  contact = ContactRecord(0, "target", "ball", (0, 0, 0), (1, 0, 0), 1)
  state_changed = np.array(_log("counterfactual").states, copy=True)
  state_changed[0, _LOG_IDS.index("target"), 0] = 0.1
  counterfactual = _log("counterfactual", states=state_changed)
  factual = _log("factual", contacts=(contact,))

  result = _qc(factual=factual, counterfactual=counterfactual)

  assert result.reason_codes == ("twin_prefix_mismatch",)


def test_qc_collects_sorted_reasons_metrics_and_honors_clip_exemption():
  changed = np.array(_log("counterfactual").states, copy=True)
  changed[2, _LOG_IDS.index("ball"), 7] = 200
  changed[2, _LOG_IDS.index("ball"), 10] = 300
  result = _qc(
      spec=_spec("non_null"),
      counterfactual=_log("counterfactual", states=changed),
      linear_velocity_ceiling=100,
      angular_velocity_ceiling=100,
  )

  assert result.reason_codes == tuple(sorted(result.reason_codes))
  assert set(result.reason_codes) == {
      "angular_velocity_ceiling", "empty_affected", "linear_velocity_ceiling"
  }
  assert result.metrics["max_linear_velocity"] == pytest.approx(200)
  assert result.metrics["max_angular_velocity"] == pytest.approx(300)
  assert "target_static_clip" not in result.reason_codes
  with pytest.raises(TypeError):
    result.metrics["new"] = 1


def test_qc_handles_malformed_doubles_without_crashing():
  malformed = SimpleNamespace(
      branch="counterfactual",
      object_ids=("target",),
      steps=(0,),
      states="not-an-array",
      contacts=None,
      step_rate="bad",
  )
  result = _qc(counterfactual=malformed)
  assert "branch_misaligned" in result.reason_codes
  assert "nonfinite_state" in result.reason_codes


@pytest.mark.parametrize("malformation", ["fractional_steps", "fractional_contact"])
def test_qc_rejects_lossy_integer_coercion(malformation):
  factual = _log("factual")
  counterfactual = _log("counterfactual")
  if malformation == "fractional_steps":
    steps = tuple(index + 0.5 for index in range(10))
    factual = SimpleNamespace(**{**factual.__dict__, "steps": steps})
    counterfactual = SimpleNamespace(
        **{**counterfactual.__dict__, "steps": steps}
    )
  else:
    contact = SimpleNamespace(
        step=0.5, object_a="target", object_b="ball"
    )
    factual = SimpleNamespace(
        **{**factual.__dict__, "contacts": (contact,)}
    )
    counterfactual = SimpleNamespace(
        **{**counterfactual.__dict__, "contacts": (contact,)}
    )

  result = _qc(factual=factual, counterfactual=counterfactual)

  assert result.reason_codes == ("branch_misaligned",)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_object",
        "permuted_ids",
        "truncated_steps",
        "wrong_step_rate",
        "missing_commanded_path",
        "wrong_factual_path",
        "counterfactual_outside_window",
    ],
)
def test_qc_requires_logs_to_match_the_exact_instance_spec(mutation):
  spec = _spec()
  factual = _log("factual")
  counterfactual = _log("counterfactual")
  if mutation == "missing_object":
    ids = ("floor", "target")
    factual = _log("factual", ids=ids, states=_states(ids))
    counterfactual = _log("counterfactual", ids=ids, states=_states(ids))
  elif mutation == "permuted_ids":
    permuted = tuple(reversed(_LOG_IDS))
    factual = replace(factual, object_ids=permuted)
    counterfactual = replace(counterfactual, object_ids=permuted)
  elif mutation == "truncated_steps":
    steps = tuple(range(9))
    factual = _log("factual", steps=steps)
    counterfactual = _log("counterfactual", steps=steps)
  elif mutation == "wrong_step_rate":
    factual = replace(factual, step_rate=120)
    counterfactual = replace(counterfactual, step_rate=120)
  elif mutation == "missing_commanded_path":
    factual = replace(factual, commanded_path=None)
    counterfactual = replace(counterfactual, commanded_path=None)
  elif mutation == "wrong_factual_path":
    path = factual.commanded_path.copy()
    path[2, 0] = 0.1
    factual = replace(factual, commanded_path=path)
  else:
    path = counterfactual.commanded_path.copy()
    path[5, 0] = 0.1
    counterfactual = replace(counterfactual, commanded_path=path)

  result = _qc(spec=spec, factual=factual, counterfactual=counterfactual)

  assert "branch_misaligned" in result.reason_codes


def test_qc_rejects_counterfactual_path_beyond_intervention_magnitude():
  intervention = Intervention(
      "target", "remove_collision", 0.05, (1, 4), push_mass=1.0
  )
  spec = replace(
      _spec(), intervention=intervention, intervention_start_step=1
  )
  factual = _log("factual")
  counterfactual_path = factual.commanded_path.copy()
  counterfactual_path[2, 0] = 0.1

  result = _qc(
      spec=spec,
      factual=factual,
      counterfactual=_log(
          "counterfactual", commanded_path=counterfactual_path
      ),
  )

  assert "branch_misaligned" in result.reason_codes


def test_qc_uses_oriented_target_volume_for_scene_bounds():
  half_angle = np.pi / 8
  quaternion = (
      np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)
  )
  scene = _scene()
  target = replace(
      scene.objects[1], size=(0.5, 0.1, 0.1), quaternion=quaternion
  )
  scene = replace(scene, objects=(scene.objects[0], target, scene.objects[2]))
  spec = _spec(scene=scene)
  states = _states()
  target_index = _LOG_IDS.index("target")
  states[2, target_index, :3] = (4.6, 0.0, 0.2)
  states[2, target_index, 3:7] = quaternion

  result = _qc(
      spec=spec,
      factual=_log("factual", states=states),
      counterfactual=_log("counterfactual", states=states),
  )

  assert "target_out_of_bounds" in result.reason_codes


@pytest.mark.parametrize(
    ("component", "reason"),
    [
        (7, "linear_velocity_ceiling"),
        (10, "angular_velocity_ceiling"),
    ],
)
def test_qc_handles_huge_finite_speed_without_overflow(component, reason):
  states = _states()
  states[2, _LOG_IDS.index("ball"), component] = 1e308

  result = _qc(counterfactual=_log("counterfactual", states=states))

  assert result.reason_codes == (reason,)
  assert np.isfinite(result.metrics[
      "max_linear_velocity" if component == 7 else "max_angular_velocity"
  ])


@pytest.mark.parametrize(
    ("truth", "category"),
    [
        (_truth(added=(_edge(),)), "contact_added"),
        (_truth(removed=(_edge(),)), "contact_removed"),
        (_truth(changed=(_edge(),)), "contact_changed"),
        (_truth(added=(_edge(),), removed=(_edge("target", "floor"),)),
         "mixed_contact_delta"),
        (_truth(hard=("ball",)), "state_only"),
        (_truth(), "null_effect"),
    ],
)
def test_primary_category(truth, category):
  assert primary_category(truth) == category


def test_propagation_hop_depth_and_bucket():
  truth = _truth(
      hard=("a", "b"),
      paths={"a": ("target", "a"), "b": ("target", "a", "x", "b")},
  )
  assert propagation_hop_depth(truth) == (3, "3+")
  assert propagation_hop_depth(_truth()) == (0, "0")


def _topology_fixture(prefix=""):
  ids = (prefix + "floor", prefix + "target", prefix + "ball")
  scene = SceneConfig(
      objects=(
          ObjectConfig(ids[0], "cube", mass=0, static=True),
          ObjectConfig(ids[1], "cube", mass=0, static=True),
          ObjectConfig(ids[2], "sphere"),
      ),
      gravity=(0, 0, 0),
      scene_bounds=((-5, -5, -5), (5, 5, 5)),
      frame_range=(0, 1), frame_rate=24, step_rate=240,
  )
  states = np.zeros((1, 3, 13))
  states[:, :, 3] = 1
  contact = ContactRecord(
      0, ids[1], ids[2], (0, 0, 0), (1, 0, 0), 1
  )
  log = _log("factual", ids=ids, steps=(0,), states=states, contacts=(contact,))
  return scene, log, ids[1]


def test_topology_signature_is_id_invariant_and_edge_sensitive():
  scene_a, log_a, target_a = _topology_fixture("")
  scene_b, log_b, target_b = _topology_fixture("renamed_")

  signature = topology_signature(scene_a, log_a, target_a)
  assert signature == topology_signature(scene_b, log_b, target_b)
  no_edge = _log(
      "factual", ids=log_a.object_ids, steps=(0,), states=log_a.states,
  )
  assert signature != topology_signature(scene_a, no_edge, target_a)


def _regular_topology(edges, prefix=""):
  ids = (prefix + "target",) + tuple(
      prefix + "node_{}".format(index) for index in range(6)
  )
  scene = SceneConfig(
      objects=(ObjectConfig(ids[0], "cube", mass=0, static=True),) + tuple(
          ObjectConfig(object_id, "sphere") for object_id in ids[1:]
      ),
      gravity=(0, 0, 0),
      scene_bounds=((-5, -5, -5), (5, 5, 5)),
      frame_range=(0, 1), frame_rate=24, step_rate=240,
  )
  states = np.zeros((1, len(ids), 13), dtype=np.float64)
  states[:, :, 3] = 1.0
  contacts = tuple(
      ContactRecord(
          0, ids[left + 1], ids[right + 1], (0, 0, 0), (1, 0, 0), 1
      )
      for left, right in edges
  )
  return scene, SimulationLog(
      "factual", ids, (0,), states, contacts, 240,
      commanded_path=_commanded_path(1),
  ), ids[0]


def test_topology_signature_distinguishes_cycle_from_two_triangles():
  cycle_edges = tuple((index, (index + 1) % 6) for index in range(6))
  triangle_edges = (
      (0, 1), (1, 2), (2, 0),
      (3, 4), (4, 5), (5, 3),
  )
  cycle_scene, cycle_log, cycle_target = _regular_topology(cycle_edges)
  triangles_scene, triangles_log, triangles_target = _regular_topology(
      triangle_edges
  )

  assert topology_signature(
      cycle_scene, cycle_log, cycle_target
  ) != topology_signature(triangles_scene, triangles_log, triangles_target)


def test_topology_signature_exactly_distinguishes_k33_from_triangular_prism():
  k33_edges = tuple(
      (left, right) for left in range(3) for right in range(3, 6)
  )
  prism_edges = (
      (0, 1), (1, 2), (2, 0),
      (3, 4), (4, 5), (5, 3),
      (0, 3), (1, 4), (2, 5),
  )
  k33_scene, k33_log, k33_target = _regular_topology(k33_edges)
  prism_scene, prism_log, prism_target = _regular_topology(prism_edges)
  renamed_scene, renamed_log, renamed_target = _regular_topology(
      prism_edges, "renamed_"
  )

  prism_signature = topology_signature(prism_scene, prism_log, prism_target)
  assert topology_signature(k33_scene, k33_log, k33_target) != prism_signature
  assert topology_signature(
      renamed_scene, renamed_log, renamed_target
  ) == prism_signature


def _candidate(index, category, bucket, topology=None):
  return CandidateSummary(
      instance_id="id_{:02d}".format(index),
      attempt_index=index,
      category=category,
      hop_depth=3 if bucket == "3+" else int(bucket),
      hop_bucket=bucket,
      topology_signature=topology or "topology_{:02d}".format(index),
      artifact_path="instances/id_{:02d}".format(index),
  )


def test_balanced_selection_is_permutation_invariant_and_round_robin_fair():
  candidates = tuple(
      _candidate(index, category, bucket)
      for index, (category, bucket) in enumerate(
          [("contact_added", "1")] * 4
          + [("contact_removed", "2")] * 4
          + [("state_only", "3+")] * 4
      )
  )
  first = select_balanced(candidates, 6, seed=99)
  second = select_balanced(tuple(reversed(candidates)), 6, seed=99)

  assert [item.instance_id for item in first] == [item.instance_id for item in second]
  counts = {}
  for item in first:
    counts[(item.category, item.hop_bucket)] = counts.get(
        (item.category, item.hop_bucket), 0
    ) + 1
  assert max(counts.values()) - min(counts.values()) <= 1
  assert select_balanced(candidates, 99, seed=99) == select_balanced(
      candidates, len(candidates), seed=99
  )


def test_grouped_split_is_deterministic_and_never_splits_topology():
  candidates = tuple(
      _candidate(index, "state_only", "1", "group_{}".format(index // 2))
      for index in range(20)
  )
  fractions = {"train": 0.6, "val": 0.2, "test": 0.2}
  first = assign_grouped_splits(candidates, fractions, seed=12)
  second = assign_grouped_splits(tuple(reversed(candidates)), fractions, seed=12)

  assert first == second
  for index in range(0, 20, 2):
    assert first["id_{:02d}".format(index)] == first["id_{:02d}".format(index + 1)]
  counts = {name: tuple(first.values()).count(name) for name in fractions}
  assert abs(counts["train"] - 12) <= 2
  assert abs(counts["val"] - 4) <= 2
  assert abs(counts["test"] - 4) <= 2


def _patch_fast_batch(monkeypatch, *, fail_indices=(), truth_by_index=None):
  import interventions.dataset as dataset

  def fake_generate(spec):
    if spec.attempt_index in fail_indices:
      raise RuntimeError("synthetic failure {}".format(spec.attempt_index))
    truth = _truth() if truth_by_index is None else truth_by_index(spec.attempt_index)
    return _log("factual"), _log("counterfactual"), truth

  monkeypatch.setattr(dataset, "generate_candidate", fake_generate)
  monkeypatch.setattr(
      dataset,
      "evaluate_qc",
      lambda *args, **kwargs: QCResult(True, (), {"max_linear_velocity": 0}),
  )

  def fake_publish(root, spec, factual, counterfactual, ground_truth=None):
    artifact = root / "instances" / spec.instance_id
    if artifact.exists():
      return artifact
    artifact.mkdir()
    payload = b"synthetic artifact\n"
    (artifact / "payload.bin").write_bytes(payload)
    manifest = {
        "instance_id": spec.instance_id,
        "files": {
            "payload.bin": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        },
    }
    (artifact / "instance_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact

  monkeypatch.setattr(dataset, "_publish_instance", fake_publish)


def test_batch_resume_uses_attempt_truth_and_refuses_mismatch(monkeypatch, tmp_path):
  _patch_fast_batch(monkeypatch)
  output = tmp_path / "dataset"

  exhausted = run_batch(_ranges(), output, 5, 2, 1)
  assert exhausted["status"] == "capacity_exhausted"
  complete = run_batch(_ranges(), output, 5, 2, 3, resume=True)

  assert complete["status"] == "complete"
  assert sorted(path.name for path in (output / "attempts").glob("*.json")) == [
      "00000000.json", "00000001.json", "00000002.json"
  ]
  assert len(complete["selected_ids"]) == 2
  (output / "manifest.json").unlink()
  rebuilt = run_batch(_ranges(), output, 5, 2, 3, resume=True)
  assert rebuilt["selected_ids"] == complete["selected_ids"]
  with pytest.raises(ValueError, match="seed"):
    run_batch(_ranges(), output, 6, 2, 3, resume=True)
  with pytest.raises(ValueError, match="count"):
    run_batch(_ranges(), output, 5, 1, 3, resume=True)
  changed = _ranges()
  changed["target"]["x"] = [-0.5, -0.5]
  with pytest.raises(ValueError, match="config"):
    run_batch(changed, output, 5, 2, 3, resume=True)


def test_batch_journals_errors_and_continues(monkeypatch, tmp_path):
  _patch_fast_batch(monkeypatch, fail_indices=(0,))
  output = tmp_path / "dataset"

  result = run_batch(_ranges(), output, 8, 1, 3)

  assert result["status"] == "complete"
  error = json.loads((output / "errors" / "00000000.json").read_text())
  attempt = json.loads((output / "attempts" / "00000000.json").read_text())
  assert error["error_type"] == "RuntimeError"
  assert attempt["status"] == "error"
  assert (output / "attempts" / "00000001.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "rewritten_manifest"])
def test_batch_resume_rejects_missing_or_corrupt_candidate_artifact(
    mutation, monkeypatch, tmp_path
):
  _patch_fast_batch(monkeypatch)
  output = tmp_path / "dataset"
  result = run_batch(_ranges(), output, 8, 1, 1)
  artifact = output / "instances" / result["selected_ids"][0]
  if mutation == "missing":
    artifact.rename(tmp_path / "moved-artifact")
  elif mutation == "corrupt":
    (artifact / "payload.bin").write_bytes(b"tampered\n")
  else:
    payload = b"tampered and rehashed\n"
    (artifact / "payload.bin").write_bytes(payload)
    rewritten = {
        "instance_id": result["selected_ids"][0],
        "files": {
            "payload.bin": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        },
    }
    (artifact / "instance_manifest.json").write_text(
        json.dumps(rewritten, sort_keys=True) + "\n", encoding="utf-8"
    )

  with pytest.raises(ValueError, match="artifact|manifest|integrity|missing|corrupt"):
    run_batch(_ranges(), output, 8, 1, 1, resume=True)


def test_batch_recovers_complete_orphan_after_attempt_journal_crash(
    monkeypatch, tmp_path
):
  import interventions.dataset as dataset

  factual = _log("factual")
  counterfactual = _log("counterfactual")
  truth = _truth()
  monkeypatch.setattr(
      dataset,
      "generate_candidate",
      lambda spec: (factual, counterfactual, truth),
  )
  monkeypatch.setattr(
      dataset,
      "evaluate_qc",
      lambda *args, **kwargs: QCResult(True, (), {"max_linear_velocity": 0}),
  )

  def fake_pair_writer(directory, *args, **kwargs):
    destination = Path(directory)
    destination.mkdir()
    (destination / "pair.bin").write_bytes(b"paired artifact\n")

  monkeypatch.setattr(dataset, "write_paired_artifact", fake_pair_writer)
  expected_spec = sample_instance_spec(_ranges(), 11, 0)
  monkeypatch.setattr(
      dataset,
      "read_paired_artifact",
      lambda directory: (
          factual,
          counterfactual,
          truth,
          {
              "schema_version": expected_spec.scene_config.schema_version,
              "target_id": expected_spec.target_id,
              "rng_seed": expected_spec.instance_seed,
              "scene_config": expected_spec.scene_config.to_dict(),
              "intervention": expected_spec.intervention.to_dict(),
              "factual": "factual",
              "counterfactual": "counterfactual",
              "tags": ["null_effect", "target_only"],
              "extraction_thresholds": {
                  "force_threshold": 0.0,
                  "min_episode_impulse": 0.0,
                  "force_tolerance": 1e-6,
                  "position_epsilon": 1e-3,
                  "velocity_epsilon": 1e-3,
                  "quaternion_epsilon": 1e-3,
              },
          },
      ),
      raising=False,
  )
  real_write_once = dataset._write_once
  crashed = False

  def crash_before_attempt_record(path, payload):
    nonlocal crashed
    if path.parent.name == "attempts" and not crashed:
      crashed = True
      raise KeyboardInterrupt("synthetic crash")
    return real_write_once(path, payload)

  monkeypatch.setattr(dataset, "_write_once", crash_before_attempt_record)
  output = tmp_path / "dataset"
  with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
    run_batch(_ranges(), output, 11, 1, 1)
  assert len(tuple((output / "instances").iterdir())) == 1
  assert not (output / "attempts" / "00000000.json").exists()

  monkeypatch.setattr(dataset, "_write_once", real_write_once)
  resumed = run_batch(_ranges(), output, 11, 1, 1, resume=True)

  assert resumed["status"] == "complete"
  assert len(resumed["selected_ids"]) == 1


def test_batch_rejects_divergent_complete_orphan_rerun(monkeypatch, tmp_path):
  import interventions.dataset as dataset

  persisted_factual = _log("factual")
  persisted_counterfactual = _log("counterfactual")
  divergent_states = np.array(persisted_counterfactual.states, copy=True)
  divergent_states[5, _LOG_IDS.index("ball"), 0] += 0.25
  divergent_counterfactual = _log(
      "counterfactual", states=divergent_states
  )
  truth = _truth()
  generated = 0

  def changing_generate(spec):
    nonlocal generated
    generated += 1
    counterfactual = (
        persisted_counterfactual if generated == 1 else divergent_counterfactual
    )
    return persisted_factual, counterfactual, truth

  monkeypatch.setattr(dataset, "generate_candidate", changing_generate)
  monkeypatch.setattr(
      dataset,
      "evaluate_qc",
      lambda *args, **kwargs: QCResult(True, (), {"max_linear_velocity": 0}),
  )

  def fake_pair_writer(directory, *args, **kwargs):
    destination = Path(directory)
    destination.mkdir()
    (destination / "pair.bin").write_bytes(b"first generated pair\n")

  monkeypatch.setattr(dataset, "write_paired_artifact", fake_pair_writer)
  expected_spec = sample_instance_spec(_ranges(), 29, 0)
  provenance = {
      "schema_version": expected_spec.scene_config.schema_version,
      "target_id": expected_spec.target_id,
      "rng_seed": expected_spec.instance_seed,
      "scene_config": expected_spec.scene_config.to_dict(),
      "intervention": expected_spec.intervention.to_dict(),
      "factual": "factual",
      "counterfactual": "counterfactual",
      "tags": ["null_effect", "target_only"],
      "extraction_thresholds": {
          "force_threshold": 0.0,
          "min_episode_impulse": 0.0,
          "force_tolerance": 1e-6,
          "position_epsilon": 1e-3,
          "velocity_epsilon": 1e-3,
          "quaternion_epsilon": 1e-3,
      },
  }
  monkeypatch.setattr(
      dataset,
      "read_paired_artifact",
      lambda directory: (
          persisted_factual, persisted_counterfactual, truth, provenance
      ),
      raising=False,
  )
  real_write_once = dataset._write_once
  crashed = False

  def crash_before_attempt(path, payload):
    nonlocal crashed
    if path.parent.name == "attempts" and not crashed:
      crashed = True
      raise KeyboardInterrupt("synthetic orphan crash")
    return real_write_once(path, payload)

  monkeypatch.setattr(dataset, "_write_once", crash_before_attempt)
  output = tmp_path / "dataset"
  with pytest.raises(KeyboardInterrupt, match="orphan crash"):
    run_batch(_ranges(), output, 29, 1, 1)

  monkeypatch.setattr(dataset, "_write_once", real_write_once)
  resumed = run_batch(_ranges(), output, 29, 1, 1, resume=True)

  assert generated == 2
  assert resumed["status"] == "capacity_exhausted"
  assert resumed["selected_ids"] == ()
  attempt = json.loads((output / "attempts" / "00000000.json").read_text())
  assert attempt["status"] == "error"
  assert "orphan" in attempt["message"].lower() or "mismatch" in attempt[
      "message"
  ].lower()


def test_instance_publisher_rejects_orphan_with_nondefault_thresholds(tmp_path):
  import interventions.dataset as dataset

  spec = sample_instance_spec(_ranges(), 43, 0)
  factual, counterfactual, truth = generate_candidate(spec)
  root = tmp_path / "dataset"
  artifact = root / "instances" / spec.instance_id
  artifact.parent.mkdir(parents=True)
  persisted_truth = dataset.write_paired_artifact(
      artifact,
      spec.scene_config,
      spec.intervention,
      spec.instance_seed,
      factual,
      counterfactual,
      position_epsilon=0.5,
  )
  assert persisted_truth == truth
  dataset._write_once(
      artifact / "spec.json", dataset._canonical_bytes(spec.to_dict())
  )
  instance_manifest = {
      "instance_id": spec.instance_id,
      "files": dataset._file_manifest(artifact),
  }
  dataset._write_once(
      artifact / "instance_manifest.json",
      dataset._canonical_bytes(instance_manifest),
  )
  persisted_factual, persisted_counterfactual, persisted_truth, provenance = (
      read_paired_artifact(artifact)
  )
  assert persisted_factual == factual
  assert persisted_counterfactual == counterfactual
  assert persisted_truth == truth
  assert provenance["extraction_thresholds"]["position_epsilon"] == 0.5

  with pytest.raises(ValueError, match="orphan|provenance|threshold|mismatch"):
    dataset._publish_instance(
        root, spec, factual, counterfactual, truth
    )


def test_batch_recovers_error_record_after_attempt_journal_crash(
    monkeypatch, tmp_path
):
  import interventions.dataset as dataset

  generated = 0

  def changing_failure(spec):
    nonlocal generated
    generated += 1
    raise RuntimeError("failure-{}".format(generated))

  monkeypatch.setattr(dataset, "generate_candidate", changing_failure)
  real_write_once = dataset._write_once
  crashed = False

  def crash_after_error_record(path, payload):
    nonlocal crashed
    if path.parent.name == "attempts" and not crashed:
      crashed = True
      raise KeyboardInterrupt("synthetic crash")
    return real_write_once(path, payload)

  monkeypatch.setattr(dataset, "_write_once", crash_after_error_record)
  output = tmp_path / "dataset"
  with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
    run_batch(_ranges(), output, 19, 1, 1)
  assert (output / "errors" / "00000000.json").exists()
  assert not (output / "attempts" / "00000000.json").exists()
  original_error = (output / "errors" / "00000000.json").read_bytes()

  monkeypatch.setattr(dataset, "_write_once", real_write_once)
  resumed = run_batch(_ranges(), output, 19, 1, 1, resume=True)

  assert resumed["status"] == "capacity_exhausted"
  assert generated == 1
  assert (output / "errors" / "00000000.json").read_bytes() == original_error
  attempt = json.loads((output / "attempts" / "00000000.json").read_text())
  assert attempt["status"] == "error"
  assert attempt["message"] == "failure-1"


def test_batch_rejects_a_concurrent_writer(monkeypatch, tmp_path):
  import interventions.dataset as dataset

  _patch_fast_batch(monkeypatch)
  generate = dataset.generate_candidate
  entered = threading.Event()
  release = threading.Event()

  def slow_generate(spec):
    entered.set()
    assert release.wait(10)
    return generate(spec)

  monkeypatch.setattr(dataset, "generate_candidate", slow_generate)
  output = tmp_path / "dataset"
  with ThreadPoolExecutor(max_workers=2) as executor:
    first = executor.submit(run_batch, _ranges(), output, 13, 1, 1)
    assert entered.wait(10)
    second = executor.submit(
        run_batch, _ranges(), output, 13, 1, 1, resume=True
    )
    try:
      second_error = second.exception(timeout=1)
    except TimeoutError as error:
      second_error = error
    finally:
      release.set()
    first_result = first.result(timeout=10)

  assert first_result["status"] == "complete"
  assert isinstance(second_error, RuntimeError)
  assert "lock" in str(second_error).lower()


def test_batch_lock_cannot_be_bypassed_with_output_symlink_alias(
    monkeypatch, tmp_path
):
  import interventions.dataset as dataset

  _patch_fast_batch(monkeypatch)
  generate = dataset.generate_candidate
  entered = threading.Event()
  release = threading.Event()
  calls = 0

  def block_first_generate(spec):
    nonlocal calls
    calls += 1
    if calls == 1:
      entered.set()
      assert release.wait(10)
    return generate(spec)

  monkeypatch.setattr(dataset, "generate_candidate", block_first_generate)
  output = tmp_path / "dataset"
  alias = tmp_path / "dataset-alias"
  with ThreadPoolExecutor(max_workers=1) as executor:
    first = executor.submit(run_batch, _ranges(), output, 31, 1, 1)
    assert entered.wait(10)
    alias.symlink_to(output, target_is_directory=True)
    try:
      try:
        run_batch(_ranges(), alias, 31, 1, 1, resume=True)
      except Exception as error:  # Captured so the first writer is always released.
        second_error = error
      else:
        second_error = None
    finally:
      release.set()
    first.result(timeout=10)

  assert isinstance(second_error, RuntimeError)
  assert "lock" in str(second_error).lower()


def test_fresh_batch_initialization_is_atomic_across_interrupt(
    monkeypatch, tmp_path
):
  import interventions.dataset as dataset

  _patch_fast_batch(monkeypatch)
  real_write_once = dataset._write_once
  interrupted = False

  def interrupt_first_write(path, payload):
    nonlocal interrupted
    if not interrupted:
      interrupted = True
      raise KeyboardInterrupt("initialization interrupted")
    return real_write_once(path, payload)

  monkeypatch.setattr(dataset, "_write_once", interrupt_first_write)
  output = tmp_path / "dataset"
  with pytest.raises(KeyboardInterrupt, match="initialization interrupted"):
    run_batch(_ranges(), output, 37, 1, 1)

  assert not output.exists()
  assert not tuple(tmp_path.glob(".dataset.init-*"))
  monkeypatch.setattr(dataset, "_write_once", real_write_once)
  result = run_batch(_ranges(), output, 37, 1, 1)
  assert result["status"] == "complete"


def test_journal_publication_fsyncs_parent_directories(monkeypatch, tmp_path):
  import interventions.dataset as dataset

  directories = []
  monkeypatch.setattr(
      dataset,
      "_fsync_directory",
      lambda directory: directories.append(Path(directory)),
      raising=False,
  )

  dataset._write_once(tmp_path / "immutable.json", b"{}\n")
  dataset._write_atomic(tmp_path / "manifest.json", b"{}\n")

  assert directories.count(tmp_path) >= 2


def test_immutable_write_is_idempotent_only_for_identical_bytes(tmp_path):
  import interventions.dataset as dataset

  destination = tmp_path / "record.json"
  dataset._write_once(destination, b"{\"value\":1}\n")
  dataset._write_once(destination, b"{\"value\":1}\n")

  with pytest.raises(FileExistsError, match="immutable|exists|conflict"):
    dataset._write_once(destination, b"{\"value\":2}\n")


def test_batch_offline_pool_allows_late_stratum_to_displace_early_surplus(
    monkeypatch, tmp_path
):
  def truth_for(index):
    if index < 3:
      return _truth(added=(_edge(),))
    return _truth(removed=(_edge(),))

  _patch_fast_batch(monkeypatch, truth_by_index=truth_for)
  result = run_batch(_ranges(), tmp_path / "pool", 9, 2, 4)
  by_id = {item["instance_id"]: item for item in result["candidates"]}
  categories = {by_id[instance_id]["category"] for instance_id in result["selected_ids"]}

  assert result["attempt_count"] == 4
  assert categories == {"contact_added", "contact_removed"}


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("objects", "friction"),
        ("objects", "restitution"),
        ("target", "friction"),
        ("target", "restitution"),
    ],
)
def test_sampling_prevalidates_material_ranges(section, key):
  ranges = _ranges()
  ranges[section][key] = [1.01, 1.01]
  with pytest.raises(ValueError, match=key):
    sample_instance_spec(ranges, 0, 0)


def test_sampling_prevalidates_full_target_path_volume_inside_bounds():
  ranges = _ranges()
  ranges["target"]["x"] = [3.9, 3.9]
  ranges["target"]["displacement_x"] = [0.0, 0.0]
  with pytest.raises(ValueError, match="target.*bounds"):
    sample_instance_spec(ranges, 0, 0)


def test_cli_modules_are_importable():
  instance_cli = importlib.import_module("scripts.generate_instance")
  dataset_cli = importlib.import_module("scripts.generate_dataset")
  assert callable(instance_cli.main)
  assert callable(dataset_cli.main)


def test_generate_instance_cli_passes_truth_to_publisher(
    monkeypatch, tmp_path, capsys
):
  import scripts.generate_instance as instance_cli

  spec = _spec()
  factual = _log("factual")
  counterfactual = _log("counterfactual")
  truth = _truth()
  published = []
  monkeypatch.setattr(instance_cli, "load_ranges", lambda path: _ranges())
  monkeypatch.setattr(
      instance_cli, "sample_instance_spec", lambda *args: spec
  )
  monkeypatch.setattr(
      instance_cli,
      "generate_candidate",
      lambda candidate: (factual, counterfactual, truth),
  )
  monkeypatch.setattr(
      instance_cli,
      "evaluate_qc",
      lambda *args, **kwargs: QCResult(True, (), {}),
  )

  def publish(root, candidate, factual_log, counterfactual_log, candidate_truth):
    published.append(
        (candidate, factual_log, counterfactual_log, candidate_truth)
    )
    return root / "instances" / candidate.instance_id

  monkeypatch.setattr(instance_cli, "_publish_instance", publish)
  exit_code = instance_cli.main([
      "--config", str(tmp_path / "ranges.yaml"),
      "--output", str(tmp_path / "output"),
      "--seed", "3",
      "--attempt-index", "0",
  ])

  assert exit_code == 0
  assert published == [(spec, factual, counterfactual, truth)]
  assert json.loads(capsys.readouterr().out)["status"] == "accepted"


@pytest.mark.parametrize(
    "script_name", ["generate_instance.py", "generate_dataset.py"]
)
def test_cli_files_run_directly_outside_the_repository(script_name, tmp_path):
  project_root = Path(__file__).resolve().parents[1]
  environment = dict(os.environ)
  environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")
  result = subprocess.run(
      [sys.executable, str(project_root / "scripts" / script_name), "--help"],
      cwd=tmp_path,
      env=environment,
      capture_output=True,
      text=True,
      timeout=60,
      check=False,
  )

  assert result.returncode == 0, result.stderr


def test_real_two_branch_candidate_publishes_and_roundtrips(tmp_path):
  output = tmp_path / "real"
  result = run_batch(_ranges(), output, 44, 1, 1)

  assert result["status"] == "complete"
  instance_id = result["selected_ids"][0]
  artifact = output / "instances" / instance_id
  manifest = json.loads((artifact / "instance_manifest.json").read_text())
  assert (artifact / "spec.json").exists()
  assert manifest["instance_id"] == instance_id
  factual, counterfactual, truth, provenance = read_paired_artifact(artifact)
  assert factual.branch == "factual"
  assert counterfactual.branch == "counterfactual"
  assert isinstance(truth, GroundTruth)
  assert provenance["target_id"] == "target"
  np.testing.assert_array_equal(factual.states, counterfactual.states)
