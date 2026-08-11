"""Dataset sampling, QC, balancing, split, and journal tests."""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
from interventions.logging import ContactRecord, SimulationLog, read_simulation_log
from interventions.schema import (
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    ObjectConfig,
    SceneConfig,
)


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


def _states(ids=("floor", "target", "ball"), steps=3):
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
    ids=("floor", "target", "ball"),
    steps=(0, 1, 2),
    contacts=(),
):
  if states is None:
    states = _states(ids, len(steps))
  return SimulationLog(
      branch=branch,
      object_ids=ids,
      steps=steps,
      states=states,
      contacts=contacts,
      step_rate=240,
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


def test_load_ranges_copies_and_freezes_yaml(tmp_path):
  config = tmp_path / "ranges.yaml"
  import yaml

  config.write_text(yaml.safe_dump(_ranges(), sort_keys=True), encoding="utf-8")
  loaded = load_ranges(config)

  assert loaded["scene"]["step_rate"] == 240
  with pytest.raises(TypeError):
    loaded["scene"]["step_rate"] = 10


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
    changed[0, 1, 0] += 0.01
    counterfactual = _log("counterfactual", states=changed)
  elif case == "prefix_contact":
    contact = ContactRecord(0, "target", "ball", (0, 0, 0), (1, 0, 0), 1)
    factual = _log("factual", contacts=(contact,))
  elif case == "nonfinite":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, 2, 0] = np.nan
    counterfactual = SimpleNamespace(
        **{**counterfactual.__dict__, "states": changed}
    )
  elif case == "linear":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, 2, 7] = 101
    counterfactual = _log("counterfactual", states=changed)
  elif case == "angular":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, 2, 10] = 101
    counterfactual = _log("counterfactual", states=changed)
  elif case == "bounds":
    changed = np.array(counterfactual.states, copy=True)
    changed[2, 1, 0] = 6
    counterfactual = _log("counterfactual", states=changed)
  elif case == "clip":
    scene = _scene()
    support = replace(
        scene.objects[0], object_id="wall", position=(2, 0, 0.2),
        size=(0.2, 0.2, 0.2), metadata={},
    )
    scene = replace(scene, objects=(support,) + scene.objects[1:])
    spec = _spec(scene=scene)
    changed = np.array(counterfactual.states, copy=True)
    changed[2, 1, :3] = (2, 0, 0.2)
    counterfactual = _log("counterfactual", states=changed)
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
  state_changed[0, 1, 0] = 0.1
  counterfactual = _log("counterfactual", states=state_changed)
  factual = _log("factual", contacts=(contact,))

  result = _qc(factual=factual, counterfactual=counterfactual)

  assert result.reason_codes == ("twin_prefix_mismatch",)


def test_qc_collects_sorted_reasons_metrics_and_honors_clip_exemption():
  changed = np.array(_log("counterfactual").states, copy=True)
  changed[2, 2, 7] = 200
  changed[2, 2, 10] = 300
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

  def fake_publish(root, spec, factual, counterfactual):
    artifact = root / "instances" / spec.instance_id
    artifact.mkdir()
    (artifact / "instance_manifest.json").write_text("{}\n", encoding="utf-8")
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


def test_real_two_branch_candidate_publishes_and_roundtrips(tmp_path):
  output = tmp_path / "real"
  result = run_batch(_ranges(), output, 44, 1, 1)

  assert result["status"] == "complete"
  instance_id = result["selected_ids"][0]
  artifact = output / "instances" / instance_id
  manifest = json.loads((artifact / "instance_manifest.json").read_text())
  assert (artifact / "spec.json").exists()
  assert manifest["instance_id"] == instance_id
  factual = read_simulation_log(artifact / "factual")
  counterfactual = read_simulation_log(artifact / "counterfactual")
  assert factual.branch == "factual"
  assert counterfactual.branch == "counterfactual"
  np.testing.assert_array_equal(factual.states, counterfactual.states)
