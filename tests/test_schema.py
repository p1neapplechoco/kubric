"""Tests for immutable intervention artifact schemas."""

import json
from dataclasses import dataclass, FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from interventions.schema import (
    CameraConfig,
    GraphEdgeDelta,
    GroundTruth,
    Intervention,
    ObjectConfig,
    SceneConfig,
    to_jsonable,
)
from interventions import schema as schema_module


def _object(object_id="ball", **overrides):
  values = {
      "object_id": object_id,
      "shape": "sphere",
      "size": 0.5,
      "mass": 1.0,
      "friction": 0.4,
      "restitution": 0.7,
      "position": (0.0, 0.0, 1.0),
      "quaternion": (1.0, 0.0, 0.0, 0.0),
      "linear_velocity": (0.0, 0.0, 0.0),
      "angular_velocity": (0.0, 0.0, 0.0),
      "static": False,
      "metadata": {"tags": ["round", "red"]},
  }
  values.update(overrides)
  return ObjectConfig(**values)


def _scene(**overrides):
  values = {
      "objects": (_object(),),
      "camera": CameraConfig(
          position=(4.0, -4.0, 3.0),
          look_at=(0.0, 0.0, 1.0),
          focal_length=35.0,
      ),
      "seed": 7,
      "scene_bounds": ((-5.0, -5.0, 0.0), (5.0, 5.0, 5.0)),
      "gravity": (0.0, 0.0, -9.81),
      "frame_range": (0, 25),
      "frame_rate": 24,
      "step_rate": 240,
  }
  values.update(overrides)
  return SceneConfig(**values)


def _edge(object_a="a", object_b="b", start_step=0, end_step=2, **payload):
  return {
      "object_a": object_a,
      "object_b": object_b,
      "start_step": start_step,
      "end_step": end_step,
      **payload,
  }


def test_object_config_normalizes_size_quaternion_and_metadata():
  config = _object(
      size=2,
      quaternion=(2.0, 0.0, 0.0, 0.0),
      metadata={"labels": {"z", "a"}},
  )

  assert config.size == (2.0, 2.0, 2.0)
  assert config.quaternion == (1.0, 0.0, 0.0, 0.0)
  assert config.metadata["labels"] == ("a", "z")
  assert config.schema_version == "1.0"
  with pytest.raises(FrozenInstanceError):
    config.mass = 2.0
  with pytest.raises(TypeError):
    config.metadata["new"] = "value"


def test_object_config_normalizes_extreme_finite_quaternion_safely():
  config = _object(quaternion=(1e308, 0.0, 0.0, 0.0))

  assert config.quaternion == (1.0, 0.0, 0.0, 0.0)
  assert np.linalg.norm(config.quaternion) == pytest.approx(1.0)


def test_schema_numeric_overflow_is_reported_as_value_error():
  oversized = Fraction(10**10000, 1)

  with pytest.raises(ValueError, match="finite"):
    _object(size=oversized)
  with pytest.raises(ValueError, match="finite"):
    to_jsonable(oversized)


def test_schema_rejects_integer_leaves_that_json_cannot_encode():
  oversized = 10**10000

  with pytest.raises(ValueError, match="JSON"):
    _object(metadata={"oversized": oversized})
  with pytest.raises(ValueError, match="JSON"):
    to_jsonable(oversized)


@pytest.mark.parametrize(
    "overrides",
    [
        {"object_id": ""},
        {"shape": "cone"},
        {"size": 0.0},
        {"size": (1.0, 2.0)},
        {"mass": 0.0},
        {"friction": -0.1},
        {"restitution": 1.1},
        {"position": (0.0, np.inf, 0.0)},
        {"quaternion": (0.0, 0.0, 0.0, 0.0)},
        {"linear_velocity": (0.0, 0.0)},
        {"static": 1},
        {"schema_version": "2.0"},
    ],
)
def test_object_config_rejects_invalid_values(overrides):
  with pytest.raises((TypeError, ValueError)):
    _object(**overrides)


def test_camera_config_validates_and_serializes():
  camera = CameraConfig(
      position=[4, -4, 3], look_at=[0, 0, 1], focal_length=50
  )

  assert camera.position == (4.0, -4.0, 3.0)
  assert camera.look_at == (0.0, 0.0, 1.0)
  assert camera.to_dict() == {
      "position": [4.0, -4.0, 3.0],
      "look_at": [0.0, 0.0, 1.0],
      "focal_length": 50.0,
      "schema_version": "1.0",
  }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": (0, 0), "look_at": (0, 0, 0), "focal_length": 35},
        {"position": (0, 0, 0), "look_at": (0, 0, 0), "focal_length": 35},
        {"position": (1, 0, 0), "look_at": (0, 0, 0), "focal_length": 0},
    ],
)
def test_camera_config_rejects_invalid_values(kwargs):
  with pytest.raises((TypeError, ValueError)):
    CameraConfig(**kwargs)


def test_scene_config_normalizes_objects_and_is_json_serializable():
  scene = _scene(objects=[_object("b"), _object("a")])

  assert isinstance(scene.objects, tuple)
  assert scene.frame_range == (0, 25)
  payload = scene.to_dict()
  assert payload["schema_version"] == "1.0"
  assert payload["objects"][0]["object_id"] == "b"
  assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "overrides",
    [
        {"objects": (_object("same"), _object("same"))},
        {"scene_bounds": ((1, -1, 0), (1, 1, 2))},
        {"scene_bounds": ((-1, -1, 0), (1, 1, np.inf))},
        {"objects": (_object(position=(8, 0, 1)),)},
        {"frame_range": (4, 4)},
        {"frame_range": (0.5, 4)},
        {"frame_rate": 0},
        {"step_rate": 100},
        {"seed": -1},
        {"gravity": (0, 0)},
        {"camera": "camera"},
    ],
)
def test_scene_config_rejects_invalid_values(overrides):
  with pytest.raises((TypeError, ValueError)):
    _scene(**overrides)


def test_intervention_normalizes_and_serializes():
  intervention = Intervention(
      target_id="ball",
      recipe="retime",
      magnitude=0.25,
      time_window=[3, 9],
      push_mass=2,
      metadata={"reason": "counterfactual"},
  )

  assert intervention.time_window == (3.0, 9.0)
  assert intervention.push_mass == 2.0
  assert intervention.to_dict()["recipe"] == "retime"
  assert intervention.schema_version == "1.0"


def test_intervention_defaults_and_serializes_push_mass():
  intervention = Intervention(
      target_id="ball",
      recipe="remove_collision",
      magnitude=0.1,
      time_window=(0, 10),
  )

  payload = intervention.to_dict()
  assert intervention.push_mass == 1.0
  assert payload["push_mass"] == 1.0
  assert json.loads(json.dumps(payload))["push_mass"] == 1.0


def test_intervention_push_mass_is_immutable():
  intervention = Intervention(
      target_id="ball",
      recipe="remove_collision",
      magnitude=0.1,
      time_window=(0, 10),
  )

  with pytest.raises(FrozenInstanceError):
    intervention.push_mass = 2.0


@pytest.mark.parametrize(
    "push_mass,error,message",
    [
        (True, TypeError, "push_mass must be a real number"),
        (0, ValueError, "push_mass must be positive"),
        (-1, ValueError, "push_mass must be positive"),
        (np.nan, ValueError, "push_mass must be finite"),
        (np.inf, ValueError, "push_mass must be finite"),
        (complex(1, 0), TypeError, "push_mass must be a real number"),
        ("1.0", TypeError, "push_mass must be a real number"),
    ],
)
def test_intervention_rejects_invalid_push_mass(push_mass, error, message):
  with pytest.raises(error, match=message):
    Intervention(
        target_id="ball",
        recipe="remove_collision",
        magnitude=0.1,
        time_window=(0, 10),
        push_mass=push_mass,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_id": ""},
        {"recipe": "teleport"},
        {"magnitude": -0.1},
        {"magnitude": np.nan},
        {"time_window": (2, 2)},
        {"time_window": (-1, 2)},
        {"time_window": (1, np.inf)},
    ],
)
def test_intervention_rejects_invalid_values(overrides):
  values = {
      "target_id": "ball",
      "recipe": "remove_collision",
      "magnitude": 0.1,
      "time_window": (0, 10),
  }
  values.update(overrides)
  with pytest.raises((TypeError, ValueError)):
    Intervention(**values)


def test_graph_delta_and_ground_truth_are_deterministic_and_json_safe():
  delta = GraphEdgeDelta(
      added=({
          "object_a": "b",
          "object_b": "a",
          "start_step": 2,
          "end_step": 4,
      },),
      removed=[{
          "object_a": "c",
          "object_b": "b",
          "start_step": 1,
          "end_step": 2,
      }],
      changed=({
          "object_a": "d",
          "object_b": "a",
          "start_step": 5,
          "end_step": 7,
          "factual": {"force": 1.0},
          "counterfactual": {"force": 2.0},
      },),
  )
  truth = GroundTruth(
      graph_delta=delta,
      hard_affected={"c", "a"},
      soft_affected=["d", "b", "b"],
      propagation_path={"c": ["c", "a"], "a": ["a"]},
  )

  assert truth.hard_affected == ("a", "c")
  assert truth.soft_affected == ("b", "d")
  assert tuple(truth.propagation_path) == ("a", "c")
  assert truth.propagation_path["c"] == ("c", "a")
  assert delta.added[0]["object_a"] == "a"
  assert delta.added[0]["object_b"] == "b"
  with pytest.raises(TypeError):
    delta.added[0]["object_a"] = "x"
  payload = truth.to_dict()
  assert payload["graph_delta"]["changed"][0]["counterfactual"] == {
      "force": 2.0
  }
  assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "field,record",
    [
        ("added", 7),
        ("removed", {"object_a": "a", "object_b": "b"}),
        (
            "changed",
            {
                "object_a": "a",
                "object_b": "b",
                "start_step": True,
                "end_step": 2,
            },
        ),
        (
            "added",
            {
                "object_a": "a",
                "object_b": "b",
                "start_step": -1,
                "end_step": 2,
            },
        ),
        (
            "removed",
            {
                "object_a": "a",
                "object_b": "b",
                "start_step": 2,
                "end_step": 2,
            },
        ),
    ],
)
def test_graph_delta_rejects_malformed_temporal_edges(field, record):
  with pytest.raises((TypeError, ValueError)):
    GraphEdgeDelta(**{field: (record,)})


def test_graph_delta_sorts_and_deduplicates_canonical_edge_identities():
  delta = GraphEdgeDelta(added=(
      _edge("z", "y", 4, 6, force=2.0),
      _edge("b", "a", 2, 3, force=1.0),
      _edge("a", "b", 2, 3, force=1.0),
  ))

  identities = [
      (edge["object_a"], edge["object_b"], edge["start_step"], edge["end_step"])
      for edge in delta.added
  ]
  assert identities == [("a", "b", 2, 3), ("y", "z", 4, 6)]


def test_graph_delta_rejects_conflicting_payload_for_one_identity():
  with pytest.raises(ValueError, match="conflicting"):
    GraphEdgeDelta(added=(
        _edge("a", "b", force=1.0),
        _edge("b", "a", force=2.0),
    ))


@pytest.mark.parametrize(
    "left_bucket,right_bucket",
    [("added", "removed"), ("added", "changed"), ("removed", "changed")],
)
def test_graph_delta_rejects_identity_shared_across_buckets(
    left_bucket, right_bucket
):
  values = {left_bucket: (_edge(),), right_bucket: (_edge("b", "a"),)}

  with pytest.raises(ValueError, match="multiple"):
    GraphEdgeDelta(**values)


def test_schema_construction_rejects_nested_dataclass_values():
  @dataclass
  class MutableMetadata:
    values: list

  nested = MutableMetadata(values=[1])
  with pytest.raises(TypeError, match="dataclass"):
    _object(metadata={"nested": nested})
  with pytest.raises(TypeError, match="dataclass"):
    GraphEdgeDelta(added=({
        "object_a": "a",
        "object_b": "b",
        "start_step": 0,
        "end_step": 1,
        "payload": nested,
    },))


@pytest.mark.parametrize(
    "build",
    [
        lambda: _object(metadata={"valid": 1, 2: "invalid"}),
        lambda: GroundTruth(
            graph_delta=GraphEdgeDelta(),
            propagation_path={"valid": ["a"], 2: ["b"]},
        ),
        lambda: to_jsonable({"valid": 1, 2: "invalid"}),
    ],
)
def test_schema_mappings_reject_non_string_keys_consistently(build):
  with pytest.raises(ValueError, match="keys must be strings"):
    build()


@pytest.mark.parametrize(
    "path",
    [
        {"a", "b"},
        frozenset(("a", "b")),
        iter(("a", "b")),
        np.array(["a", "b"]),
    ],
)
def test_ground_truth_rejects_unordered_or_non_sequence_propagation_paths(path):
  with pytest.raises(ValueError, match="ordered sequence"):
    GroundTruth(
        graph_delta=GraphEdgeDelta(),
        propagation_path={"b": path},
    )


def test_intervention_recipes_have_one_authoritative_constant():
  assert schema_module.INTERVENTION_RECIPES == frozenset((
      "remove_collision",
      "create_collision",
      "retime",
      "break_contact",
      "maintain_contact",
  ))


def test_ground_truth_rejects_overlap_and_invalid_delta():
  with pytest.raises(ValueError):
    GroundTruth(
        graph_delta=GraphEdgeDelta(),
        hard_affected={"a"},
        soft_affected={"a"},
        propagation_path={},
    )
  with pytest.raises(TypeError):
    GroundTruth(
        graph_delta={},
        hard_affected=(),
        soft_affected=(),
        propagation_path={},
    )


def test_to_jsonable_handles_nested_numpy_values_and_sorted_sets():
  value = {
      "array": np.array([1.0, 2.0]),
      "integer": np.int64(3),
      "members": {"z", "a"},
  }

  assert to_jsonable(value) == {
      "array": [1.0, 2.0],
      "integer": 3,
      "members": ["a", "z"],
  }
  with pytest.raises(TypeError):
    to_jsonable(object())
