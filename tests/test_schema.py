"""Tests for immutable intervention artifact schemas."""

import json
from dataclasses import dataclass, FrozenInstanceError

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
      metadata={"reason": "counterfactual"},
  )

  assert intervention.time_window == (3.0, 9.0)
  assert intervention.to_dict()["recipe"] == "retime"
  assert intervention.schema_version == "1.0"


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
      propagation_path={"c": {"c", "a"}, "a": ["a"]},
  )

  assert truth.hard_affected == ("a", "c")
  assert truth.soft_affected == ("b", "d")
  assert tuple(truth.propagation_path) == ("a", "c")
  assert truth.propagation_path["c"] == ("a", "c")
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
