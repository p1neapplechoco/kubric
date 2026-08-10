"""Tests for simulator-independent contact and state logging."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from interventions.logging import (
    ANGULAR_VELOCITY_SLICE,
    LINEAR_VELOCITY_SLICE,
    POSITION_SLICE,
    QUATERNION_SLICE,
    ContactLogger,
    ContactRecord,
    SimulationLog,
    read_simulation_log,
    state_index,
    write_simulation_log,
)


def _bullet_contact(
    body_a,
    body_b,
    *,
    position=(1.0, 2.0, 3.0),
    normal=(0.0, 1.0, 0.0),
    distance=-0.01,
    force=4.0,
):
  return (
      0,
      body_a,
      body_b,
      -1,
      -1,
      (9.0, 9.0, 9.0),
      position,
      normal,
      distance,
      force,
      0.0,
      (1.0, 0.0, 0.0),
      0.0,
      (0.0, 0.0, 1.0),
  )


def _states(num_steps=3, num_objects=2):
  states = np.zeros((num_steps, num_objects, 13), dtype=float)
  states[:, :, 3] = 1.0
  return states


def _simulation_log(**overrides):
  values = {
      "branch": "factual",
      "object_ids": ("b", "a"),
      "steps": (2, 3, 5),
      "states": _states(),
      "contacts": (
          ContactRecord(
              step=3,
              object_a="a",
              object_b="b",
              position=(0, 0, 0),
              normal=(1, 0, 0),
              normal_force=2,
          ),
      ),
      "step_rate": 240,
      "commanded_path": np.array(
          [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], dtype=float
      ),
      "metadata": {"seed": 7, "labels": ["demo"]},
  }
  values.update(overrides)
  return SimulationLog(**values)


def test_contact_record_canonicalizes_endpoints_and_flips_normal():
  record = ContactRecord(
      step=7,
      object_a="z",
      object_b="a",
      position=[1, 2, 3],
      normal=[0, 1, 0],
      normal_force=2.5,
      contact_distance=-0.01,
  )

  assert (record.object_a, record.object_b) == ("a", "z")
  assert record.normal == (0.0, -1.0, 0.0)
  assert record.to_dict() == {
      "step": 7,
      "object_a": "a",
      "object_b": "z",
      "position": [1.0, 2.0, 3.0],
      "normal": [0.0, -1.0, 0.0],
      "normal_force": 2.5,
      "contact_distance": -0.01,
      "schema_version": "1.0",
  }
  with pytest.raises(FrozenInstanceError):
    record.step = 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"step": -1},
        {"step": True},
        {"object_a": "same", "object_b": "same"},
        {"normal": (0, np.nan, 0)},
        {"normal_force": -1},
        {"contact_distance": np.inf},
        {"schema_version": "2.0"},
    ],
)
def test_contact_record_rejects_invalid_values(overrides):
  values = {
      "step": 0,
      "object_a": "a",
      "object_b": "b",
      "position": (0, 0, 0),
      "normal": (1, 0, 0),
      "normal_force": 0,
  }
  values.update(overrides)

  with pytest.raises((TypeError, ValueError)):
    ContactRecord(**values)


def test_contact_logger_parses_bullet_fields_filters_and_sorts_records():
  logger = ContactLogger({4: "z", 2: "a", 8: "unknown"}, 240, force_epsilon=0.5)

  accepted = logger.log(
      3,
      (
          _bullet_contact(4, 2, position=(2, 0, 0), normal=(0, 1, 0), force=3),
          _bullet_contact(2, 4, position=(1, 0, 0), normal=(1, 0, 0), force=2),
          _bullet_contact(4, 2, force=0.5),
          _bullet_contact(999, 2, force=10),
          _bullet_contact(4, 4, force=10),
      ),
  )

  assert isinstance(accepted, tuple)
  assert len(accepted) == 2
  assert [record.position for record in accepted] == [
      (1.0, 0.0, 0.0),
      (2.0, 0.0, 0.0),
  ]
  assert accepted[0].normal == (1.0, 0.0, 0.0)
  assert accepted[1].normal == (0.0, -1.0, 0.0)
  assert logger.records == accepted
  logger.clear()
  assert logger.records == ()


def test_contact_logger_supports_callable_mapping_and_validates_raw_tuples():
  names = {1: "a", 2: "b"}
  logger = ContactLogger(lambda body: names.get(body), step_rate=60)

  assert len(logger.log(0, [_bullet_contact(1, 2)])) == 1
  with pytest.raises(ValueError, match="14"):
    logger.log(1, [(0, 1, 2)])
  with pytest.raises((TypeError, ValueError)):
    logger.log(1, [_bullet_contact(1, 2, force=np.nan)])


def test_simulation_log_defensively_copies_and_freezes_arrays_and_metadata():
  source_states = _states()
  source_path = np.zeros((3, 7), dtype=float)
  source_path[:, 3] = 1.0
  source_metadata = {"nested": [1, 2]}

  log = _simulation_log(
      states=source_states,
      commanded_path=source_path,
      metadata=source_metadata,
  )
  source_states[0, 0, 0] = 99
  source_path[0, 0] = 99
  source_metadata["nested"].append(3)

  assert log.states[0, 0, 0] == 0
  assert log.commanded_path[0, 0] == 0
  assert log.metadata["nested"] == (1, 2)
  assert not log.states.flags.writeable
  assert not log.commanded_path.flags.writeable
  with pytest.raises(ValueError):
    log.states[0, 0, 0] = 1
  with pytest.raises(TypeError):
    log.metadata["new"] = 1
  assert state_index("position") == POSITION_SLICE == slice(0, 3)
  assert state_index("quaternion") == QUATERNION_SLICE == slice(3, 7)
  assert state_index("linear_velocity") == LINEAR_VELOCITY_SLICE == slice(7, 10)
  assert state_index("angular_velocity") == ANGULAR_VELOCITY_SLICE == slice(10, 13)


@pytest.mark.parametrize(
    "overrides",
    [
        {"branch": ""},
        {"object_ids": ("a", "a")},
        {"steps": (0, 0, 1)},
        {"steps": (0, 1), "states": _states()},
        {"states": np.full((3, 2, 13), np.nan)},
        {"states": np.zeros((3, 2, 13))},
        {"step_rate": 0},
        {"commanded_path": np.zeros((2, 3))},
        {
            "contacts": (
                ContactRecord(9, "a", "b", (0, 0, 0), (1, 0, 0), 1),
            )
        },
    ],
)
def test_simulation_log_rejects_invalid_alignment_and_values(overrides):
  with pytest.raises((TypeError, ValueError)):
    _simulation_log(**overrides)


def test_simulation_log_roundtrip_and_overwrite_refusal(tmp_path):
  log = _simulation_log()
  directory = tmp_path / "artifact"

  write_simulation_log(log, directory)
  restored = read_simulation_log(directory)

  assert restored.branch == log.branch
  assert restored.object_ids == log.object_ids
  assert restored.steps == log.steps
  assert restored.contacts == log.contacts
  assert restored.step_rate == log.step_rate
  assert restored.metadata == log.metadata
  np.testing.assert_array_equal(restored.states, log.states)
  np.testing.assert_array_equal(restored.commanded_path, log.commanded_path)
  assert (directory / "states.npy").read_bytes()[:6] == b"\x93NUMPY"
  metadata = json.loads((directory / "metadata.json").read_text())
  assert metadata["schema_version"] == "1.0"
  assert json.loads((directory / "contacts.jsonl").read_text()) == log.contacts[0].to_dict()
  with pytest.raises(FileExistsError):
    write_simulation_log(log, directory)

  replacement = _simulation_log(branch="counterfactual", contacts=())
  write_simulation_log(replacement, directory, overwrite=True)
  assert read_simulation_log(directory).branch == "counterfactual"


def test_read_simulation_log_rejects_corrupt_artifacts(tmp_path):
  directory = tmp_path / "artifact"
  write_simulation_log(_simulation_log(), directory)
  metadata_path = directory / "metadata.json"
  payload = json.loads(metadata_path.read_text())
  payload["schema_version"] = "999"
  metadata_path.write_text(json.dumps(payload))

  with pytest.raises(ValueError, match="schema_version"):
    read_simulation_log(directory)
