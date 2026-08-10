"""Tests for simulator-independent contact and state logging."""

import hashlib
import io
import json
import math
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import interventions.logging as logging_module
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


def _timedelta_states(num_steps=3, num_objects=2):
  states = np.zeros(
      (num_steps, num_objects, 13), dtype="timedelta64[ns]"
  )
  states[:, :, 3] = np.timedelta64(1, "ns")
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


_PAYLOAD_FILENAMES = ("contacts.jsonl", "states.npy", "metadata.json")


def _manifest_for_payloads(payloads):
  files = {}
  for filename in _PAYLOAD_FILENAMES:
    payload = payloads[filename]
    files[filename] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
  generation_material = "".join(
      files[filename]["sha256"] for filename in _PAYLOAD_FILENAMES
  ).encode("ascii")
  generation = hashlib.sha256(generation_material).hexdigest()
  for filename in _PAYLOAD_FILENAMES:
    files[filename]["path"] = "generations/{}/{}".format(
        generation, filename
    )
  return {
      "files": files,
      "generation": generation,
      "schema_version": "1.0",
  }


def _read_manifest(directory):
  return json.loads((directory / "manifest.json").read_text())


def _payload_path(directory, filename):
  return directory / _read_manifest(directory)["files"][filename]["path"]


def _expected_manifest(directory):
  manifest = _read_manifest(directory)
  payloads = {
      filename: (
          directory / "generations" / manifest["generation"] / filename
      ).read_bytes()
      for filename in _PAYLOAD_FILENAMES
  }
  return _manifest_for_payloads(payloads)


def _refresh_manifest(directory):
  manifest = _read_manifest(directory)
  old_generation = manifest["generation"]
  old_directory = directory / "generations" / old_generation
  payloads = {
      filename: (old_directory / filename).read_bytes()
      for filename in _PAYLOAD_FILENAMES
  }
  refreshed = _manifest_for_payloads(payloads)
  new_generation = refreshed["generation"]
  if new_generation != old_generation:
    old_directory.rename(directory / "generations" / new_generation)
  payload = json.dumps(
      refreshed,
      sort_keys=True,
      separators=(",", ":"),
  ) + "\n"
  (directory / "manifest.json").write_text(payload)


def _artifact_snapshot(directory):
  return {
      str(path.relative_to(directory)): (
          None if path.is_dir() else path.read_bytes()
      )
      for path in sorted(directory.rglob("*"))
  }


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


def test_contact_record_canonicalizes_signed_zero_after_endpoint_reversal():
  canonical = ContactRecord(
      step=0,
      object_a="a",
      object_b="b",
      position=(-0.0, 0.0, -0.0),
      normal=(0.0, -0.0, 1.0),
      normal_force=-0.0,
      contact_distance=-0.0,
  )
  reversed_record = ContactRecord(
      step=0,
      object_a="b",
      object_b="a",
      position=(0.0, -0.0, 0.0),
      normal=(-0.0, 0.0, -1.0),
      normal_force=0.0,
      contact_distance=0.0,
  )

  canonical_bytes = json.dumps(
      canonical.to_dict(), sort_keys=True, separators=(",", ":")
  ).encode()
  reversed_bytes = json.dumps(
      reversed_record.to_dict(), sort_keys=True, separators=(",", ":")
  ).encode()
  assert canonical_bytes == reversed_bytes
  zero_values = (
      canonical.position
      + canonical.normal[:2]
      + (canonical.normal_force, canonical.contact_distance)
  )
  assert all(math.copysign(1.0, value) == 1.0 for value in zero_values)


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
  with pytest.raises(ValueError):
    log.states.setflags(write=True)
  with pytest.raises(ValueError):
    log.commanded_path.setflags(write=True)
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"object_ids": {"a", "b"}},
        {"object_ids": frozenset(("a", "b"))},
        {"steps": {2, 3, 5}},
        {"steps": frozenset((2, 3, 5))},
    ],
)
def test_simulation_log_rejects_unordered_object_and_step_associations(overrides):
  with pytest.raises((TypeError, ValueError), match="ordered"):
    _simulation_log(**overrides)


@pytest.mark.parametrize(
    "states",
    [
        _states().astype(str),
        _states().astype(object),
    ],
)
def test_simulation_log_rejects_nonnumeric_source_dtypes(states):
  with pytest.raises((TypeError, ValueError), match="numeric|real"):
    _simulation_log(states=states)


def test_simulation_log_rejects_timedelta_states():
  with pytest.raises((TypeError, ValueError), match="numeric|real"):
    _simulation_log(states=_timedelta_states())


def test_simulation_log_rejects_nonnumeric_commanded_path_dtype():
  path = np.array(
      [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], dtype=float
  ).astype(str)

  with pytest.raises((TypeError, ValueError), match="numeric|real"):
    _simulation_log(commanded_path=path)


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
  assert _payload_path(directory, "states.npy").read_bytes()[:6] == b"\x93NUMPY"
  metadata = json.loads(_payload_path(directory, "metadata.json").read_text())
  assert metadata["schema_version"] == "1.0"
  assert json.loads(_payload_path(directory, "contacts.jsonl").read_text()) == (
      log.contacts[0].to_dict()
  )
  manifest = _read_manifest(directory)
  assert manifest == _expected_manifest(directory)
  generation_directory = directory / "generations" / manifest["generation"]
  assert sorted(path.name for path in generation_directory.iterdir()) == list(
      sorted(_PAYLOAD_FILENAMES)
  )
  before_refusal = _artifact_snapshot(directory)
  with pytest.raises(FileExistsError):
    write_simulation_log(log, directory)
  assert before_refusal == _artifact_snapshot(directory)
  assert sorted(path.name for path in directory.iterdir()) == [
      ".publish.lock",
      "generations",
      "manifest.json",
  ]

  replacement = _simulation_log(branch="counterfactual", contacts=())
  write_simulation_log(replacement, directory, overwrite=True)
  assert read_simulation_log(directory).branch == "counterfactual"
  new_generation = _read_manifest(directory)["generation"]
  assert new_generation != manifest["generation"]
  assert sorted(
      path.name for path in (directory / "generations").iterdir()
  ) == sorted((manifest["generation"], new_generation))


def test_states_npy_is_deterministic_across_input_memory_layouts(tmp_path):
  values = _states()
  c_log = _simulation_log(states=np.ascontiguousarray(values), contacts=())
  f_log = _simulation_log(states=np.asfortranarray(values), contacts=())

  c_directory = write_simulation_log(c_log, tmp_path / "c")
  f_directory = write_simulation_log(f_log, tmp_path / "f")

  assert _payload_path(c_directory, "states.npy").read_bytes() == (
      _payload_path(f_directory, "states.npy").read_bytes()
  )
  assert _read_manifest(c_directory)["generation"] == (
      _read_manifest(f_directory)["generation"]
  )


def test_identical_overwrite_keeps_every_artifact_byte_identical(tmp_path):
  directory = write_simulation_log(_simulation_log(), tmp_path / "artifact")
  before = _artifact_snapshot(directory)

  write_simulation_log(_simulation_log(), directory, overwrite=True)

  assert before == _artifact_snapshot(directory)
  assert len(tuple((directory / "generations").iterdir())) == 1


def test_signed_zero_and_contact_permutations_produce_identical_artifacts(tmp_path):
  positive_states = _states()
  negative_states = _states()
  negative_states[negative_states == 0.0] = -0.0
  positive_path = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], dtype=float)
  negative_path = positive_path.copy()
  negative_path[negative_path == 0.0] = -0.0
  canonical_contact = ContactRecord(
      3, "a", "b", (-0.0, 0.0, -0.0), (0.0, -0.0, 1.0), 1.0, -0.0
  )
  reversed_contact = ContactRecord(
      3, "b", "a", (0.0, -0.0, 0.0), (-0.0, 0.0, -1.0), 1.0, 0.0
  )
  positive = _simulation_log(
      states=positive_states,
      commanded_path=positive_path,
      contacts=(canonical_contact,),
      metadata={"zero": 0.0},
  )
  negative = _simulation_log(
      states=negative_states,
      commanded_path=negative_path,
      contacts=(reversed_contact,),
      metadata={"zero": -0.0},
  )

  positive_directory = write_simulation_log(positive, tmp_path / "positive")
  negative_directory = write_simulation_log(negative, tmp_path / "negative")

  assert (positive_directory / "manifest.json").read_bytes() == (
      negative_directory / "manifest.json"
  ).read_bytes()
  for filename in _PAYLOAD_FILENAMES:
    assert _payload_path(positive_directory, filename).read_bytes() == (
        _payload_path(negative_directory, filename).read_bytes()
    )


def test_write_simulation_log_publishes_generation_then_manifest_and_fsyncs(
    tmp_path, monkeypatch
):
  renames = []
  replacements = []
  publication_events = []
  directory_fsyncs = []
  real_rename = logging_module.os.rename
  real_replace = logging_module.os.replace
  real_fsync = logging_module.os.fsync

  def tracked_rename(source, destination):
    renames.append((Path(source), Path(destination)))
    publication_events.append("generation")
    return real_rename(source, destination)

  def tracked_replace(source, destination):
    replacements.append((Path(source), Path(destination)))
    publication_events.append("manifest")
    return real_replace(source, destination)

  def tracked_fsync(file_descriptor):
    if stat.S_ISDIR(logging_module.os.fstat(file_descriptor).st_mode):
      directory_fsyncs.append(file_descriptor)
    return real_fsync(file_descriptor)

  monkeypatch.setattr(logging_module.os, "rename", tracked_rename)
  monkeypatch.setattr(logging_module.os, "replace", tracked_replace)
  monkeypatch.setattr(logging_module.os, "fsync", tracked_fsync)
  directory = tmp_path / "artifact"

  write_simulation_log(_simulation_log(), directory)

  assert len(renames) == 1
  temporary_generation, final_generation = renames[0]
  assert temporary_generation.parent == directory / "generations"
  assert temporary_generation.name.startswith(".tmp-")
  assert final_generation.parent == directory / "generations"
  assert final_generation.name == _read_manifest(directory)["generation"]
  assert replacements == [
      (replacements[0][0], directory / "manifest.json")
  ]
  assert replacements[0][0].parent == directory
  assert replacements[0][0].name.startswith(".manifest.")
  assert publication_events == ["generation", "manifest"]
  # Temporary generation, generations/, artifact root, and newly created
  # artifact root's parent are all durably synchronized.
  assert len(directory_fsyncs) >= 4


@pytest.mark.parametrize("failure_boundary", ["generation", "manifest"])
def test_publication_failure_during_overwrite_keeps_old_generation_readable(
    tmp_path, monkeypatch, failure_boundary
):
  old_log = _simulation_log(branch="old", contacts=())
  directory = write_simulation_log(old_log, tmp_path / "artifact")
  old_manifest = (directory / "manifest.json").read_bytes()
  new_states = _states()
  new_states[-1, 0, 0] = 1.0
  new_log = _simulation_log(branch="new", states=new_states, contacts=())
  real_rename = logging_module.os.rename
  real_replace = logging_module.os.replace

  def fail_generation(source, destination):
    if failure_boundary == "generation":
      raise OSError("injected publication failure")
    return real_rename(source, destination)

  def fail_manifest(source, destination):
    if failure_boundary == "manifest":
      raise OSError("injected publication failure")
    return real_replace(source, destination)

  monkeypatch.setattr(logging_module.os, "rename", fail_generation)
  monkeypatch.setattr(logging_module.os, "replace", fail_manifest)
  with pytest.raises(OSError, match="injected"):
    write_simulation_log(new_log, directory, overwrite=True)

  assert (directory / "manifest.json").read_bytes() == old_manifest
  assert read_simulation_log(directory) == old_log
  assert not any(
      path.name.startswith(".tmp-")
      for path in (directory / "generations").iterdir()
  )


@pytest.mark.parametrize("failure_boundary", ["generation", "manifest"])
def test_initial_publication_failure_leaves_no_readable_manifest(
    tmp_path, monkeypatch, failure_boundary
):
  directory = tmp_path / "artifact"
  real_rename = logging_module.os.rename
  real_replace = logging_module.os.replace

  def fail_generation(source, destination):
    if failure_boundary == "generation":
      raise OSError("injected publication failure")
    return real_rename(source, destination)

  def fail_manifest(source, destination):
    if failure_boundary == "manifest":
      raise OSError("injected publication failure")
    return real_replace(source, destination)

  monkeypatch.setattr(logging_module.os, "rename", fail_generation)
  monkeypatch.setattr(logging_module.os, "replace", fail_manifest)
  with pytest.raises(OSError, match="injected"):
    write_simulation_log(_simulation_log(), directory)

  assert not (directory / "manifest.json").exists()
  with pytest.raises(ValueError, match="corrupt|incomplete|manifest"):
    read_simulation_log(directory)


def test_concurrent_overwrites_publish_one_complete_generation(tmp_path):
  directory = write_simulation_log(
      _simulation_log(branch="old", contacts=()), tmp_path / "artifact"
  )
  first_states = _states()
  first_states[-1, 0, 0] = 1.0
  second_states = _states()
  second_states[-1, 0, 0] = 2.0
  first = _simulation_log(branch="first", states=first_states, contacts=())
  second = _simulation_log(branch="second", states=second_states, contacts=())
  barrier = threading.Barrier(2)

  def publish(log):
    barrier.wait()
    write_simulation_log(log, directory, overwrite=True)

  with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(publish, log) for log in (first, second)]
    for future in futures:
      future.result()

  restored = read_simulation_log(directory)
  assert restored == first or restored == second
  assert _read_manifest(directory) == _expected_manifest(directory)
  assert not any(
      path.name.startswith(".tmp-")
      for path in (directory / "generations").iterdir()
  )


def test_concurrent_non_overwriting_publishers_have_one_winner(tmp_path):
  directory = tmp_path / "artifact"
  first = _simulation_log(branch="first", contacts=())
  second = _simulation_log(branch="second", contacts=())
  barrier = threading.Barrier(2)

  def publish(log):
    barrier.wait()
    try:
      write_simulation_log(log, directory)
    except FileExistsError:
      return ("exists", log.branch)
    return ("written", log.branch)

  with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(publish, (first, second)))

  assert sorted(result for result, _ in results) == ["exists", "written"]
  winner = next(branch for result, branch in results if result == "written")
  assert read_simulation_log(directory).branch == winner


def test_read_simulation_log_requires_and_validates_manifest(tmp_path):
  directory = write_simulation_log(_simulation_log(), tmp_path / "artifact")
  assert _read_manifest(directory) == _expected_manifest(directory)
  (directory / "manifest.json").unlink()

  with pytest.raises(ValueError, match="corrupt|incomplete|manifest"):
    read_simulation_log(directory)


@pytest.mark.parametrize("filename", _PAYLOAD_FILENAMES)
def test_read_simulation_log_rejects_manifest_payload_mismatch(tmp_path, filename):
  directory = write_simulation_log(_simulation_log(), tmp_path / filename)
  with _payload_path(directory, filename).open("ab") as stream:
    stream.write(b"tampered")

  with pytest.raises(ValueError, match="corrupt|integrity|manifest"):
    read_simulation_log(directory)


@pytest.mark.parametrize("corruption", ["trailing", "concatenated"])
def test_read_simulation_log_rejects_bytes_after_npy_payload(tmp_path, corruption):
  directory = write_simulation_log(_simulation_log(), tmp_path / corruption)
  states_path = _payload_path(directory, "states.npy")
  if corruption == "trailing":
    extra = b"trailing bytes"
  else:
    stream = io.BytesIO()
    np.save(stream, _states(), allow_pickle=False)
    extra = stream.getvalue()
  with states_path.open("ab") as stream:
    stream.write(extra)
  _refresh_manifest(directory)

  with pytest.raises(ValueError, match="trailing|payload"):
    read_simulation_log(directory)


@pytest.mark.parametrize("dtype", [str, object])
def test_read_simulation_log_rejects_nonnumeric_or_pickled_states(
    tmp_path, dtype
):
  directory = write_simulation_log(_simulation_log(), tmp_path / "artifact")
  states_path = _payload_path(directory, "states.npy")
  values = _states().astype(dtype)
  with states_path.open("wb") as stream:
    np.save(stream, values, allow_pickle=dtype is object)
  _refresh_manifest(directory)

  with pytest.raises(ValueError, match="numeric|object|NumPy"):
    read_simulation_log(directory)


def test_read_simulation_log_rejects_timedelta_states(tmp_path):
  directory = write_simulation_log(_simulation_log(), tmp_path / "artifact")
  with _payload_path(directory, "states.npy").open("wb") as stream:
    np.save(stream, _timedelta_states(), allow_pickle=False)
  _refresh_manifest(directory)

  with pytest.raises(ValueError, match="numeric|real"):
    read_simulation_log(directory)


def test_read_simulation_log_rejects_corrupt_artifacts(tmp_path):
  directory = tmp_path / "artifact"
  write_simulation_log(_simulation_log(), directory)
  metadata_path = _payload_path(directory, "metadata.json")
  payload = json.loads(metadata_path.read_text())
  payload["schema_version"] = "999"
  metadata_path.write_text(json.dumps(payload))
  _refresh_manifest(directory)

  with pytest.raises(ValueError, match="schema_version"):
    read_simulation_log(directory)
