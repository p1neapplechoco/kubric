"""Simulator-independent contact and rigid-body state logging."""

from __future__ import annotations

import json
import math
import numbers
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Optional, Tuple, Union

import numpy as np

from interventions.schema import SCHEMA_VERSION, to_jsonable


POSITION_SLICE = slice(0, 3)
QUATERNION_SLICE = slice(3, 7)
LINEAR_VELOCITY_SLICE = slice(7, 10)
ANGULAR_VELOCITY_SLICE = slice(10, 13)
STATE_INDEX = MappingProxyType({
    "position": POSITION_SLICE,
    "quaternion": QUATERNION_SLICE,
    "linear_velocity": LINEAR_VELOCITY_SLICE,
    "angular_velocity": ANGULAR_VELOCITY_SLICE,
})


def state_index(component: str) -> slice:
  """Returns the slice for a named component of the 13-value state vector."""
  if not isinstance(component, str):
    raise TypeError("state component must be a string")
  try:
    return STATE_INDEX[component]
  except KeyError as error:
    raise ValueError("unknown state component: {!r}".format(component)) from error


def _identifier(value: Any, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if not value.strip():
    raise ValueError("{} must not be empty".format(name))
  return value


def _integer(value: Any, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, numbers.Integral):
    raise TypeError("{} must be an integer".format(name))
  return int(value)


def _real(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise TypeError("{} must be a real number".format(name))
  try:
    result = float(value)
  except (OverflowError, ValueError) as error:
    raise ValueError("{} must be finite".format(name)) from error
  if not math.isfinite(result):
    raise ValueError("{} must be finite".format(name))
  return result


def _vector(value: Any, length: int, name: str) -> Tuple[float, ...]:
  if isinstance(value, (str, bytes)):
    raise TypeError("{} must be a numeric sequence".format(name))
  try:
    items = tuple(value)
  except TypeError as error:
    raise TypeError("{} must be a numeric sequence".format(name)) from error
  if len(items) != length:
    raise ValueError("{} must contain {} values".format(name, length))
  return tuple(_real(item, "{}[{}]".format(name, index))
               for index, item in enumerate(items))


def _contact_sort_key(record: "ContactRecord") -> Tuple[Any, ...]:
  return (
      record.step,
      record.object_a,
      record.object_b,
      record.position,
      record.normal,
      record.normal_force,
      record.contact_distance is None,
      0.0 if record.contact_distance is None else record.contact_distance,
  )


def _freeze_json(value: Any) -> Any:
  if isinstance(value, dict):
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
  if isinstance(value, list):
    return tuple(_freeze_json(item) for item in value)
  return value


def _metadata(value: Any) -> Mapping[str, Any]:
  if value is None:
    value = {}
  if not isinstance(value, Mapping):
    raise TypeError("metadata must be a mapping")
  converted = to_jsonable(value)
  if not isinstance(converted, dict):  # Defensive: mappings serialize as objects.
    raise TypeError("metadata must serialize to a JSON object")
  return _freeze_json(converted)


def _float_array(value: Any, name: str) -> np.ndarray:
  try:
    untyped = np.asarray(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("{} must be a numeric array".format(name)) from error
  if (
      not np.issubdtype(untyped.dtype, np.number)
      or np.issubdtype(untyped.dtype, np.complexfloating)
  ):
    raise ValueError("{} must contain real numeric values".format(name))
  try:
    canonical = np.array(untyped, dtype=np.float64, order="C", copy=True)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("{} must be a numeric array".format(name)) from error
  if not np.isfinite(canonical).all():
    raise ValueError("{} must contain only finite values".format(name))
  immutable = canonical.tobytes(order="C")
  return np.frombuffer(immutable, dtype=np.float64).reshape(canonical.shape)


def _validate_unit_quaternions(quaternions: np.ndarray, name: str) -> None:
  for index, quaternion in enumerate(quaternions.reshape((-1, 4))):
    norm = math.hypot(*(float(component) for component in quaternion))
    if abs(norm - 1.0) > 1e-6:
      raise ValueError("{} quaternion {} must be unit-normalized".format(name, index))


def _readonly(array: np.ndarray) -> np.ndarray:
  array.setflags(write=False)
  return array


@dataclass(frozen=True)
class ContactRecord:
  """One canonical physical contact.

  ``normal`` is oriented from ``object_b`` toward ``object_a``. Endpoints are
  canonicalized lexicographically, with the normal reversed whenever that swaps
  the input endpoint order.
  """

  step: int
  object_a: str
  object_b: str
  position: Tuple[float, float, float]
  normal: Tuple[float, float, float]
  normal_force: float
  contact_distance: Optional[float] = None
  schema_version: str = SCHEMA_VERSION

  def __post_init__(self) -> None:
    step = _integer(self.step, "step")
    if step < 0:
      raise ValueError("step must be nonnegative")
    object_a = _identifier(self.object_a, "object_a")
    object_b = _identifier(self.object_b, "object_b")
    if object_a == object_b:
      raise ValueError("contact endpoints must be distinct")
    position = _vector(self.position, 3, "position")
    normal = _vector(self.normal, 3, "normal")
    normal_force = _real(self.normal_force, "normal_force")
    if normal_force < 0.0:
      raise ValueError("normal_force must be nonnegative")
    contact_distance = self.contact_distance
    if contact_distance is not None:
      contact_distance = _real(contact_distance, "contact_distance")
    if self.schema_version != SCHEMA_VERSION:
      raise ValueError("schema_version must be {!r}".format(SCHEMA_VERSION))

    if object_b < object_a:
      object_a, object_b = object_b, object_a
      normal = tuple(-component for component in normal)

    object.__setattr__(self, "step", step)
    object.__setattr__(self, "object_a", object_a)
    object.__setattr__(self, "object_b", object_b)
    object.__setattr__(self, "position", position)
    object.__setattr__(self, "normal", normal)
    object.__setattr__(self, "normal_force", normal_force)
    object.__setattr__(self, "contact_distance", contact_distance)
    object.__setattr__(self, "schema_version", SCHEMA_VERSION)

  def to_dict(self) -> Mapping[str, Any]:
    """Returns a new JSON-safe representation."""
    return to_jsonable(self)


BodyResolver = Union[Mapping[int, str], Callable[[int], Optional[str]]]


class ContactLogger:
  """Parses PyBullet-compatible contact tuples without importing PyBullet."""

  def __init__(
      self,
      body_to_object_id: BodyResolver,
      step_rate: float,
      force_epsilon: float = 0.0,
  ) -> None:
    self.step_rate = _real(step_rate, "step_rate")
    if self.step_rate <= 0.0:
      raise ValueError("step_rate must be positive")
    self.force_epsilon = _real(force_epsilon, "force_epsilon")
    if self.force_epsilon < 0.0:
      raise ValueError("force_epsilon must be nonnegative")

    if isinstance(body_to_object_id, Mapping):
      resolver = {}
      for body_id, object_id in body_to_object_id.items():
        normalized_body = _integer(body_id, "body id")
        resolver[normalized_body] = _identifier(object_id, "object id")
      self._body_mapping: Optional[Mapping[int, str]] = MappingProxyType(resolver)
      self._body_resolver: Optional[Callable[[int], Optional[str]]] = None
    elif callable(body_to_object_id):
      self._body_mapping = None
      self._body_resolver = body_to_object_id
    else:
      raise TypeError("body_to_object_id must be a mapping or callable")
    self._records: Tuple[ContactRecord, ...] = ()

  @property
  def records(self) -> Tuple[ContactRecord, ...]:
    return self._records

  def clear(self) -> None:
    self._records = ()

  def _resolve(self, body_id: int) -> Optional[str]:
    if self._body_mapping is not None:
      return self._body_mapping.get(body_id)
    assert self._body_resolver is not None
    try:
      value = self._body_resolver(body_id)
    except (KeyError, LookupError):
      return None
    if value is None:
      return None
    return _identifier(value, "resolved object id")

  def log(
      self, step: int, raw_contacts: Iterable[Any]
  ) -> Tuple[ContactRecord, ...]:
    """Parses and retains contacts newly accepted for ``step``."""
    normalized_step = _integer(step, "step")
    if normalized_step < 0:
      raise ValueError("step must be nonnegative")
    if isinstance(raw_contacts, (str, bytes)):
      raise TypeError("raw_contacts must be an iterable of contact tuples")
    try:
      raw_values = tuple(raw_contacts)
    except TypeError as error:
      raise TypeError("raw_contacts must be an iterable of contact tuples") from error

    accepted = []
    for index, raw_contact in enumerate(raw_values):
      if isinstance(raw_contact, (str, bytes)):
        raise TypeError("contact {} must be a sequence".format(index))
      try:
        fields = tuple(raw_contact)
      except TypeError as error:
        raise TypeError("contact {} must be a sequence".format(index)) from error
      if len(fields) != 14:
        raise ValueError("contact {} must contain 14 fields".format(index))
      body_a = _integer(fields[1], "contact {} bodyA".format(index))
      body_b = _integer(fields[2], "contact {} bodyB".format(index))
      object_a = self._resolve(body_a)
      object_b = self._resolve(body_b)
      if object_a is None or object_b is None or object_a == object_b:
        continue
      force = _real(fields[9], "contact {} normalForce".format(index))
      if force <= self.force_epsilon:
        continue
      accepted.append(ContactRecord(
          step=normalized_step,
          object_a=object_a,
          object_b=object_b,
          position=fields[6],
          normal=fields[7],
          contact_distance=fields[8],
          normal_force=force,
      ))

    result = tuple(sorted(accepted, key=_contact_sort_key))
    self._records = tuple(sorted(self._records + result, key=_contact_sort_key))
    return result


@dataclass(frozen=True, eq=False)
class SimulationLog:
  """Immutable state/contact log with states shaped ``[T, N, 13]``."""

  branch: str
  object_ids: Tuple[str, ...]
  steps: Tuple[int, ...]
  states: np.ndarray
  contacts: Tuple[ContactRecord, ...]
  step_rate: float
  commanded_path: Optional[np.ndarray] = None
  metadata: Mapping[str, Any] = field(default_factory=dict)
  schema_version: str = SCHEMA_VERSION

  POSITION: ClassVar[slice] = POSITION_SLICE
  QUATERNION: ClassVar[slice] = QUATERNION_SLICE
  LINEAR_VELOCITY: ClassVar[slice] = LINEAR_VELOCITY_SLICE
  ANGULAR_VELOCITY: ClassVar[slice] = ANGULAR_VELOCITY_SLICE

  def __post_init__(self) -> None:
    branch = _identifier(self.branch, "branch")
    if isinstance(self.object_ids, (str, bytes, set, frozenset)):
      raise TypeError("object_ids must be an ordered iterable")
    try:
      object_ids = tuple(
          _identifier(item, "object_id") for item in self.object_ids
      )
    except TypeError as error:
      raise TypeError("object_ids must be an ordered iterable") from error
    if len(set(object_ids)) != len(object_ids):
      raise ValueError("object_ids must be unique")

    if isinstance(self.steps, (str, bytes, set, frozenset)):
      raise TypeError("steps must be an ordered iterable")
    try:
      steps = tuple(_integer(item, "step") for item in self.steps)
    except TypeError as error:
      raise TypeError("steps must be an ordered iterable of integers") from error
    if any(step < 0 for step in steps):
      raise ValueError("steps must be nonnegative")
    if any(right <= left for left, right in zip(steps, steps[1:])):
      raise ValueError("steps must be strictly increasing")

    states = _float_array(self.states, "states")
    expected_shape = (len(steps), len(object_ids), 13)
    if states.shape != expected_shape:
      raise ValueError("states must have shape {!r}".format(expected_shape))
    _validate_unit_quaternions(states[:, :, QUATERNION_SLICE], "states")

    if isinstance(self.contacts, (str, bytes)):
      raise TypeError("contacts must be an iterable of ContactRecord values")
    try:
      contacts = tuple(self.contacts)
    except TypeError as error:
      raise TypeError("contacts must be an iterable of ContactRecord values") from error
    if not all(isinstance(record, ContactRecord) for record in contacts):
      raise TypeError("contacts must contain only ContactRecord values")
    if contacts and not steps:
      raise ValueError("contacts require at least one logged step")
    if contacts and any(
        record.step < steps[0] or record.step > steps[-1] for record in contacts
    ):
      raise ValueError("contact steps must lie within the logged step range")
    contacts = tuple(sorted(contacts, key=_contact_sort_key))

    step_rate = _real(self.step_rate, "step_rate")
    if step_rate <= 0.0:
      raise ValueError("step_rate must be positive")

    commanded_path = self.commanded_path
    if commanded_path is not None:
      commanded_path = _float_array(commanded_path, "commanded_path")
      if (
          commanded_path.ndim != 2
          or commanded_path.shape[0] != len(steps)
          or commanded_path.shape[1] not in (3, 7)
      ):
        raise ValueError("commanded_path must have shape [T, 3] or [T, 7]")
      if commanded_path.shape[1] == 7:
        _validate_unit_quaternions(
            commanded_path[:, 3:7], "commanded_path"
        )
      commanded_path = _readonly(commanded_path)

    if self.schema_version != SCHEMA_VERSION:
      raise ValueError("schema_version must be {!r}".format(SCHEMA_VERSION))

    object.__setattr__(self, "branch", branch)
    object.__setattr__(self, "object_ids", object_ids)
    object.__setattr__(self, "steps", steps)
    object.__setattr__(self, "states", _readonly(states))
    object.__setattr__(self, "contacts", contacts)
    object.__setattr__(self, "step_rate", step_rate)
    object.__setattr__(self, "commanded_path", commanded_path)
    object.__setattr__(self, "metadata", _metadata(self.metadata))
    object.__setattr__(self, "schema_version", SCHEMA_VERSION)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, SimulationLog):
      return NotImplemented
    commanded_equal = (
        self.commanded_path is None and other.commanded_path is None
    ) or (
        self.commanded_path is not None
        and other.commanded_path is not None
        and np.array_equal(self.commanded_path, other.commanded_path)
    )
    return bool(
        self.branch == other.branch
        and self.object_ids == other.object_ids
        and self.steps == other.steps
        and np.array_equal(self.states, other.states)
        and self.contacts == other.contacts
        and self.step_rate == other.step_rate
        and commanded_equal
        and self.metadata == other.metadata
        and self.schema_version == other.schema_version
    )


_ARTIFACT_FILENAMES = ("contacts.jsonl", "states.npy", "metadata.json")


def _json_bytes(value: Any) -> bytes:
  return (
      json.dumps(
          value,
          sort_keys=True,
          separators=(",", ":"),
          ensure_ascii=False,
          allow_nan=False,
      )
      + "\n"
  ).encode("utf-8")


def _write_temp(directory: Path, prefix: str, payload: bytes) -> Path:
  with tempfile.NamedTemporaryFile(
      mode="wb", prefix=prefix, dir=str(directory), delete=False
  ) as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    return Path(stream.name)


def write_simulation_log(
    log: SimulationLog,
    directory: Union[str, os.PathLike[str]],
    *,
    overwrite: bool = False,
) -> Path:
  """Atomically writes each file of a simulation-log artifact."""
  if not isinstance(log, SimulationLog):
    raise TypeError("log must be a SimulationLog")
  if not isinstance(overwrite, bool):
    raise TypeError("overwrite must be a bool")
  target = Path(directory)
  target.mkdir(parents=True, exist_ok=True)
  final_paths = {name: target / name for name in _ARTIFACT_FILENAMES}
  if not overwrite and any(path.exists() for path in final_paths.values()):
    raise FileExistsError("simulation-log artifact already exists: {}".format(target))

  contacts_payload = b"".join(
      _json_bytes(record.to_dict()) for record in log.contacts
  )
  metadata_payload = {
      "branch": log.branch,
      "object_ids": list(log.object_ids),
      "steps": list(log.steps),
      "step_rate": log.step_rate,
      "commanded_path": (
          None if log.commanded_path is None else to_jsonable(log.commanded_path)
      ),
      "metadata": to_jsonable(log.metadata),
      "schema_version": log.schema_version,
  }

  temporary = {}
  try:
    temporary["contacts.jsonl"] = _write_temp(
        target, ".contacts.", contacts_payload
    )
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".states.", dir=str(target), delete=False
    ) as stream:
      np.save(stream, log.states, allow_pickle=False)
      stream.flush()
      os.fsync(stream.fileno())
      temporary["states.npy"] = Path(stream.name)
    temporary["metadata.json"] = _write_temp(
        target, ".metadata.", _json_bytes(metadata_payload)
    )
    for name in _ARTIFACT_FILENAMES:
      os.replace(str(temporary[name]), str(final_paths[name]))
  finally:
    for path in temporary.values():
      try:
        path.unlink()
      except FileNotFoundError:
        pass
  return target


def read_simulation_log(
    directory: Union[str, os.PathLike[str]],
) -> SimulationLog:
  """Reads and validates a simulation-log artifact."""
  source = Path(directory)
  with (source / "metadata.json").open("r", encoding="utf-8") as stream:
    metadata_payload = json.load(stream)
  if not isinstance(metadata_payload, dict):
    raise ValueError("metadata.json must contain a JSON object")
  expected_keys = {
      "branch",
      "object_ids",
      "steps",
      "step_rate",
      "commanded_path",
      "metadata",
      "schema_version",
  }
  if set(metadata_payload) != expected_keys:
    raise ValueError("metadata.json has missing or unexpected fields")

  try:
    with (source / "states.npy").open("rb") as stream:
      states = np.load(stream, allow_pickle=False)
      trailing_payload = stream.read(1)
  except (TypeError, ValueError, OSError) as error:
    raise ValueError("states.npy is not a valid numeric NumPy array") from error
  if trailing_payload:
    raise ValueError("states.npy contains a trailing payload")
  if not isinstance(states, np.ndarray) or states.dtype.hasobject:
    raise ValueError("states.npy must contain one non-object NumPy array")

  contacts = []
  with (source / "contacts.jsonl").open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        raise ValueError(
            "contacts.jsonl contains a blank line at {}".format(line_number)
        )
      try:
        payload = json.loads(line)
      except json.JSONDecodeError as error:
        raise ValueError(
            "contacts.jsonl line {} is invalid JSON".format(line_number)
        ) from error
      if not isinstance(payload, dict):
        raise ValueError(
            "contacts.jsonl line {} must contain an object".format(line_number)
        )
      contacts.append(ContactRecord(**payload))

  return SimulationLog(
      branch=metadata_payload["branch"],
      object_ids=metadata_payload["object_ids"],
      steps=metadata_payload["steps"],
      states=states,
      contacts=tuple(contacts),
      step_rate=metadata_payload["step_rate"],
      commanded_path=metadata_payload["commanded_path"],
      metadata=metadata_payload["metadata"],
      schema_version=metadata_payload["schema_version"],
  )


__all__ = [
    "ANGULAR_VELOCITY_SLICE",
    "LINEAR_VELOCITY_SLICE",
    "POSITION_SLICE",
    "QUATERNION_SLICE",
    "STATE_INDEX",
    "ContactLogger",
    "ContactRecord",
    "SimulationLog",
    "read_simulation_log",
    "state_index",
    "write_simulation_log",
]
