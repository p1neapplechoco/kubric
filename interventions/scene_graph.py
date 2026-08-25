"""Per-step relation graphs over logged rigid-body states and contacts.

Purpose: read a replay as an evolving scene graph by deriving contact, proximity,
and approach/recede relations for every logged step.
Public API: CausalEdge, NodeState, RelationEdge, RelationSeries, SceneGraphFrame,
CONTACT_RELATION, PROXIMITY_RELATION, RELATION_KINDS, build_relation_series(),
contact_activation_steps(), and propagation_tree().
Dependencies: NumPy plus validated logging values; no simulator or plotting backend.
Trust boundary: contact edges inherit whatever authority the supplied contact log
already carries, while proximity gaps and approach rates are center-distance
readings that are exact for spheres and conservative for boxes; nothing here
re-simulates physics or attests that the log is complete.
"""

from __future__ import annotations

import math
import numbers
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from interventions.logging import (
    LINEAR_VELOCITY_SLICE,
    POSITION_SLICE,
    SimulationLog,
)

CONTACT_RELATION = "contact"
PROXIMITY_RELATION = "near"
RELATION_KINDS = (CONTACT_RELATION, PROXIMITY_RELATION)


def _identifier(value: Any, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if not value.strip():
    raise ValueError("{} must not be empty".format(name))
  return value


def _real(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise TypeError("{} must be a real number".format(name))
  result = float(value)
  if not math.isfinite(result):
    raise ValueError("{} must be finite".format(name))
  return result


def _nonnegative(value: Any, name: str) -> float:
  result = _real(value, name)
  if result < 0.0:
    raise ValueError("{} must be nonnegative".format(name))
  return result


def _positive(value: Any, name: str) -> float:
  result = _real(value, name)
  if result <= 0.0:
    raise ValueError("{} must be positive".format(name))
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


def _relation(value: Any) -> str:
  kind = _identifier(value, "relation")
  if kind not in RELATION_KINDS:
    raise ValueError("relation must be one of {!r}".format(RELATION_KINDS))
  return kind


@dataclass(frozen=True)
class NodeState:
  """One object's position, speed, and presence within a single step."""

  object_id: str
  position: Tuple[float, float, float]
  speed: float
  present: bool

  def __post_init__(self) -> None:
    object.__setattr__(
        self, "object_id", _identifier(self.object_id, "object_id")
    )
    object.__setattr__(self, "position", _vector(self.position, 3, "position"))
    object.__setattr__(self, "speed", _nonnegative(self.speed, "speed"))
    if not isinstance(self.present, (bool, np.bool_)):
      raise TypeError("present must be a boolean")
    object.__setattr__(self, "present", bool(self.present))


@dataclass(frozen=True)
class RelationEdge:
  """One unordered per-step relation with its surface gap and closing speed.

  ``gap`` is the center distance minus both collision radii, so it is negative
  while shapes overlap.  ``approach_rate`` is the rate at which that gap shrinks,
  hence positive while the endpoints close on each other.
  """

  object_a: str
  object_b: str
  relation: str
  gap: float
  approach_rate: float
  normal_force: float = 0.0

  def __post_init__(self) -> None:
    object_a = _identifier(self.object_a, "object_a")
    object_b = _identifier(self.object_b, "object_b")
    if object_a == object_b:
      raise ValueError("relation endpoints must be distinct")
    if object_b < object_a:
      object_a, object_b = object_b, object_a
    relation = _relation(self.relation)
    normal_force = _nonnegative(self.normal_force, "normal_force")
    if relation != CONTACT_RELATION and normal_force != 0.0:
      raise ValueError("only contact relations may carry a normal force")
    object.__setattr__(self, "object_a", object_a)
    object.__setattr__(self, "object_b", object_b)
    object.__setattr__(self, "relation", relation)
    object.__setattr__(self, "gap", _real(self.gap, "gap"))
    object.__setattr__(
        self, "approach_rate", _real(self.approach_rate, "approach_rate")
    )
    object.__setattr__(self, "normal_force", normal_force)

  @property
  def pair(self) -> Tuple[str, str]:
    """The canonically ordered endpoint pair identifying this relation."""
    return (self.object_a, self.object_b)

  @property
  def approaching(self) -> bool:
    """Whether the endpoints are currently closing rather than separating."""
    return self.approach_rate > 0.0


@dataclass(frozen=True)
class SceneGraphFrame:
  """The complete relation graph for one logged step."""

  step: int
  nodes: Tuple[NodeState, ...]
  edges: Tuple[RelationEdge, ...]

  def __post_init__(self) -> None:
    if isinstance(self.step, bool) or not isinstance(self.step, numbers.Integral):
      raise TypeError("step must be an integer")
    if int(self.step) < 0:
      raise ValueError("step must be nonnegative")
    nodes = tuple(self.nodes)
    if not all(isinstance(node, NodeState) for node in nodes):
      raise TypeError("nodes must contain only NodeState values")
    identifiers = tuple(node.object_id for node in nodes)
    if len(set(identifiers)) != len(identifiers):
      raise ValueError("nodes must not repeat an object_id")
    edges = tuple(self.edges)
    if not all(isinstance(edge, RelationEdge) for edge in edges):
      raise TypeError("edges must contain only RelationEdge values")
    known = set(identifiers)
    if any(
        edge.object_a not in known or edge.object_b not in known
        for edge in edges
    ):
      raise ValueError("edge endpoints must name nodes of the same frame")
    if len({edge.pair for edge in edges}) != len(edges):
      raise ValueError("edges must not repeat an endpoint pair")
    object.__setattr__(self, "step", int(self.step))
    object.__setattr__(self, "nodes", nodes)
    object.__setattr__(
        self, "edges", tuple(sorted(edges, key=lambda edge: edge.pair))
    )

  def contact_edges(self) -> Tuple[RelationEdge, ...]:
    """The edges backed by a logged contact at this step."""
    return tuple(
        edge for edge in self.edges if edge.relation == CONTACT_RELATION
    )

  def moving_ids(self, motion_epsilon: float) -> Tuple[str, ...]:
    """The present object IDs whose speed exceeds ``motion_epsilon``."""
    threshold = _nonnegative(motion_epsilon, "motion_epsilon")
    return tuple(
        node.object_id for node in self.nodes
        if node.present and node.speed > threshold
    )


@dataclass(frozen=True)
class RelationSeries:
  """An ordered per-step relation graph series with its extraction settings."""

  branch: str
  object_ids: Tuple[str, ...]
  radii: Mapping[str, float]
  near_margin: float
  motion_epsilon: float
  frames: Tuple[SceneGraphFrame, ...]

  def __post_init__(self) -> None:
    object.__setattr__(self, "branch", _identifier(self.branch, "branch"))
    object_ids = tuple(
        _identifier(item, "object_id") for item in self.object_ids
    )
    if len(set(object_ids)) != len(object_ids):
      raise ValueError("object_ids must be unique")
    radii = {
        _identifier(key, "radii key"): _positive(value, "radii value")
        for key, value in dict(self.radii).items()
    }
    if set(radii) != set(object_ids):
      raise ValueError("radii must cover exactly the logged object_ids")
    frames = tuple(self.frames)
    if not all(isinstance(frame, SceneGraphFrame) for frame in frames):
      raise TypeError("frames must contain only SceneGraphFrame values")
    steps = tuple(frame.step for frame in frames)
    if any(right <= left for left, right in zip(steps, steps[1:])):
      raise ValueError("frames must be strictly ordered by step")
    object.__setattr__(self, "object_ids", object_ids)
    object.__setattr__(self, "radii", MappingProxyType(radii))
    object.__setattr__(
        self, "near_margin", _nonnegative(self.near_margin, "near_margin")
    )
    object.__setattr__(
        self,
        "motion_epsilon",
        _nonnegative(self.motion_epsilon, "motion_epsilon"),
    )
    object.__setattr__(self, "frames", frames)

  @property
  def steps(self) -> Tuple[int, ...]:
    """The logged step numbers in frame order."""
    return tuple(frame.step for frame in self.frames)

  def frame_at(self, step: int) -> SceneGraphFrame:
    """The frame logged at ``step``, or a KeyError when it was not logged."""
    for frame in self.frames:
      if frame.step == step:
        return frame
    raise KeyError(step)


def _presence_array(
    presence: Any, shape: Tuple[int, int]
) -> np.ndarray:
  if presence is None:
    return np.ones(shape, dtype=np.bool_)
  array = np.asarray(presence)
  if array.dtype != np.bool_:
    raise TypeError("presence must be a boolean array")
  if array.shape != shape:
    raise ValueError("presence must have shape {!r}".format(shape))
  return array


def _excluded(value: Iterable[str], known: Iterable[str]) -> frozenset:
  if isinstance(value, (str, bytes)):
    raise TypeError("proximity_exclude must be an iterable of identifiers")
  excluded = frozenset(
      _identifier(item, "excluded object_id") for item in value
  )
  unknown = excluded - set(known)
  if unknown:
    raise ValueError(
        "proximity_exclude names unknown object_ids: {!r}".format(
            sorted(unknown)
        )
    )
  return excluded


def _contact_forces(
    log: SimulationLog,
) -> Dict[int, Dict[Tuple[str, str], float]]:
  grouped: Dict[int, Dict[Tuple[str, str], float]] = defaultdict(
      lambda: defaultdict(float)
  )
  for record in log.contacts:
    pair = (record.object_a, record.object_b)
    grouped[record.step][pair] += float(record.normal_force)
  return {step: dict(pairs) for step, pairs in grouped.items()}


def build_relation_series(
    log: SimulationLog,
    radii: Mapping[str, float],
    presence: Optional[np.ndarray] = None,
    near_margin: float = 0.12,
    motion_epsilon: float = 1e-3,
    proximity_exclude: Iterable[str] = (),
) -> RelationSeries:
  """Derives a contact, proximity, and approach relation graph for every step.

  A pair becomes a ``CONTACT_RELATION`` edge whenever the log records a contact
  at that step, and otherwise a ``PROXIMITY_RELATION`` edge while its surface gap
  stays within ``near_margin``.  Endpoints named by ``proximity_exclude`` still
  form contact edges but never proximity edges, which keeps unbounded supports
  such as a ground plane from linking to every object.
  """
  if not isinstance(log, SimulationLog):
    raise TypeError("log must be a SimulationLog")
  radius_map = {
      _identifier(key, "radii key"): _positive(value, "radii value")
      for key, value in dict(radii).items()
  }
  if set(radius_map) != set(log.object_ids):
    raise ValueError("radii must cover exactly the logged object_ids")
  margin = _nonnegative(near_margin, "near_margin")
  epsilon = _nonnegative(motion_epsilon, "motion_epsilon")
  excluded = _excluded(proximity_exclude, log.object_ids)

  states = np.asarray(log.states, dtype=np.float64)
  if not np.isfinite(states).all():
    raise ValueError("states must contain only finite values")
  object_ids = tuple(log.object_ids)
  visible = _presence_array(presence, (len(log.steps), len(object_ids)))
  forces_by_step = _contact_forces(log)
  radius_values = np.array(
      [radius_map[object_id] for object_id in object_ids], dtype=np.float64
  )

  frames: List[SceneGraphFrame] = []
  for time_index, step in enumerate(log.steps):
    positions = states[time_index, :, POSITION_SLICE]
    velocities = states[time_index, :, LINEAR_VELOCITY_SLICE]
    present = visible[time_index]
    speeds = np.linalg.norm(velocities, axis=1)
    nodes = tuple(
        NodeState(
            object_id=object_id,
            position=tuple(positions[index]),
            speed=float(speeds[index]),
            present=bool(present[index]),
        )
        for index, object_id in enumerate(object_ids)
    )
    step_forces = forces_by_step.get(step, {})
    edges: List[RelationEdge] = []
    for left in range(len(object_ids)):
      for right in range(left + 1, len(object_ids)):
        if not (present[left] and present[right]):
          continue
        pair = (object_ids[left], object_ids[right])
        offset = positions[right] - positions[left]
        distance = float(np.linalg.norm(offset))
        gap = distance - float(radius_values[left] + radius_values[right])
        if distance > 0.0:
          direction = offset / distance
          relative = velocities[right] - velocities[left]
          approach_rate = -float(np.dot(relative, direction))
        else:
          approach_rate = 0.0
        force = step_forces.get(pair)
        if force is not None:
          edges.append(RelationEdge(
              object_a=pair[0],
              object_b=pair[1],
              relation=CONTACT_RELATION,
              gap=gap,
              approach_rate=approach_rate,
              normal_force=force,
          ))
        elif gap <= margin and not (
            pair[0] in excluded or pair[1] in excluded
        ):
          edges.append(RelationEdge(
              object_a=pair[0],
              object_b=pair[1],
              relation=PROXIMITY_RELATION,
              gap=gap,
              approach_rate=approach_rate,
          ))
    frames.append(SceneGraphFrame(step=step, nodes=nodes, edges=tuple(edges)))

  return RelationSeries(
      branch=log.branch,
      object_ids=object_ids,
      radii=radius_map,
      near_margin=margin,
      motion_epsilon=epsilon,
      frames=tuple(frames),
  )


def contact_activation_steps(
    series: RelationSeries,
) -> Mapping[Tuple[str, str], int]:
  """Returns the first step at which each endpoint pair reached contact."""
  if not isinstance(series, RelationSeries):
    raise TypeError("series must be a RelationSeries")
  first: Dict[Tuple[str, str], int] = {}
  for frame in series.frames:
    for edge in frame.contact_edges():
      first.setdefault(edge.pair, frame.step)
  return MappingProxyType({pair: first[pair] for pair in sorted(first)})


@dataclass(frozen=True)
class CausalEdge:
  """One parent-to-child hop of an oracle propagation path."""

  parent: str
  child: str
  hop: int

  def __post_init__(self) -> None:
    parent = _identifier(self.parent, "parent")
    child = _identifier(self.child, "child")
    if parent == child:
      raise ValueError("causal edge endpoints must be distinct")
    if isinstance(self.hop, bool) or not isinstance(self.hop, numbers.Integral):
      raise TypeError("hop must be an integer")
    if int(self.hop) < 1:
      raise ValueError("hop must be at least one")
    object.__setattr__(self, "parent", parent)
    object.__setattr__(self, "child", child)
    object.__setattr__(self, "hop", int(self.hop))


def propagation_tree(
    propagation_path: Mapping[str, Sequence[str]],
) -> Tuple[CausalEdge, ...]:
  """Collapses oracle propagation walks into deduplicated parent-child hops.

  Reachability witnesses may revisit a node when an undirected delta closes a
  cycle, so each parent-child pair keeps only its shallowest observed hop.
  """
  if not isinstance(propagation_path, Mapping):
    raise TypeError("propagation_path must be a mapping")
  shallowest: Dict[Tuple[str, str], int] = {}
  for affected, path in propagation_path.items():
    _identifier(affected, "propagation_path key")
    if isinstance(path, (str, bytes)):
      raise TypeError("propagation_path values must be sequences of IDs")
    walk = tuple(_identifier(item, "propagation_path item") for item in path)
    if not walk:
      raise ValueError("propagation_path values must not be empty")
    if walk[-1] != affected:
      raise ValueError("propagation paths must end at their key")
    for index, child in enumerate(walk[1:], start=1):
      parent = walk[index - 1]
      if parent == child:
        raise ValueError("propagation paths must not repeat a node in place")
      key = (parent, child)
      if index < shallowest.get(key, index + 1):
        shallowest[key] = index
  return tuple(
      CausalEdge(parent=parent, child=child, hop=shallowest[(parent, child)])
      for parent, child in sorted(shallowest)
  )


__all__ = [
    "CONTACT_RELATION",
    "CausalEdge",
    "NodeState",
    "PROXIMITY_RELATION",
    "RELATION_KINDS",
    "RelationEdge",
    "RelationSeries",
    "SceneGraphFrame",
    "build_relation_series",
    "contact_activation_steps",
    "propagation_tree",
]
