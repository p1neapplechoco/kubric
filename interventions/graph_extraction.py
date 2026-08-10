"""Pure post-processing for temporal contact graphs and affected-object labels."""

from __future__ import annotations

import heapq
import math
import numbers
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from interventions.logging import (
    ANGULAR_VELOCITY_SLICE,
    LINEAR_VELOCITY_SLICE,
    POSITION_SLICE,
    QUATERNION_SLICE,
    ContactRecord,
    SimulationLog,
)
from interventions.schema import GraphEdgeDelta, GroundTruth, to_jsonable


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


def _nonnegative(value: Any, name: str) -> float:
  result = _real(value, name)
  if result < 0.0:
    raise ValueError("{} must be nonnegative".format(name))
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


def _normalized(vector: Sequence[float]) -> Optional[Tuple[float, ...]]:
  norm = math.hypot(*(float(item) for item in vector))
  if norm == 0.0:
    return None
  return tuple(float(item) / norm for item in vector)


@dataclass(frozen=True)
class AggregatedContactStep:
  """The deterministic aggregate of all contact points for one pair and step."""

  step: int
  object_a: str
  object_b: str
  position: Tuple[float, float, float]
  normal: Tuple[float, float, float]
  normal_force: float
  impulse: float

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
    normal_force = _nonnegative(self.normal_force, "normal_force")
    impulse = _nonnegative(self.impulse, "impulse")
    if object_b < object_a:
      object_a, object_b = object_b, object_a
      normal = tuple(-component for component in normal)
    object.__setattr__(self, "step", step)
    object.__setattr__(self, "object_a", object_a)
    object.__setattr__(self, "object_b", object_b)
    object.__setattr__(self, "position", position)
    object.__setattr__(self, "normal", normal)
    object.__setattr__(self, "normal_force", normal_force)
    object.__setattr__(self, "impulse", impulse)

  def to_dict(self) -> Mapping[str, Any]:
    return to_jsonable(self)

  def __getitem__(self, key: str) -> Any:
    """Supports concise field access when consuming generic graph records."""
    if key not in (
        "step", "object_a", "object_b", "position", "normal",
        "normal_force", "impulse",
    ):
      raise KeyError(key)
    return getattr(self, key)


def _coerce_contact_step(value: Any) -> AggregatedContactStep:
  if isinstance(value, AggregatedContactStep):
    return value
  if isinstance(value, Mapping):
    return AggregatedContactStep(**dict(value))
  raise TypeError("contact_steps must contain AggregatedContactStep values")


@dataclass(frozen=True)
class TemporalEdge:
  """A maximal consecutive contact episode with an exclusive end step."""

  object_a: str
  object_b: str
  start_step: int
  end_step: int
  total_impulse: float
  peak_force: float
  contact_steps: Tuple[AggregatedContactStep, ...] = ()

  def __post_init__(self) -> None:
    object_a = _identifier(self.object_a, "object_a")
    object_b = _identifier(self.object_b, "object_b")
    if object_a == object_b:
      raise ValueError("temporal edge endpoints must be distinct")
    object_a, object_b = sorted((object_a, object_b))
    start_step = _integer(self.start_step, "start_step")
    end_step = _integer(self.end_step, "end_step")
    if start_step < 0 or end_step <= start_step:
      raise ValueError("steps must satisfy 0 <= start_step < end_step")
    total_impulse = _nonnegative(self.total_impulse, "total_impulse")
    peak_force = _nonnegative(self.peak_force, "peak_force")
    if isinstance(self.contact_steps, (str, bytes)):
      raise TypeError("contact_steps must be an ordered iterable")
    try:
      contact_steps = tuple(
          _coerce_contact_step(item) for item in self.contact_steps
      )
    except TypeError as error:
      raise TypeError("contact_steps must be an ordered iterable") from error
    step_numbers = tuple(item.step for item in contact_steps)
    if any(right <= left for left, right in zip(step_numbers, step_numbers[1:])):
      raise ValueError("contact_steps must be strictly ordered by step")
    if any(
        item.object_a != object_a or item.object_b != object_b
        for item in contact_steps
    ):
      raise ValueError("contact_steps must use the temporal edge endpoints")
    if contact_steps and (
        step_numbers[0] != start_step
        or step_numbers[-1] + 1 != end_step
        or any(right != left + 1 for left, right in zip(step_numbers, step_numbers[1:]))
    ):
      raise ValueError("contact_steps must exactly cover the consecutive episode")

    object.__setattr__(self, "object_a", object_a)
    object.__setattr__(self, "object_b", object_b)
    object.__setattr__(self, "start_step", start_step)
    object.__setattr__(self, "end_step", end_step)
    object.__setattr__(self, "total_impulse", total_impulse)
    object.__setattr__(self, "peak_force", peak_force)
    object.__setattr__(self, "contact_steps", contact_steps)

  @property
  def identity(self) -> Tuple[str, str, int, int]:
    return (self.object_a, self.object_b, self.start_step, self.end_step)

  def to_dict(self) -> Mapping[str, Any]:
    return to_jsonable(self)


def _coerce_edge(value: Any) -> TemporalEdge:
  if isinstance(value, TemporalEdge):
    return value
  if isinstance(value, Mapping):
    return TemporalEdge(**dict(value))
  raise TypeError("edges must contain TemporalEdge values")


@dataclass(frozen=True)
class TemporalGraph:
  """A deterministic undirected temporal contact graph."""

  nodes: Tuple[str, ...] = ()
  edges: Tuple[TemporalEdge, ...] = ()

  def __post_init__(self) -> None:
    if isinstance(self.nodes, (str, bytes)):
      raise TypeError("nodes must be an iterable of identifiers")
    try:
      nodes = {_identifier(item, "node") for item in self.nodes}
    except TypeError as error:
      raise TypeError("nodes must be an iterable of identifiers") from error
    if isinstance(self.edges, (str, bytes, Mapping)):
      raise TypeError("edges must be an iterable of TemporalEdge values")
    try:
      edges = tuple(_coerce_edge(item) for item in self.edges)
    except TypeError as error:
      raise TypeError("edges must be an iterable of TemporalEdge values") from error
    by_identity: Dict[Tuple[str, str, int, int], TemporalEdge] = {}
    for edge in edges:
      existing = by_identity.get(edge.identity)
      if existing is not None and existing != edge:
        raise ValueError(
            "conflicting temporal edges for identity {!r}".format(edge.identity)
        )
      by_identity[edge.identity] = edge
      nodes.update((edge.object_a, edge.object_b))
    object.__setattr__(self, "nodes", tuple(sorted(nodes)))
    object.__setattr__(
        self,
        "edges",
        tuple(by_identity[identity] for identity in sorted(by_identity)),
    )

  def to_dict(self) -> Mapping[str, Any]:
    return to_jsonable(self)


def _contact_sort_key(record: ContactRecord) -> Tuple[Any, ...]:
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


def _weighted_vector(
    values: Sequence[Sequence[float]], weights: Sequence[float]
) -> Tuple[float, ...]:
  denominator = math.fsum(weights)
  if denominator == 0.0:
    count = len(values)
    return tuple(
        math.fsum(float(value[axis]) / count for value in values)
        for axis in range(len(values[0]))
    )
  return tuple(
      math.fsum(float(value[axis]) * weight / denominator
                for value, weight in zip(values, weights))
      for axis in range(len(values[0]))
  )


def _aggregate_normal(
    normals: Sequence[Sequence[float]], weights: Sequence[float]
) -> Tuple[float, float, float]:
  weighted = _weighted_vector(normals, weights)
  normalized = _normalized(weighted)
  if normalized is not None:
    return normalized  # type: ignore[return-value]
  for normal in sorted(tuple(tuple(item) for item in normals)):
    normalized = _normalized(normal)
    if normalized is not None:
      return normalized  # type: ignore[return-value]
  return (0.0, 0.0, 0.0)


def aggregate_contact_steps(
    records: Iterable[ContactRecord], step_rate: float
) -> Tuple[AggregatedContactStep, ...]:
  """Aggregates contact points into one force/geometry record per step and pair."""
  rate = _real(step_rate, "step_rate")
  if rate <= 0.0:
    raise ValueError("step_rate must be positive")
  if isinstance(records, (str, bytes, Mapping)):
    raise TypeError("records must be an iterable of ContactRecord values")
  try:
    values = tuple(records)
  except TypeError as error:
    raise TypeError("records must be an iterable of ContactRecord values") from error
  if not all(isinstance(record, ContactRecord) for record in values):
    raise TypeError("records must contain only ContactRecord values")

  grouped: Dict[Tuple[int, str, str], List[ContactRecord]] = defaultdict(list)
  for record in values:
    grouped[(record.step, record.object_a, record.object_b)].append(record)

  result = []
  for (step, object_a, object_b), group in sorted(grouped.items()):
    ordered = sorted(group, key=_contact_sort_key)
    forces = tuple(record.normal_force for record in ordered)
    try:
      total_force = math.fsum(forces)
    except OverflowError as error:
      raise ValueError("aggregated normal force must be finite") from error
    if not math.isfinite(total_force):
      raise ValueError("aggregated normal force must be finite")
    position = _weighted_vector(
        tuple(record.position for record in ordered), forces
    )
    normal = _aggregate_normal(
        tuple(record.normal for record in ordered), forces
    )
    result.append(AggregatedContactStep(
        step=step,
        object_a=object_a,
        object_b=object_b,
        position=position,
        normal=normal,
        normal_force=total_force,
        impulse=total_force / rate,
    ))
  return tuple(result)


def contact_log_to_temporal_graph(
    records: Iterable[ContactRecord],
    step_rate: float,
    force_threshold: float = 0.0,
    min_episode_impulse: float = 0.0,
) -> TemporalGraph:
  """Converts a contact log into maximal consecutive contact episodes."""
  threshold = _nonnegative(force_threshold, "force_threshold")
  minimum_impulse = _nonnegative(
      min_episode_impulse, "min_episode_impulse"
  )
  if isinstance(records, (str, bytes, Mapping)):
    raise TypeError("records must be an iterable of ContactRecord values")
  try:
    values = tuple(records)
  except TypeError as error:
    raise TypeError("records must be an iterable of ContactRecord values") from error
  if not all(isinstance(record, ContactRecord) for record in values):
    raise TypeError("records must contain only ContactRecord values")
  nodes = {endpoint for record in values
           for endpoint in (record.object_a, record.object_b)}
  aggregates = aggregate_contact_steps(values, step_rate)
  per_pair: Dict[Tuple[str, str], List[AggregatedContactStep]] = defaultdict(list)
  for aggregate in aggregates:
    if aggregate.normal_force >= threshold:
      per_pair[(aggregate.object_a, aggregate.object_b)].append(aggregate)

  edges = []
  for pair in sorted(per_pair):
    episode: List[AggregatedContactStep] = []
    for aggregate in per_pair[pair]:
      if episode and aggregate.step != episode[-1].step + 1:
        edge = _episode_edge(pair, episode)
        if edge.total_impulse >= minimum_impulse:
          edges.append(edge)
        episode = []
      episode.append(aggregate)
    if episode:
      edge = _episode_edge(pair, episode)
      if edge.total_impulse >= minimum_impulse:
        edges.append(edge)
  return TemporalGraph(nodes=tuple(nodes), edges=tuple(edges))


def _episode_edge(
    pair: Tuple[str, str], episode: Sequence[AggregatedContactStep]
) -> TemporalEdge:
  try:
    total_impulse = math.fsum(item.impulse for item in episode)
  except OverflowError as error:
    raise ValueError("episode impulse must be finite") from error
  return TemporalEdge(
      object_a=pair[0],
      object_b=pair[1],
      start_step=episode[0].step,
      end_step=episode[-1].step + 1,
      total_impulse=total_impulse,
      peak_force=max(item.normal_force for item in episode),
      contact_steps=tuple(episode),
  )


def graph_delta(
    factual: TemporalGraph,
    counterfactual: TemporalGraph,
    force_tolerance: float = 1e-6,
) -> GraphEdgeDelta:
  """Returns added, removed, and metric-changed temporal edge identities."""
  if not isinstance(factual, TemporalGraph):
    raise TypeError("factual must be a TemporalGraph")
  if not isinstance(counterfactual, TemporalGraph):
    raise TypeError("counterfactual must be a TemporalGraph")
  tolerance = _nonnegative(force_tolerance, "force_tolerance")
  factual_edges = {edge.identity: edge for edge in factual.edges}
  counterfactual_edges = {edge.identity: edge for edge in counterfactual.edges}
  factual_ids = set(factual_edges)
  counterfactual_ids = set(counterfactual_edges)

  added = [counterfactual_edges[identity].to_dict()
           for identity in sorted(counterfactual_ids - factual_ids)]
  removed = [factual_edges[identity].to_dict()
             for identity in sorted(factual_ids - counterfactual_ids)]
  changed = []
  for identity in sorted(factual_ids.intersection(counterfactual_ids)):
    factual_edge = factual_edges[identity]
    counterfactual_edge = counterfactual_edges[identity]
    if (
        abs(factual_edge.total_impulse - counterfactual_edge.total_impulse)
        > tolerance
        or abs(factual_edge.peak_force - counterfactual_edge.peak_force)
        > tolerance
    ):
      changed.append({
          "object_a": identity[0],
          "object_b": identity[1],
          "start_step": identity[2],
          "end_step": identity[3],
          "factual": {
              "total_impulse": factual_edge.total_impulse,
              "peak_force": factual_edge.peak_force,
          },
          "counterfactual": {
              "total_impulse": counterfactual_edge.total_impulse,
              "peak_force": counterfactual_edge.peak_force,
          },
      })
  return GraphEdgeDelta(added=added, removed=removed, changed=changed)


def _excluded_nodes(value: Iterable[str]) -> frozenset[str]:
  if isinstance(value, (str, bytes)):
    raise TypeError("exclude_nodes must be an iterable of identifiers")
  try:
    return frozenset(_identifier(item, "excluded node") for item in value)
  except TypeError as error:
    raise TypeError("exclude_nodes must be an iterable of identifiers") from error


def temporal_reachability(
    factual: TemporalGraph,
    counterfactual: TemporalGraph,
    target_id: str,
    intervention_start: int,
    exclude_nodes: Iterable[str] = (),
) -> Tuple[Tuple[str, ...], Mapping[str, Tuple[str, ...]]]:
  """Traverses the union graph along nondecreasing contact times."""
  if not isinstance(factual, TemporalGraph):
    raise TypeError("factual must be a TemporalGraph")
  if not isinstance(counterfactual, TemporalGraph):
    raise TypeError("counterfactual must be a TemporalGraph")
  target = _identifier(target_id, "target_id")
  start = _integer(intervention_start, "intervention_start")
  if start < 0:
    raise ValueError("intervention_start must be nonnegative")
  excluded = set(_excluded_nodes(exclude_nodes))
  excluded.discard(target)

  by_signature: Dict[Tuple[str, str, int, int], TemporalEdge] = {}
  for edge in factual.edges + counterfactual.edges:
    by_signature[edge.identity] = edge
  adjacency: Dict[str, List[TemporalEdge]] = defaultdict(list)
  for edge in by_signature.values():
    if edge.object_a in excluded or edge.object_b in excluded:
      continue
    adjacency[edge.object_a].append(edge)
    adjacency[edge.object_b].append(edge)
  for edges in adjacency.values():
    edges.sort(key=lambda edge: edge.identity)

  # Rank is arrival time, path length, then full lexicographic path.
  best: Dict[str, Tuple[int, int, Tuple[str, ...]]] = {
      target: (start, 0, (target,))
  }
  queue: List[Tuple[int, int, Tuple[str, ...], str]] = [
      (start, 0, (target,), target)
  ]
  while queue:
    arrival, hops, path, node = heapq.heappop(queue)
    if best.get(node) != (arrival, hops, path):
      continue
    for edge in adjacency.get(node, ()):
      other = edge.object_b if node == edge.object_a else edge.object_a
      contact_time = max(arrival, edge.start_step)
      if contact_time >= edge.end_step:
        continue
      candidate_path = path + (other,)
      candidate = (contact_time, hops + 1, candidate_path)
      if other not in best or candidate < best[other]:
        best[other] = candidate
        heapq.heappush(
            queue,
            (contact_time, hops + 1, candidate_path, other),
        )

  affected = tuple(sorted(node for node in best if node != target))
  paths = MappingProxyType({node: best[node][2] for node in affected})
  return affected, paths


def _distance(left: np.ndarray, right: np.ndarray) -> float:
  return math.hypot(*(
      float(left[index]) - float(right[index]) for index in range(len(left))
  ))


def state_affected(
    factual_log: SimulationLog,
    counterfactual_log: SimulationLog,
    target_id: str,
    position_epsilon: float = 1e-3,
    velocity_epsilon: float = 1e-3,
    quaternion_epsilon: float = 1e-3,
) -> Tuple[str, ...]:
  """Returns non-target IDs whose aligned rigid-body states diverge."""
  if not isinstance(factual_log, SimulationLog):
    raise TypeError("factual_log must be a SimulationLog")
  if not isinstance(counterfactual_log, SimulationLog):
    raise TypeError("counterfactual_log must be a SimulationLog")
  target = _identifier(target_id, "target_id")
  position_threshold = _nonnegative(position_epsilon, "position_epsilon")
  velocity_threshold = _nonnegative(velocity_epsilon, "velocity_epsilon")
  quaternion_threshold = _nonnegative(
      quaternion_epsilon, "quaternion_epsilon"
  )
  if factual_log.object_ids != counterfactual_log.object_ids:
    raise ValueError("factual and counterfactual object_ids must be identical")
  if factual_log.steps != counterfactual_log.steps:
    raise ValueError("factual and counterfactual steps must be identical")
  if factual_log.states.shape != counterfactual_log.states.shape:
    raise ValueError("factual and counterfactual state shapes must be identical")
  expected_shape = (
      len(factual_log.steps), len(factual_log.object_ids), 13
  )
  if factual_log.states.shape != expected_shape:
    raise ValueError("state arrays are not aligned to steps and object_ids")
  if not (
      np.isfinite(factual_log.states).all()
      and np.isfinite(counterfactual_log.states).all()
  ):
    raise ValueError("state arrays must contain only finite values")
  if target not in factual_log.object_ids:
    raise ValueError("target_id must appear in object_ids")

  affected = []
  for object_index, object_id in enumerate(factual_log.object_ids):
    if object_id == target:
      continue
    diverged = False
    for time_index in range(len(factual_log.steps)):
      factual = factual_log.states[time_index, object_index]
      counterfactual = counterfactual_log.states[time_index, object_index]
      position_distance = _distance(
          factual[POSITION_SLICE], counterfactual[POSITION_SLICE]
      )
      linear_distance = _distance(
          factual[LINEAR_VELOCITY_SLICE],
          counterfactual[LINEAR_VELOCITY_SLICE],
      )
      angular_distance = _distance(
          factual[ANGULAR_VELOCITY_SLICE],
          counterfactual[ANGULAR_VELOCITY_SLICE],
      )
      factual_quaternion = factual[QUATERNION_SLICE]
      counterfactual_quaternion = counterfactual[QUATERNION_SLICE]
      dot = math.fsum(
          float(left) * float(right)
          for left, right in zip(factual_quaternion, counterfactual_quaternion)
      )
      quaternion_distance = 2.0 * math.acos(min(1.0, abs(dot)))
      if (
          position_distance > position_threshold
          or linear_distance > velocity_threshold
          or angular_distance > velocity_threshold
          or quaternion_distance > quaternion_threshold
      ):
        diverged = True
        break
    if diverged:
      affected.append(object_id)
  return tuple(sorted(affected))


def extract_ground_truth(
    factual_log: SimulationLog,
    counterfactual_log: SimulationLog,
    target_id: str,
    intervention_start: int,
    exclude_nodes: Iterable[str] = (),
    force_threshold: float = 0.0,
    min_episode_impulse: float = 0.0,
    force_tolerance: float = 1e-6,
    position_epsilon: float = 1e-3,
    velocity_epsilon: float = 1e-3,
    quaternion_epsilon: float = 1e-3,
) -> GroundTruth:
  """Extracts graph deltas, causal reachability, and residual state effects."""
  factual_graph = contact_log_to_temporal_graph(
      factual_log.contacts,
      factual_log.step_rate,
      force_threshold=force_threshold,
      min_episode_impulse=min_episode_impulse,
  )
  counterfactual_graph = contact_log_to_temporal_graph(
      counterfactual_log.contacts,
      counterfactual_log.step_rate,
      force_threshold=force_threshold,
      min_episode_impulse=min_episode_impulse,
  )
  delta = graph_delta(
      factual_graph, counterfactual_graph, force_tolerance=force_tolerance
  )
  hard_affected, paths = temporal_reachability(
      factual_graph,
      counterfactual_graph,
      target_id,
      intervention_start,
      exclude_nodes=exclude_nodes,
  )
  state_ids = state_affected(
      factual_log,
      counterfactual_log,
      target_id,
      position_epsilon=position_epsilon,
      velocity_epsilon=velocity_epsilon,
      quaternion_epsilon=quaternion_epsilon,
  )
  hard_set = set(hard_affected)
  soft_affected = tuple(sorted(set(state_ids) - hard_set - {target_id}))
  propagation_paths = {node: paths[node] for node in hard_affected}
  return GroundTruth(
      graph_delta=delta,
      hard_affected=hard_affected,
      soft_affected=soft_affected,
      propagation_path=propagation_paths,
  )


__all__ = [
    "AggregatedContactStep",
    "TemporalEdge",
    "TemporalGraph",
    "aggregate_contact_steps",
    "contact_log_to_temporal_graph",
    "extract_ground_truth",
    "graph_delta",
    "state_affected",
    "temporal_reachability",
]
