"""Tests for deterministic temporal contact-graph extraction."""

import json
import math
import random
from collections import defaultdict, deque

import numpy as np
import pytest

from interventions.graph_extraction import (
    AggregatedContactStep,
    TemporalEdge,
    TemporalGraph,
    aggregate_contact_steps,
    contact_log_to_temporal_graph,
    extract_ground_truth,
    graph_delta,
    state_affected,
    temporal_reachability,
)
from interventions.logging import ContactRecord, SimulationLog


def _contact(
    step,
    object_a="a",
    object_b="b",
    *,
    force=1.0,
    position=(0.0, 0.0, 0.0),
    normal=(1.0, 0.0, 0.0),
):
  return ContactRecord(
      step=step,
      object_a=object_a,
      object_b=object_b,
      position=position,
      normal=normal,
      normal_force=force,
  )


def _edge(
    object_a,
    object_b,
    start,
    end=None,
    *,
    impulse=1.0,
    peak=1.0,
):
  return TemporalEdge(
      object_a=object_a,
      object_b=object_b,
      start_step=start,
      end_step=start + 1 if end is None else end,
      total_impulse=impulse,
      peak_force=peak,
      contact_steps=(),
  )


def _graph(*edges):
  return TemporalGraph(nodes=(), edges=edges)


def _brute_triggered_nodes(factual, counterfactual, target, start):
  """Independent finite-state oracle for randomized hard-set checks."""
  delta = graph_delta(factual, counterfactual)
  trigger_ids = {
      (
          record["object_a"],
          record["object_b"],
          record["start_step"],
          record["end_step"],
      )
      for bucket in (delta.added, delta.removed, delta.changed)
      for record in bucket
  }
  union = {edge.identity: edge for edge in factual.edges + counterfactual.edges}
  adjacency = defaultdict(list)
  for edge in union.values():
    adjacency[edge.object_a].append(edge)
    adjacency[edge.object_b].append(edge)

  initial = (target, start, False)
  reachable = {initial}
  queue = deque((initial,))
  reverse = defaultdict(set)
  while queue:
    state = queue.popleft()
    node, arrival, triggered = state
    for edge in adjacency[node]:
      contact_time = max(arrival, edge.start_step)
      if contact_time >= edge.end_step:
        continue
      other = edge.object_b if node == edge.object_a else edge.object_a
      successor = (
          other,
          contact_time,
          triggered or edge.identity in trigger_ids,
      )
      reverse[successor].add(state)
      if successor not in reachable:
        reachable.add(successor)
        queue.append(successor)

  causal_states = {state for state in reachable if state[2]}
  queue = deque(causal_states)
  while queue:
    state = queue.popleft()
    for predecessor in reverse[state]:
      if predecessor not in causal_states:
        causal_states.add(predecessor)
        queue.append(predecessor)
  return tuple(sorted({state[0] for state in causal_states} - {target}))


def _states(num_steps, num_objects):
  states = np.zeros((num_steps, num_objects, 13), dtype=float)
  states[:, :, 3] = 1.0
  return states


def _log(
    branch,
    object_ids,
    steps,
    *,
    states=None,
    contacts=(),
    step_rate=10,
):
  if states is None:
    states = _states(len(steps), len(object_ids))
  return SimulationLog(
      branch=branch,
      object_ids=object_ids,
      steps=steps,
      states=states,
      contacts=contacts,
      step_rate=step_rate,
  )


def test_aggregate_contact_steps_sums_force_and_uses_weighted_geometry():
  records = (
      _contact(3, force=3, position=(4, 0, 0), normal=(0, 1, 0)),
      _contact(3, force=1, position=(0, 0, 0), normal=(1, 0, 0)),
      _contact(1, "c", "a", force=2, position=(0, 0, 2), normal=(0, 0, 1)),
  )

  result = aggregate_contact_steps(records, step_rate=2)

  assert all(isinstance(item, AggregatedContactStep) for item in result)
  assert [(item.step, item.object_a, item.object_b) for item in result] == [
      (1, "a", "c"),
      (3, "a", "b"),
  ]
  aggregate = result[1]
  assert aggregate.normal_force == pytest.approx(4.0)
  assert aggregate.impulse == pytest.approx(2.0)
  assert aggregate.position == pytest.approx((3.0, 0.0, 0.0))
  assert aggregate.normal == pytest.approx(
      (1 / math.sqrt(10), 3 / math.sqrt(10), 0.0)
  )
  assert json.loads(json.dumps(aggregate.to_dict())) == aggregate.to_dict()


def test_aggregate_zero_force_fallback_is_finite_and_order_independent():
  records = (
      _contact(0, force=0, position=(2, 0, 0), normal=(1, 0, 0)),
      _contact(0, force=0, position=(0, 0, 0), normal=(-1, 0, 0)),
  )

  forward = aggregate_contact_steps(records, 10)
  reverse = aggregate_contact_steps(reversed(records), 10)

  assert forward == reverse
  assert forward[0].position == pytest.approx((1.0, 0.0, 0.0))
  assert forward[0].normal == (0.0, 0.0, 0.0)


def test_contact_log_to_temporal_graph_collapses_consecutive_thresholded_steps():
  records = (
      _contact(1, force=2),
      _contact(1, force=3, position=(1, 0, 0)),
      _contact(2, force=4),
      _contact(3, force=0.5),
      _contact(4, force=5),
      _contact(2, "c", "d", force=1),
  )

  graph = contact_log_to_temporal_graph(
      records,
      step_rate=10,
      force_threshold=1,
      min_episode_impulse=0.15,
  )

  assert graph.nodes == ("a", "b", "c", "d")
  assert [
      (edge.object_a, edge.object_b, edge.start_step, edge.end_step)
      for edge in graph.edges
  ] == [("a", "b", 1, 3), ("a", "b", 4, 5)]
  assert graph.edges[0].total_impulse == pytest.approx(0.9)
  assert graph.edges[0].peak_force == pytest.approx(5.0)
  assert tuple(item.step for item in graph.edges[0].contact_steps) == (1, 2)
  assert json.loads(json.dumps(graph.to_dict())) == graph.to_dict()


def test_temporal_edge_and_graph_canonicalize_and_validate():
  edge = _edge("z", "a", 2, 4)
  graph = TemporalGraph(nodes=("z", "a", "a"), edges=(edge,))

  assert (edge.object_a, edge.object_b) == ("a", "z")
  assert graph.nodes == ("a", "z")
  with pytest.raises(ValueError):
    _edge("a", "a", 0)
  with pytest.raises(ValueError):
    _edge("a", "b", 2, 2)


def test_temporal_edge_validates_summaries_against_contact_steps():
  contact_steps = (
      AggregatedContactStep(
          step=2,
          object_a="a",
          object_b="b",
          position=(0, 0, 0),
          normal=(1, 0, 0),
          normal_force=2,
          impulse=0.1,
      ),
      AggregatedContactStep(
          step=3,
          object_a="a",
          object_b="b",
          position=(0, 0, 0),
          normal=(1, 0, 0),
          normal_force=3,
          impulse=0.2,
      ),
  )

  edge = TemporalEdge("a", "b", 2, 4, 0.3, 3.0, contact_steps)

  assert edge.total_impulse == pytest.approx(0.3)
  with pytest.raises(ValueError, match="total_impulse"):
    TemporalEdge("a", "b", 2, 4, 0.4, 3.0, contact_steps)
  with pytest.raises(ValueError, match="peak_force"):
    TemporalEdge("a", "b", 2, 4, 0.3, 4.0, contact_steps)
  assert _edge("a", "b", 2, 4, impulse=99, peak=99).contact_steps == ()


def test_graph_delta_reports_added_removed_and_metric_changes():
  factual = _graph(
      _edge("a", "b", 0, 2, impulse=1, peak=3),
      _edge("c", "d", 2, impulse=2, peak=4),
      _edge("e", "f", 5, impulse=1, peak=1),
  )
  counterfactual = _graph(
      _edge("b", "a", 0, 2, impulse=1.5, peak=3),
      _edge("e", "f", 5, impulse=1, peak=1),
      _edge("g", "h", 4, impulse=3, peak=8),
  )

  delta = graph_delta(factual, counterfactual)

  assert [(item["object_a"], item["object_b"]) for item in delta.added] == [
      ("g", "h")
  ]
  assert [(item["object_a"], item["object_b"]) for item in delta.removed] == [
      ("c", "d")
  ]
  assert len(delta.changed) == 1
  assert delta.changed[0]["factual"] == {
      "peak_force": 3.0,
      "total_impulse": 1.0,
  }
  assert delta.changed[0]["counterfactual"] == {
      "peak_force": 3.0,
      "total_impulse": 1.5,
  }
  assert not graph_delta(factual, counterfactual, force_tolerance=1).changed


def test_temporal_reachability_closes_same_step_across_union_branches():
  factual = _graph(_edge("target", "b", 5), _edge("c", "d", 4))
  counterfactual = _graph(_edge("b", "c", 5), _edge("d", "e", 6))

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=5
  )

  assert affected == ("b", "c")
  assert paths == {
      "b": ("target", "b"),
      "c": ("target", "b", "c"),
  }


def test_temporal_reachability_requires_a_graph_delta_trigger():
  unchanged = _graph(_edge("target", "peer", 1))

  affected, paths = temporal_reachability(
      unchanged, unchanged, "target", intervention_start=0
  )

  assert affected == ()
  assert paths == {}


def test_temporal_reachability_propagates_after_target_direct_delta():
  factual = _graph(
      _edge("target", "a", 1),
      _edge("a", "b", 3),
  )
  counterfactual = _graph(_edge("a", "b", 3))

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a", "b")
  assert paths == {
      "a": ("target", "a"),
      "b": ("target", "a", "b"),
  }


def test_temporal_reachability_keeps_unchanged_prefix_to_downstream_delta():
  shared_prefix = _edge("target", "a", 1)
  factual = _graph(shared_prefix, _edge("a", "b", 3, impulse=1, peak=1))
  counterfactual = _graph(
      shared_prefix,
      _edge("a", "b", 3, impulse=2, peak=2),
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a", "b")
  assert paths == {
      "a": ("target", "a"),
      "b": ("target", "a", "b"),
  }


def test_temporal_reachability_propagates_from_both_delta_endpoints():
  shared_prefix = _edge("target", "a", 1)
  shared_suffix = _edge("a", "c", 3)
  factual = _graph(
      shared_prefix,
      _edge("a", "b", 2, impulse=1, peak=1),
      shared_suffix,
  )
  counterfactual = _graph(
      shared_prefix,
      _edge("a", "b", 2, impulse=2, peak=2),
      shared_suffix,
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a", "b", "c")
  assert paths["a"] == ("target", "a")
  assert paths["b"] == ("target", "a", "b")
  assert paths["c"] == ("target", "a", "b", "a", "c")


def test_temporal_reachability_cycle_closing_delta_triggers_later_suffix():
  factual = _graph(
      _edge("target", "a", 1),
      _edge("a", "b", 2),
      _edge("a", "b", 3, impulse=1, peak=1),
      _edge("a", "c", 4),
  )
  counterfactual = _graph(
      _edge("target", "a", 1),
      _edge("a", "b", 2),
      _edge("a", "b", 3, impulse=2, peak=2),
      _edge("a", "c", 4),
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a", "b", "c")
  assert paths["a"] == ("target", "a")
  assert paths["b"] == ("target", "a", "b")
  assert paths["c"] == ("target", "a", "b", "a", "c")


def test_temporal_reachability_respects_start_and_excluded_delta_endpoint():
  factual = _graph(
      _edge("target", "before", 1),
      _edge("target", "floor", 3),
      _edge("floor", "hidden", 4),
  )
  counterfactual = TemporalGraph()

  affected, paths = temporal_reachability(
      factual,
      counterfactual,
      "target",
      intervention_start=2,
      exclude_nodes=("floor",),
  )

  assert affected == ()
  assert paths == {}


def test_temporal_reachability_matches_randomized_finite_state_oracle():
  rng = random.Random(20260814)
  nodes = ("target", "a", "b", "c")
  possible = [
      (left, right, step)
      for left_index, left in enumerate(nodes)
      for right in nodes[left_index + 1:]
      for step in range(4)
  ]

  for _ in range(1000):
    factual_edges = []
    counterfactual_edges = []
    for left, right, step in possible:
      factual_present = rng.random() < 0.22
      counterfactual_present = rng.random() < 0.22
      factual_force = rng.choice((1.0, 2.0))
      counterfactual_force = (
          factual_force if rng.random() < 0.5 else 3.0 - factual_force
      )
      if factual_present:
        factual_edges.append(
            _edge(
                left,
                right,
                step,
                impulse=factual_force,
                peak=factual_force,
            )
        )
      if counterfactual_present:
        counterfactual_edges.append(
            _edge(
                left,
                right,
                step,
                impulse=counterfactual_force,
                peak=counterfactual_force,
            )
        )
    factual = _graph(*factual_edges)
    counterfactual = _graph(*counterfactual_edges)
    start = rng.randrange(3)

    affected, paths = temporal_reachability(
        factual, counterfactual, "target", intervention_start=start
    )

    assert affected == _brute_triggered_nodes(
        factual, counterfactual, "target", start
    )
    assert tuple(sorted(paths)) == affected


def test_temporal_reachability_ignores_delta_disconnected_from_target():
  shared_target_component = _edge("target", "a", 1)
  factual = _graph(
      shared_target_component,
      _edge("x", "y", 2, impulse=1, peak=1),
  )
  counterfactual = _graph(
      shared_target_component,
      _edge("x", "y", 2, impulse=2, peak=2),
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ()
  assert paths == {}


def test_temporal_reachability_treats_metric_changed_edge_as_trigger():
  factual = _graph(
      _edge("target", "a", 1, impulse=1, peak=1),
  )
  counterfactual = _graph(
      _edge("target", "a", 1, impulse=2, peak=2),
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a",)
  assert paths == {"a": ("target", "a")}


def test_temporal_reachability_rejects_backward_paths_and_excluded_transit():
  graph = _graph(
      _edge("target", "a", 5),
      _edge("a", "past", 4),
      _edge("a", "floor", 6),
      _edge("floor", "hidden", 7),
      _edge("a", "future", 8),
  )

  affected, paths = temporal_reachability(
      graph,
      TemporalGraph(),
      "target",
      intervention_start=5,
      exclude_nodes=("floor",),
  )

  assert affected == ("a", "future")
  assert paths["future"] == ("target", "a", "future")
  assert "past" not in paths
  assert "floor" not in paths
  assert "hidden" not in paths


def test_temporal_reachability_chooses_earliest_then_shortest_lexical_path():
  graph = _graph(
      _edge("target", "b", 1),
      _edge("target", "a", 1),
      _edge("a", "c", 2),
      _edge("b", "c", 2),
      _edge("c", "destination", 2),
      _edge("target", "destination", 3),
  )

  affected, paths = temporal_reachability(
      graph, TemporalGraph(), "target", intervention_start=0
  )

  assert affected == ("a", "b", "c", "destination")
  assert paths["c"] == ("target", "a", "c")
  assert paths["destination"] == ("target", "a", "c", "destination")


def test_temporal_reachability_preserves_later_shorter_intermediate_label():
  graph = _graph(
      _edge("target", "x", 0),
      _edge("x", "a", 0),
      _edge("target", "a", 1),
      _edge("a", "d", 2),
  )

  affected, paths = temporal_reachability(
      graph, TemporalGraph(), "target", intervention_start=0
  )

  assert affected == ("a", "d", "x")
  assert paths["a"] == ("target", "x", "a")
  assert paths["d"] == ("target", "a", "d")


def test_temporal_reachability_keeps_prefix_of_dominated_trigger_route():
  factual = _graph(
      _edge("target", "x", 0),
      _edge("x", "a", 0),
      _edge("target", "a", 1),
      _edge("a", "b", 2, impulse=1, peak=1),
  )
  counterfactual = _graph(
      _edge("target", "x", 0),
      _edge("x", "a", 0),
      _edge("target", "a", 1),
      _edge("a", "b", 2, impulse=2, peak=2),
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a", "b", "x")
  assert paths["x"] == ("target", "x")
  assert paths["b"] == ("target", "a", "b")


def test_temporal_reachability_keeps_equal_arrival_dominated_prefix_route():
  shared_prefix = (
      _edge("target", "x", 0),
      _edge("x", "a", 0),
      _edge("target", "a", 0),
  )
  factual = _graph(
      *shared_prefix,
      _edge("a", "b", 1, impulse=1, peak=1),
  )
  counterfactual = _graph(
      *shared_prefix,
      _edge("a", "b", 1, impulse=2, peak=2),
  )

  affected, paths = temporal_reachability(
      factual, counterfactual, "target", intervention_start=0
  )

  assert affected == ("a", "b", "x")
  assert paths["x"] == ("target", "x")
  assert paths["b"] == ("target", "a", "b")


def test_temporal_reachability_preserves_later_lexical_tie_label():
  graph = _graph(
      _edge("target", "z", 0),
      _edge("z", "a", 0),
      _edge("target", "b", 1),
      _edge("b", "a", 1),
      _edge("a", "d", 2),
  )

  _, paths = temporal_reachability(
      graph, TemporalGraph(), "target", intervention_start=0
  )

  assert paths["a"] == ("target", "z", "a")
  assert paths["d"] == ("target", "b", "a", "d")


def test_state_affected_uses_thresholds_and_sign_invariant_quaternions():
  object_ids = ("target", "a", "b")
  factual_states = _states(2, 3)
  counterfactual_states = factual_states.copy()
  counterfactual_states[:, 1, QUATERNION_INDEX] *= -1
  counterfactual_states[1, 1, 0] = 0.002
  counterfactual_states[1, 2, 7] = 0.002
  factual = _log("factual", object_ids, (0, 1), states=factual_states)
  counterfactual = _log(
      "counterfactual", object_ids, (0, 1), states=counterfactual_states
  )

  assert state_affected(
      factual,
      counterfactual,
      "target",
      position_epsilon=0.001,
      velocity_epsilon=0.001,
  ) == ("a", "b")
  assert state_affected(
      factual,
      counterfactual,
      "target",
      position_epsilon=0.01,
      velocity_epsilon=0.01,
  ) == ()


QUATERNION_INDEX = slice(3, 7)


def test_state_affected_detects_quaternion_angle_but_ignores_sign_only():
  factual_states = _states(1, 2)
  counterfactual_states = factual_states.copy()
  counterfactual_states[0, 0, QUATERNION_INDEX] *= -1
  angle = 0.02
  counterfactual_states[0, 1, QUATERNION_INDEX] = (
      math.cos(angle / 2),
      0,
      0,
      math.sin(angle / 2),
  )
  factual = _log("factual", ("target", "a"), (0,), states=factual_states)
  counterfactual = _log(
      "counterfactual", ("target", "a"), (0,), states=counterfactual_states
  )

  assert state_affected(
      factual, counterfactual, "target", quaternion_epsilon=0.01
  ) == ("a",)


def test_state_affected_normalizes_accepted_quaternions_before_angle():
  factual_states = _states(1, 2)
  counterfactual_states = factual_states.copy()
  quaternion = (0.9999995, 0.0, 0.0, 0.0)
  factual_states[0, 1, QUATERNION_INDEX] = quaternion
  counterfactual_states[0, 1, QUATERNION_INDEX] = quaternion
  factual = _log("factual", ("target", "a"), (0,), states=factual_states)
  counterfactual = _log(
      "counterfactual", ("target", "a"), (0,), states=counterfactual_states
  )

  assert state_affected(factual, counterfactual, "target") == ()


def test_state_affected_rejects_misaligned_logs_and_invalid_thresholds():
  factual = _log("factual", ("target", "a"), (0, 1))
  reordered = _log("counterfactual", ("a", "target"), (0, 1))
  shifted = _log("counterfactual", ("target", "a"), (1, 2))
  different_rate = _log(
      "counterfactual", ("target", "a"), (0, 1), step_rate=20
  )

  with pytest.raises(ValueError, match="object_ids"):
    state_affected(factual, reordered, "target")
  with pytest.raises(ValueError, match="steps"):
    state_affected(factual, shifted, "target")
  with pytest.raises(ValueError, match="step_rate"):
    state_affected(factual, different_rate, "target")
  with pytest.raises(ValueError, match="step_rate"):
    extract_ground_truth(factual, different_rate, "target", 0)
  with pytest.raises(ValueError):
    state_affected(factual, factual, "target", position_epsilon=-1)


def test_state_affected_start_step_excludes_preintervention_divergence():
  object_ids = ("target", "a")
  steps = (0, 5, 10)
  factual_states = _states(3, 2)
  counterfactual_states = factual_states.copy()
  counterfactual_states[0, 1, 0] = 0.1
  factual = _log("factual", object_ids, steps, states=factual_states)
  counterfactual = _log(
      "counterfactual", object_ids, steps, states=counterfactual_states
  )

  assert state_affected(factual, counterfactual, "target") == ("a",)
  assert state_affected(
      factual, counterfactual, "target", start_step=5
  ) == ()
  with pytest.raises(ValueError, match="start_step"):
    state_affected(factual, counterfactual, "target", start_step=11)

  truth = extract_ground_truth(
      factual, counterfactual, "target", intervention_start=5
  )
  assert truth.soft_affected == ()


def test_extract_ground_truth_combines_graph_reachability_and_state_effects():
  object_ids = ("target", "a", "b", "c", "d")
  steps = (0, 1, 2, 3)
  factual_states = _states(4, 5)
  counterfactual_states = factual_states.copy()
  counterfactual_states[3, 4, 0] = 0.1
  factual = _log(
      "factual",
      object_ids,
      steps,
      states=factual_states,
      contacts=(
          _contact(1, "target", "a", force=1),
          _contact(2, "a", "b", force=1),
      ),
  )
  counterfactual = _log(
      "counterfactual",
      object_ids,
      steps,
      states=counterfactual_states,
      contacts=(
          _contact(1, "target", "a", force=2),
          _contact(2, "a", "b", force=1),
          _contact(2, "a", "c", force=1),
      ),
  )

  truth = extract_ground_truth(
      factual,
      counterfactual,
      target_id="target",
      intervention_start=0,
  )

  assert truth.hard_affected == ("a", "b", "c")
  assert truth.soft_affected == ("d",)
  assert truth.propagation_path["b"] == ("target", "a", "b")
  assert set(truth.propagation_path) == set(truth.hard_affected)
  assert len(truth.graph_delta.added) == 1
  assert len(truth.graph_delta.changed) == 1


def test_extract_ground_truth_ignores_target_only_change_with_unchanged_graph():
  object_ids = ("target", "peer")
  steps = (0, 1)
  factual_states = _states(2, 2)
  counterfactual_states = factual_states.copy()
  counterfactual_states[1, 0, 0] = 1.0
  contacts = (_contact(1, "target", "peer", force=1),)
  factual = _log(
      "factual",
      object_ids,
      steps,
      states=factual_states,
      contacts=contacts,
  )
  counterfactual = _log(
      "counterfactual",
      object_ids,
      steps,
      states=counterfactual_states,
      contacts=contacts,
  )

  truth = extract_ground_truth(
      factual,
      counterfactual,
      target_id="target",
      intervention_start=1,
  )

  assert truth.graph_delta.added == ()
  assert truth.graph_delta.removed == ()
  assert truth.graph_delta.changed == ()
  assert truth.hard_affected == ()
  assert truth.soft_affected == ()
  assert truth.propagation_path == {}


def test_extract_ground_truth_reuses_force_tolerant_delta_for_reachability():
  object_ids = ("target", "peer")
  steps = (0, 1)
  factual = _log(
      "factual",
      object_ids,
      steps,
      contacts=(_contact(1, "target", "peer", force=1),),
  )
  counterfactual = _log(
      "counterfactual",
      object_ids,
      steps,
      contacts=(_contact(1, "target", "peer", force=1.01),),
  )

  truth = extract_ground_truth(
      factual,
      counterfactual,
      target_id="target",
      intervention_start=0,
      force_tolerance=1.0,
  )

  assert truth.graph_delta.changed == ()
  assert truth.hard_affected == ()
  assert truth.propagation_path == {}
