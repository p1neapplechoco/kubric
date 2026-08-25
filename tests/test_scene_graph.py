"""Relation-graph extraction contracts for interventions.scene_graph."""

from __future__ import annotations

import numpy as np
import pytest

from interventions.logging import ContactRecord, SimulationLog
from interventions.scene_graph import (
    CONTACT_RELATION,
    PROXIMITY_RELATION,
    CausalEdge,
    NodeState,
    RelationEdge,
    SceneGraphFrame,
    build_relation_series,
    contact_activation_steps,
    propagation_tree,
)

_OBJECT_IDS = ("left", "right")
_RADII = {"left": 0.5, "right": 0.5}


def _log(
    positions,
    velocities=None,
    contacts=(),
    object_ids=_OBJECT_IDS,
    branch="factual",
):
  positions = np.asarray(positions, dtype=np.float64)
  states = np.zeros((positions.shape[0], len(object_ids), 13))
  states[:, :, 0:3] = positions
  states[:, :, 3] = 1.0
  if velocities is not None:
    states[:, :, 7:10] = np.asarray(velocities, dtype=np.float64)
  return SimulationLog(
      branch=branch,
      object_ids=object_ids,
      steps=tuple(range(positions.shape[0])),
      states=states,
      contacts=tuple(contacts),
      step_rate=240.0,
  )


def _contact(step, normal_force, object_a="left", object_b="right"):
  return ContactRecord(
      step=step,
      object_a=object_a,
      object_b=object_b,
      position=(0.0, 0.0, 0.0),
      normal=(1.0, 0.0, 0.0),
      normal_force=normal_force,
  )


def _pair_positions(distances):
  return [
      [(0.0, 0.0, 0.0), (float(distance), 0.0, 0.0)]
      for distance in distances
  ]


def test_logged_contacts_become_contact_edges_with_summed_force():
  log = _log(
      _pair_positions([1.0, 1.0]),
      contacts=(_contact(1, 30.0), _contact(1, 12.0)),
  )

  series = build_relation_series(log, _RADII)

  assert not series.frames[0].contact_edges()
  edge, = series.frames[1].contact_edges()
  assert edge.relation == CONTACT_RELATION
  assert edge.normal_force == pytest.approx(42.0)
  assert edge.gap == pytest.approx(0.0)


def test_proximity_edges_appear_only_within_the_near_margin():
  log = _log(_pair_positions([1.05, 1.30]))

  series = build_relation_series(log, _RADII, near_margin=0.12)

  edge, = series.frames[0].edges
  assert edge.relation == PROXIMITY_RELATION
  assert edge.gap == pytest.approx(0.05)
  assert series.frames[1].edges == ()


def test_a_contacting_pair_yields_one_edge_rather_than_two():
  log = _log(_pair_positions([1.0]), contacts=(_contact(0, 5.0),))

  series = build_relation_series(log, _RADII, near_margin=0.5)

  assert len(series.frames[0].edges) == 1
  assert series.frames[0].edges[0].relation == CONTACT_RELATION


def test_approach_rate_is_positive_only_while_endpoints_close():
  closing = _log(
      _pair_positions([1.05]), velocities=[[(2.0, 0.0, 0.0), (0.0, 0.0, 0.0)]]
  )
  separating = _log(
      _pair_positions([1.05]), velocities=[[(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0)]]
  )

  closing_edge, = build_relation_series(closing, _RADII).frames[0].edges
  separating_edge, = build_relation_series(separating, _RADII).frames[0].edges

  assert closing_edge.approach_rate == pytest.approx(2.0)
  assert closing_edge.approaching
  assert separating_edge.approach_rate == pytest.approx(-2.0)
  assert not separating_edge.approaching


def test_excluded_endpoints_keep_contacts_but_lose_proximity_edges():
  log = _log(_pair_positions([1.05, 1.05]), contacts=(_contact(1, 9.0),))

  series = build_relation_series(
      log, _RADII, proximity_exclude=("left",)
  )

  assert series.frames[0].edges == ()
  edge, = series.frames[1].edges
  assert edge.relation == CONTACT_RELATION


def test_absent_objects_form_no_edges():
  log = _log(_pair_positions([1.05, 1.05]))
  presence = np.array([[True, True], [True, False]])

  series = build_relation_series(log, _RADII, presence=presence)

  assert len(series.frames[0].edges) == 1
  assert series.frames[1].edges == ()
  assert not series.frames[1].nodes[1].present


def test_moving_ids_report_only_present_objects_above_the_threshold():
  log = _log(
      _pair_positions([1.05]), velocities=[[(0.5, 0.0, 0.0), (0.0, 0.0, 0.0)]]
  )

  frame = build_relation_series(log, _RADII).frames[0]

  assert frame.moving_ids(1e-3) == ("left",)
  assert frame.moving_ids(1.0) == ()


def test_radii_must_cover_exactly_the_logged_object_ids():
  log = _log(_pair_positions([1.05]))

  with pytest.raises(ValueError, match="radii must cover"):
    build_relation_series(log, {"left": 0.5})


def test_unknown_excluded_object_ids_are_rejected():
  log = _log(_pair_positions([1.05]))

  with pytest.raises(ValueError, match="unknown object_ids"):
    build_relation_series(log, _RADII, proximity_exclude=("ghost",))


def test_relation_edges_canonicalize_their_endpoint_order():
  edge = RelationEdge(
      object_a="right", object_b="left",
      relation=PROXIMITY_RELATION, gap=0.02, approach_rate=-1.0,
  )

  assert edge.pair == ("left", "right")


def test_only_contact_relations_may_carry_a_normal_force():
  with pytest.raises(ValueError, match="only contact relations"):
    RelationEdge(
        object_a="left", object_b="right", relation=PROXIMITY_RELATION,
        gap=0.02, approach_rate=0.0, normal_force=3.0,
    )


def test_frames_reject_repeated_pairs_and_unknown_endpoints():
  nodes = tuple(
      NodeState(
          object_id=object_id, position=(0.0, 0.0, 0.0), speed=0.0,
          present=True,
      )
      for object_id in _OBJECT_IDS
  )
  edge = RelationEdge(
      object_a="left", object_b="right", relation=PROXIMITY_RELATION,
      gap=0.02, approach_rate=0.0,
  )
  stranger = RelationEdge(
      object_a="left", object_b="ghost", relation=PROXIMITY_RELATION,
      gap=0.02, approach_rate=0.0,
  )

  with pytest.raises(ValueError, match="must not repeat an endpoint pair"):
    SceneGraphFrame(step=0, nodes=nodes, edges=(edge, edge))
  with pytest.raises(ValueError, match="must name nodes of the same frame"):
    SceneGraphFrame(step=0, nodes=nodes, edges=(stranger,))


def test_contact_activation_steps_report_the_first_contact_per_pair():
  log = _log(
      _pair_positions([2.0, 1.0, 2.0, 1.0]),
      contacts=(_contact(1, 4.0), _contact(3, 6.0)),
  )

  activations = contact_activation_steps(build_relation_series(log, _RADII))

  assert dict(activations) == {("left", "right"): 1}


def test_propagation_tree_collapses_shared_prefixes_into_unique_hops():
  edges = propagation_tree({
      "b": ("t", "b"),
      "c": ["t", "b", "c"],
      "d": ("t", "b", "c", "d"),
  })

  assert edges == (
      CausalEdge(parent="b", child="c", hop=2),
      CausalEdge(parent="c", child="d", hop=3),
      CausalEdge(parent="t", child="b", hop=1),
  )


def test_propagation_tree_keeps_the_shallowest_hop_for_a_repeated_pair():
  edges = propagation_tree({
      "b": ("t", "b"),
      "z": ("t", "x", "t", "b", "z"),
  })

  assert CausalEdge(parent="t", child="b", hop=1) in edges
  assert not any(
      edge.parent == "t" and edge.child == "b" and edge.hop != 1
      for edge in edges
  )


def test_propagation_tree_rejects_paths_that_do_not_end_at_their_key():
  with pytest.raises(ValueError, match="must end at their key"):
    propagation_tree({"b": ("t", "c")})


def test_series_exposes_ordered_steps_and_frame_lookup():
  log = _log(_pair_positions([1.05, 1.05, 1.05]))

  series = build_relation_series(log, _RADII)

  assert series.steps == (0, 1, 2)
  assert series.frame_at(2) is series.frames[2]
  with pytest.raises(KeyError):
    series.frame_at(9)
