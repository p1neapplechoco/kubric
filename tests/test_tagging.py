"""Tests for deterministic ground-truth tag derivation."""

from interventions.schema import GraphEdgeDelta, GroundTruth
from interventions.tagging import derive_tags


def _edge(object_a, object_b, start=0, end=1):
  return {
      "object_a": object_a,
      "object_b": object_b,
      "start_step": start,
      "end_step": end,
  }


def test_derive_tags_reports_delta_direct_contact_and_cascade_rules():
  truth = GroundTruth(
      graph_delta=GraphEdgeDelta(
          added=(_edge("target", "a"),),
          removed=(_edge("target", "b", 1, 2),),
          changed=({
              **_edge("a", "b", 2, 3),
              "factual": {"peak_force": 1},
              "counterfactual": {"peak_force": 2},
          },),
      ),
      hard_affected=("a", "b"),
      propagation_path={
          "a": ("target", "a"),
          "b": ("target", "a", "b"),
      },
  )

  assert derive_tags(truth, target_id="target") == (
      "cascade",
      "contact_added",
      "contact_changed",
      "contact_removed",
      "direct_contact",
  )


def test_derive_tags_identifies_environment_only_delta():
  truth = GroundTruth(
      graph_delta=GraphEdgeDelta(added=(
          _edge("target", "floor"),
          _edge("floor", "wall", 2, 3),
      )),
  )

  tags = derive_tags(
      truth,
      target_id="target",
      environment_ids=("wall", "floor"),
  )

  assert tags == ("contact_added", "environment_only", "target_only")


def test_derive_tags_target_only_allows_target_incident_non_environment_delta():
  truth = GroundTruth(
      graph_delta=GraphEdgeDelta(removed=(_edge("target", "object"),)),
  )

  assert derive_tags(truth, target_id="target") == (
      "contact_removed",
      "target_only",
  )


def test_derive_tags_target_only_rejects_non_environment_edge_away_from_target():
  truth = GroundTruth(
      graph_delta=GraphEdgeDelta(changed=({
          **_edge("a", "b"),
          "factual": {"total_impulse": 1},
          "counterfactual": {"total_impulse": 2},
      },)),
  )

  assert derive_tags(
      truth, target_id="target", environment_ids=("floor",)
  ) == ("contact_changed",)


def test_derive_tags_null_effect_and_quality_flags_are_independent_and_sorted():
  truth = GroundTruth(graph_delta=GraphEdgeDelta())

  first = derive_tags(
      truth,
      target_id="target",
      unstable=True,
      preintervention_mismatch=True,
  )
  second = derive_tags(
      truth,
      target_id="target",
      unstable=True,
      preintervention_mismatch=True,
  )

  assert first == second == (
      "null_effect",
      "preintervention_mismatch",
      "target_only",
      "unstable",
  )


def test_derive_tags_does_not_mark_target_only_when_an_object_is_affected():
  truth = GroundTruth(
      graph_delta=GraphEdgeDelta(),
      soft_affected=("a",),
  )

  assert derive_tags(truth, target_id="target") == ()


def test_logging_graph_and_tagging_api_is_exported_from_package():
  import interventions

  assert interventions.ContactRecord.__name__ == "ContactRecord"
  assert interventions.TemporalGraph.__name__ == "TemporalGraph"
  assert interventions.extract_ground_truth.__name__ == "extract_ground_truth"
  assert interventions.derive_tags is derive_tags
