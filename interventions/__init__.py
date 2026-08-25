"""Stable public exports for counterfactual trajectory interventions.

Purpose: expose the supported backend-neutral intervention API from one package.
Public API: schemas, trajectories, logs, graph extraction, scene-graph relations,
tags, and lazily loaded simulator/twin-runner entry points listed in ``__all__``.
Dependencies: eager exports use the standard library and NumPy-facing modules;
Kubric and PyBullet backends are imported only when a lazy export is requested.
Trust boundary: this namespace defines stable access paths, but does not add
provenance or simulator-origin attestation to values returned by its modules.
"""

from importlib import import_module

from interventions.schema import (
    CameraConfig,
    GraphEdgeDelta,
    GroundTruth,
    INTERVENTION_RECIPES,
    Intervention,
    ObjectConfig,
    SceneConfig,
    to_jsonable,
)
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
from interventions.scene_graph import (
    CONTACT_RELATION,
    PROXIMITY_RELATION,
    RELATION_KINDS,
    CausalEdge,
    NodeState,
    RelationEdge,
    RelationSeries,
    SceneGraphFrame,
    build_relation_series,
    contact_activation_steps,
    propagation_tree,
)
from interventions.logging import (
    ANGULAR_VELOCITY_SLICE,
    LINEAR_VELOCITY_SLICE,
    POSITION_SLICE,
    QUATERNION_SLICE,
    STATE_INDEX,
    ContactLogger,
    ContactRecord,
    SimulationLog,
    read_simulation_log,
    state_index,
    write_simulation_log,
)
from interventions.tagging import derive_tags
from interventions.trajectory import (
    RECIPE_PROFILE_SEMANTICS,
    build_path,
    max_position_deviation,
    perturb_path,
    validate_path,
)


_LAZY_EXPORTS = {
    "KinematicDragSimulator": (
        "interventions.kinematic_simulator",
        "KinematicDragSimulator",
    ),
    "KinematicSimulator": (
        "interventions.kinematic_simulator",
        "KinematicSimulator",
    ),
    "extract_pair_ground_truth": (
        "interventions.twin_runner",
        "extract_pair_ground_truth",
    ),
    "generate_paired_instance": (
        "interventions.twin_runner",
        "generate_paired_instance",
    ),
    "read_paired_artifact": (
        "interventions.twin_runner",
        "read_paired_artifact",
    ),
    "write_paired_artifact": (
        "interventions.twin_runner",
        "write_paired_artifact",
    ),
}


def __getattr__(name):
  try:
    module_name, attribute_name = _LAZY_EXPORTS[name]
  except KeyError as error:
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name)
    ) from error
  value = getattr(import_module(module_name), attribute_name)
  globals()[name] = value
  return value


def __dir__():
  return sorted(set(globals()) | set(__all__))


__all__ = [
    "ANGULAR_VELOCITY_SLICE",
    "AggregatedContactStep",
    "CONTACT_RELATION",
    "CameraConfig",
    "CausalEdge",
    "ContactLogger",
    "ContactRecord",
    "GraphEdgeDelta",
    "GroundTruth",
    "INTERVENTION_RECIPES",
    "Intervention",
    "KinematicDragSimulator",
    "LINEAR_VELOCITY_SLICE",
    "KinematicSimulator",
    "NodeState",
    "ObjectConfig",
    "PROXIMITY_RELATION",
    "POSITION_SLICE",
    "QUATERNION_SLICE",
    "RECIPE_PROFILE_SEMANTICS",
    "RELATION_KINDS",
    "RelationEdge",
    "RelationSeries",
    "STATE_INDEX",
    "SceneConfig",
    "SceneGraphFrame",
    "SimulationLog",
    "TemporalEdge",
    "TemporalGraph",
    "aggregate_contact_steps",
    "build_path",
    "build_relation_series",
    "contact_activation_steps",
    "contact_log_to_temporal_graph",
    "derive_tags",
    "extract_ground_truth",
    "extract_pair_ground_truth",
    "generate_paired_instance",
    "graph_delta",
    "max_position_deviation",
    "perturb_path",
    "propagation_tree",
    "read_paired_artifact",
    "read_simulation_log",
    "state_affected",
    "state_index",
    "temporal_reachability",
    "to_jsonable",
    "validate_path",
    "write_paired_artifact",
    "write_simulation_log",
]
