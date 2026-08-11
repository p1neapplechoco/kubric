"""Backend-independent tools for counterfactual trajectory interventions."""

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
from interventions.kinematic_simulator import (
    KinematicDragSimulator,
    KinematicSimulator,
)
from interventions.tagging import derive_tags
from interventions.trajectory import (
    RECIPE_PROFILE_SEMANTICS,
    build_path,
    max_position_deviation,
    perturb_path,
    validate_path,
)


__all__ = [
    "ANGULAR_VELOCITY_SLICE",
    "AggregatedContactStep",
    "CameraConfig",
    "ContactLogger",
    "ContactRecord",
    "GraphEdgeDelta",
    "GroundTruth",
    "INTERVENTION_RECIPES",
    "Intervention",
    "KinematicDragSimulator",
    "LINEAR_VELOCITY_SLICE",
    "KinematicSimulator",
    "ObjectConfig",
    "POSITION_SLICE",
    "QUATERNION_SLICE",
    "RECIPE_PROFILE_SEMANTICS",
    "STATE_INDEX",
    "SceneConfig",
    "SimulationLog",
    "TemporalEdge",
    "TemporalGraph",
    "aggregate_contact_steps",
    "build_path",
    "contact_log_to_temporal_graph",
    "derive_tags",
    "extract_ground_truth",
    "graph_delta",
    "max_position_deviation",
    "perturb_path",
    "read_simulation_log",
    "state_affected",
    "state_index",
    "temporal_reachability",
    "to_jsonable",
    "validate_path",
    "write_simulation_log",
]
