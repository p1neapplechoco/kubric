import dataclasses
import hashlib
import json
import math

import pytest

from scripts import trajectory_demo_spec as demo_spec


EXPECTED_IDS = (
    "breaker", "floor", "rack_01", "rack_02", "rack_03", "rack_04",
    "rack_05", "rack_06", "side_01", "side_02", "target",
)


def _replace_object(spec, object_id, **changes):
    objects = tuple(
        dataclasses.replace(obj, **changes) if obj.object_id == object_id else obj
        for obj in spec.objects
    )
    return dataclasses.replace(spec, objects=objects)


def test_canonical_spec_contract():
    spec = demo_spec.FORKED_RACK_SPEC
    assert spec.version == "forked_rack_v1"
    assert spec.object_ids == EXPECTED_IDS
    assert len(spec.objects) == 11
    assert spec.ball_ids == ("breaker", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05", "rack_06", "side_01", "side_02")
    assert spec.main_ball_ids == ("breaker", "rack_01", "rack_02", "rack_03", "rack_04", "rack_05", "rack_06")
    assert spec.side_ball_ids == ("side_01", "side_02")
    assert (spec.num_steps, spec.frame_range, spec.frame_rate, spec.step_rate) == (200, (0, 20), 24, 240)
    assert spec.intervention_window == (40, 160)
    assert spec.target_id == "target"
    assert all(obj.quaternion == (1.0, 0.0, 0.0, 0.0) for obj in spec.objects)
    assert dataclasses.is_dataclass(spec) and dataclasses.is_dataclass(spec.objects[0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.num_steps = 1


def test_calibrated_scene_values_are_exact():
    spec = demo_spec.FORKED_RACK_SPEC
    assert (spec.scene_bounds, spec.gravity, spec.intervention_recipe,
            spec.intervention_magnitude, spec.push_mass, spec.path_start,
            spec.path_end) == (((-4.5, -4.5, -1.0), (4.5, 4.5, 2.0)),
                                (0, 0, 0), "create_collision", 1.2, 2.0,
                                (-2.0, -0.25, 0.18), (2.0, -0.25, 0.18))
    expected = {
        "breaker": ("sphere", 0.22, 1.0, (0.25, 0.60, 0.22), False, 0.02, 0.65, "ball", "main", (0.95, 0.45, 0.08), 1, False),
        "floor": ("cube", (4.0, 4.0, 0.25), 1.0, (0, 0, -0.25), True, 0, 0, "floor", None, (0.055, 0.19, 0.12), None, False),
        "rack_01": ("sphere", 0.22, 1.0, (0.72, 0.60, 0.22), False, 0.02, 0.65, "ball", "main", (0.15, 0.55, 0.92), 2, False),
        "rack_02": ("sphere", 0.22, 1.0, (1.18, 0.37, 0.22), False, 0.02, 0.65, "ball", "main", (0.84, 0.16, 0.20), 3, False),
        "rack_03": ("sphere", 0.22, 1.0, (1.18, 0.83, 0.22), False, 0.02, 0.65, "ball", "main", (0.48, 0.22, 0.78), 4, False),
        "rack_04": ("sphere", 0.22, 1.0, (2.00, 1.32, 0.22), False, 0.02, 0.65, "ball", "main", (0.96, 0.78, 0.08), 5, False),
        "rack_05": ("sphere", 0.22, 1.0, (1.64, 0.60, 0.22), False, 0.02, 0.65, "ball", "main", (0.10, 0.62, 0.30), 6, False),
        "rack_06": ("sphere", 0.22, 1.0, (1.64, 1.06, 0.22), False, 0.02, 0.65, "ball", "main", (0.50, 0.18, 0.10), 7, False),
        "side_01": ("sphere", 0.22, 1.0, (0.25, -0.48, 0.22), False, 0.02, 0.65, "ball", "side", (0.08, 0.42, 0.90), 8, False),
        "side_02": ("sphere", 0.22, 1.0, (0.72, -0.48, 0.22), False, 0.02, 0.65, "ball", "side", (0.82, 0.12, 0.18), 9, True),
        "target": ("cube", 0.18, 2.0, (-2.0, -0.25, 0.18), True, 0.02, 0.65, "target", None, (0.58, 0.27, 0.08), None, False),
    }
    for obj in spec.objects:
        assert (obj.shape, obj.size, obj.mass, obj.position, obj.static,
                obj.friction, obj.restitution, obj.visual_role, obj.group,
                obj.color, obj.ball_number, obj.striped) == expected[obj.object_id]


def test_payload_digest_and_summary_preserve_order():
    spec = demo_spec.FORKED_RACK_SPEC
    payload = demo_spec.canonical_spec_payload(spec)
    assert list(obj["object_id"] for obj in payload["objects"]) == list(EXPECTED_IDS)
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert demo_spec.spec_sha256(spec) == expected
    assert demo_spec.demo_spec_summary(spec) == {"object_count": 11, "sha256": expected, "source_frames": 200, "version": "forked_rack_v1"}
    changed = _replace_object(spec, "breaker", quaternion=(0.0, 0.0, 0.0, 1.0))
    assert demo_spec.spec_sha256(changed) != expected


@pytest.mark.parametrize("objects, pattern", [
    (lambda s: tuple(o for o in s.objects if o.object_id != "floor"), "floor"),
    (lambda s: tuple(o for o in s.objects if o.object_id != "side_02"), "nine balls"),
    (lambda s: (s.objects[1], s.objects[0], *s.objects[2:]), "canonical.*order"),
    (lambda s: (dataclasses.replace(s.objects[0], object_id="rack_01"), *s.objects[1:]), "unique|duplicate"),
])
def test_invalid_object_layout_rejected(objects, pattern):
    with pytest.raises((TypeError, ValueError), match=pattern):
        demo_spec.validate_demo_spec(dataclasses.replace(demo_spec.FORKED_RACK_SPEC, objects=objects(demo_spec.FORKED_RACK_SPEC)))


@pytest.mark.parametrize("object_id, changes, pattern", [
    ("breaker", {"object_id": ""}, "object_id"), ("breaker", {"shape": "capsule"}, "shape"),
    ("breaker", {"size": 0.0}, "size"), ("breaker", {"size": True}, "size"),
    ("breaker", {"size": (0.22, 0.22)}, "three components"), ("breaker", {"mass": 0.0}, "mass"),
    ("breaker", {"position": (math.nan, 0.6, 0.22)}, "finite"), ("breaker", {"quaternion": (0, 0, 0, 0)}, "quaternion"),
    ("breaker", {"visual_role": "prop"}, "visual_role"), ("breaker", {"group": "side"}, "group"),
    ("breaker", {"ball_number": True}, "ball numbers"), ("breaker", {"static": True}, "dynamic sphere"),
    ("breaker", {"shape": "cube"}, "dynamic sphere"), ("target", {"static": False}, "static cube"),
    ("floor", {"shape": "sphere"}, "static cube"),
])
def test_invalid_object_mutants_rejected(object_id, changes, pattern):
    with pytest.raises((TypeError, ValueError), match=pattern):
        demo_spec.validate_demo_spec(_replace_object(demo_spec.FORKED_RACK_SPEC, object_id, **changes))


@pytest.mark.parametrize("changes, pattern", [
    ({"target_id": "missing_target"}, "target"), ({"intervention_window": (-1, 160)}, "window"),
    ({"seed": -1}, "seed"), ({"frame_range": (0, 21)}, "200"),
    ({"frame_range": (0, 8), "step_rate": 25}, "integral"),
])
def test_invalid_scene_mutants_rejected(changes, pattern):
    with pytest.raises((TypeError, ValueError), match=pattern):
        demo_spec.validate_demo_spec(dataclasses.replace(demo_spec.FORKED_RACK_SPEC, **changes))


@pytest.mark.parametrize("objects, pattern", [
    (None, "objects"),
    ((None,), "DemoObjectSpec"),
])
def test_malformed_objects_rejected(objects, pattern):
    with pytest.raises((TypeError, ValueError), match=pattern):
        demo_spec.validate_demo_spec(dataclasses.replace(demo_spec.FORKED_RACK_SPEC, objects=objects))


@pytest.mark.parametrize("field, value", [
    ("frame_range", None), ("frame_range", (0,)), ("frame_range", (0, 20, 1)),
    ("intervention_window", None), ("intervention_window", (0,)),
    ("scene_bounds", None), ("scene_bounds", ((0, 0, 0),)),
    ("scene_bounds", ((0, 0), (1, 1, 1))),
])
def test_malformed_scene_containers_rejected(field, value):
    with pytest.raises((TypeError, ValueError), match="must|bounds|window|range|objects"):
        demo_spec.validate_demo_spec(dataclasses.replace(demo_spec.FORKED_RACK_SPEC, **{field: value}))


def test_role_counts_and_role_invariants_are_structural():
    objects = tuple(dataclasses.replace(obj, visual_role="target") if obj.object_id == "floor" else obj for obj in demo_spec.FORKED_RACK_SPEC.objects)
    with pytest.raises(ValueError, match="nine balls|target|floor"):
        demo_spec.validate_demo_spec(dataclasses.replace(demo_spec.FORKED_RACK_SPEC, objects=objects))
