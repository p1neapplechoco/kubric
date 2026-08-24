import dataclasses
import hashlib
import json
import math
import re

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
