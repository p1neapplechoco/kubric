"""Bundle loading and drawing contracts for the scene-graph renderer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import render_scene_graph_video as renderer
from scripts.trajectory_demo_spec import FORKED_RACK_SPEC

_BRANCHES = ("normal", "trajectory_changed", "target_removed")


def _states(num_steps: int) -> np.ndarray:
  states = np.zeros((num_steps, len(FORKED_RACK_SPEC.object_ids), 13))
  for index, item in enumerate(FORKED_RACK_SPEC.objects):
    states[:, index, 0:3] = np.asarray(item.position, dtype=np.float64)
  states[:, :, 3] = 1.0
  states[:, 0, 7] = 1.5
  return states


def _contact_record(step: int) -> dict:
  return {
      "step": step,
      "object_a": "breaker",
      "object_b": "rack_01",
      "position": [0.5, 0.6, 0.22],
      "normal": [1.0, 0.0, 0.0],
      "normal_force": 120.0,
      "contact_distance": -0.001,
      "schema_version": "1.0",
  }


def _write_bundle(directory, num_steps: int = 4):
  start, end = FORKED_RACK_SPEC.intervention_window
  summary = {
      "branches": {},
      "demo_spec": {},
      "ground_truth": {
          "graph_delta": {"added": [], "removed": [], "changed": []},
          "hard_affected": ["rack_01"],
          "soft_affected": [],
          "propagation_path": {"rack_01": ["target", "breaker", "rack_01"]},
          "schema_version": "1.0",
      },
      "intervention_end": end,
      "intervention_start": start,
      "intervention_window": [start, end],
      "object_ids": list(FORKED_RACK_SPEC.object_ids),
      "seed": FORKED_RACK_SPEC.seed,
      "step_rate": float(FORKED_RACK_SPEC.step_rate),
  }
  (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
  (directory / "contacts.json").write_text(
      json.dumps({
          "normal": [],
          "trajectory_changed": [_contact_record(2)],
          "target_removed": [],
      }),
      encoding="utf-8",
  )
  presence = np.ones(
      (num_steps, len(FORKED_RACK_SPEC.object_ids)), dtype=np.bool_
  )
  removed = presence.copy()
  removed[2:, list(FORKED_RACK_SPEC.object_ids).index("target")] = False
  for branch in _BRANCHES:
    np.save(directory / f"{branch}_states.npy", _states(num_steps))
    np.save(
        directory / f"{branch}_presence.npy",
        removed if branch == "target_removed" else presence,
    )
  return summary


def test_collision_radii_cover_every_canonical_object():
  radii = renderer._collision_radii(FORKED_RACK_SPEC)

  assert set(radii) == set(FORKED_RACK_SPEC.object_ids)
  assert radii["breaker"] == pytest.approx(0.22)
  assert radii["floor"] == pytest.approx(0.25)
  assert all(value > 0.0 for value in radii.values())


def test_object_styles_label_balls_by_number_and_mark_the_target():
  styles = renderer._object_styles(FORKED_RACK_SPEC)

  assert styles["breaker"]["label"] == "1"
  assert styles["target"]["label"] == "T"
  assert styles["floor"]["role"] == "floor"


def test_load_demo_bundle_derives_a_series_for_every_branch(tmp_path):
  _write_bundle(tmp_path)

  summary, bundles = renderer.load_demo_bundle(tmp_path)

  assert set(bundles) == set(_BRANCHES)
  assert summary["object_ids"] == list(FORKED_RACK_SPEC.object_ids)
  for bundle in bundles.values():
    assert len(bundle.series.frames) == 4
    assert bundle.steps == (0, 1, 2, 3)
  assert bundles["trajectory_changed"].series.frames[2].contact_edges()
  assert not bundles["normal"].series.frames[2].contact_edges()


def test_loaded_series_never_links_the_floor_by_proximity(tmp_path):
  _write_bundle(tmp_path)

  _, bundles = renderer.load_demo_bundle(tmp_path)

  for bundle in bundles.values():
    for frame in bundle.series.frames:
      for edge in frame.edges:
        if edge.relation != renderer.CONTACT_RELATION:
          assert "floor" not in edge.pair


def test_removed_targets_drop_out_of_the_series(tmp_path):
  _write_bundle(tmp_path)

  _, bundles = renderer.load_demo_bundle(tmp_path)
  frames = bundles["target_removed"].series.frames

  present = {
      node.object_id: node.present for node in frames[3].nodes
  }
  assert not present["target"]
  assert all("target" not in edge.pair for edge in frames[3].edges)


def test_load_demo_bundle_rejects_object_ids_from_another_spec(tmp_path):
  _write_bundle(tmp_path)
  summary = json.loads((tmp_path / "summary.json").read_text())
  summary["object_ids"] = ["a", "b"]
  (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

  with pytest.raises(ValueError, match="canonical demo spec"):
    renderer.load_demo_bundle(tmp_path)


def test_spatial_limits_frame_every_branch_identically(tmp_path):
  _write_bundle(tmp_path)
  _, bundles = renderer.load_demo_bundle(tmp_path)
  radii = renderer._collision_radii(FORKED_RACK_SPEC)
  styles = renderer._object_styles(FORKED_RACK_SPEC)

  low_x, high_x, low_y, high_y = renderer._spatial_limits(
      bundles, radii, styles
  )

  assert low_x < high_x and low_y < high_y
  assert low_x <= FORKED_RACK_SPEC.path_start[0]
  assert high_x >= max(
      item.position[0] for item in FORKED_RACK_SPEC.objects
      if item.visual_role == "ball"
  )


def test_render_scene_graph_media_rejects_unknown_branches(tmp_path):
  with pytest.raises(ValueError, match="unknown branches"):
    renderer.render_scene_graph_media(tmp_path, branches=("sideways",))
  with pytest.raises(ValueError, match="at least one branch"):
    renderer.render_scene_graph_media(tmp_path, branches=())


def _context(bundles, summary):
  radii = renderer._collision_radii(FORKED_RACK_SPEC)
  styles = renderer._object_styles(FORKED_RACK_SPEC)
  from interventions.scene_graph import (
      contact_activation_steps,
      propagation_tree,
  )

  return {
      "summary": summary,
      "limits": renderer._spatial_limits(bundles, radii, styles),
      "radii": radii,
      "styles": styles,
      "floor_ids": ("floor",),
      "causal_edges": propagation_tree(
          summary["ground_truth"]["propagation_path"]
      ),
      "activations": {
          name: contact_activation_steps(bundle.series)
          for name, bundle in bundles.items()
      },
  }


def test_rendering_reuses_one_figure_without_stacking_artists(tmp_path):
  summary = _write_bundle(tmp_path)
  _, bundles = renderer.load_demo_bundle(tmp_path)
  context = _context(bundles, summary)
  figure, axes = renderer._new_figure()

  renderer._render_frame(figure, axes, bundles["trajectory_changed"], 2, context)
  first_texts = len(figure.texts)
  first_legends = len(figure.legends)
  for frame_index in range(4):
    renderer._render_frame(
        figure, axes, bundles["trajectory_changed"], frame_index, context
    )

  assert len(figure.texts) == first_texts
  assert len(figure.legends) == first_legends


def test_static_summary_frame_renders_without_a_current_step(tmp_path):
  summary = _write_bundle(tmp_path)
  _, bundles = renderer.load_demo_bundle(tmp_path)
  context = _context(bundles, summary)
  figure, axes = renderer._new_figure()

  renderer._render_frame(figure, axes, bundles["normal"], None, context)

  assert "WHOLE-REPLAY" in figure.texts[0].get_text()
  figure.savefig(tmp_path / "static.png")
  assert (tmp_path / "static.png").is_file()
