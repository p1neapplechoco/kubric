"""Render per-branch scene-graph media from the canonical replay bundle.

Purpose: turn each logged branch into a frame-by-frame relation-graph animation
and a static whole-replay scene graph, both laid out in true simulated x/y.
Public API: BranchBundle, load_demo_bundle(), render_scene_graph_media(), and main().
Dependencies: NumPy, Matplotlib's Agg backend, and the external ffmpeg binary;
interventions.scene_graph supplies relations and trajectory_demo_spec supplies the
canonical geometry, colours, and object roles.
Trust boundary: rendering reads the published bundle and never reruns physics, so
contact edges carry only the authority of the logged contacts, proximity edges are
center-distance readings, and the causal overlay restates the bundled oracle rather
than proving it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from interventions.logging import ContactRecord, SimulationLog
from interventions.scene_graph import (
    CONTACT_RELATION,
    build_relation_series,
    contact_activation_steps,
    propagation_tree,
)
from scripts.trajectory_demo_spec import FORKED_RACK_SPEC

_BRANCHES = ("normal", "trajectory_changed", "target_removed")
_BRANCH_LABELS = {
    "normal": "NORMAL",
    "trajectory_changed": "TRAJECTORY CHANGED",
    "target_removed": "TARGET REMOVED",
}
_BRANCH_ACCENTS = {
    "normal": "#8bd5ca",
    "trajectory_changed": "#f5a97f",
    "target_removed": "#c6a0f6",
}
_BACKGROUND = "#181926"
_PANEL = "#1e2030"
_GRID = "#3b3f51"
_TEXT = "#cad3f5"
_MUTED = "#8087a2"
_CONTACT_COLOR = "#ed8796"
_APPROACH_COLOR = "#eed49f"
_RECEDE_COLOR = "#7dc4e4"
_CAUSAL_COLOR = "#f5bde6"
_INTERVENTION_COLOR = "#ed8796"

_FIGURE_SIZE = (16.0, 9.0)
_FIGURE_DPI = 80
_OUTPUT_FPS = 24
_TRAIL_STEPS = 18
_NODE_RADIUS_FRACTION = 0.46
_NEAR_MARGIN = 0.12
_MOTION_EPSILON = 1e-3
_VELOCITY_SCALE = 0.09
_FFMPEG_TIMEOUT_SECONDS = 900


def _require_regular_file(path: Path, name: str) -> Path:
  if path.is_symlink():
    raise ValueError(f"{name} must not be a symlink: {path}")
  if not path.is_file():
    raise FileNotFoundError(f"missing {name}: {path}")
  return path


def _collision_radii(spec: Any) -> dict[str, float]:
  """Return each object's conservative collision radius in metres."""
  radii = {}
  for item in spec.objects:
    size = item.size
    radii[item.object_id] = (
        float(size) if isinstance(size, (int, float)) else float(min(size))
    )
  return radii


def _object_styles(spec: Any) -> dict[str, dict[str, Any]]:
  """Return per-object colour, label, and role used by every panel."""
  styles = {}
  for item in spec.objects:
    if item.ball_number is not None:
      label = str(item.ball_number)
    elif item.visual_role == "target":
      label = "T"
    else:
      label = item.object_id[:2].upper()
    styles[item.object_id] = {
        "color": tuple(float(channel) for channel in item.color),
        "label": label,
        "role": item.visual_role,
        "group": item.group,
    }
  return styles


@dataclass(frozen=True)
class BranchBundle:
  """One branch's log, presence mask, and derived relation-graph series."""

  name: str
  log: SimulationLog
  presence: np.ndarray
  series: Any

  @property
  def steps(self) -> tuple[int, ...]:
    """The logged step numbers of this branch in frame order."""
    return tuple(self.log.steps)


def load_demo_bundle(
    states_dir: str | Path,
    near_margin: float = _NEAR_MARGIN,
    motion_epsilon: float = _MOTION_EPSILON,
) -> tuple[dict[str, Any], dict[str, BranchBundle]]:
  """Load the published replay bundle and derive every branch relation series."""
  directory = Path(states_dir)
  if not directory.is_dir():
    raise FileNotFoundError(f"states directory does not exist: {directory}")
  summary = json.loads(
      _require_regular_file(directory / "summary.json", "summary.json")
      .read_text(encoding="utf-8")
  )
  contacts = json.loads(
      _require_regular_file(directory / "contacts.json", "contacts.json")
      .read_text(encoding="utf-8")
  )
  object_ids = tuple(summary["object_ids"])
  if object_ids != tuple(FORKED_RACK_SPEC.object_ids):
    raise ValueError("bundle object_ids do not match the canonical demo spec")
  radii = _collision_radii(FORKED_RACK_SPEC)
  floor_ids = tuple(
      item.object_id for item in FORKED_RACK_SPEC.objects
      if item.visual_role == "floor"
  )

  bundles = {}
  for branch in _BRANCHES:
    states = np.load(
        _require_regular_file(
            directory / f"{branch}_states.npy", f"{branch} states"
        )
    )
    presence = np.load(
        _require_regular_file(
            directory / f"{branch}_presence.npy", f"{branch} presence"
        )
    )
    records = tuple(
        ContactRecord(**record) for record in contacts[branch]
    )
    log = SimulationLog(
        branch=branch,
        object_ids=object_ids,
        steps=tuple(range(states.shape[0])),
        states=states,
        contacts=records,
        step_rate=float(summary["step_rate"]),
    )
    series = build_relation_series(
        log,
        radii,
        presence=presence,
        near_margin=near_margin,
        motion_epsilon=motion_epsilon,
        proximity_exclude=floor_ids,
    )
    bundles[branch] = BranchBundle(
        name=branch, log=log, presence=presence, series=series
    )
  return summary, bundles


def _spatial_limits(
    bundles: Mapping[str, BranchBundle],
    radii: Mapping[str, float],
    styles: Mapping[str, dict[str, Any]],
    padding: float = 0.30,
) -> tuple[float, float, float, float]:
  """Return one x/y window that frames every branch identically."""
  lows = []
  highs = []
  for bundle in bundles.values():
    for index, object_id in enumerate(bundle.log.object_ids):
      if styles[object_id]["role"] == "floor":
        continue
      visible = bundle.presence[:, index]
      if not visible.any():
        continue
      points = np.asarray(bundle.log.states)[visible, index, :2]
      margin = radii[object_id] + padding
      lows.append(points.min(axis=0) - margin)
      highs.append(points.max(axis=0) + margin)
  low = np.min(np.stack(lows), axis=0)
  high = np.max(np.stack(highs), axis=0)
  return float(low[0]), float(high[0]), float(low[1]), float(high[1])


def _style_axes(axes: Any, title: str | None = None) -> None:
  axes.set_facecolor(_PANEL)
  for spine in axes.spines.values():
    spine.set_color(_GRID)
  axes.tick_params(colors=_MUTED, labelsize=8)
  axes.grid(color=_GRID, alpha=0.35, linewidth=0.6)
  axes.set_axisbelow(True)
  if title is not None:
    axes.set_title(title, color=_TEXT, fontsize=11, pad=8)


def _floor_contact_ids(frame: Any, floor_ids: Sequence[str]) -> set[str]:
  grounded = set()
  for edge in frame.contact_edges():
    if edge.object_a in floor_ids:
      grounded.add(edge.object_b)
    elif edge.object_b in floor_ids:
      grounded.add(edge.object_a)
  return grounded


def _draw_spatial(
    axes: Any,
    bundle: BranchBundle,
    frame_index: int,
    limits: tuple[float, float, float, float],
    radii: Mapping[str, float],
    styles: Mapping[str, dict[str, Any]],
    floor_ids: Sequence[str],
    hard_affected: Sequence[str],
) -> None:
  """Draw one step as a top-down relation graph over true simulated x/y."""
  frame = bundle.series.frames[frame_index]
  states = np.asarray(bundle.log.states)
  axes.set_xlim(limits[0], limits[1])
  axes.set_ylim(limits[2], limits[3])
  axes.set_aspect("equal", adjustable="box")
  _style_axes(axes)
  axes.set_xlabel("x (m)", color=_MUTED, fontsize=9)
  axes.set_ylabel("y (m)", color=_MUTED, fontsize=9)

  positions = {node.object_id: node.position for node in frame.nodes}
  present = {node.object_id: node.present for node in frame.nodes}
  grounded = _floor_contact_ids(frame, floor_ids)
  affected = set(hard_affected)
  trail_start = max(0, frame_index - _TRAIL_STEPS)

  for index, object_id in enumerate(bundle.log.object_ids):
    if styles[object_id]["role"] == "floor":
      continue
    visible = bundle.presence[trail_start:frame_index + 1, index]
    if not visible.any():
      continue
    trail = states[trail_start:frame_index + 1, index, :2][visible]
    if len(trail) > 1:
      axes.plot(
          trail[:, 0], trail[:, 1],
          color=styles[object_id]["color"], alpha=0.35, linewidth=1.4,
          solid_capstyle="round", zorder=2,
      )

  # A body drawn at true scale would swallow every centre-to-centre segment, so
  # each object shows a translucent true-size footprint with a smaller solid
  # graph node inside it, leaving the relation layer readable between nodes.
  for object_id, style in styles.items():
    if style["role"] == "floor":
      continue
    position = positions[object_id]
    radius = radii[object_id]
    if not present[object_id]:
      axes.add_patch(Circle(
          position[:2], radius, facecolor="none", edgecolor=_MUTED,
          linewidth=1.2, linestyle=(0, (3, 3)), alpha=0.55, zorder=3,
      ))
      axes.text(
          position[0], position[1] - radius - 0.10, "REMOVED",
          color=_MUTED, fontsize=7, ha="center", va="top", zorder=7,
      )
      continue
    axes.add_patch(Circle(
        position[:2], radius, facecolor=style["color"], alpha=0.20,
        edgecolor=style["color"], linewidth=1.0, zorder=3,
    ))
    axes.add_patch(Circle(
        position[:2], radius * _NODE_RADIUS_FRACTION,
        facecolor=style["color"],
        edgecolor=_CAUSAL_COLOR if object_id in affected else "#11111b",
        linewidth=2.2 if object_id in affected else 1.0,
        zorder=6,
    ))
    if object_id in grounded:
      axes.add_patch(Circle(
          position[:2], radius * (_NODE_RADIUS_FRACTION + 0.16),
          facecolor="none", edgecolor="#ffffff", linewidth=1.1,
          linestyle=(0, (2, 2)), alpha=0.85, zorder=6,
      ))
    axes.text(
        position[0], position[1], style["label"],
        color="#11111b", fontsize=7.5, fontweight="bold",
        ha="center", va="center", zorder=7,
    )

  for edge in frame.edges:
    if edge.object_a in floor_ids or edge.object_b in floor_ids:
      continue
    start = positions[edge.object_a][:2]
    end = positions[edge.object_b][:2]
    if edge.relation == CONTACT_RELATION:
      width = 2.2 + 1.8 * float(np.log10(1.0 + edge.normal_force))
      axes.plot(
          (start[0], end[0]), (start[1], end[1]),
          color=_CONTACT_COLOR, linewidth=width, alpha=0.95,
          solid_capstyle="round", zorder=5,
      )
      axes.plot(
          (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0,
          marker="*", color="#ffffff", markersize=13, zorder=5.5,
      )
    else:
      closeness = 1.0 - min(1.0, max(0.0, edge.gap) / bundle.series.near_margin)
      color = _APPROACH_COLOR if edge.approaching else _RECEDE_COLOR
      axes.plot(
          (start[0], end[0]), (start[1], end[1]),
          color=color, linewidth=1.1 + 1.4 * closeness,
          alpha=0.45 + 0.45 * closeness,
          linestyle=(0, (5, 3)) if edge.approaching else (0, (1, 3)),
          solid_capstyle="round", zorder=4,
      )

  for node in frame.nodes:
    if not node.present or styles[node.object_id]["role"] == "floor":
      continue
    if node.speed <= bundle.series.motion_epsilon:
      continue
    index = bundle.log.object_ids.index(node.object_id)
    velocity = states[frame_index, index, 7:9] * _VELOCITY_SCALE
    if float(np.linalg.norm(velocity)) < 0.02:
      continue
    axes.add_patch(FancyArrowPatch(
        (node.position[0], node.position[1]),
        (node.position[0] + velocity[0], node.position[1] + velocity[1]),
        arrowstyle="-|>", mutation_scale=10, color="#ffffff",
        alpha=0.7, linewidth=1.2, zorder=10,
    ))


def _draw_causal_tree(
    axes: Any,
    edges: Sequence[Any],
    target_id: str,
    activations: Mapping[tuple[str, str], int],
    current_step: int | None,
    intervention_start: int,
    styles: Mapping[str, dict[str, Any]],
) -> None:
  """Draw the oracle propagation tree, lit up as its contacts activate."""
  _style_axes(axes, "ORACLE CAUSAL PROPAGATION")
  axes.grid(False)
  axes.set_xticks(())
  axes.set_yticks(())
  if not edges:
    axes.text(
        0.5, 0.5, "no propagation path", color=_MUTED, fontsize=9,
        ha="center", va="center", transform=axes.transAxes,
    )
    return

  hops = {target_id: 0}
  for edge in edges:
    hops[edge.child] = min(hops.get(edge.child, edge.hop), edge.hop)
  levels: dict[int, list[str]] = {}
  for object_id, hop in sorted(hops.items(), key=lambda item: (item[1], item[0])):
    levels.setdefault(hop, []).append(object_id)
  coordinates = {}
  for hop, members in levels.items():
    for order, object_id in enumerate(members):
      coordinates[object_id] = (
          float(hop), float(order) - (len(members) - 1) / 2.0
      )

  active = current_step is None
  for edge in edges:
    start = coordinates[edge.parent]
    end = coordinates[edge.child]
    step = activations.get(tuple(sorted((edge.parent, edge.child))))
    lit = step is not None and (active or step <= current_step)
    axes.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        color=_CAUSAL_COLOR if lit else _GRID,
        alpha=0.95 if lit else 0.5,
        linewidth=1.8 if lit else 1.0,
        linestyle="solid" if lit else (0, (3, 3)),
        shrinkA=13, shrinkB=13, zorder=3,
    ))
    if lit and step is not None:
      axes.text(
          (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0 + 0.16,
          str(step), color=_CAUSAL_COLOR, fontsize=6.5,
          ha="center", va="bottom", zorder=4,
      )

  for object_id, (x, y) in coordinates.items():
    incoming = [
        activations.get(tuple(sorted((edge.parent, edge.child))))
        for edge in edges if edge.child == object_id
    ]
    reached = [step for step in incoming if step is not None]
    if object_id == target_id:
      lit = active or current_step >= intervention_start
    else:
      lit = bool(reached) and (
          active or min(reached) <= current_step
      )
    axes.scatter(
        x, y, s=210,
        color=styles[object_id]["color"] if lit else _PANEL,
        edgecolors=_CAUSAL_COLOR if lit else _GRID,
        linewidths=1.8 if lit else 1.0, zorder=5,
    )
    axes.text(
        x, y, styles[object_id]["label"],
        color="#11111b" if lit else _MUTED, fontsize=7.5,
        fontweight="bold", ha="center", va="center", zorder=6,
    )

  axes.set_xlim(-0.6, max(hops.values()) + 0.6)
  span = max(
      (len(members) for members in levels.values()), default=1
  )
  axes.set_ylim(-span / 2.0 - 0.6, span / 2.0 + 0.6)
  for hop in sorted(levels):
    axes.text(
        hop, span / 2.0 + 0.32, f"hop {hop}", color=_MUTED, fontsize=7,
        ha="center", va="bottom",
    )


def _draw_timeline(
    axes: Any,
    bundle: BranchBundle,
    current_step: int | None,
    intervention_window: Sequence[int],
) -> None:
  """Draw contact and proximity edge counts across the whole replay."""
  _style_axes(axes, "RELATION COUNT PER STEP")
  steps = np.asarray(bundle.series.steps, dtype=np.float64)
  contacts = np.asarray(
      [len(frame.contact_edges()) for frame in bundle.series.frames],
      dtype=np.float64,
  )
  near = np.asarray(
      [
          len(frame.edges) - len(frame.contact_edges())
          for frame in bundle.series.frames
      ],
      dtype=np.float64,
  )
  axes.axvspan(
      intervention_window[0], intervention_window[1],
      color=_INTERVENTION_COLOR, alpha=0.12, zorder=1,
  )
  axes.fill_between(
      steps, near, color=_RECEDE_COLOR, alpha=0.35, linewidth=0,
      zorder=2, label="proximity",
  )
  axes.bar(
      steps, contacts, width=1.6, color=_CONTACT_COLOR, zorder=3,
      label="contact",
  )
  if current_step is not None:
    axes.axvline(current_step, color="#ffffff", linewidth=1.4, zorder=4)
  axes.set_xlim(float(steps[0]), float(steps[-1]))
  upper = max(1.0, float(max(near.max(), contacts.max())))
  axes.set_ylim(0.0, upper * 1.25)
  axes.set_xlabel("bullet step", color=_MUTED, fontsize=8)
  legend = axes.legend(
      loc="upper left", fontsize=7, framealpha=0.0, ncol=2
  )
  for text in legend.get_texts():
    text.set_color(_TEXT)


def _status_lines(
    bundle: BranchBundle,
    frame_index: int,
    floor_ids: Sequence[str],
    step_rate: float,
) -> list[str]:
  frame = bundle.series.frames[frame_index]
  contacts = frame.contact_edges()
  near_count = len(frame.edges) - len(contacts)
  moving = frame.moving_ids(bundle.series.motion_epsilon)
  lines = [
      f"step {frame.step:>3d}   t = {frame.step / step_rate:6.3f} s",
      f"contact edges {len(contacts):>2d}    proximity edges {near_count:>2d}"
      f"    moving {len(moving):>2d}",
  ]
  if contacts:
    for edge in contacts[:4]:
      if edge.object_a in floor_ids or edge.object_b in floor_ids:
        other = (
            edge.object_b if edge.object_a in floor_ids else edge.object_a
        )
        lines.append(f"  {other} on floor   {edge.normal_force:8.1f} N")
      else:
        lines.append(
            f"  {edge.object_a} - {edge.object_b}   "
            f"{edge.normal_force:8.1f} N"
        )
    if len(contacts) > 4:
      lines.append(f"  (+{len(contacts) - 4} more)")
  return lines


def _legend_handles() -> list[Line2D]:
  return [
      Line2D([], [], color=_CONTACT_COLOR, linewidth=3.0, label="contact"),
      Line2D(
          [], [], color=_APPROACH_COLOR, linewidth=2.0,
          linestyle=(0, (5, 3)), label="near, approaching",
      ),
      Line2D(
          [], [], color=_RECEDE_COLOR, linewidth=2.0,
          linestyle=(0, (1, 3)), label="near, receding",
      ),
      Line2D(
          [], [], color=_CAUSAL_COLOR, linewidth=2.4, label="causally affected",
      ),
  ]


def _new_figure() -> tuple[Figure, dict[str, Any]]:
  figure = Figure(figsize=_FIGURE_SIZE, dpi=_FIGURE_DPI)
  FigureCanvasAgg(figure)
  figure.patch.set_facecolor(_BACKGROUND)
  axes = {
      "spatial": figure.add_axes((0.040, 0.075, 0.595, 0.815)),
      "causal": figure.add_axes((0.680, 0.535, 0.295, 0.355)),
      "timeline": figure.add_axes((0.680, 0.085, 0.295, 0.265)),
  }
  return figure, axes


def _render_frame(
    figure: Figure,
    axes: Mapping[str, Any],
    bundle: BranchBundle,
    frame_index: int | None,
    context: Mapping[str, Any],
) -> None:
  for axis in axes.values():
    axis.cla()
  # Figure-level artists survive axis clears and would otherwise stack up
  # across the hundreds of frames drawn into this one reused figure.
  figure.texts.clear()
  figure.legends.clear()

  summary = context["summary"]
  step_rate = float(summary["step_rate"])
  current_step = (
      None if frame_index is None else bundle.series.frames[frame_index].step
  )
  accent = _BRANCH_ACCENTS[bundle.name]

  if frame_index is None:
    _draw_static_spatial(axes["spatial"], bundle, context)
    headline = f"{_BRANCH_LABELS[bundle.name]}  ·  WHOLE-REPLAY SCENE GRAPH"
  else:
    _draw_spatial(
        axes["spatial"], bundle, frame_index, context["limits"],
        context["radii"], context["styles"], context["floor_ids"],
        summary["ground_truth"]["hard_affected"],
    )
    headline = (
        f"{_BRANCH_LABELS[bundle.name]}  ·  SCENE GRAPH  ·  "
        f"step {current_step}/{bundle.series.frames[-1].step}"
    )
  _draw_causal_tree(
      axes["causal"], context["causal_edges"], FORKED_RACK_SPEC.target_id,
      context["activations"][bundle.name], current_step,
      summary["intervention_start"], context["styles"],
  )
  _draw_timeline(
      axes["timeline"], bundle, current_step, summary["intervention_window"]
  )

  figure.text(
      0.040, 0.955, headline, color=accent, fontsize=17, fontweight="bold",
      ha="left", va="center",
  )
  figure.text(
      0.040, 0.921,
      "contact and proximity relations from the logged replay; "
      "causal overlay restates the bundled oracle",
      color=_MUTED, fontsize=9, ha="left", va="center",
  )
  legend = figure.legend(
      handles=_legend_handles(), loc="upper right",
      bbox_to_anchor=(0.978, 0.975), ncol=2, fontsize=8.5,
      framealpha=0.0,
  )
  for text in legend.get_texts():
    text.set_color(_TEXT)

  if frame_index is not None:
    lines = _status_lines(
        bundle, frame_index, context["floor_ids"], step_rate
    )
    figure.text(
        0.680, 0.470, "\n".join(lines), color=_TEXT, fontsize=9,
        family="monospace", ha="left", va="top", linespacing=1.5,
    )


def _draw_static_spatial(
    axes: Any, bundle: BranchBundle, context: Mapping[str, Any]
) -> None:
  """Draw full trajectories with every contact episode of the replay."""
  limits = context["limits"]
  radii = context["radii"]
  styles = context["styles"]
  floor_ids = context["floor_ids"]
  states = np.asarray(bundle.log.states)
  axes.set_xlim(limits[0], limits[1])
  axes.set_ylim(limits[2], limits[3])
  axes.set_aspect("equal", adjustable="box")
  _style_axes(axes)
  axes.set_xlabel("x (m)", color=_MUTED, fontsize=9)
  axes.set_ylabel("y (m)", color=_MUTED, fontsize=9)
  affected = set(context["summary"]["ground_truth"]["hard_affected"])

  for index, object_id in enumerate(bundle.log.object_ids):
    if styles[object_id]["role"] == "floor":
      continue
    visible = bundle.presence[:, index]
    if not visible.any():
      continue
    path = states[visible, index, :2]
    axes.plot(
        path[:, 0], path[:, 1], color=styles[object_id]["color"],
        linewidth=1.6, alpha=0.55, zorder=2,
    )
    axes.add_patch(Circle(
        path[0], radii[object_id], facecolor="none",
        edgecolor=styles[object_id]["color"], linewidth=1.1,
        linestyle=(0, (3, 3)), alpha=0.7, zorder=3,
    ))
    axes.add_patch(Circle(
        path[-1], radii[object_id], facecolor=styles[object_id]["color"],
        edgecolor=_CAUSAL_COLOR if object_id in affected else "#11111b",
        linewidth=2.4 if object_id in affected else 1.0, zorder=4,
    ))
    axes.text(
        path[-1][0], path[-1][1], styles[object_id]["label"],
        color="#11111b", fontsize=8, fontweight="bold",
        ha="center", va="center", zorder=5,
    )
    if not visible.all():
      axes.text(
          path[-1][0], path[-1][1] - radii[object_id] - 0.10, "REMOVED",
          color=_MUTED, fontsize=7, ha="center", va="top", zorder=6,
      )

  for pair, step in context["activations"][bundle.name].items():
    if pair[0] in floor_ids or pair[1] in floor_ids:
      continue
    frame = bundle.series.frame_at(step)
    positions = {node.object_id: node.position for node in frame.nodes}
    start = positions[pair[0]][:2]
    end = positions[pair[1]][:2]
    axes.plot(
        (start[0], end[0]), (start[1], end[1]),
        color=_CONTACT_COLOR, linewidth=2.2, alpha=0.9, zorder=6,
    )
    axes.text(
        (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, str(step),
        color=_CONTACT_COLOR, fontsize=7.5, ha="center", va="bottom",
        zorder=7,
    )
  axes.add_patch(Rectangle(
      (limits[0], limits[2]), limits[1] - limits[0], limits[3] - limits[2],
      facecolor="none", edgecolor=_GRID, linewidth=1.0, zorder=1,
  ))


def _encode_video(frame_dir: Path, output: Path, ffmpeg: str) -> None:
  command = [
      ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
      "-framerate", str(_OUTPUT_FPS),
      "-i", str(frame_dir / "frame_%05d.png"),
      "-c:v", "libx264", "-pix_fmt", "yuv420p",
      "-r", str(_OUTPUT_FPS), "-movflags", "+faststart",
      str(output),
  ]
  try:
    subprocess.run(
        command, check=True, capture_output=True, text=True,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
  except (OSError, subprocess.CalledProcessError) as error:
    detail = getattr(error, "stderr", "") or str(error)
    raise RuntimeError(f"ffmpeg encoding failed: {detail.strip()}") from error
  except subprocess.TimeoutExpired as error:
    raise RuntimeError(
        f"ffmpeg encoding timed out after {_FFMPEG_TIMEOUT_SECONDS} seconds"
    ) from error


def render_scene_graph_media(
    states_dir: str | Path = "output/demo_collision_intervention",
    output_dir: str | Path | None = None,
    branches: Sequence[str] = _BRANCHES,
    near_margin: float = _NEAR_MARGIN,
) -> dict[str, Any]:
  """Render and atomically publish per-branch scene-graph videos and summaries."""
  requested = tuple(branches)
  unknown = sorted(set(requested) - set(_BRANCHES))
  if unknown:
    raise ValueError(f"unknown branches requested: {unknown!r}")
  if not requested:
    raise ValueError("at least one branch must be requested")

  directory = Path(states_dir)
  destination = (
      directory / "scene_graphs" if output_dir is None else Path(output_dir)
  )
  summary, bundles = load_demo_bundle(directory, near_margin=near_margin)
  radii = _collision_radii(FORKED_RACK_SPEC)
  styles = _object_styles(FORKED_RACK_SPEC)
  floor_ids = tuple(
      object_id for object_id, style in styles.items()
      if style["role"] == "floor"
  )
  context = {
      "summary": summary,
      "limits": _spatial_limits(bundles, radii, styles),
      "radii": radii,
      "styles": styles,
      "floor_ids": floor_ids,
      "causal_edges": propagation_tree(
          summary["ground_truth"]["propagation_path"]
      ),
      "activations": {
          name: contact_activation_steps(bundle.series)
          for name, bundle in bundles.items()
      },
  }

  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError("ffmpeg was not found on PATH")
  destination.mkdir(parents=True, exist_ok=True)

  results = {}
  figure, axes = _new_figure()
  for name in requested:
    bundle = bundles[name]
    video_path = destination / f"{name}_scene_graph.mp4"
    still_path = destination / f"{name}_scene_graph.png"
    with tempfile.TemporaryDirectory(
        prefix=f".{name}.frames.", dir=destination
    ) as staging_name:
      staging = Path(staging_name)
      for frame_index in range(len(bundle.series.frames)):
        _render_frame(figure, axes, bundle, frame_index, context)
        figure.savefig(
            staging / f"frame_{frame_index:05d}.png",
            facecolor=_BACKGROUND,
        )
      _render_frame(figure, axes, bundle, None, context)
      figure.savefig(staging / "summary.png", facecolor=_BACKGROUND)
      _encode_video(staging, staging / "scene_graph.mp4", ffmpeg)
      os.replace(staging / "scene_graph.mp4", video_path)
      os.replace(staging / "summary.png", still_path)
    contact_pairs = context["activations"][name]
    results[name] = {
        "frames": len(bundle.series.frames),
        "contact_pairs": {
            "|".join(pair): step for pair, step in contact_pairs.items()
        },
        "peak_proximity_edges": max(
            len(frame.edges) - len(frame.contact_edges())
            for frame in bundle.series.frames
        ),
        "still": str(still_path),
        "video": str(video_path),
    }

  metadata = {
      "branches": results,
      "fps": _OUTPUT_FPS,
      "near_margin": near_margin,
      "output_dir": str(destination),
      "size": [
          int(_FIGURE_SIZE[0] * _FIGURE_DPI),
          int(_FIGURE_SIZE[1] * _FIGURE_DPI),
      ],
  }
  _atomic_write_json(destination / "scene_graph_summary.json", metadata)
  return metadata


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
  encoded = (
      json.dumps(payload, indent=2, sort_keys=True) + "\n"
  ).encode("utf-8")
  descriptor, temporary_name = tempfile.mkstemp(
      dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
  )
  try:
    with os.fdopen(descriptor, "wb") as handle:
      handle.write(encoded)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary_name, path)
  except BaseException:
    Path(temporary_name).unlink(missing_ok=True)
    raise


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Render scene-graph animations for the replay branches."
  )
  parser.add_argument(
      "--states-dir",
      type=Path,
      default=Path("output/demo_collision_intervention"),
      help="directory containing summary.json, contacts.json, and state arrays",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=None,
      help="output directory (default: STATES_DIR/scene_graphs)",
  )
  parser.add_argument(
      "--branches",
      nargs="+",
      default=list(_BRANCHES),
      choices=list(_BRANCHES),
      help="branches to render (default: all three)",
  )
  parser.add_argument(
      "--near-margin",
      type=float,
      default=_NEAR_MARGIN,
      help="surface gap in metres below which a proximity edge is drawn",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Renders per-branch scene-graph media atomically and returns zero."""
  args = _parser().parse_args(argv)
  metadata = render_scene_graph_media(
      args.states_dir, args.output_dir, args.branches, args.near_margin
  )
  print(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
