#!/usr/bin/env python3
"""Generate a batch of velocity-intervention instances, rendered and published.

Purpose: drive the whole pipeline for many instances -- sample, simulate the
three branches, re-frame the camera on the rollout, derive ground truth, run QC,
render Video/Mask/Graph/Tracking, and publish a manifest with train/val/test
splits.
Public API: assign_split, generate_dataset, main.
Dependencies: NumPy and the standard library; Kubric and ``bpy`` only when
rendering is enabled (``--no-render`` runs the physics half on a machine with no
Blender).
Trust boundary: unchanged from the single-instance harness. The manifest records
what was sampled, simulated, and handed to Blender; it does not attest that the
encoded pixels match the logged physics, and nothing here re-simulates to check.
Rejected attempts are recorded rather than hidden, so the acceptance rate of a
published batch is auditable from its own manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interventions import appearance, velocity_scenes
from interventions.schema import to_jsonable
from scripts import render_velocity_scene


#: Splits are named in the config's ``split.fractions``; this is only the order
#: they are laid out in when bucketing, so the mapping stays stable as fractions
#: change.
DEFAULT_SPLITS = ("train", "val", "test")


def assign_split(
    instance_id: str, fractions: Mapping[str, float], split_seed: int
) -> str:
  """Assigns one instance to a split, deterministically and independently.

  The bucket comes from a hash of ``(split_seed, instance_id)``, not from the
  instance's position in the accepted sequence. That independence is the point:
  a resumed run, a longer run, or a run whose rejection pattern differs still
  puts a given instance in the same split, so a batch can be extended without
  leaking a previously-held-out scene into training.
  """
  names = [name for name in DEFAULT_SPLITS if name in fractions]
  names += sorted(set(fractions) - set(names))
  if not names:
    raise ValueError("split.fractions is empty")

  weights = [float(fractions[name]) for name in names]
  if any(weight < 0.0 for weight in weights):
    raise ValueError("split fractions must be non-negative")
  total = sum(weights)
  if total <= 0.0:
    raise ValueError("split fractions must sum to a positive number")

  digest = hashlib.sha256(
      "{}:{}".format(int(split_seed), instance_id).encode("utf-8")
  ).digest()
  # 53 bits is what a float64 can hold exactly, so this is a uniform draw in
  # [0, 1) with no rounding bias at the bucket edges.
  draw = int.from_bytes(digest[:7], "big") / float(1 << 56)

  cursor = 0.0
  for name, weight in zip(names, weights):
    cursor += weight / total
    if draw < cursor:
      return name
  return names[-1]


def _profile_from_args(
    ranges: Mapping[str, Any], args: argparse.Namespace
) -> appearance.RenderProfile:
  """The config's render block with the CLI's overrides applied."""
  profile = render_velocity_scene.render_profile_from_ranges(ranges)
  if args.device is not None:
    profile = dataclasses.replace(profile, device=args.device)
  if args.samples_per_pixel is not None:
    profile = dataclasses.replace(
        profile, samples_per_pixel=args.samples_per_pixel
    )
  if args.resolution is not None:
    profile = dataclasses.replace(profile, resolution=tuple(args.resolution))
  return profile


def _is_complete(instance_dir: Path, branches: Sequence[str]) -> bool:
  """Whether a directory already holds every artifact this run would write.

  Resume checks the artifacts, not a journal entry, because a run killed
  mid-render leaves the journal claiming success for a branch whose mp4 is a
  truncated file. Every required output is named and stat-ed.
  """
  if not (instance_dir / "instance.json").is_file():
    return False
  if not (instance_dir / "render.json").is_file():
    return False
  for branch in branches:
    branch_dir = instance_dir / branch
    for name in ("video.mp4", "segmentation.npz", "tracking.npz", "graph.json"):
      path = branch_dir / name
      if not path.is_file() or path.stat().st_size == 0:
        return False
  return True


def _physics_complete(instance_dir: Path) -> bool:
  """The ``--no-render`` counterpart of :func:`_is_complete`."""
  return (instance_dir / "instance.json").is_file()


def _attempt(
    ranges: Mapping[str, Any], seed: int, index: int
) -> Dict[str, Any]:
  """Samples and simulates one index, or reports why it cannot be used.

  Placement and camera failures raise ``ValueError`` out of ``sample_instance``;
  they are ordinary outcomes of a rejection sampler, not faults, so they are
  turned into a rejection record here rather than allowed to kill the batch.
  """
  try:
    spec = velocity_scenes.sample_instance(ranges, seed, index)
  except ValueError as error:
    return {"status": "sample_failed", "index": index, "reason": str(error)}

  logs = velocity_scenes.generate_triplet(spec)
  # The camera can only be framed on the rollout once the rollout exists.
  spec = velocity_scenes.refit_camera(ranges, spec, logs)

  qc = ranges.get("qc", {})
  pair_truth = velocity_scenes.pair_ground_truth(
      logs["factual"], logs["counterfactual"], spec.subject_id, qc
  )
  removal_truth = velocity_scenes.removal_ground_truth(
      logs["factual"], logs["subject_removed"], spec.subject_id, qc
  )
  qc_result = velocity_scenes.evaluate_qc(spec, logs, pair_truth, qc)
  return {
      "status": "accepted" if qc_result.accepted else "qc_rejected",
      "index": index,
      "spec": spec,
      "logs": logs,
      "pair_truth": pair_truth,
      "removal_truth": removal_truth,
      "qc_result": qc_result,
  }


def _manifest_entry(
    summary: Mapping[str, Any],
    split: str,
    render_record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
  """The per-instance row of the manifest: enough to filter without loading."""
  scene = summary["scene"]
  entry = {
      "instance_id": summary["instance_id"],
      "index": summary["index"],
      "split": split,
      "subject_id": summary["subject_id"],
      "branches": list(summary["branches"]),
      "num_objects": len(scene["object_ids"]),
      "object_ids": list(scene["object_ids"]),
      "material_families": scene["material_families"],
      "visual_scene_hash": summary["visual_scene_hash"],
      "motion": summary["motion"],
      "qc_metrics": summary["qc"]["metrics"],
      "rendered": render_record is not None,
  }
  if render_record is not None:
    entry["render_profile_hash"] = render_record["render_profile_hash"]
    entry["render_seconds"] = round(
        sum(
            float(branch["render_seconds"])
            for branch in render_record["branches"].values()
        ),
        3,
    )
    entry["contact_edges"] = {
        name: branch["contact_edges"]
        for name, branch in sorted(render_record["branches"].items())
    }
  return entry


def generate_dataset(
    ranges: Mapping[str, Any],
    output: Path,
    seed: int,
    num_instances: int,
    max_attempts: int,
    *,
    profile: Optional[appearance.RenderProfile] = None,
    branches: Sequence[str] = velocity_scenes.BRANCHES,
    render: bool = True,
    resume: bool = False,
    start_index: int = 0,
    save_blend: bool = False,
    verbose: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
  """Runs the batch and returns the manifest it published.

  Attempts are indexed from ``start_index`` upward and each index is an
  independent draw, so the batch is reproducible from ``(seed, start_index)``
  regardless of how many attempts were rejected along the way.
  """
  output = Path(output)
  instances_root = output / "instances"
  instances_root.mkdir(parents=True, exist_ok=True)

  branches = list(branches)
  unknown = [name for name in branches if name not in velocity_scenes.BRANCHES]
  if unknown:
    raise ValueError("unknown branches: {}".format(sorted(unknown)))

  split_config = dict(ranges.get("split", {}))
  fractions = dict(split_config.get("fractions", {"train": 1.0}))
  split_seed = int(split_config.get("seed", seed))

  if render and profile is None:
    profile = render_velocity_scene.render_profile_from_ranges(ranges)

  entries: List[Dict[str, Any]] = []
  rejections: List[Dict[str, Any]] = []
  attempts = 0
  index = int(start_index)
  started = time.perf_counter()

  while len(entries) < num_instances and attempts < max_attempts:
    attempts += 1
    current_index = index
    index += 1

    # Resume is decided before anything expensive runs. The instance id is a
    # pure function of (seed, index), so an already-published index costs a
    # stat() here instead of a three-branch rollout it would only throw away.
    if resume:
      instance_id = velocity_scenes.instance_id_for(seed, current_index)
      instance_dir = instances_root / instance_id
      done = (
          _is_complete(instance_dir, branches)
          if render
          else _physics_complete(instance_dir)
      )
      if done:
        summary = json.loads(
            (instance_dir / "instance.json").read_text(encoding="utf-8")
        )
        record = None
        if render:
          record = json.loads(
              (instance_dir / "render.json").read_text(encoding="utf-8")
          )
        split = summary.get("split") or assign_split(
            instance_id, fractions, split_seed
        )
        entries.append(_manifest_entry(summary, split, record))
        if progress:
          print(
              "  [{:>3}/{}] {} resumed".format(
                  len(entries), num_instances, instance_id
              ),
              file=sys.stderr,
          )
        continue

    result = _attempt(ranges, seed, current_index)
    if result["status"] != "accepted":
      reason = result.get("reason")
      if reason is None:
        reason = ", ".join(result["qc_result"].reasons)
      rejections.append({
          "index": current_index,
          "status": result["status"],
          "reason": reason,
          "instance_id": (
              result["spec"].instance_id if "spec" in result else None
          ),
      })
      if progress:
        print(
            "  index {:>4} rejected ({}): {}".format(
                current_index, result["status"], reason
            ),
            file=sys.stderr,
        )
      continue

    spec = result["spec"]
    instance_dir = instances_root / spec.instance_id
    split = assign_split(spec.instance_id, fractions, split_seed)

    instance_dir.mkdir(parents=True, exist_ok=True)
    summary = velocity_scenes.instance_summary(
        spec,
        result["logs"],
        result["pair_truth"],
        result["removal_truth"],
        result["qc_result"],
    )
    summary["split"] = split
    (instance_dir / "instance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    record = None
    if render:
      record = render_velocity_scene.render_instance(
          spec,
          result["logs"],
          profile,
          instance_dir,
          branches=branches,
          use_gpu=profile.device == "GPU",
          save_blend=save_blend,
          verbose=verbose,
      )

    entries.append(_manifest_entry(summary, split, record))
    if progress:
      elapsed = "" if record is None else " in {:.0f}s".format(
          sum(
              float(item["wall_seconds"])
              for item in record["branches"].values()
          )
      )
      print(
          "  [{:>3}/{}] {} split={}{}".format(
              len(entries), num_instances, spec.instance_id, split, elapsed
          ),
          file=sys.stderr,
      )

  split_counts: Dict[str, int] = {}
  for entry in entries:
    split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1

  manifest = {
      "schema_version": "1.0",
      "trust_model": velocity_scenes.VELOCITY_TRUST_MODEL,
      "master_seed": seed,
      "start_index": int(start_index),
      "next_index": index,
      "requested": num_instances,
      "accepted": len(entries),
      "attempts": attempts,
      "max_attempts": max_attempts,
      "status": "complete" if len(entries) >= num_instances else "incomplete",
      "branches": branches,
      "rendered": render,
      "render_profile": None if not render else to_jsonable(profile),
      "render_profile_hash": (
          None if not render else appearance.render_profile_hash(profile)
      ),
      "splits": {
          "seed": split_seed,
          "fractions": fractions,
          "counts": dict(sorted(split_counts.items())),
      },
      "wall_seconds": round(time.perf_counter() - started, 3),
      "instances": entries,
      "rejections": rejections,
  }
  (output / "manifest.json").write_text(
      json.dumps(to_jsonable(manifest), indent=2, sort_keys=True),
      encoding="utf-8",
  )
  return manifest


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description=(
          "Generate a batch of three-branch velocity-intervention instances "
          "with Video/Mask/Graph/Tracking outputs."
      )
  )
  parser.add_argument(
      "--config", type=Path, default=Path("configs/scene_ranges_velocity.yaml")
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--num-instances", type=int, required=True)
  parser.add_argument(
      "--max-attempts", type=int, default=None,
      help="Defaults to 4x --num-instances, which the sampler's measured "
           "acceptance rate leaves ample headroom under.",
  )
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument(
      "--branches", nargs="+", default=list(velocity_scenes.BRANCHES)
  )
  parser.add_argument(
      "--no-render", action="store_true",
      help="Sample, simulate, and publish instance.json only. Runs anywhere; "
           "needs no Blender.",
  )
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--device", choices=("GPU", "CPU"), default=None)
  parser.add_argument("--samples-per-pixel", type=int, default=None)
  parser.add_argument(
      "--resolution", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT")
  )
  parser.add_argument("--save-blend", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  parser.add_argument(
      "--quiet", action="store_true", help="Suppress the per-instance progress log."
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Runs resumable batch generation and returns its stable CLI exit status."""
  args = _parser().parse_args(argv)
  ranges = velocity_scenes.load_ranges(args.config)
  render = not args.no_render
  profile = _profile_from_args(ranges, args) if render else None
  max_attempts = (
      args.max_attempts
      if args.max_attempts is not None
      else max(1, 4 * args.num_instances)
  )

  try:
    manifest = generate_dataset(
        ranges,
        args.output,
        args.seed,
        args.num_instances,
        max_attempts,
        profile=profile,
        branches=args.branches,
        render=render,
        resume=args.resume,
        start_index=args.start_index,
        save_blend=args.save_blend,
        verbose=args.verbose,
        progress=not args.quiet,
    )
  except Exception as error:  # CLI boundary emits machine-readable failures.
    print(
        json.dumps(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            sort_keys=True,
        )
    )
    return 1

  # The manifest itself is on disk; stdout gets the part a caller polls on.
  print(
      json.dumps(
          {
              key: manifest[key]
              for key in (
                  "status",
                  "accepted",
                  "requested",
                  "attempts",
                  "next_index",
                  "splits",
                  "wall_seconds",
              )
          },
          indent=2,
          sort_keys=True,
      )
  )
  return 0 if manifest["status"] == "complete" else 2


__all__ = ["assign_split", "generate_dataset", "main"]


if __name__ == "__main__":
  raise SystemExit(main())
