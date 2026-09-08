#!/usr/bin/env python3
"""Publish a generated velocity-intervention batch to a HuggingFace dataset repo.

Purpose: turn an output directory written by ``generate_velocity_dataset.py``
into an uploaded dataset repo, with a card and a split index derived from the
manifest rather than hand-written.
Public API: build_dataset_card, build_split_index, resolve_token, upload_dataset,
main.
Dependencies: ``huggingface_hub``; NumPy and the standard library.
Trust boundary: the token is read from the environment or from an interactive
prompt and is never written to disk, never echoed, and never placed in the card,
the split index, or the commit message. Uploading publishes the batch to a
third-party host: for a public repo that is an irreversible disclosure, which is
why ``--private`` is the default and the CLI states the destination before it
sends anything.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


#: Environment variables checked for a token, in order. ``HF_TOKEN`` is what
#: ``huggingface_hub`` itself reads; the others are common aliases.
TOKEN_VARIABLES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

#: Everything the generator writes per branch, in the order the card lists them.
BRANCH_ARTIFACTS = (
    ("video.mp4", "H.264 clip of the branch, one frame per rendered step."),
    ("segmentation.npz", "`segmentation` [T,H,W,1] uint16 instance mask; label "
                         "0 is background and label *i+1* is row *i* of the "
                         "tracking arrays."),
    ("tracking.npz", "Per-frame states, image-space projections, bounding "
                     "boxes, visibility, and the camera matrices."),
    ("graph.json", "Temporal contact graph: nodes per body, edges carrying "
                   "contact interval, total impulse, and peak force."),
    ("depth.npz", "`depth` [T,H,W,1] float32, in metres."),
)


def resolve_token(explicit: Optional[str] = None, *, prompt: bool = True) -> str:
  """Finds the access token without ever putting it on a command line.

  Order: an explicitly passed value, then the environment, then an interactive
  no-echo prompt. There is deliberately no ``--token`` flag: an argument lands
  in the shell history and in the process table, where anything on the machine
  can read it.
  """
  if explicit:
    return explicit.strip()
  for name in TOKEN_VARIABLES:
    value = os.environ.get(name)
    if value and value.strip():
      return value.strip()
  if not prompt or not sys.stdin.isatty():
    raise RuntimeError(
        "no access token: set one of {} or run interactively".format(
            ", ".join(TOKEN_VARIABLES)
        )
    )
  token = getpass.getpass("HuggingFace access token (input hidden): ").strip()
  if not token:
    raise RuntimeError("no access token supplied")
  return token


def build_split_index(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
  """One flat row per instance, so a consumer can filter without walking dirs."""
  rows = []
  for entry in manifest.get("instances", ()):
    rows.append({
        "instance_id": entry["instance_id"],
        "split": entry["split"],
        "index": entry["index"],
        "subject_id": entry["subject_id"],
        "num_objects": entry["num_objects"],
        "object_ids": entry["object_ids"],
        "branches": entry["branches"],
        "visual_scene_hash": entry["visual_scene_hash"],
        "path": "instances/{}".format(entry["instance_id"]),
    })
  rows.sort(key=lambda row: (row["split"], row["instance_id"]))
  return rows


def _fmt_counts(counts: Mapping[str, int]) -> str:
  if not counts:
    return "none"
  return ", ".join("{} {}".format(value, key) for key, value in sorted(counts.items()))


def build_dataset_card(manifest: Mapping[str, Any], repo_id: str) -> str:
  """Writes the README from the manifest, so the card cannot drift from the data."""
  splits = manifest.get("splits", {})
  counts = splits.get("counts", {})
  profile = manifest.get("render_profile") or {}
  resolution = profile.get("resolution") or ["?", "?"]
  branches = manifest.get("branches", [])

  yaml_counts = "\n".join(
      "  - name: {}\n    num_examples: {}".format(name, value)
      for name, value in sorted(counts.items())
  )
  artifacts = "\n".join(
      "| `{}` | {} |".format(name, description)
      for name, description in BRANCH_ARTIFACTS
  )

  return """---
license: apache-2.0
task_categories:
  - video-classification
tags:
  - physics
  - counterfactual
  - causal-reasoning
  - synthetic
  - kubric
configs:
  - config_name: default
    data_files: split_index.json
dataset_info:
  splits:
{yaml_counts}
---

# {repo_id}

Synthetic three-branch counterfactual physics videos, generated with
[Kubric](https://github.com/google-research/kubric) (PyBullet physics, Blender
Cycles rendering) from `configs/scene_ranges_velocity.yaml`.

Each **instance** is one scene rendered three times from **one shared
appearance record**. The bodies, materials, textures, lights, background, and
camera are identical across the three branches; the only thing that differs is
the intervention.

| Branch | What it is |
| --- | --- |
| `factual` | The subject is given one initial linear velocity at t=0 and released. |
| `counterfactual` | The same scene, but the subject's initial velocity is **replaced** with a different one — a new speed and a turned direction — applied once at t=0. The factual velocity is discarded, not perturbed. |
| `subject_removed` | The same scene with the subject absent from the start. |

Because all three share one `VisualSceneSpec`, any pixel difference between
branches is caused by the intervention rather than by re-sampled appearance.
The shared record's SHA-256 is published per instance as `visual_scene_hash`.

## Scene configuration

- **Mass** is fixed at a constant for every body; the material sampler moves
  friction and restitution but never mass.
- **Velocity** is a single initial impulse in one direction. Nothing is driven
  along a prescribed path and no force is applied after t=0 — every body is a
  free dynamic body for the whole rollout.
- **Material** is sampled per body from {{metal, rubber, plastic, ceramic, wood,
  stone}}, with friction and restitution coupled to the sampled family.
- **{objects} bodies** per scene including the subject. Some sit in the
  subject's corridor and are struck; others are bystanders that are never
  touched.
- **Motion** covers both modes: cubes slide and spheres roll, and a scene is
  rejected unless it contains both.
- **Camera** is fixed for the whole clip and resampled per instance. It is
  framed against the real frustum of the sampled lens over the whole rollout,
  so every body stays inside the image on every frame.

## Layout

```
manifest.json                      the full generation record
split_index.json                   one row per instance: id, split, path
instances/<instance_id>/
  instance.json                    spec, ground truth, QC, motion modes, split
  render.json                      render profile, device, per-branch timings
  {branch_dirs}/
    video.mp4  segmentation.npz  tracking.npz  graph.json  depth.npz
```

### Per-branch artifacts

| File | Contents |
| --- | --- |
{artifacts}

`tracking.npz` holds `object_ids`, `frame_steps`, `states` `[T,N,13]`
(position, quaternion wxyz, linear velocity, angular velocity), `image_positions`
`[T,N,2]`, `in_front_of_camera`, `bboxes` `[T,N,4]` as inclusive
`(min_row, min_col, max_row, max_col)` with `-1` where the body is not visible,
`visible_pixels`, and the per-frame camera `matrix_world` and `intrinsics`.
All arrays load without `allow_pickle`.

## Ground truth

`instance.json` carries, for both the counterfactual and the removal branch:

- `hard_affected` — bodies whose trajectory diverges from the factual branch by
  more than the configured position/velocity/quaternion epsilons.
- `soft_affected` — bodies that diverge only below those thresholds.
- `propagation_path` — for each affected body, the contact chain through which
  the intervention reached it, starting at the subject.
- `graph_delta` — the temporal contact graph difference against the factual
  branch, as `added`, `removed`, and `changed` edges.

## Generation

```bash
python -m scripts.generate_velocity_dataset \\
  --config configs/scene_ranges_velocity.yaml \\
  --output output/velocity_dataset \\
  --seed {seed} --num-instances {requested}
```

- Master seed: `{seed}`
- Instances: **{accepted}** accepted from {attempts} attempts
- Splits: {counts_text} (assigned by hashing the instance id, so extending a
  batch never moves an existing instance between splits)
- Render: {width}x{height}, {spp} samples/pixel, Blender Cycles on
  `{device}`

## Trust boundary

Pixels do not attest physics. Every annotation here is derived from the logged
simulation states and contacts, or from Blender's own segmentation pass; nothing
is re-simulated at render time. The published hashes are claims about the inputs
handed to the renderer, not about the encoded frames.
""".format(
      repo_id=repo_id,
      yaml_counts=yaml_counts or "  - name: train\n    num_examples: 0",
      artifacts=artifacts,
      branch_dirs="{" + ",".join(branches) + "}" if branches else "<branch>",
      objects=_object_range(manifest),
      seed=manifest.get("master_seed", "?"),
      requested=manifest.get("requested", "?"),
      accepted=manifest.get("accepted", "?"),
      attempts=manifest.get("attempts", "?"),
      counts_text=_fmt_counts(counts),
      width=resolution[0],
      height=resolution[1],
      spp=profile.get("samples_per_pixel", "?"),
      device=profile.get("device", "?"),
  )


def _object_range(manifest: Mapping[str, Any]) -> str:
  """The observed body count, reported as the range that actually occurred."""
  counts = [
      int(entry["num_objects"]) - 1  # the floor is not a scene body
      for entry in manifest.get("instances", ())
      if "num_objects" in entry
  ]
  if not counts:
    return "4-6"
  low, high = min(counts), max(counts)
  return str(low) if low == high else "{}-{}".format(low, high)


def upload_dataset(
    dataset_dir: Path,
    repo_id: str,
    *,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
    write_card: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
  """Uploads a generated batch and returns what it published."""
  from huggingface_hub import HfApi

  dataset_dir = Path(dataset_dir)
  manifest_path = dataset_dir / "manifest.json"
  if not manifest_path.is_file():
    raise FileNotFoundError(
        "{} has no manifest.json -- point this at a directory written by "
        "generate_velocity_dataset.py".format(dataset_dir)
    )
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if not manifest.get("instances"):
    raise ValueError("manifest lists no instances; nothing to upload")

  split_index = build_split_index(manifest)
  (dataset_dir / "split_index.json").write_text(
      json.dumps(split_index, indent=2, sort_keys=True), encoding="utf-8"
  )
  if write_card:
    (dataset_dir / "README.md").write_text(
        build_dataset_card(manifest, repo_id), encoding="utf-8"
    )

  files = [path for path in dataset_dir.rglob("*") if path.is_file()]
  total_bytes = sum(path.stat().st_size for path in files)
  plan = {
      "repo_id": repo_id,
      "private": private,
      "instances": len(manifest["instances"]),
      "files": len(files),
      "total_bytes": total_bytes,
      "total_mib": round(total_bytes / (1024 * 1024), 1),
      "splits": manifest.get("splits", {}).get("counts", {}),
      "dry_run": dry_run,
  }
  if dry_run:
    plan["uploaded"] = False
    return plan

  api = HfApi(token=resolve_token(token))
  api.create_repo(
      repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
  )
  commit = api.upload_folder(
      repo_id=repo_id,
      repo_type="dataset",
      folder_path=str(dataset_dir),
      commit_message=commit_message
      or "Add {} velocity-intervention instances (seed {})".format(
          len(manifest["instances"]), manifest.get("master_seed")
      ),
  )
  plan["uploaded"] = True
  plan["url"] = "https://huggingface.co/datasets/{}".format(repo_id)
  plan["commit"] = getattr(commit, "oid", None) or str(commit)
  return plan


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Upload a generated velocity-intervention batch to HuggingFace."
  )
  parser.add_argument("--dataset-dir", type=Path, required=True)
  parser.add_argument(
      "--repo-id", required=True, help="Target dataset repo, e.g. user/name."
  )
  parser.add_argument(
      "--public", action="store_true",
      help="Publish publicly. Off by default: uploading is hard to undo, and a "
           "public dataset can be mirrored before it can be deleted.",
  )
  parser.add_argument("--commit-message", default=None)
  parser.add_argument("--no-card", action="store_true")
  parser.add_argument(
      "--dry-run", action="store_true",
      help="Write README.md and split_index.json, report the plan, upload nothing.",
  )
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Publishes one generated batch and returns its stable CLI exit status."""
  args = _parser().parse_args(argv)
  try:
    result = upload_dataset(
        args.dataset_dir,
        args.repo_id,
        private=not args.public,
        commit_message=args.commit_message,
        write_card=not args.no_card,
        dry_run=args.dry_run,
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
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


__all__ = [
    "BRANCH_ARTIFACTS",
    "TOKEN_VARIABLES",
    "build_dataset_card",
    "build_split_index",
    "main",
    "resolve_token",
    "upload_dataset",
]


if __name__ == "__main__":
  raise SystemExit(main())
