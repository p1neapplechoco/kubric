"""Uploads a built velocity-intervention dataset to the Hugging Face Hub.

Purpose:
  Writes a dataset card (``README.md`` with YAML front matter) next to the
  ``manifest.jsonl`` produced by :mod:`scripts.build_velocity_dataset`, then
  uploads the whole output directory to a ``dataset`` repository with
  ``huggingface_hub``. Large folders are uploaded in resumable chunks; reruns only
  upload files whose content changed.

Public API:
  ``resolve_token``, ``write_dataset_card``, ``publish_dataset``, ``main``.

Dependencies:
  ``huggingface_hub`` (>= 0.24) and the JSON artifacts written by the build
  driver. No Kubric or Blender import.

Trust boundary:
  The access token is read from ``--token``, the ``HF_TOKEN`` environment
  variable or the cached ``huggingface-cli login`` credentials, in that order,
  and is never written to disk or printed. The card is generated from
  ``dataset_summary.json``; nothing else in the output directory is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CARD_TEMPLATE = """---
license: apache-2.0
task_categories:
  - video-classification
  - other
tags:
  - kubric
  - synthetic
  - physics
  - counterfactual
  - intervention
  - rigid-body
pretty_name: {pretty_name}
size_categories:
  - {size_category}
---

# {pretty_name}

Synthetic multi-object rigid-body videos generated with
[Kubric](https://github.com/{source_repo}) (PyBullet physics + Blender/Cycles
rendering{gpu_note}). Every scene has **4-6 fixed-mass bodies** on a table; exactly one
body (the *subject*) receives an **initial velocity along a single heading** and nothing
else is actuated. Some bodies sit in the subject's corridor and get struck, others are
bystanders that are never touched. Spheres roll, boxes slide, and a sphere launched
without spin slides first and then rolls. Materials (and the coupled friction /
restitution) are sampled per object; the camera is fixed within a clip and re-sampled
between clips.

Each instance has three branches sharing one visual scene and one camera:

| branch | description |
| --- | --- |
| `factual` | subject launched with the sampled initial velocity |
| `counterfactual` | same scene, the factual velocity is discarded and a **new** initial velocity is applied at step 0 |
| `subject_removed` | same scene without the subject |

## Contents

```
manifest.jsonl                 one row per instance (split, roles, velocities, QC, ground truth summary)
dataset_summary.json           counts and render devices
config.yaml                    sampling ranges used for this build
instances/<index>/
  instance.json                full scene spec (objects, physics, materials, camera)
  ground_truth.json            contact-graph delta, hard/soft affected objects, propagation paths
  qc.json                      QC metrics: rolling / sliding labels, struck & untouched objects
  <branch>/video.mp4           RGB clip ({frames} frames @ {frame_rate} fps)
  <branch>/mask.mp4            colourised instance segmentation preview
  <branch>/segmentation.npz    uint8 [T,H,W] instance ids (0 = floor/background)
  <branch>/depth.npz           float16 [T,H,W] depth
  <branch>/tracking.npz        per-frame poses, velocities, 2D projections, boxes, visibility, presence
  <branch>/graph.json          temporal contact graph (object-object episodes; floor episodes listed separately)
  <branch>/sim_log/            immutable per-physics-step states + contacts (manifest-hashed)
```

`segmentation_ids` / `object_ids` in `tracking.npz` are shared across the three branches,
so masks and tracks line up between factual, counterfactual and subject-removed clips.

## Build summary

```json
{summary_json}
```

## Reproduce

```bash
git clone https://github.com/{source_repo}
cd kubric
python scripts/build_velocity_dataset.py --output out --seed {seed} --count {count} --require-gpu
python scripts/publish_velocity_dataset.py --output out --repo-id <user>/<name>
```
"""


def resolve_token(explicit: Optional[str] = None) -> Optional[str]:
  """Returns the first available token: argument, ``HF_TOKEN``, cached login."""
  if explicit:
    return explicit
  for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    if os.environ.get(name):
      return os.environ[name]
  try:
    from huggingface_hub import HfFolder  # pylint: disable=import-outside-toplevel
    return HfFolder.get_token()
  except Exception:  # pylint: disable=broad-except
    return None


def _size_category(count: int) -> str:
  if count < 1000:
    return "n<1K"
  if count < 10_000:
    return "1K<n<10K"
  if count < 100_000:
    return "10K<n<100K"
  return "100K<n<1M"


def write_dataset_card(
    output: Path, repo_id: str, *, source_repo: str = "p1neapplechoco/kubric",
    seed: Optional[int] = None,
) -> Path:
  """Writes ``README.md`` (dataset card) into ``output`` from the build summary."""
  summary_path = output / "dataset_summary.json"
  summary: Mapping[str, Any] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
  config = {}
  if (output / "config.yaml").exists():
    try:
      import yaml  # pylint: disable=import-outside-toplevel
      config = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # pylint: disable=broad-except
      config = {}
  scene = config.get("scene", {})
  frame_range = scene.get("frame_range", [0, 48])
  devices = summary.get("render_devices", [])
  gpu_note = ", GPU-rendered" if devices == ["GPU"] else ""
  rows = []
  manifest = output / "manifest.jsonl"
  if manifest.exists():
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
  inferred_seed = seed if seed is not None else (rows[0]["master_seed"] if rows else 0)
  card = _CARD_TEMPLATE.format(
      pretty_name=repo_id.split("/")[-1],
      size_category=_size_category(int(summary.get("instances", len(rows)))),
      source_repo=source_repo,
      gpu_note=gpu_note,
      frames=int(frame_range[1]) - int(frame_range[0]),
      frame_rate=scene.get("frame_rate", 24),
      summary_json=json.dumps(summary, indent=2, sort_keys=True),
      seed=inferred_seed,
      count=int(summary.get("instances", len(rows))),
  )
  path = output / "README.md"
  path.write_text(card, encoding="utf-8")
  return path


def publish_dataset(
    output: Path, repo_id: str, *, token: Optional[str] = None, private: bool = True,
    commit_message: str = "Upload velocity-intervention dataset", source_repo: str = "p1neapplechoco/kubric",
    seed: Optional[int] = None, ignore_patterns: Sequence[str] = ("**/.render-*", "**/*.tmp", "**/__pycache__/**"),
) -> str:
  """Creates the dataset repo if needed and uploads ``output``; returns its URL."""
  from huggingface_hub import HfApi  # pylint: disable=import-outside-toplevel

  output = Path(output)
  if not (output / "manifest.jsonl").exists():
    raise FileNotFoundError("{} has no manifest.jsonl; run build_velocity_dataset.py first".format(output))
  resolved = resolve_token(token)
  if not resolved:
    raise RuntimeError("no Hugging Face token: pass --token, set HF_TOKEN, or run `huggingface-cli login`")
  write_dataset_card(output, repo_id, source_repo=source_repo, seed=seed)
  api = HfApi(token=resolved)
  api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
  if hasattr(api, "upload_large_folder"):
    # Resumable, chunked, and skips files already present with the same hash.
    api.upload_large_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(output),
        ignore_patterns=list(ignore_patterns), print_report=True,
    )
  else:  # pragma: no cover - older huggingface_hub
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(output),
        commit_message=commit_message, ignore_patterns=list(ignore_patterns),
    )
  return "https://huggingface.co/datasets/{}".format(repo_id)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--output", type=Path, required=True, help="directory built by build_velocity_dataset.py")
  parser.add_argument("--repo-id", required=True, help="e.g. username/kubric-velocity-intervention")
  parser.add_argument("--token", default=None, help="HF access token (defaults to $HF_TOKEN / cached login)")
  parser.add_argument("--public", action="store_true", help="create a public repo (default private)")
  parser.add_argument("--seed", type=int, default=None, help="master seed to print in the card")
  parser.add_argument("--source-repo", default="p1neapplechoco/kubric")
  parser.add_argument("--card-only", action="store_true", help="only (re)write README.md, do not upload")
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Writes the dataset card, uploads the folder and prints the dataset URL."""
  args = _parser().parse_args(argv)
  if args.card_only:
    print(write_dataset_card(args.output, args.repo_id, source_repo=args.source_repo, seed=args.seed))
    return 0
  url = publish_dataset(
      args.output, args.repo_id, token=args.token, private=not args.public,
      source_repo=args.source_repo, seed=args.seed,
  )
  print(url)
  return 0


if __name__ == "__main__":
  sys.exit(main())
