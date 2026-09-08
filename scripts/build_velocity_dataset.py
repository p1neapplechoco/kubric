"""Resumable batch driver for the velocity-intervention dataset.

Purpose:
  Generates ``count`` instances starting at ``start`` from one master seed:
  sample -> simulate three branches -> QC -> ground truth -> write physics
  artifacts, then (optionally) render every branch with Blender via
  :mod:`scripts.render_velocity_intervention`. Rendering may run in-process or
  through a separate command (another interpreter that has ``bpy``, or a
  ``docker run`` prefix) so that physics and rendering environments can differ.
  Finally a ``manifest.jsonl`` row per instance and a ``dataset_summary.json``
  are (re)written by scanning the output directory, which makes reruns idempotent.

Public API:
  ``instance_dirname``, ``assign_split``, ``build_manifest``, ``build_dataset``,
  ``main``.

Dependencies:
  :mod:`interventions.velocity_intervention` (NumPy, PyBullet, Kubric core).
  Rendering additionally needs ``bpy`` in whichever process executes
  ``scripts/render_velocity_intervention.py``.

Trust boundary:
  Existing ``instance.json`` / ``render_info.json`` files are treated as completed
  work and skipped; delete a directory to regenerate it. Render subprocess failures
  are recorded in ``render_failures.jsonl`` and do not abort the batch unless
  ``--strict`` is set. The manifest is derived from on-disk artifacts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from interventions import velocity_intervention as vi  # noqa: E402  pylint: disable=wrong-import-position
from interventions.schema import to_jsonable  # noqa: E402  pylint: disable=wrong-import-position

DEFAULT_CONFIG = _REPO_ROOT / "configs" / "velocity_intervention.yaml"
RENDER_SCRIPT = _REPO_ROOT / "scripts" / "render_velocity_intervention.py"
INSTANCES_DIRNAME = "instances"
DEFAULT_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def instance_dirname(index: int) -> str:
  """Directory name for instance ``index`` (zero padded, seed independent)."""
  return "{:06d}".format(int(index))


def assign_split(instance_id: str, fractions: Mapping[str, float] = DEFAULT_SPLITS, salt: str = "split") -> str:
  """Deterministically hashes ``instance_id`` into a named split."""
  digest = hashlib.sha256("{}\0{}".format(salt, instance_id).encode("utf-8")).digest()
  point = int.from_bytes(digest[:8], "big") / float(1 << 64)
  total = sum(fractions.values())
  running = 0.0
  names = sorted(fractions)
  for name in names:
    running += fractions[name] / total
    if point < running:
      return name
  return names[-1]


def _render_inline(instance_dir: Path, branches: Sequence[str], render_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
  from scripts import render_velocity_intervention as rvi  # pylint: disable=import-outside-toplevel

  return rvi.render_instance(instance_dir, branches, **render_kwargs)


def _render_subprocess(
    command: Sequence[str], instance_dir: Path, branches: Sequence[str],
    resolution: int, samples: int, layers: Sequence[str], denoise: bool, require_gpu: bool,
) -> None:
  argv = list(command) + [str(instance_dir), "--branches", *branches,
                          "--resolution", str(resolution), "--samples", str(samples),
                          "--layers", *layers]
  if not denoise:
    argv.append("--no-denoise")
  if require_gpu:
    argv.append("--require-gpu")
  subprocess.run(argv, check=True)


def _rendered(instance_dir: Path, branches: Sequence[str]) -> bool:
  return all((instance_dir / branch / "render_info.json").exists() for branch in branches)


def build_manifest(output: Path, splits: Mapping[str, float] = DEFAULT_SPLITS) -> List[Mapping[str, Any]]:
  """Scans ``output/instances`` and rewrites ``manifest.jsonl`` + ``dataset_summary.json``."""
  rows: List[Mapping[str, Any]] = []
  instances_dir = output / INSTANCES_DIRNAME
  for instance_dir in sorted(instances_dir.iterdir()) if instances_dir.exists() else []:
    spec_path = instance_dir / vi.SPEC_FILENAME
    if not spec_path.exists():
      continue
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    truth = json.loads((instance_dir / "ground_truth.json").read_text(encoding="utf-8"))
    qc = json.loads((instance_dir / "qc.json").read_text(encoding="utf-8"))
    renders = {}
    for branch in vi.BRANCHES:
      info_path = instance_dir / branch / "render_info.json"
      if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        renders[branch] = {"device": info["device"]["device"], "backend": info["device"].get("backend"),
                           "resolution": info["resolution"], "frames": info["frames"]}
    motion = qc["report"]["metrics"]["motion"]["factual"]
    rows.append({
        "index": payload["index"],
        "instance_id": payload["instance_id"],
        "path": instance_dir.relative_to(output).as_posix(),
        "split": assign_split(payload["instance_id"], splits),
        "master_seed": payload["master_seed"],
        "attempt": payload["attempt"],
        "object_count": payload["metadata"]["object_count"],
        "subject_id": payload["subject_id"],
        "subject_shape": next(o["shape"] for o in payload["scene"]["objects"] if o["object_id"] == payload["subject_id"]),
        "roles": payload["roles"],
        "material_families": {oid: p["material_family"] for oid, p in payload["physics"].items()},
        "factual_velocity": payload["factual_velocity"],
        "counterfactual_velocity": payload["counterfactual_velocity"],
        "heading_change_rad": payload["metadata"]["heading_change_rad"],
        "factual_struck": qc["report"]["metrics"]["factual_struck"],
        "factual_untouched": qc["report"]["metrics"]["factual_untouched"],
        "motion_labels": {oid: info["label"] for oid, info in motion.items()},
        "hard_affected": truth["hard_affected"],
        "soft_affected": truth["soft_affected"],
        "graph_delta_counts": {k: len(truth["graph_delta"][k]) for k in ("added", "removed", "changed")},
        "rendered_branches": sorted(renders),
        "renders": renders,
        "camera": payload["scene"]["camera"],
    })
  (output / "manifest.jsonl").write_text(
      "".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
  )
  summary = {
      "instances": len(rows),
      "rendered": sum(1 for row in rows if len(row["rendered_branches"]) == len(vi.BRANCHES)),
      "splits": {name: sum(1 for row in rows if row["split"] == name) for name in sorted(splits)},
      "object_counts": {str(k): sum(1 for row in rows if row["object_count"] == k)
                        for k in sorted({row["object_count"] for row in rows})},
      "subject_shapes": {shape: sum(1 for row in rows if row["subject_shape"] == shape)
                         for shape in sorted({row["subject_shape"] for row in rows})},
      "render_devices": sorted({r["device"] for row in rows for r in row["renders"].values()}),
      "branches": list(vi.BRANCHES),
      "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  }
  (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  return rows


def build_dataset(
    output: Path, *, config: Path = DEFAULT_CONFIG, seed: int = 0, start: int = 0, count: int = 1,
    render: bool = True, render_command: Optional[Sequence[str]] = None, branches: Sequence[str] = vi.BRANCHES,
    resolution: int = 512, samples: int = 64, layers: Sequence[str] = ("rgba", "segmentation", "depth"),
    denoise: bool = True, require_gpu: bool = False, strict: bool = False, log=print,
) -> List[Mapping[str, Any]]:
  """Generates (and renders) instances ``start .. start+count-1`` under ``output``."""
  ranges = vi.load_ranges(config)
  output = Path(output)
  instances_dir = output / INSTANCES_DIRNAME
  instances_dir.mkdir(parents=True, exist_ok=True)
  (output / "config.yaml").write_text(Path(config).read_text(encoding="utf-8"), encoding="utf-8")
  failures_path = output / "render_failures.jsonl"

  for index in range(start, start + count):
    instance_dir = instances_dir / instance_dirname(index)
    if not (instance_dir / vi.SPEC_FILENAME).exists():
      started = time.time()
      generated = vi.generate_instance(ranges, seed, index)
      vi.write_instance(instance_dir, generated, overwrite=True)
      log("[physics] index={} id={} attempts={} objects={} struck={} {:.1f}s".format(
          index, generated.spec.instance_id, generated.spec.attempt + 1,
          generated.spec.metadata["object_count"], generated.qc.metrics["factual_struck"],
          time.time() - started))
    else:
      log("[physics] index={} already generated".format(index))

    if render and not _rendered(instance_dir, branches):
      started = time.time()
      try:
        if render_command:
          _render_subprocess(render_command, instance_dir, branches, resolution, samples,
                             layers, denoise, require_gpu)
        else:
          info = _render_inline(instance_dir, branches, dict(
              resolution=resolution, samples=samples, layers=layers, denoise=denoise))
          if require_gpu and any(rec["device"]["device"] != "GPU" for rec in info.values()):
            raise RuntimeError("GPU required but Cycles rendered on CPU")
        log("[render] index={} branches={} {:.1f}s".format(index, list(branches), time.time() - started))
      except Exception as error:  # pylint: disable=broad-except
        with failures_path.open("a", encoding="utf-8") as handle:
          handle.write(json.dumps({"index": index, "error": repr(error)}) + "\n")
        log("[render] index={} FAILED: {!r}".format(index, error))
        if strict:
          raise

  return build_manifest(output)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--count", type=int, default=1)
  parser.add_argument("--no-render", action="store_true", help="physics artifacts only")
  parser.add_argument("--render-command", type=str, default=None,
                      help="command prefix that runs scripts/render_velocity_intervention.py "
                           "(e.g. a different interpreter or a `docker run ...` prefix); "
                           "default renders in-process")
  parser.add_argument("--branches", nargs="+", default=list(vi.BRANCHES), choices=vi.BRANCHES)
  parser.add_argument("--resolution", type=int, default=512,
                      help="render resolution (primary control for image quality and GPU compute/memory scaling)")
  parser.add_argument("--samples", type=int, default=64)
  parser.add_argument("--layers", nargs="*", default=["rgba", "segmentation", "depth"])
  parser.add_argument("--no-denoise", action="store_true")
  parser.add_argument("--require-gpu", action="store_true")
  parser.add_argument("--strict", action="store_true", help="abort on the first render failure")
  parser.add_argument("--manifest-only", action="store_true", help="only rebuild manifest/summary")
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Runs resumable batch generation and returns zero when the manifest is written."""
  args = _parser().parse_args(argv)
  if args.manifest_only:
    rows = build_manifest(args.output)
    print("[manifest] {} instances".format(len(rows)))
    return 0
  render_command = shlex.split(args.render_command) if args.render_command else None
  rows = build_dataset(
      args.output, config=args.config, seed=args.seed, start=args.start, count=args.count,
      render=not args.no_render, render_command=render_command, branches=args.branches,
      resolution=args.resolution, samples=args.samples, layers=args.layers,
      denoise=not args.no_denoise, require_gpu=args.require_gpu, strict=args.strict,
  )
  print("[manifest] {} instances -> {}".format(len(rows), args.output / "manifest.jsonl"))
  return 0


if __name__ == "__main__":
  sys.exit(main())
