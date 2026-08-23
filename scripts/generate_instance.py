"""Generate one inspectable factual/counterfactual candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interventions.dataset import (
    _publish_instance,
    evaluate_qc,
    generate_candidate,
    load_ranges,
    sample_instance_spec,
)
from interventions.schema import to_jsonable


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--config", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--attempt-index", required=True, type=int)
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = _parser().parse_args(argv)
  try:
    ranges = load_ranges(args.config)
    spec = sample_instance_spec(ranges, args.seed, args.attempt_index)
    factual, counterfactual, truth = generate_candidate(spec)
    qc = evaluate_qc(spec, factual, counterfactual, truth, ranges.get("qc", {}))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    artifact = _publish_instance(root, spec, factual, counterfactual, truth)
    payload = {
        "status": "accepted" if qc.accepted else "rejected",
        "instance_id": spec.instance_id,
        "artifact_path": str(artifact),
        "qc": qc.to_dict(),
    }
  except Exception as error:  # CLI boundary emits machine-readable failures.
    print(json.dumps({
        "status": "error",
        "error_type": type(error).__name__,
        "message": str(error),
    }, sort_keys=True))
    return 1
  print(json.dumps(to_jsonable(payload), sort_keys=True))
  return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module.
  raise SystemExit(main())
