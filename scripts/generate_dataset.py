"""CLI for resumable, balanced intervention-dataset generation.

Purpose: parse one batch request, run the dataset pipeline, and emit stable JSON.
Public API: main().
Dependencies: argparse plus interventions.dataset and schema serialization.
Trust boundary: the CLI delegates sampling, QC, journaling, resume checks, and
publication to run_batch(); its JSON status does not authenticate producer origin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interventions.dataset import load_ranges, run_batch
from interventions.schema import to_jsonable


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(allow_abbrev=False)
  parser.add_argument("--config", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--num-instances", required=True, type=int)
  parser.add_argument("--max-attempts", required=True, type=int)
  parser.add_argument("--resume", action="store_true")
  return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Runs resumable batch generation and returns its stable CLI exit status."""
  args = _parser().parse_args(argv)
  try:
    result = run_batch(
        load_ranges(args.config),
        args.output,
        args.seed,
        args.num_instances,
        args.max_attempts,
        resume=args.resume,
        workers=1,
    )
  except Exception as error:  # CLI boundary emits machine-readable failures.
    print(json.dumps({
        "status": "error",
        "error_type": type(error).__name__,
        "message": str(error),
    }, sort_keys=True))
    return 1
  print(json.dumps(to_jsonable(result), sort_keys=True))
  return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover - exercised as a module.
  raise SystemExit(main())
