"""Pure deterministic tag derivation for intervention ground truth."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Tuple

from interventions.schema import GroundTruth


def _identifier(value: object, name: str) -> str:
  if not isinstance(value, str):
    raise TypeError("{} must be a string".format(name))
  if not value.strip():
    raise ValueError("{} must not be empty".format(name))
  return value


def _environment_ids(value: Iterable[str]) -> frozenset[str]:
  if isinstance(value, (str, bytes)):
    raise TypeError("environment_ids must be an iterable of identifiers")
  try:
    return frozenset(_identifier(item, "environment id") for item in value)
  except TypeError as error:
    raise TypeError("environment_ids must be an iterable of identifiers") from error


def derive_tags(
    ground_truth: GroundTruth,
    *,
    target_id: str,
    environment_ids: Iterable[str] = (),
    unstable: bool = False,
    preintervention_mismatch: bool = False,
) -> Tuple[str, ...]:
  """Derives all applicable deterministic tags from validated ground truth."""
  if not isinstance(ground_truth, GroundTruth):
    raise TypeError("ground_truth must be a GroundTruth")
  target = _identifier(target_id, "target_id")
  environment = _environment_ids(environment_ids)
  if not isinstance(unstable, bool):
    raise TypeError("unstable must be a bool")
  if not isinstance(preintervention_mismatch, bool):
    raise TypeError("preintervention_mismatch must be a bool")

  delta = ground_truth.graph_delta
  records = delta.added + delta.removed + delta.changed
  tags = set()
  if delta.added:
    tags.add("contact_added")
  if delta.removed:
    tags.add("contact_removed")
  if delta.changed:
    tags.add("contact_changed")
  paths = tuple(ground_truth.propagation_path.values())
  if any(len(path) == 2 for path in paths):
    tags.add("direct_contact")
  if any(len(path) >= 3 for path in paths):
    tags.add("cascade")

  no_affected = not ground_truth.hard_affected and not ground_truth.soft_affected
  has_non_environment_edge_away_from_target = any(
      target not in (record["object_a"], record["object_b"])
      and not (
          record["object_a"] in environment
          and record["object_b"] in environment
      )
      for record in records
  )
  if no_affected and not has_non_environment_edge_away_from_target:
    tags.add("target_only")

  if records and all(
      endpoint == target or endpoint in environment
      for record in records
      for endpoint in (record["object_a"], record["object_b"])
  ):
    tags.add("environment_only")
  if not records and no_affected:
    tags.add("null_effect")
  if unstable:
    tags.add("unstable")
  if preintervention_mismatch:
    tags.add("preintervention_mismatch")
  return tuple(sorted(tags))


__all__ = ["derive_tags"]
