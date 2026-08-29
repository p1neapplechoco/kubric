"""Material families coupling appearance priors to simulated physical properties.

Purpose: hold the shipped per-family visual and physical priors and realize them.
Public API: COUPLING_MODES, MASS_BOUNDS, MATERIAL_DRAW_COUNT, PHYSICS_DRAW_COUNT,
FamilyPriors, FAMILY_PRIORS, sample_material, coupled_physics, proxy_volume, and
is_held_out.
Dependencies: Python's standard library, NumPy generators supplied by the caller, and
interventions.appearance; no renderer or simulator import.
Trust boundary: priors are dataset-scale conventions, not physical measurements, and
clamping is recorded rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from interventions import appearance
from interventions.schema import _real, _SchemaMixin


COUPLING_MODES = frozenset(("coupled", "independent", "held_out"))

# The published mass envelope. Target push mass is sampled from the same bounds,
# so a target and a dynamic object of the same family stay comparable.
MASS_BOUNDS: Tuple[float, float] = (0.25, 4.0)

# Draw counts are part of the sampling contract: callers that interleave other
# draws depend on a family never changing how much of the stream it consumes.
MATERIAL_DRAW_COUNT = 5
PHYSICS_DRAW_COUNT = 3

_SPECULAR = (0.4, 0.6)
_NO_EMISSION = (0.0, 0.0)


def _range(value: Any, name: str) -> Tuple[float, float]:
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    raise TypeError("{} must be a (low, high) pair".format(name))
  if len(value) != 2:
    raise ValueError("{} must be a (low, high) pair".format(name))
  low = _real(value[0], "{}[0]".format(name))
  high = _real(value[1], "{}[1]".format(name))
  if low > high:
    raise ValueError("{} lower bound must not exceed its upper bound".format(name))
  return (low, high)


@dataclass(frozen=True)
class FamilyPriors(_SchemaMixin):
  """Bounded visual and physical priors for one material family."""

  family: str
  density: Tuple[float, float]
  friction: Tuple[float, float]
  restitution: Tuple[float, float]
  metallic: Tuple[float, float]
  roughness: Tuple[float, float]
  ior: Tuple[float, float]
  transmission: Tuple[float, float]
  specular: Tuple[float, float]
  emission_strength: Tuple[float, float]
  texture_kinds: Tuple[str, ...]

  def __post_init__(self) -> None:
    if self.family not in appearance.MATERIAL_FAMILIES:
      raise ValueError("unknown material family: {!r}".format(self.family))
    for name in (
        "density", "friction", "restitution", "metallic", "roughness", "ior",
        "transmission", "specular", "emission_strength",
    ):
      object.__setattr__(self, name, _range(getattr(self, name), name))
    kinds = tuple(self.texture_kinds)
    if not kinds:
      raise ValueError("texture_kinds must not be empty")
    if len(set(kinds)) != len(kinds):
      raise ValueError("texture_kinds must not repeat a kind")
    unknown = set(kinds) - appearance.TEXTURE_KINDS
    if unknown:
      raise ValueError("unknown texture kinds: {}".format(sorted(unknown)))
    object.__setattr__(self, "texture_kinds", kinds)


def _priors(family, density, friction, restitution, metallic, roughness, ior,
            transmission, texture_kinds) -> FamilyPriors:
  return FamilyPriors(
      family=family,
      density=density,
      friction=friction,
      restitution=restitution,
      metallic=metallic,
      roughness=roughness,
      ior=ior,
      transmission=transmission,
      specular=_SPECULAR,
      emission_strength=_NO_EMISSION,
      texture_kinds=texture_kinds,
  )


FAMILY_PRIORS: Mapping[str, FamilyPriors] = MappingProxyType({
    "metal": _priors(
        "metal", (55.0, 100.0), (0.15, 0.45), (0.10, 0.35),
        (0.85, 1.00), (0.12, 0.45), (1.45, 2.50), (0.0, 0.0),
        ("solid", "noise", "speckle")),
    "rubber": _priors(
        "rubber", (18.0, 35.0), (0.65, 0.95), (0.55, 0.85),
        (0.0, 0.0), (0.65, 0.95), (1.20, 1.60), (0.0, 0.0),
        ("solid", "noise", "speckle")),
    "plastic": _priors(
        "plastic", (12.0, 30.0), (0.25, 0.55), (0.20, 0.55),
        (0.0, 0.0), (0.20, 0.60), (1.35, 1.60), (0.0, 0.0),
        ("solid", "noise", "checker", "speckle")),
    "ceramic": _priors(
        "ceramic", (30.0, 65.0), (0.30, 0.60), (0.10, 0.35),
        (0.0, 0.05), (0.15, 0.45), (1.45, 1.65), (0.0, 0.0),
        ("solid", "marble", "speckle")),
    "glass": _priors(
        "glass", (35.0, 70.0), (0.20, 0.50), (0.05, 0.25),
        (0.0, 0.0), (0.02, 0.18), (1.45, 1.55), (0.85, 1.00),
        ("solid",)),
    "wood": _priors(
        "wood", (10.0, 28.0), (0.35, 0.70), (0.15, 0.45),
        (0.0, 0.0), (0.35, 0.75), (1.35, 1.55), (0.0, 0.0),
        ("wood", "solid", "noise")),
    "stone": _priors(
        "stone", (45.0, 85.0), (0.55, 0.90), (0.02, 0.20),
        (0.0, 0.10), (0.55, 0.95), (1.40, 1.70), (0.0, 0.0),
        ("marble", "noise", "speckle", "solid")),
})


def proxy_volume(shape: str, half_extents: Sequence[float]) -> float:
  """Returns the volume of the collision proxy ``shape`` with ``half_extents``.

  For a capsule, ``half_extents[2]`` is the cylinder half-height, matching
  ObjectConfig.size rather than schema.half_extents, which adds the cap radius.
  """
  extents = tuple(_real(component, "half_extents") for component in half_extents)
  if len(extents) != 3:
    raise ValueError("half_extents must have three components")
  radius = extents[0]
  if shape == "cube":
    return 8.0 * extents[0] * extents[1] * extents[2]
  if shape == "sphere":
    return 4.0 / 3.0 * math.pi * radius ** 3
  if shape == "cylinder":
    return math.pi * radius ** 2 * (2.0 * extents[2])
  if shape == "capsule":
    return (
        math.pi * radius ** 2 * (2.0 * extents[2])
        + 4.0 / 3.0 * math.pi * radius ** 3
    )
  raise ValueError("unsupported shape: {!r}".format(shape))


def _draw(rng, bounds: Tuple[float, float]) -> float:
  """Consumes exactly one uniform draw and returns a value inside ``bounds``.

  A degenerate range still draws, so the number of values a family consumes is
  independent of its priors and one family's ranges cannot shift another's
  samples downstream.
  """
  sample = float(rng.uniform(0.0, 1.0))
  if bounds[0] == bounds[1]:
    return bounds[0]
  return bounds[0] + sample * (bounds[1] - bounds[0])


def sample_material(
    rng, family: str, color_rgba: Sequence[float], texture
) -> appearance.MaterialSpec:
  """Realizes one material of ``family`` from its priors."""
  priors = FAMILY_PRIORS[family]
  metallic = _draw(rng, priors.metallic)
  roughness = _draw(rng, priors.roughness)
  specular = _draw(rng, priors.specular)
  ior = _draw(rng, priors.ior)
  transmission = _draw(rng, priors.transmission)
  return appearance.MaterialSpec(
      family=family,
      base_color=tuple(color_rgba),
      metallic=metallic,
      roughness=roughness,
      specular=specular,
      ior=ior,
      transmission=transmission,
      emission=(0.0, 0.0, 0.0, 1.0),
      texture=texture,
  )


def coupled_physics(rng, family: str, proxy_volume: float) -> Mapping[str, float]:
  """Derives mass, friction, and restitution from ``family`` and proxy size.

  The unclamped mass is returned alongside the clamped one so that a proxy whose
  size pushes it outside MASS_BOUNDS is visible in the record instead of silently
  looking like a deliberately chosen boundary value.
  """
  priors = FAMILY_PRIORS[family]
  effective_density = _draw(rng, priors.density)
  friction = _draw(rng, priors.friction)
  restitution = _draw(rng, priors.restitution)
  unclamped_mass = effective_density * float(proxy_volume)
  return {
      "effective_density": effective_density,
      "unclamped_mass": unclamped_mass,
      "mass": min(max(unclamped_mass, MASS_BOUNDS[0]), MASS_BOUNDS[1]),
      "friction": friction,
      "restitution": restitution,
  }


def is_held_out(
    combination: Mapping[str, str], holdouts: Sequence[Mapping[str, str]]
) -> bool:
  """Reports whether ``combination`` matches every key of any holdout pattern.

  A holdout names only the axes it constrains, so it excludes a family of
  combinations rather than a single fully specified one.
  """
  return any(
      all(combination.get(key) == value for key, value in holdout.items())
      for holdout in holdouts
  )
