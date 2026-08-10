"""Interpolation and bounded heuristic perturbation of XYZ/XYZ+WXYZ paths.

Recipe names describe the contact outcome a caller is trying to produce. This module
only generates heuristic candidate path profiles; a physics simulator and downstream
quality control must determine whether the named contact outcome actually occurred.
"""

from __future__ import annotations

import math
import numbers
import warnings
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from interventions.schema import INTERVENTION_RECIPES


RECIPE_PROFILE_SEMANTICS = MappingProxyType({
    "remove_collision": (
        "Heuristic clearance candidate; physics and QC determine collision removal."
    ),
    "create_collision": (
        "Heuristic approach candidate; physics and QC determine collision creation."
    ),
    "retime": (
        "Heuristic timing-warp candidate; physics and QC determine contact changes."
    ),
    "break_contact": (
        "Heuristic lift candidate; physics and QC determine whether contact breaks."
    ),
    "maintain_contact": (
        "Heuristic lateral candidate; physics and QC determine whether contact persists."
    ),
})
_QUATERNION_TOLERANCE = 1e-6
_FLOAT_MAX = np.finfo(float).max


def _numeric_array(value: Any, name: str) -> np.ndarray:
  try:
    untyped = np.asarray(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("{} must be numeric".format(name)) from error
  contains_object_complex = (
      untyped.dtype.kind == "O"
      and any(
          isinstance(item, numbers.Complex) and not isinstance(item, numbers.Real)
          for item in untyped.flat
      )
  )
  if np.iscomplexobj(untyped) or contains_object_complex:
    raise ValueError("{} must not contain complex values".format(name))
  try:
    array = np.asarray(untyped, dtype=float)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("{} must be numeric".format(name)) from error
  return array


def _stable_euclidean_norm(vector: np.ndarray) -> float:
  """Computes a Euclidean norm without overflow during squaring."""
  components = np.abs(np.asarray(vector, dtype=float).reshape(-1))
  scale = float(np.max(components)) if components.size else 0.0
  if scale == 0.0:
    return 0.0
  if not math.isfinite(scale):
    return math.inf
  scaled_norm = math.hypot(*(float(item / scale) for item in components))
  if scale > _FLOAT_MAX / scaled_norm:
    return math.inf
  return scale * scaled_norm


def _validate_path_array(
    value: Any,
    name: str,
    *,
    require_continuous_quaternions: bool,
) -> np.ndarray:
  array = _numeric_array(value, name)
  if array.ndim != 2:
    raise ValueError("{} must be a 2-D array".format(name))
  if array.shape[0] < 2:
    raise ValueError("{} must contain at least two samples".format(name))
  if array.shape[1] not in (3, 7):
    raise ValueError("{} must have dimension 3 or 7".format(name))
  if not np.isfinite(array).all():
    raise ValueError("{} must contain only finite values".format(name))
  if array.shape[1] == 7:
    quaternions = array[:, 3:]
    norms = np.array([_stable_euclidean_norm(item) for item in quaternions])
    if not np.all(np.abs(norms - 1.0) <= _QUATERNION_TOLERANCE):
      raise ValueError("{} quaternions must be unit-normalized".format(name))
    if require_continuous_quaternions and len(quaternions) > 1:
      adjacent_dots = np.einsum("ij,ij->i", quaternions[:-1], quaternions[1:])
      if np.any(adjacent_dots < -_QUATERNION_TOLERANCE):
        raise ValueError("{} quaternions must be sign-continuous".format(name))
  return array


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
  result = np.array(quaternions, dtype=float, copy=True)
  result /= np.linalg.norm(result, axis=1, keepdims=True)
  for index in range(1, len(result)):
    if np.dot(result[index - 1], result[index]) < 0.0:
      result[index] *= -1.0
  return result


def _linear_interpolate(
    source_times: np.ndarray,
    values: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
  """Linearly interpolates with convex weights to avoid subtractive overflow."""
  queries = np.clip(np.asarray(query_times, dtype=float), source_times[0], source_times[-1])
  indices = np.searchsorted(source_times, queries, side="right") - 1
  indices = np.clip(indices, 0, len(source_times) - 2)
  fractions = (
      (queries - source_times[indices])
      / (source_times[indices + 1] - source_times[indices])
  )[:, None]
  return (1.0 - fractions) * values[indices] + fractions * values[indices + 1]


def _slerp(
    source_times: np.ndarray,
    quaternions: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
  """Piecewise spherical interpolation for WXYZ quaternions."""
  aligned = _continuous_quaternions(quaternions)
  queries = np.clip(np.asarray(query_times, dtype=float), source_times[0], source_times[-1])
  indices = np.searchsorted(source_times, queries, side="right") - 1
  indices = np.clip(indices, 0, len(source_times) - 2)
  left_times = source_times[indices]
  right_times = source_times[indices + 1]
  fractions = (queries - left_times) / (right_times - left_times)

  result = np.empty((len(queries), 4), dtype=float)
  for output_index, (source_index, fraction) in enumerate(zip(indices, fractions)):
    left = aligned[source_index]
    right = aligned[source_index + 1]
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
      value = (1.0 - fraction) * left + fraction * right
    else:
      angle = math.acos(dot)
      denominator = math.sin(angle)
      value = (
          math.sin((1.0 - fraction) * angle) / denominator * left
          + math.sin(fraction * angle) / denominator * right
      )
    result[output_index] = value / np.linalg.norm(value)

  result = _continuous_quaternions(result)
  # Avoid search/interpolation roundoff at the exact endpoints.
  endpoint_start = np.isclose(queries, source_times[0], rtol=0.0, atol=1e-15)
  endpoint_end = np.isclose(queries, source_times[-1], rtol=0.0, atol=1e-15)
  result[endpoint_start] = aligned[0]
  result[endpoint_end] = aligned[-1]
  return result


def build_path(
    waypoints: Any,
    num_frames: int,
    method: str = "linear",
) -> np.ndarray:
  """Builds a uniformly sampled trajectory through uniformly timed waypoints.

  Args:
    waypoints: A finite ``[N, 3]`` XYZ or ``[N, 7]`` XYZ+WXYZ array.
    num_frames: Number of output samples, at least two.
    method: ``"linear"`` or ``"spline"`` positional interpolation. Quaternion
      components always use spherical interpolation.

  Returns:
    A new floating-point array with shape ``[num_frames, D]``.

  Raises:
    TypeError: If ``num_frames`` or ``method`` has the wrong type.
    ValueError: If an argument is unsupported, malformed, or non-finite.
  """
  if isinstance(num_frames, bool) or not isinstance(num_frames, numbers.Integral):
    raise TypeError("num_frames must be an integer")
  num_frames = int(num_frames)
  if num_frames < 2:
    raise ValueError("num_frames must be at least two")
  if not isinstance(method, str):
    raise TypeError("method must be a string")
  if method not in ("linear", "spline"):
    raise ValueError("method must be 'linear' or 'spline'")

  source = _validate_path_array(
      waypoints, "waypoints", require_continuous_quaternions=False
  )
  source_times = np.linspace(0.0, 1.0, len(source))
  output_times = np.linspace(0.0, 1.0, num_frames)
  positions = source[:, :3]
  if method == "linear":
    output_positions = _linear_interpolate(source_times, positions, output_times)
  elif len(source) == 2:
    output_positions = _linear_interpolate(source_times, positions, output_times)
  else:
    try:
      from scipy.interpolate import CubicSpline
    except ImportError as error:  # pragma: no cover - exercised only without SciPy.
      raise ImportError("method='spline' requires SciPy") from error
    scales = np.max(np.abs(positions), axis=0)
    safe_scales = np.where(scales == 0.0, 1.0, scales)
    scaled_positions = positions / safe_scales
    try:
      with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scaled_output = CubicSpline(
            source_times, scaled_positions, axis=0
        )(output_times)
    except (FloatingPointError, RuntimeWarning, ValueError) as error:
      raise ValueError("spline interpolation failed") from error
    with np.errstate(over="ignore", invalid="ignore"):
      output_positions = scaled_output * safe_scales

  if source.shape[1] == 3:
    result = np.asarray(output_positions, dtype=float)
  else:
    output_quaternions = _slerp(source_times, source[:, 3:], output_times)
    result = np.column_stack((output_positions, output_quaternions))
  result[0, :3] = source[0, :3]
  result[-1, :3] = source[-1, :3]
  if result.shape[1] == 7:
    aligned = _continuous_quaternions(source[:, 3:])
    result[0, 3:] = aligned[0]
    result[-1, 3:] = aligned[-1]
  if not np.isfinite(result).all():
    raise ValueError("interpolation produced non-finite values")
  return result


def _bounds_array(bounds: Any, name: str) -> Optional[np.ndarray]:
  if bounds is None:
    return None
  array = _numeric_array(bounds, name)
  if array.shape != (2, 3):
    raise ValueError("{} must have shape [2, 3]".format(name))
  if not np.isfinite(array).all():
    raise ValueError("{} must contain only finite values".format(name))
  if np.any(array[0] >= array[1]):
    raise ValueError("{} minimum must be below maximum on every axis".format(name))
  return array


def _aabb_array(static_aabbs: Iterable[Any]) -> np.ndarray:
  if static_aabbs is None:
    return np.empty((0, 2, 3), dtype=float)
  if isinstance(static_aabbs, Mapping):
    static_aabbs = (static_aabbs,)
  try:
    values = tuple(static_aabbs)
  except TypeError as error:
    raise ValueError("static_aabbs must be an iterable of AABBs") from error
  if not values:
    return np.empty((0, 2, 3), dtype=float)

  normalized = []
  for index, aabb in enumerate(values):
    if isinstance(aabb, Mapping):
      try:
        aabb = (aabb["min"], aabb["max"])
      except KeyError as error:
        raise ValueError("AABB mappings require 'min' and 'max'") from error
    array = _numeric_array(aabb, "static_aabbs[{}]".format(index))
    if array.shape != (2, 3):
      raise ValueError("every static AABB must have shape [2, 3]")
    if not np.isfinite(array).all():
      raise ValueError("static AABBs must contain only finite values")
    if np.any(array[0] >= array[1]):
      raise ValueError("AABB minimum must be below maximum on every axis")
    normalized.append(array)
  return np.stack(normalized)


def _nonnegative_real(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise TypeError("{} must be a real number".format(name))
  try:
    result = float(value)
  except (OverflowError, ValueError) as error:
    raise ValueError("{} must be finite".format(name)) from error
  if not math.isfinite(result):
    raise ValueError("{} must be finite".format(name))
  if result < 0.0:
    raise ValueError("{} must be nonnegative".format(name))
  return result


def _segment_intersects_aabb(
    start: np.ndarray,
    end: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
  """Returns whether a closed line segment intersects a closed AABB."""
  entry = 0.0
  exit_ = 1.0
  for axis in range(3):
    start_value = float(start[axis])
    end_value = float(end[axis])
    minimum_value = float(minimum[axis])
    maximum_value = float(maximum[axis])
    segment_minimum = min(start_value, end_value)
    segment_maximum = max(start_value, end_value)
    if segment_maximum < minimum_value or segment_minimum > maximum_value:
      return False
    if start_value == end_value:
      continue
    clipped_minimum = max(minimum_value, segment_minimum)
    clipped_maximum = min(maximum_value, segment_maximum)
    scale = max(
        abs(start_value),
        abs(end_value),
        abs(clipped_minimum),
        abs(clipped_maximum),
    )
    if scale == 0.0:
      continue
    scaled_start = start_value / scale
    scaled_end = end_value / scale
    delta = scaled_end - scaled_start
    if delta == 0.0:
      continue
    first = (clipped_minimum / scale - scaled_start) / delta
    second = (clipped_maximum / scale - scaled_start) / delta
    entry = max(entry, min(first, second))
    exit_ = min(exit_, max(first, second))
    if entry > exit_:
      return False
  return True


def _validate_geometry(
    positions: np.ndarray,
    bounds: Optional[np.ndarray],
    static_aabbs: np.ndarray,
    clearance: float,
) -> None:
  if not np.isfinite(positions).all():
    raise ValueError("path positions must be finite")
  if bounds is not None:
    if np.any(positions < bounds[0]) or np.any(positions > bounds[1]):
      raise ValueError("path positions fall outside bounds")
  for aabb in static_aabbs:
    with np.errstate(over="ignore"):
      expanded_minimum = np.maximum(aabb[0] - clearance, -_FLOAT_MAX)
      expanded_maximum = np.minimum(aabb[1] + clearance, _FLOAT_MAX)
    inside = np.all(positions >= expanded_minimum, axis=1) & np.all(
        positions <= expanded_maximum, axis=1
    )
    if np.any(inside):
      raise ValueError("path intersects a static AABB")
    if any(
        _segment_intersects_aabb(start, end, expanded_minimum, expanded_maximum)
        for start, end in zip(positions[:-1], positions[1:])
    ):
      raise ValueError("path intersects a static AABB")


def validate_path(
    path: Any,
    *,
    bounds: Any = None,
    static_aabbs: Iterable[Any] = (),
    clearance: float = 0.0,
) -> None:
  """Validates a sampled XYZ or XYZ+WXYZ path and optional spatial constraints.

  A path is valid when it is finite, has at least two samples, uses unit and
  sign-continuous quaternions when present, remains within ``bounds``, and has no
  line segment intersecting a static AABB expanded by ``clearance``.

  Raises:
    ValueError: If the path or any spatial constraint is invalid.
  """
  try:
    clearance_value = _nonnegative_real(clearance, "clearance")
  except TypeError as error:
    raise ValueError(str(error)) from error
  array = _validate_path_array(
      path, "path", require_continuous_quaternions=True
  )
  bounds_array = _bounds_array(bounds, "bounds")
  aabbs_array = _aabb_array(static_aabbs)
  _validate_geometry(array[:, :3], bounds_array, aabbs_array, clearance_value)


def max_position_deviation(factual: Any, perturbed: Any) -> float:
  """Returns the maximum per-frame Euclidean XYZ deviation between two paths."""
  factual_array = _numeric_array(factual, "factual")
  perturbed_array = _numeric_array(perturbed, "perturbed")
  if factual_array.ndim != 2 or factual_array.shape[1] not in (3, 7):
    raise ValueError("factual must have shape [T, 3] or [T, 7]")
  if perturbed_array.shape != factual_array.shape:
    raise ValueError("factual and perturbed paths must have identical shapes")
  if factual_array.shape[0] < 2:
    raise ValueError("paths must contain at least two samples")
  if not np.isfinite(factual_array).all() or not np.isfinite(perturbed_array).all():
    raise ValueError("paths must contain only finite values")
  with np.errstate(over="ignore", invalid="ignore"):
    differences = factual_array[:, :3] - perturbed_array[:, :3]
  return max(_stable_euclidean_norm(item) for item in differences)


def _resample_path(path: np.ndarray, query_times: np.ndarray) -> np.ndarray:
  source_times = np.linspace(0.0, 1.0, len(path))
  positions = _linear_interpolate(source_times, path[:, :3], query_times)
  if path.shape[1] == 3:
    return positions
  return np.column_stack(
      (positions, _slerp(source_times, path[:, 3:], query_times))
  )


def _retimed_candidate(
    path: np.ndarray,
    magnitude: float,
    rng: Any,
) -> np.ndarray:
  output_times = np.linspace(0.0, 1.0, len(path))
  try:
    random_value = float(rng.uniform(-1.0, 1.0))
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("rng.uniform must return a finite scalar") from error
  if not math.isfinite(random_value):
    raise ValueError("rng returned a non-finite value")
  sign = -1.0 if random_value < 0.0 else 1.0
  strength_fraction = 0.5 + 0.5 * min(abs(random_value), 1.0)
  base_strength = sign * min(0.3, max(magnitude, 1e-6)) * strength_fraction

  def candidate(scale: float) -> np.ndarray:
    warped = output_times + scale * base_strength * np.sin(np.pi * output_times)
    warped = np.clip(warped, 0.0, 1.0)
    result = _resample_path(path, warped)
    result[0] = path[0]
    result[-1] = path[-1]
    return result

  result = candidate(1.0)
  if max_position_deviation(path, result) <= magnitude:
    return result
  lower = 0.0
  upper = 1.0
  for _ in range(52):
    middle = (lower + upper) / 2.0
    trial = candidate(middle)
    if max_position_deviation(path, trial) <= magnitude:
      lower = middle
      result = trial
    else:
      upper = middle
  return result


def _random_direction(rng: Any, recipe: str) -> np.ndarray:
  for _ in range(8):
    direction = _numeric_array(rng.normal(size=3), "rng direction")
    if direction.shape != (3,) or not np.isfinite(direction).all():
      raise ValueError("rng.normal(size=3) must return three finite values")
    if recipe == "maintain_contact":
      direction[2] = 0.0
    elif recipe == "break_contact":
      direction[2] = abs(direction[2]) + 0.25
    scale = float(np.max(np.abs(direction)))
    if scale > 0.0:
      direction /= scale
      norm = _stable_euclidean_norm(direction)
      direction /= norm
      return -direction if recipe == "create_collision" else direction
  raise ValueError("rng repeatedly returned a zero displacement direction")


def _spatial_candidate(
    path: np.ndarray,
    recipe: str,
    magnitude: float,
    rng: Any,
) -> np.ndarray:
  direction = _random_direction(rng, recipe)
  try:
    amplitude_draw = float(rng.uniform(0.5, 1.0))
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("rng.uniform must return a finite scalar") from error
  if not math.isfinite(amplitude_draw):
    raise ValueError("rng returned a non-finite value")
  amplitude = magnitude * float(np.clip(amplitude_draw, 0.0, 1.0))
  times = np.linspace(0.0, 1.0, len(path))
  if recipe == "remove_collision":
    profile = np.sin(np.pi * times) ** 2
  elif recipe == "create_collision":
    profile = np.sin(np.pi * times)
  elif recipe == "break_contact":
    profile = np.sin(np.pi * times) ** 2
  else:  # maintain_contact
    profile = np.sin(np.pi * times)
  result = np.array(path, dtype=float, copy=True)
  with np.errstate(over="ignore", invalid="ignore"):
    displacement = amplitude * profile[:, None] * direction[None, :]
    result[:, :3] = result[:, :3] + displacement
  result[0] = path[0]
  result[-1] = path[-1]
  return result


def perturb_path(
    path_factual: Any,
    recipe: str,
    magnitude: float,
    rng: Any,
    *,
    bounds: Any = None,
    static_aabbs: Iterable[Any] = (),
    clearance: float = 0.0,
    max_attempts: int = 64,
) -> np.ndarray:
  """Returns a deterministic, bounded perturbation of a sampled trajectory.

  Spatial recipes add a smooth displacement that is zero at both endpoints.
  ``retime`` instead samples the same piecewise trajectory at smoothly warped times.
  These profiles are heuristic candidates: only physics simulation and downstream QC
  can establish whether the recipe's named contact outcome occurred. Candidates
  violating bounds or expanded static AABBs are resampled up to ``max_attempts`` times.

  Args:
    path_factual: A valid ``[T, 3]`` or ``[T, 7]`` path.
    recipe: One of the five supported intervention recipes.
    magnitude: Nonnegative maximum positional deviation.
    rng: NumPy-style random generator exposing ``normal`` and ``uniform``.
    bounds: Optional ``[minimum_xyz, maximum_xyz]`` scene bounds.
    static_aabbs: Static ``[minimum_xyz, maximum_xyz]`` obstacle boxes.
    clearance: Nonnegative expansion applied to every static AABB.
    max_attempts: Positive number of candidates to sample.

  Raises:
    TypeError: If an argument has an incompatible type.
    ValueError: If validation fails or no valid candidate can be sampled.
  """
  if not isinstance(recipe, str):
    raise TypeError("recipe must be a string")
  if recipe not in INTERVENTION_RECIPES:
    raise ValueError("unsupported recipe: {!r}".format(recipe))
  magnitude_value = _nonnegative_real(magnitude, "magnitude")
  clearance_value = _nonnegative_real(clearance, "clearance")
  if isinstance(max_attempts, bool) or not isinstance(max_attempts, numbers.Integral):
    raise TypeError("max_attempts must be an integer")
  max_attempts = int(max_attempts)
  if max_attempts <= 0:
    raise ValueError("max_attempts must be positive")
  if not callable(getattr(rng, "uniform", None)):
    raise TypeError("rng must provide a uniform method")
  if recipe != "retime" and not callable(getattr(rng, "normal", None)):
    raise TypeError("rng must provide a normal method for spatial recipes")

  factual = _validate_path_array(
      path_factual, "path_factual", require_continuous_quaternions=True
  )
  bounds_array = _bounds_array(bounds, "bounds")
  aabbs_array = _aabb_array(static_aabbs)
  if magnitude_value == 0.0:
    result = np.array(factual, dtype=float, copy=True)
    _validate_geometry(result[:, :3], bounds_array, aabbs_array, clearance_value)
    return result

  for _ in range(max_attempts):
    if recipe == "retime":
      candidate = _retimed_candidate(factual, magnitude_value, rng)
    else:
      candidate = _spatial_candidate(factual, recipe, magnitude_value, rng)
    try:
      _validate_geometry(
          candidate[:, :3], bounds_array, aabbs_array, clearance_value
      )
    except ValueError:
      continue
    deviation = max_position_deviation(factual, candidate)
    if deviation > magnitude_value + 1e-12:
      continue
    # The construction leaves spatial-recipe quaternions untouched and uses SLERP
    # for retiming, but validate here to guard unusual numeric inputs from custom RNGs.
    validate_path(candidate)
    return candidate
  raise ValueError(
      "unable to sample a valid perturbation within max_attempts={}".format(
          max_attempts
      )
  )
