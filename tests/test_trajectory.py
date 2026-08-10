"""Tests for trajectory interpolation and perturbation."""

import math
import warnings

import numpy as np
import pytest

from interventions import schema as schema_module
from interventions import trajectory as trajectory_module
from interventions.trajectory import (
    build_path,
    max_position_deviation,
    perturb_path,
    validate_path,
)


RECIPES = tuple(sorted(schema_module.INTERVENTION_RECIPES))


def _curved_path(num_frames=9):
  t = np.linspace(0.0, 1.0, num_frames)
  return np.column_stack((t, 0.2 * np.sin(np.pi * t), t * (1.0 - t)))


def _pose_path(num_frames=9):
  positions = _curved_path(num_frames)
  angles = np.linspace(0.0, np.pi / 2.0, num_frames)
  quaternions = np.column_stack(
      (np.cos(angles / 2.0), np.zeros((num_frames, 2)), np.sin(angles / 2.0))
  )
  return np.column_stack((positions, quaternions))


def test_build_linear_path_has_expected_shape_values_and_endpoints():
  waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

  path = build_path(waypoints, 5, method="linear")

  assert path.shape == (5, 3)
  np.testing.assert_allclose(path[0], waypoints[0])
  np.testing.assert_allclose(path[-1], waypoints[-1])
  np.testing.assert_allclose(path[2], [0.5, 1.0, 1.5])


def test_build_linear_path_interpolates_extreme_finite_endpoints_safely():
  waypoints = np.array([[1e308, 0.0, 0.0], [-1e308, 0.0, 0.0]])

  path = build_path(waypoints, 3, method="linear")

  assert np.isfinite(path).all()
  np.testing.assert_array_equal(path[[0, -1]], waypoints)
  assert path[1, 0] == pytest.approx(0.0, abs=1.0)


def test_build_spline_path_preserves_waypoints_endpoints():
  waypoints = np.array(
      [[0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, 0.0, 1.0]]
  )

  path = build_path(waypoints, 11, method="spline")

  assert path.shape == (11, 3)
  np.testing.assert_allclose(path[[0, -1]], waypoints[[0, -1]])
  assert np.isfinite(path).all()


def test_build_two_waypoint_spline_uses_overflow_safe_linear_interpolation():
  waypoints = np.array([[1e308, 0.0, 0.0], [-1e308, 0.0, 0.0]])

  spline = build_path(waypoints, 5, method="spline")
  linear = build_path(waypoints, 5, method="linear")

  assert np.isfinite(spline).all()
  np.testing.assert_array_equal(spline, linear)


def test_build_spline_scales_three_extreme_waypoints_before_interpolation():
  waypoints = np.array(
      [[1e308, 0.0, 0.0], [0.0, 1.0, 0.0], [-1e308, 0.0, 0.0]]
  )

  path = build_path(waypoints, 7, method="spline")

  assert np.isfinite(path).all()
  np.testing.assert_array_equal(path[[0, -1]], waypoints[[0, -1]])


@pytest.mark.parametrize("method", ["linear", "spline"])
def test_build_pose_path_uses_unit_sign_continuous_quaternions(method):
  root_half = np.sqrt(0.5)
  waypoints = np.array(
      [
          [0, 0, 0, 1, 0, 0, 0],
          [1, 0, 0, root_half, 0, 0, root_half],
          [2, 1, 0, 0, 0, 0, -1],
      ],
      dtype=float,
  )

  path = build_path(waypoints, 17, method=method)

  assert path.shape == (17, 7)
  np.testing.assert_allclose(path[0, :3], waypoints[0, :3])
  np.testing.assert_allclose(path[-1, :3], waypoints[-1, :3])
  np.testing.assert_allclose(np.linalg.norm(path[:, 3:], axis=1), 1.0)
  assert np.all(np.sum(path[:-1, 3:] * path[1:, 3:], axis=1) >= -1e-12)
  assert abs(np.dot(path[-1, 3:], waypoints[-1, 3:])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "waypoints,num_frames,method",
    [
        ([[0, 0, 0]], 2, "linear"),
        ([[0, 0], [1, 1]], 2, "linear"),
        ([[0, 0, 0], [1, np.nan, 0]], 2, "linear"),
        ([[0, 0, 0], [1, 1, 1]], 1, "linear"),
        ([[0, 0, 0], [1, 1, 1]], 2, "bezier"),
        ([[0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]], 2, "linear"),
        ([[0, 0, 0, 2, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]], 2, "linear"),
    ],
)
def test_build_path_rejects_invalid_inputs(waypoints, num_frames, method):
  with pytest.raises((TypeError, ValueError)):
    build_path(waypoints, num_frames, method)


@pytest.mark.parametrize("recipe", RECIPES)
def test_perturb_path_supports_every_recipe_without_mutating_input(recipe):
  factual = _curved_path()
  before = factual.copy()

  perturbed = perturb_path(
      factual, recipe, 0.15, np.random.default_rng(1234)
  )

  np.testing.assert_array_equal(factual, before)
  assert perturbed.shape == factual.shape
  assert np.isfinite(perturbed).all()
  np.testing.assert_allclose(perturbed[[0, -1]], factual[[0, -1]])
  assert max_position_deviation(factual, perturbed) <= 0.15 + 1e-12
  assert not np.allclose(perturbed[1:-1], factual[1:-1])


@pytest.mark.parametrize("recipe", RECIPES)
def test_perturb_path_is_deterministic_for_equal_rng_state(recipe):
  factual = _pose_path()

  first = perturb_path(factual, recipe, 0.1, np.random.default_rng(88))
  second = perturb_path(factual, recipe, 0.1, np.random.default_rng(88))

  np.testing.assert_allclose(first, second)
  np.testing.assert_allclose(np.linalg.norm(first[:, 3:], axis=1), 1.0)
  assert np.all(np.sum(first[:-1, 3:] * first[1:, 3:], axis=1) >= -1e-12)


def test_retime_only_resamples_the_existing_spatial_path():
  t = np.linspace(0.0, 1.0, 13)
  factual = np.column_stack((2.0 * t, np.zeros_like(t), np.zeros_like(t)))

  perturbed = perturb_path(
      factual, "retime", 0.2, np.random.default_rng(4)
  )

  assert np.all(np.diff(perturbed[:, 0]) >= 0.0)
  np.testing.assert_allclose(perturbed[:, 1:], 0.0)
  np.testing.assert_allclose(perturbed[[0, -1]], factual[[0, -1]])
  assert max_position_deviation(factual, perturbed) <= 0.2 + 1e-12


def test_perturb_path_obeys_bounds():
  factual = _curved_path()
  bounds = ((-0.25, -0.25, -0.25), (1.25, 0.5, 0.5))

  perturbed = perturb_path(
      factual,
      "break_contact",
      0.2,
      np.random.default_rng(3),
      bounds=bounds,
  )

  validate_path(perturbed, bounds=bounds)


def test_validate_path_rejects_bounds_and_expanded_static_aabbs():
  path = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

  with pytest.raises(ValueError, match="bounds"):
    validate_path(path, bounds=((0.1, -1, -1), (2, 1, 1)))
  with pytest.raises(ValueError, match="AABB"):
    validate_path(
        path,
        static_aabbs=(((0.6, 0.2, 0.2), (0.8, 0.4, 0.4)),),
        clearance=0.25,
    )


def test_validate_path_rejects_segment_crossing_static_aabb():
  path = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

  with pytest.raises(ValueError, match="AABB"):
    validate_path(
        path,
        static_aabbs=(((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),),
    )


def test_extreme_segment_aabb_test_avoids_false_positive():
  path = np.array([[0.0, 0.0, 0.0], [1e-320, 1.0, 0.0]])
  aabb = ((5e-321, -0.1, -1.0), (1e308, 0.1, 1.0))

  validate_path(path, static_aabbs=(aabb,))


def test_extreme_segment_aabb_test_avoids_false_negative():
  path = np.array([[-1e-200, 0.0, -1e100], [1.0, 0.0, 1e308]])
  aabb = ((-1e300, -1.0, -5e-324), (1e-320, 1.0, 1e-200))

  with pytest.raises(ValueError, match="AABB"):
    validate_path(path, static_aabbs=(aabb,))


def test_perturb_path_raises_when_no_collision_free_candidate_exists():
  factual = _curved_path(5)
  enclosing_box = (((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),)

  with pytest.raises(ValueError, match="max_attempts"):
    perturb_path(
        factual,
        "create_collision",
        0.2,
        np.random.default_rng(1),
        static_aabbs=enclosing_box,
        max_attempts=3,
    )


def test_zero_magnitude_returns_an_independent_copy():
  factual = _pose_path()

  perturbed = perturb_path(
      factual, "maintain_contact", 0.0, np.random.default_rng(2)
  )

  np.testing.assert_array_equal(perturbed, factual)
  assert perturbed is not factual


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recipe": "unknown", "magnitude": 0.1},
        {"recipe": "retime", "magnitude": -0.1},
        {"recipe": "retime", "magnitude": np.inf},
        {"recipe": "retime", "magnitude": 0.1, "clearance": -1},
        {"recipe": "retime", "magnitude": 0.1, "max_attempts": 0},
        {
            "recipe": "retime",
            "magnitude": 0.1,
            "bounds": ((1, 1, 1), (0, 2, 2)),
        },
        {
            "recipe": "retime",
            "magnitude": 0.1,
            "static_aabbs": (((1, 1, 1), (0, 2, 2)),),
        },
    ],
)
def test_perturb_path_rejects_invalid_arguments(kwargs):
  values = {"recipe": "retime", "magnitude": 0.1}
  values.update(kwargs)
  recipe = values.pop("recipe")
  magnitude = values.pop("magnitude")
  with pytest.raises((TypeError, ValueError)):
    perturb_path(
        _curved_path(), recipe, magnitude, np.random.default_rng(0), **values
    )


def test_validate_path_rejects_invalid_shape_finiteness_and_quaternion():
  with pytest.raises(ValueError):
    validate_path(np.zeros((1, 3)))
  with pytest.raises(ValueError):
    validate_path(np.zeros((3, 4)))
  with pytest.raises(ValueError):
    validate_path(np.array([[0, 0, 0], [np.nan, 0, 0]]))
  invalid_pose = _pose_path()
  invalid_pose[2, 3:] *= 2.0
  with pytest.raises(ValueError, match="unit"):
    validate_path(invalid_pose)


def test_max_position_deviation_uses_only_xyz_and_validates_shape():
  factual = _pose_path()
  perturbed = factual.copy()
  perturbed[3, 1] += 0.25
  perturbed[:, 3:] *= -1.0

  assert max_position_deviation(factual, perturbed) == pytest.approx(0.25)
  with pytest.raises(ValueError):
    max_position_deviation(factual, factual[:-1])


def test_max_position_deviation_is_stable_for_huge_finite_coordinates():
  factual = np.zeros((2, 3))
  perturbed = factual.copy()
  perturbed[1] = (1e308, 1e308, 0.0)

  deviation = max_position_deviation(factual, perturbed)

  assert math.isfinite(deviation)
  assert deviation == pytest.approx(math.hypot(1e308, 1e308))


def test_spatial_perturbation_normalizes_huge_finite_rng_direction():
  class HugeDirectionRng:
    def normal(self, size):
      assert size == 3
      return np.array([1e308, 1e308, 0.0])

    def uniform(self, *unused_args):
      return 1.0

  factual = np.zeros((3, 3))

  perturbed = perturb_path(
      factual, "maintain_contact", 0.1, HugeDirectionRng()
  )

  assert np.isfinite(perturbed).all()
  assert max_position_deviation(factual, perturbed) == pytest.approx(0.1)


def test_spatial_candidate_overflow_resamples_without_runtime_warning():
  class PositiveXRng:
    def normal(self, size):
      assert size == 3
      return np.array([1.0, 0.0, 0.0])

    def uniform(self, *unused_args):
      return 1.0

  factual = np.array([[0.0, 0.0, 0.0], [1e308, 0.0, 0.0], [0.0, 0.0, 0.0]])

  with pytest.raises(ValueError, match="max_attempts"):
    perturb_path(
        factual,
        "remove_collision",
        1e308,
        PositiveXRng(),
        max_attempts=2,
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda value: build_path(value, 3),
        lambda value: validate_path(value),
        lambda value: perturb_path(
            value, "retime", 0.1, np.random.default_rng(0)
        ),
        lambda value: max_position_deviation(value, value),
        lambda value: validate_path(np.zeros((2, 3)), bounds=value),
        lambda value: validate_path(
            np.zeros((2, 3)), static_aabbs=(value,)
        ),
    ],
)
def test_public_trajectory_apis_reject_complex_arrays_without_warning(call):
  complex_value = np.array(
      [[0.0 + 1.0j, 0.0, 0.0], [1.0 + 2.0j, 1.0, 1.0]]
  )

  with warnings.catch_warnings():
    warnings.simplefilter("error")
    with pytest.raises(ValueError, match="complex"):
      call(complex_value)


def test_trajectory_numeric_overflow_is_reported_as_value_error():
  with pytest.raises(ValueError, match="numeric"):
    build_path([[10**10000, 0, 0], [0, 0, 0]], 2)


def test_recipe_profiles_are_documented_as_heuristic_candidates():
  assert trajectory_module.INTERVENTION_RECIPES is schema_module.INTERVENTION_RECIPES
  descriptions = trajectory_module.RECIPE_PROFILE_SEMANTICS
  assert frozenset(descriptions) == schema_module.INTERVENTION_RECIPES
  assert all("heuristic" in description.lower() for description in descriptions.values())
  assert "physics" in perturb_path.__doc__.lower()
