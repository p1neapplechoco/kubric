import math

import numpy as np
import pytest

from interventions import appearance, materials


def test_family_priors_match_the_designed_tables():
  assert set(materials.FAMILY_PRIORS) == set(appearance.MATERIAL_FAMILIES)
  metal = materials.FAMILY_PRIORS["metal"]
  assert metal.density == (55.0, 100.0)
  assert metal.friction == (0.15, 0.45)
  assert metal.restitution == (0.10, 0.35)
  assert metal.metallic == (0.85, 1.00)
  assert metal.roughness == (0.12, 0.45)
  assert metal.ior == (1.45, 2.50)
  assert metal.transmission == (0.0, 0.0)
  glass = materials.FAMILY_PRIORS["glass"]
  assert glass.transmission == (0.85, 1.00)
  assert glass.roughness == (0.02, 0.18)
  stone = materials.FAMILY_PRIORS["stone"]
  assert stone.density == (45.0, 85.0)
  assert stone.restitution == (0.02, 0.20)


@pytest.mark.parametrize(
    ("shape", "half_extents", "expected"),
    (
        ("cube", (0.5, 0.5, 0.5), 1.0),
        ("sphere", (0.5, 0.5, 0.5), 4.0 / 3.0 * math.pi * 0.125),
        ("cylinder", (0.5, 0.5, 1.0), math.pi * 0.25 * 2.0),
        ("capsule", (0.5, 0.5, 1.0),
         math.pi * 0.25 * 2.0 + 4.0 / 3.0 * math.pi * 0.125),
    ),
)
def test_proxy_volume_matches_closed_form(shape, half_extents, expected):
  assert materials.proxy_volume(shape, half_extents) == pytest.approx(expected)


def test_proxy_volume_rejects_an_unknown_shape():
  with pytest.raises(ValueError):
    materials.proxy_volume("torus", (0.5, 0.5, 0.5))


@pytest.mark.parametrize("family", sorted(appearance.MATERIAL_FAMILIES))
def test_sample_material_stays_inside_family_priors(family):
  rng = np.random.default_rng(11)
  texture = appearance.TextureSpec(
      kind="solid", seed=1, colors=((0.5, 0.5, 0.5, 1.0),), scale=1.0)
  spec = materials.sample_material(rng, family, (0.5, 0.5, 0.5, 1.0), texture)
  priors = materials.FAMILY_PRIORS[family]
  assert spec.family == family
  assert priors.metallic[0] <= spec.metallic <= priors.metallic[1]
  assert priors.roughness[0] <= spec.roughness <= priors.roughness[1]
  assert priors.ior[0] <= spec.ior <= priors.ior[1]
  assert priors.transmission[0] <= spec.transmission <= priors.transmission[1]


@pytest.mark.parametrize("family", sorted(appearance.MATERIAL_FAMILIES))
def test_sample_material_consumes_the_same_number_of_draws_per_family(family):
  texture = appearance.TextureSpec(
      kind="solid", seed=1, colors=((0.5, 0.5, 0.5, 1.0),), scale=1.0)
  rng = np.random.default_rng(7)
  materials.sample_material(rng, family, (0.5, 0.5, 0.5, 1.0), texture)
  after_sample = rng.uniform(0.0, 1.0)

  reference = np.random.default_rng(7)
  reference.uniform(0.0, 1.0, size=materials.MATERIAL_DRAW_COUNT)
  assert after_sample == reference.uniform(0.0, 1.0)


def test_sample_material_returns_the_bound_exactly_for_a_degenerate_range():
  rng = np.random.default_rng(2)
  texture = appearance.TextureSpec(
      kind="solid", seed=1, colors=((0.5, 0.5, 0.5, 1.0),), scale=1.0)
  spec = materials.sample_material(rng, "rubber", (0.2, 0.2, 0.2, 1.0), texture)

  assert spec.metallic == 0.0
  assert spec.transmission == 0.0


def test_coupled_physics_clamps_mass_and_records_unclamped_value():
  rng = np.random.default_rng(3)
  result = materials.coupled_physics(rng, "metal", proxy_volume=1.0)
  assert materials.MASS_BOUNDS[0] <= result["mass"] <= materials.MASS_BOUNDS[1]
  assert result["mass"] == materials.MASS_BOUNDS[1]
  assert result["unclamped_mass"] > materials.MASS_BOUNDS[1]
  assert result["unclamped_mass"] == pytest.approx(
      result["effective_density"] * 1.0)
  priors = materials.FAMILY_PRIORS["metal"]
  assert priors.friction[0] <= result["friction"] <= priors.friction[1]
  assert priors.restitution[0] <= result["restitution"] <= priors.restitution[1]


def test_coupled_physics_clamps_a_tiny_proxy_up_to_the_lower_bound():
  result = materials.coupled_physics(np.random.default_rng(3), "wood", 1e-6)

  assert result["mass"] == materials.MASS_BOUNDS[0]
  assert result["unclamped_mass"] < materials.MASS_BOUNDS[0]


def test_coupled_physics_is_deterministic_for_a_given_seed():
  first = materials.coupled_physics(np.random.default_rng(5), "wood", 0.01)
  second = materials.coupled_physics(np.random.default_rng(5), "wood", 0.01)
  assert first == second


def test_is_held_out_matches_on_all_declared_keys_only():
  holdouts = ({"material_family": "glass", "texture_kind": "checker"},)
  assert materials.is_held_out(
      {"material_family": "glass", "texture_kind": "checker", "shape": "cube"},
      holdouts)
  assert not materials.is_held_out(
      {"material_family": "glass", "texture_kind": "noise", "shape": "cube"},
      holdouts)


def test_is_held_out_is_false_when_no_holdouts_are_declared():
  assert not materials.is_held_out({"material_family": "glass"}, ())


def test_family_declares_permitted_texture_kinds():
  for family, priors in materials.FAMILY_PRIORS.items():
    assert priors.texture_kinds
    assert set(priors.texture_kinds) <= appearance.TEXTURE_KINDS
  assert "wood" in materials.FAMILY_PRIORS["wood"].texture_kinds
  assert "checker" not in materials.FAMILY_PRIORS["glass"].texture_kinds


def test_family_priors_reject_an_inverted_range():
  with pytest.raises(ValueError):
    materials.FamilyPriors(
        family="metal",
        density=(100.0, 55.0),
        friction=(0.15, 0.45),
        restitution=(0.10, 0.35),
        metallic=(0.85, 1.00),
        roughness=(0.12, 0.45),
        ior=(1.45, 2.50),
        transmission=(0.0, 0.0),
        specular=(0.4, 0.6),
        emission_strength=(0.0, 0.0),
        texture_kinds=("solid",),
    )


def test_family_priors_and_coupling_modes_are_immutable_constants():
  assert materials.COUPLING_MODES == frozenset(
      ("coupled", "independent", "held_out"))
  assert materials.MASS_BOUNDS == (0.25, 4.0)
  with pytest.raises(TypeError):
    materials.FAMILY_PRIORS["metal"] = None
