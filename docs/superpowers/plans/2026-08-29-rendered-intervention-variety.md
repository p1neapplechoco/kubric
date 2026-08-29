# Rendered Intervention Variety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the physics-only intervention dataset pipeline into a deterministic, resumable, end-to-end pipeline that samples immutable per-instance appearance (geometry, materials, textures, camera, lights, background), simulates cube/sphere/cylinder/capsule proxies, and renders factual and counterfactual branches with Kubric's full annotation layer set.

**Architecture:** Four bounded layers, added in order. (1) `interventions/appearance.py` holds renderer-independent frozen visual schemas. (2) `interventions/materials.py` + `interventions/appearance_sampling.py` turn validated YAML ranges into one realized `VisualSceneSpec` using domain-separated seeds. (3) `kubric/core/objects.py`, `kubric/simulator/pybullet.py`, and `kubric/renderer/blender.py` gain `Cylinder` and `Capsule` primitives. (4) `interventions/rendering.py` + `scripts/render_dataset.py` replay logged physics onto a render-only scene and publish validated, atomically renamed render artifacts. Physics artifacts and their canonical bytes never change when appearance is absent.

**Tech Stack:** Python 3.11 in the dedicated `thesis` Conda env (no Docker), `bpy==4.2.0` (pip-installed Blender module), PyBullet, NumPy, PyYAML, Pillow/imageio/pypng/OpenEXR, pytest.

**Design source of truth:** `docs/superpowers/specs/2026-08-29-rendered-intervention-variety-design.md`. Read it before Task 1 and re-read the relevant section at the start of each task.

---

## Global Constraints

- **Environment:** every Python and pytest command uses the `thesis` Conda env interpreter
  `C:\Users\uya7hc\.conda\envs\thesis\python.exe`. Never use Docker, `base`, or `kubric-demo`.
- **No Docker anywhere.** Rendering uses the pip-installed `bpy` module in-process. Do not add
  `docker run` to any new script, test, or Makefile target.
- Clear `PYTHONPATH` before running commands in the persistent PowerShell session:
  `Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue`.
- Set `$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"` before pytest.
- **Indentation is 2 spaces** in `interventions/`, `scripts/`, and `tests/`; `kubric/` core files
  keep their existing local style (2 spaces in `kubric/core` and `kubric/simulator`, 4 spaces in
  `kubric/renderer/blender.py`). Match the file you are editing.
- Every new module gets a docstring with these four headings, in this order, each with non-empty
  content: `Purpose:`, `Public API:`, `Dependencies:`, `Trust boundary:`. `tests/test_module_documentation.py`
  enforces this and must be extended for each new module.
- Heavy backends (`kubric`, `bpy`, `pybullet`) must be imported lazily inside functions, never at
  module import time, for anything reachable from `import interventions`.
  `tests/test_offline_imports.py` enforces this and must be extended.
- Canonical JSON everywhere: `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`.
- All new schema containers are `@dataclass(frozen=True)`, validate in `__post_init__` via
  `object.__setattr__`, reuse `interventions.schema` helpers (`_real`, `_integer`, `_vector`,
  `_nonempty_string`, `_metadata`, `_freeze`, `to_jsonable`, `_SchemaMixin`), and raise `TypeError`
  for wrong types and `ValueError` for out-of-range values.
- `SCHEMA_VERSION` in `interventions/schema.py` stays `"1.0"`. Appearance carries its own
  `APPEARANCE_SCHEMA_VERSION = "1.0"` in `interventions/appearance.py`.
- Backward compatibility is a hard requirement: when `InstanceSpec.visual_scene is None`,
  `InstanceSpec.to_dict()` must omit the key entirely, and `instance_id` bytes must be
  byte-identical to today's.
- Intervention targets stay `cube` or `sphere`. `cylinder` and `capsule` are legal for dynamic and
  non-target static objects only.
- Never commit generated data: no `output/`, `*.blend`, `*.mp4`, `*.npy`, `*.png`, `*.tiff`,
  dataset roots, or asset caches.
- Do not weaken or delete an existing test to make new code pass.
- Red-green-refactor for every task. Commit at the end of each task.

## Environment reference

```powershell
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
& $py -m pytest tests/ -q
```

## File map

**Create**
- `interventions/appearance.py` — frozen visual schemas + `visual_scene_hash()`.
- `interventions/materials.py` — material family tables, coupling modes, mass coupling.
- `interventions/appearance_sampling.py` — YAML ranges → one realized `VisualSceneSpec`.
- `interventions/asset_catalog.py` — digest-pinned manifests, content-addressed cache, safe extraction.
- `interventions/rendering.py` — render-only scene build, state replay, layer output, validation.
- `scripts/render_dataset.py` — CLI that renders an existing dataset selection.
- `tests/test_appearance.py`, `tests/test_materials.py`, `tests/test_appearance_sampling.py`,
  `tests/test_asset_catalog.py`, `tests/test_rendering.py`, `tests/test_render_dataset.py`.
- `configs/scene_ranges_visual.yaml` — shipped procedural appearance config.

**Modify**
- `kubric/core/objects.py` — add `Cylinder`, `Capsule`.
- `kubric/__init__.py`, `kubric/core/__init__.py` — export them.
- `kubric/simulator/pybullet.py` — `GEOM_CYLINDER` / `GEOM_CAPSULE` registrations.
- `kubric/renderer/blender.py` — cylinder/capsule mesh registrations.
- `interventions/schema.py` — `SUPPORTED_SHAPES`, size semantics, oriented-AABB helper.
- `interventions/dataset.py` — visual scene on `InstanceSpec`, sampling hook, `CandidateSummary`
  variation fields, balancing axes, component-based splits, render journal, `run_batch` render stage.
- `scripts/generate_dataset.py` — `--render-profile`, exit code 3.
- `tests/test_schema.py`, `tests/test_dataset.py` — extend, never weaken.
- `tests/test_module_documentation.py`, `tests/test_offline_imports.py` — cover new modules.
- `test/test_core.py`, `test/test_pybullet.py`, `test/test_blender.py` — cover new primitives.
- `docs/trajectory_interventions.md` — new sections.
- `README.md` — pointer to the docker-free environment doc.
- `.gitignore` — ignore the asset cache.

Also created as documentation output: `docs/environment_thesis.md` (Task 0) and
`notes/session-logs/2026-08-29-rendered-intervention-variety.md` (Task 15).

**Task order:** 0 → 15, sequentially. Tasks 3-5 (kubric primitives) are independent of Tasks 1-2
(appearance schemas) and may be swapped, but Task 6 depends on Tasks 3-5 and Task 9 depends on
Tasks 6-8.

---

## Task 0: Provision and pin the docker-free `thesis` environment

**Files:**
- Create: `docs/environment_thesis.md`
- Test: manual verification commands below

**Interfaces:**
- Produces: a working `C:\Users\uya7hc\.conda\envs\thesis\python.exe` with `bpy==4.2.0`,
  `pybullet`, and every kubric runtime dependency. All later tasks consume it.

- [ ] **Step 1: Verify the interpreter and Blender module**

```powershell
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $py -c "import sys, bpy; print(sys.version); print(bpy.app.version_string)"
```

Expected: Python `3.11.x` and Blender `4.2.0`.

- [ ] **Step 2: Install the remaining runtime dependencies**

```powershell
& $py -m pip install pybullet pyquaternion traitlets munch "etils[epath]" imageio imageio-ffmpeg pypng trimesh absl-py PyYAML scipy OpenEXR Pillow bidict pytest
```

If the `pybullet` wheel is unavailable for cp311 and a source build fails, fall back to a cp310
environment: `conda create -y -n thesis -c conda-forge python=3.10` and `bpy==4.0.0`. Everything
else in this plan is unchanged; only the pinned versions in `docs/environment_thesis.md` change.

- [ ] **Step 3: Verify kubric imports and a real one-frame render**

```powershell
& $py -c "import kubric as kb; import kubric.renderer.blender as kbb; import kubric.simulator.pybullet as kbp; print('kubric ok')"
& $py -c "
import kubric as kb
from kubric.renderer.blender import Blender
scene = kb.Scene(resolution=(64, 64))
scene += kb.PerspectiveCamera(position=(3, 3, 3), look_at=(0, 0, 0))
scene += kb.DirectionalLight(position=(2, 2, 4), look_at=(0, 0, 0), intensity=2.0)
scene += kb.Cube(scale=(0.5, 0.5, 0.5), position=(0, 0, 0))
renderer = Blender(scene, samples_per_pixel=1, adaptive_sampling=False, use_denoising=False)
out = renderer.render_still(return_layers=('rgba', 'segmentation', 'depth'))
print({k: (v.shape, str(v.dtype)) for k, v in out.items()})
"
```

Expected: a dict with `rgba (64, 64, 4)`, `segmentation (64, 64, 1)`, `depth (64, 64, 1)`.
This is the gate for the whole rendering half of the plan. If it fails, stop and report the exact
Blender error rather than reintroducing Docker.

- [ ] **Step 4: Verify the existing suite is green in this env**

```powershell
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
& $py -m pytest tests/ -q
```

Expected: the pre-existing suite passes. Record any pre-existing failure verbatim in
`docs/environment_thesis.md` as a known-baseline item; do not fix unrelated failures here.

- [ ] **Step 5: Write `docs/environment_thesis.md`**

Record: the env name and absolute interpreter path, `python --version`, `bpy.app.version_string`,
the exact `pip install` line used, the `pip freeze` output for the packages above, the render smoke
command from Step 3 with its observed output shapes, the pytest baseline, and an explicit statement
that Docker is not used and `KUBRIC_USE_GPU` stays unset (CPU Cycles).

- [ ] **Step 6: Commit**

```powershell
git add docs/environment_thesis.md
git commit -m "docs: pin docker-free thesis conda environment for rendering"
```

---

## Task 1: Frozen appearance value schemas (textures, materials, objects)

**Files:**
- Create: `interventions/appearance.py`
- Create: `tests/test_appearance.py`
- Modify: `tests/test_module_documentation.py`
- Modify: `tests/test_offline_imports.py`

**Interfaces:**
- Consumes: `interventions.schema` helpers `_real`, `_integer`, `_vector`, `_nonempty_string`,
  `_metadata`, `to_jsonable`, `_SchemaMixin`.
- Produces:
  - `APPEARANCE_SCHEMA_VERSION: str = "1.0"`
  - `TEXTURE_KINDS: frozenset` = `{"solid", "noise", "checker", "wood", "marble", "speckle", "image"}`
  - `MATERIAL_FAMILIES: frozenset` = `{"metal", "rubber", "plastic", "ceramic", "glass", "wood", "stone"}`
  - `SOURCE_KINDS: frozenset` = `{"procedural", "kubasic", "gso"}`
  - `MATERIAL_MODES: frozenset` = `{"native", "override"}`
  - `class ImageReference(_SchemaMixin)`: `role: str`, `uri: str`, `sha256: str`, `color_space: str`
  - `class TextureSpec(_SchemaMixin)`: `kind`, `seed`, `colors`, `scale`, `detail`, `roughness`,
    `distortion`, `rotation`, `images`
  - `class MaterialSpec(_SchemaMixin)`: `family`, `base_color`, `metallic`, `roughness`, `specular`,
    `ior`, `transmission`, `emission`, `texture`
  - `class AssetReference(_SchemaMixin)`: `source_kind`, `manifest_uri`, `manifest_sha256`,
    `asset_id`, `archive_sha256`, `material_mode`
  - `class VisualObjectSpec(_SchemaMixin)`: `object_id`, `source_kind`, `asset`, `collision_proxy_id`,
    `scale`, `origin_offset`, `alignment_quaternion`, `material`

- [ ] **Step 1: Write the failing schema tests**

Create `tests/test_appearance.py`:

```python
import dataclasses
import json

import pytest

from interventions import appearance


def _texture(**overrides):
  values = {
      "kind": "checker",
      "seed": 7,
      "colors": ((0.1, 0.2, 0.3, 1.0), (0.9, 0.8, 0.7, 1.0)),
      "scale": 4.0,
      "detail": 2.0,
      "roughness": 0.5,
      "distortion": 0.0,
      "rotation": 0.25,
  }
  values.update(overrides)
  return appearance.TextureSpec(**values)


def _material(**overrides):
  values = {
      "family": "plastic",
      "base_color": (0.4, 0.5, 0.6, 1.0),
      "metallic": 0.0,
      "roughness": 0.4,
      "specular": 0.5,
      "ior": 1.45,
      "transmission": 0.0,
      "emission": (0.0, 0.0, 0.0, 1.0),
      "texture": _texture(),
  }
  values.update(overrides)
  return appearance.MaterialSpec(**values)


def test_texture_spec_is_frozen_and_canonical():
  texture = _texture()
  assert dataclasses.is_dataclass(texture)
  with pytest.raises(dataclasses.FrozenInstanceError):
    texture.scale = 2.0
  assert isinstance(texture.colors, tuple)
  assert texture.to_dict()["kind"] == "checker"
  assert json.dumps(texture.to_dict(), sort_keys=True)


def test_texture_spec_rejects_unknown_kind():
  with pytest.raises(ValueError, match="kind"):
    _texture(kind="plaid")


def test_texture_spec_rejects_nonfinite_and_out_of_range_colors():
  with pytest.raises(ValueError):
    _texture(colors=((0.0, 0.0, 0.0, float("nan")),))
  with pytest.raises(ValueError, match="colors"):
    _texture(colors=((0.0, 0.0, 0.0, 1.5),))
  with pytest.raises(ValueError, match="colors"):
    _texture(colors=())


def test_image_texture_requires_pinned_images():
  with pytest.raises(ValueError, match="images"):
    _texture(kind="image", images=())


def test_image_reference_requires_hex_digest_and_known_color_space():
  with pytest.raises(ValueError, match="sha256"):
    appearance.ImageReference(
        role="base_color", uri="file:///t.png", sha256="zz", color_space="sRGB")
  with pytest.raises(ValueError, match="color_space"):
    appearance.ImageReference(
        role="roughness", uri="file:///t.png", sha256="a" * 64, color_space="sRGB")
  reference = appearance.ImageReference(
      role="roughness", uri="file:///t.png", sha256="a" * 64, color_space="Non-Color")
  assert reference.color_space == "Non-Color"


def test_material_spec_rejects_unknown_family_and_bad_ranges():
  with pytest.raises(ValueError, match="family"):
    _material(family="unobtanium")
  with pytest.raises(ValueError, match="metallic"):
    _material(metallic=1.5)
  with pytest.raises(ValueError, match="ior"):
    _material(ior=0.5)


def test_material_spec_rejects_alpha_below_one_without_transmission():
  with pytest.raises(ValueError, match="transmission"):
    _material(base_color=(0.4, 0.5, 0.6, 0.5))
  translucent = _material(
      family="glass", base_color=(0.4, 0.5, 0.6, 0.5), transmission=0.9, roughness=0.05)
  assert translucent.base_color[3] == 0.5


def test_asset_reference_pins_manifest_and_archive_digests():
  reference = appearance.AssetReference(
      source_kind="gso",
      manifest_uri="file:///manifest.json",
      manifest_sha256="b" * 64,
      asset_id="Mug_001",
      archive_sha256="c" * 64,
      material_mode="native")
  assert reference.to_dict()["asset_id"] == "Mug_001"
  with pytest.raises(ValueError, match="material_mode"):
    dataclasses.replace(reference, material_mode="inherit")
  with pytest.raises(ValueError, match="source_kind"):
    dataclasses.replace(reference, source_kind="procedural")


def test_visual_object_spec_requires_asset_for_external_sources():
  with pytest.raises(ValueError, match="asset"):
    appearance.VisualObjectSpec(
        object_id="obj_0",
        source_kind="gso",
        asset=None,
        collision_proxy_id="obj_0",
        scale=(1.0, 1.0, 1.0),
        origin_offset=(0.0, 0.0, 0.0),
        alignment_quaternion=(1.0, 0.0, 0.0, 0.0),
        material=_material())


def test_visual_object_spec_normalizes_alignment_quaternion():
  spec = appearance.VisualObjectSpec(
      object_id="obj_0",
      source_kind="procedural",
      asset=None,
      collision_proxy_id="obj_0",
      scale=(1.0, 2.0, 3.0),
      origin_offset=(0.0, 0.0, 0.5),
      alignment_quaternion=(2.0, 0.0, 0.0, 0.0),
      material=_material())
  assert spec.alignment_quaternion == (1.0, 0.0, 0.0, 0.0)
  assert spec.to_dict()["scale"] == [1.0, 2.0, 3.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_appearance.py -q
```

Expected: collection error `ModuleNotFoundError: No module named 'interventions.appearance'`.

- [ ] **Step 3: Implement `interventions/appearance.py`**

Module docstring first:

```python
"""Immutable renderer-independent appearance schemas for intervention instances.

Purpose: define frozen, validated visual values and their canonical JSON form.
Public API: APPEARANCE_SCHEMA_VERSION, TEXTURE_KINDS, MATERIAL_FAMILIES, SOURCE_KINDS,
MATERIAL_MODES, ImageReference, TextureSpec, MaterialSpec, AssetReference, and
VisualObjectSpec.
Dependencies: Python's standard library and interventions.schema helpers only, so
appearance never imports Kubric, Blender, or a simulator backend.
Trust boundary: validation enforces value ranges, enum membership, and JSON safety;
it does not verify that a referenced asset or image exists or is authentic.
"""
```

Implementation rules:

- Reuse `from interventions.schema import _real, _integer, _vector, _nonempty_string, _metadata, to_jsonable, _SchemaMixin`.
- Add module-local helpers `_unit(value, name)` (finite, `0.0 <= v <= 1.0`), `_positive(value, name)`,
  `_rgba(value, name)` (`_vector(value, 4, name)` with each component in `[0, 1]`),
  `_hex_digest(value, name)` (lowercase 64-char hex), and `_enum(value, allowed, name)`.
- `COLOR_SPACES = frozenset(("sRGB", "Non-Color"))`; `_COLOR_ROLES = frozenset(("base_color", "emission"))`;
  `_NON_COLOR_ROLES = frozenset(("roughness", "metallic", "normal", "height"))`.
  `ImageReference.__post_init__` requires `color_space == "sRGB"` for color roles and
  `"Non-Color"` for non-color roles, and rejects unknown roles.
- `TextureSpec` fields and defaults:
  `kind: str`, `seed: int = 0`, `colors: Tuple[Tuple[float, ...], ...] = ()`, `scale: float = 1.0`,
  `detail: float = 2.0`, `roughness: float = 0.5`, `distortion: float = 0.0`, `rotation: float = 0.0`,
  `images: Tuple[ImageReference, ...] = ()`, `schema_version: str = APPEARANCE_SCHEMA_VERSION`.
  Validation: `kind in TEXTURE_KINDS`; `1 <= len(colors) <= 4` for every kind except `image`;
  each color is `_rgba`; `scale > 0`; `detail >= 0`; `roughness`, `distortion` in `[0, 1]`;
  `rotation` in `[0, 1]`; `images` non-empty **iff** `kind == "image"`; image roles unique.
- `MaterialSpec` fields: `family`, `base_color`, `metallic`, `roughness`, `specular`, `ior`,
  `transmission`, `emission`, `texture: TextureSpec`, `schema_version`.
  Validation: `family in MATERIAL_FAMILIES`; `metallic`, `roughness`, `specular`, `transmission`
  are `_unit`; `1.0 <= ior <= 3.0`; `base_color`/`emission` are `_rgba`;
  `base_color[3] < 1.0` requires `transmission > 0.0` and raises `ValueError("... transmission ...")`.
- `AssetReference` fields: `source_kind`, `manifest_uri`, `manifest_sha256`, `asset_id`,
  `archive_sha256`, `material_mode`, `schema_version`. Validation: `source_kind in {"kubasic", "gso"}`
  (raise `ValueError("source_kind ...")` for `"procedural"`), digests are `_hex_digest`,
  `material_mode in MATERIAL_MODES`.
- `VisualObjectSpec` fields: `object_id`, `source_kind`, `asset: Optional[AssetReference]`,
  `collision_proxy_id`, `scale`, `origin_offset`, `alignment_quaternion`, `material: MaterialSpec`,
  `schema_version`. Validation: `source_kind in SOURCE_KINDS`; `asset is None` iff
  `source_kind == "procedural"` (otherwise `ValueError("asset ...")`) and, when present,
  `asset.source_kind == source_kind`; `scale` is a 3-vector of positives; `origin_offset` is a
  3-vector; `alignment_quaternion` is a 4-vector normalized to unit norm (reject zero norm).

Follow the existing `interventions/schema.py` `__post_init__` + `object.__setattr__` pattern exactly.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_appearance.py -q
```

Expected: all pass.

- [ ] **Step 5: Extend the documentation and offline-import contracts**

In `tests/test_module_documentation.py`, add `"interventions/appearance.py"` to `_MODULE_PATHS`.
In `tests/test_offline_imports.py`, add `appearance` to the set of submodules that must import while
`kubric` and `pybullet` are blocked.

```powershell
& $py -m pytest tests/test_module_documentation.py tests/test_offline_imports.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add interventions/appearance.py tests/test_appearance.py tests/test_module_documentation.py tests/test_offline_imports.py
git commit -m "feat(appearance): add frozen texture, material, and visual object schemas"
```

---

## Task 2: Scene-level appearance schemas and visual scene hash

**Files:**
- Modify: `interventions/appearance.py`
- Modify: `tests/test_appearance.py`

**Interfaces:**
- Consumes: Task 1 types; `interventions.schema.SceneConfig`.
- Produces:
  - `class CameraRenderSpec(_SchemaMixin)`: `positions`, `look_ats`, `focal_length`,
    `sensor_width`, `clipping_range`
  - `class LightSpec(_SchemaMixin)`: `light_id`, `kind`, `position`, `look_at`, `color`,
    `intensity`, `width`, `height`, `spot_size`, `spot_blend`
  - `class BackgroundSpec(_SchemaMixin)`: `kind`, `color`, `hdri`, `rotation`, `strength`, `exposure`
  - `class RenderProfile(_SchemaMixin)`: `name`, `resolution`, `samples_per_pixel`,
    `adaptive_sampling`, `use_denoising`, `background_transparency`, `layers`, `device`
  - `class VisualSceneSpec(_SchemaMixin)`: `objects`, `camera`, `lights`, `background`,
    `render_seed`, `frame_steps`
  - `def visual_scene_hash(spec: VisualSceneSpec) -> str` — 64-char SHA-256 of canonical bytes
  - `def render_profile_hash(profile: RenderProfile) -> str`
  - `def validate_scene_correspondence(visual: VisualSceneSpec, scene: SceneConfig) -> None`
  - `def frame_steps_for(scene: SceneConfig) -> Tuple[int, ...]`
  - `SMOKE_PROFILE: RenderProfile`, `PRODUCTION_PROFILE: RenderProfile`, `RENDER_LAYERS: Tuple[str, ...]`
  - `PROFILES_BY_NAME: Mapping[str, RenderProfile]` — an immutable `MappingProxyType` mapping
    `"smoke"` and `"production"` to the two profiles. Every CLI and `run_batch` resolves
    profile *names* to `RenderProfile` objects through this mapping; only `RenderProfile`
    objects cross into `interventions.rendering`.

- [ ] **Step 1: Write the failing scene-level tests**

Append to `tests/test_appearance.py`:

```python
from interventions import schema


def _visual_object(object_id="obj_0"):
  return appearance.VisualObjectSpec(
      object_id=object_id,
      source_kind="procedural",
      asset=None,
      collision_proxy_id=object_id,
      scale=(1.0, 1.0, 1.0),
      origin_offset=(0.0, 0.0, 0.0),
      alignment_quaternion=(1.0, 0.0, 0.0, 0.0),
      material=_material())


def _camera(num_frames=2):
  return appearance.CameraRenderSpec(
      positions=tuple((3.0 + i, 3.0, 2.0) for i in range(num_frames)),
      look_ats=tuple((0.0, 0.0, 0.0) for _ in range(num_frames)),
      focal_length=35.0,
      sensor_width=36.0,
      clipping_range=(0.1, 100.0))


def _light(light_id="key"):
  return appearance.LightSpec(
      light_id=light_id,
      kind="rect_area",
      position=(2.0, 2.0, 4.0),
      look_at=(0.0, 0.0, 0.0),
      color=(1.0, 0.98, 0.95, 1.0),
      intensity=120.0,
      width=1.5,
      height=1.5)


def _background():
  return appearance.BackgroundSpec(
      kind="color",
      color=(0.2, 0.2, 0.22, 1.0),
      hdri=None,
      rotation=0.0,
      strength=1.0,
      exposure=0.0)


def _visual_scene(object_ids=("obj_0", "obj_1"), frame_steps=(0, 10)):
  return appearance.VisualSceneSpec(
      objects=tuple(_visual_object(name) for name in object_ids),
      camera=_camera(len(frame_steps)),
      lights=(_light("key"), _light("fill")),
      background=_background(),
      render_seed=99,
      frame_steps=frame_steps)


def _scene_config(object_ids=("obj_0", "obj_1")):
  return schema.SceneConfig(
      objects=tuple(
          schema.ObjectConfig(object_id=name, shape="cube", size=0.2)
          for name in object_ids),
      camera=schema.CameraConfig(
          position=(3.0, 3.0, 2.0), look_at=(0.0, 0.0, 0.0), focal_length=35.0),
      frame_range=(0, 2),
      frame_rate=24,
      step_rate=240)


def test_camera_render_spec_requires_matching_path_lengths():
  with pytest.raises(ValueError, match="look_ats"):
    appearance.CameraRenderSpec(
        positions=((1.0, 1.0, 1.0),),
        look_ats=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        focal_length=35.0,
        sensor_width=36.0,
        clipping_range=(0.1, 100.0))
  with pytest.raises(ValueError, match="clipping_range"):
    appearance.CameraRenderSpec(
        positions=((1.0, 1.0, 1.0),),
        look_ats=((0.0, 0.0, 0.0),),
        focal_length=35.0,
        sensor_width=36.0,
        clipping_range=(10.0, 1.0))


def test_visual_scene_spec_requires_unique_objects_and_frame_alignment():
  with pytest.raises(ValueError, match="object_id"):
    _visual_scene(object_ids=("obj_0", "obj_0"))
  with pytest.raises(ValueError, match="frame_steps"):
    appearance.VisualSceneSpec(
        objects=(_visual_object(),),
        camera=_camera(2),
        lights=(_light(),),
        background=_background(),
        render_seed=1,
        frame_steps=(10, 0))


def test_frame_steps_for_maps_output_frames_to_physics_steps():
  scene = _scene_config()
  assert appearance.frame_steps_for(scene) == (0, 10)


def test_validate_scene_correspondence_requires_exact_object_match():
  visual = _visual_scene()
  appearance.validate_scene_correspondence(visual, _scene_config())
  with pytest.raises(ValueError, match="object"):
    appearance.validate_scene_correspondence(
        visual, _scene_config(object_ids=("obj_0", "obj_2")))


def test_visual_scene_hash_is_stable_and_sensitive():
  first = appearance.visual_scene_hash(_visual_scene())
  assert len(first) == 64
  assert first == appearance.visual_scene_hash(_visual_scene())
  changed = dataclasses.replace(_visual_scene(), render_seed=100)
  assert appearance.visual_scene_hash(changed) != first


def test_render_profiles_are_distinct_and_cover_all_layers():
  assert appearance.SMOKE_PROFILE.resolution == (64, 64)
  assert appearance.SMOKE_PROFILE.samples_per_pixel == 1
  assert appearance.SMOKE_PROFILE.adaptive_sampling is False
  assert appearance.SMOKE_PROFILE.use_denoising is False
  assert appearance.PRODUCTION_PROFILE.resolution == (256, 256)
  assert appearance.PRODUCTION_PROFILE.samples_per_pixel == 64
  assert appearance.PRODUCTION_PROFILE.adaptive_sampling is True
  assert appearance.PRODUCTION_PROFILE.use_denoising is True
  assert set(appearance.SMOKE_PROFILE.layers) == set(appearance.RENDER_LAYERS)
  assert (appearance.render_profile_hash(appearance.SMOKE_PROFILE)
          != appearance.render_profile_hash(appearance.PRODUCTION_PROFILE))
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_appearance.py -q
```

Expected: `AttributeError: module 'interventions.appearance' has no attribute 'CameraRenderSpec'`.

- [ ] **Step 3: Implement the scene-level schemas**

- `RENDER_LAYERS = ("rgba", "segmentation", "depth", "normal", "forward_flow", "backward_flow", "object_coordinates")`.
- `LIGHT_KINDS = frozenset(("directional", "point", "spot", "rect_area"))`;
  `BACKGROUND_KINDS = frozenset(("color", "hdri"))`; `RENDER_DEVICES = frozenset(("CPU", "GPU"))`.
- `CameraRenderSpec`: `positions` and `look_ats` are non-empty tuples of 3-vectors of equal length;
  `focal_length > 0`; `sensor_width > 0`; `clipping_range` is a 2-vector with `0 < near < far`.
- `LightSpec`: `kind in LIGHT_KINDS`; `intensity > 0`; `color` is `_rgba`; `width`/`height` are
  positive and required for `rect_area`, otherwise must be `None`; `spot_size`/`spot_blend` are
  required for `spot` (`spot_size` in `(0, math.pi]`, `spot_blend` in `[0, 1]`), otherwise `None`.
- `BackgroundSpec`: `kind in BACKGROUND_KINDS`; exactly one of `color` (an `_rgba`) or
  `hdri` (an `ImageReference` with `role == "base_color"`) is set to match `kind`;
  `rotation` in `[0, 1]`; `strength > 0`; `-10.0 <= exposure <= 10.0`.
- `RenderProfile`: `name` non-empty; `resolution` a 2-tuple of positive ints;
  `samples_per_pixel >= 1`; three bools; `layers` a non-empty, deduplicated, sorted-stable tuple
  drawn from `RENDER_LAYERS`; `device in RENDER_DEVICES`.
- `VisualSceneSpec`: `objects` non-empty with unique `object_id`s (sorted by `object_id` in
  `__post_init__` for canonical order); `lights` non-empty with unique `light_id`s;
  `render_seed >= 0`; `frame_steps` a strictly increasing tuple of non-negative ints whose length
  equals `len(camera.positions)` (raise `ValueError("frame_steps ...")`);
  `schema_version = APPEARANCE_SCHEMA_VERSION`.
- `frame_steps_for(scene)`: assert `scene.step_rate % scene.frame_rate == 0`, then return
  `tuple((frame - scene.frame_range[0]) * scene.step_rate // scene.frame_rate
          for frame in range(scene.frame_range[0], scene.frame_range[1]))`.
- `validate_scene_correspondence(visual, scene)`: object id sets must be equal (raise
  `ValueError` naming the symmetric difference); every `collision_proxy_id` must name an object in
  `scene`; `visual.frame_steps` must equal `frame_steps_for(scene)`.
- `visual_scene_hash` / `render_profile_hash`: `hashlib.sha256(canonical_bytes(spec.to_dict())).hexdigest()`
  using a module-private `_canonical_bytes` identical to the one in `interventions/dataset.py`.
- `SMOKE_PROFILE = RenderProfile(name="smoke", resolution=(64, 64), samples_per_pixel=1, adaptive_sampling=False, use_denoising=False, background_transparency=False, layers=RENDER_LAYERS, device="CPU")`
  and `PRODUCTION_PROFILE = RenderProfile(name="production", resolution=(256, 256), samples_per_pixel=64, adaptive_sampling=True, use_denoising=True, background_transparency=False, layers=RENDER_LAYERS, device="CPU")`.

Update the module docstring `Public API:` line to list the new names.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_appearance.py tests/test_module_documentation.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add interventions/appearance.py tests/test_appearance.py
git commit -m "feat(appearance): add camera, light, background, profile, and scene schemas"
```

---

## Task 3: Cylinder and capsule primitives in kubric core

**Files:**
- Modify: `kubric/core/objects.py`
- Modify: `kubric/core/__init__.py`
- Modify: `kubric/__init__.py`
- Modify: `test/test_core.py`

**Interfaces:**
- Produces: `kubric.Cylinder` and `kubric.Capsule`, both `PhysicalObject` subclasses whose
  `bounds` default is `(-1, -1, -1), (1, 1, 1)` and whose local axis is +Z.
  `scale` semantics: cylinder `(radius, radius, half_height)`; capsule
  `(radius, radius, cylinder_half_height)` with total half-extent `cylinder_half_height + radius`.
  `Capsule.aabbox` accounts for the hemispherical caps.

- [ ] **Step 1: Write the failing core tests**

Append to `test/test_core.py`:

```python
def test_cylinder_defaults_to_unit_bounds():
  cylinder = kb.Cylinder(scale=(0.5, 0.5, 1.5))
  assert cylinder.bounds == ((-1, -1, -1), (1, 1, 1))
  aabbox = cylinder.aabbox
  assert aabbox[0] == pytest.approx([-0.5, -0.5, -1.5])
  assert aabbox[1] == pytest.approx([0.5, 0.5, 1.5])


def test_capsule_aabbox_includes_hemispherical_caps():
  capsule = kb.Capsule(scale=(0.5, 0.5, 1.0))
  aabbox = capsule.aabbox
  assert aabbox[0] == pytest.approx([-0.5, -0.5, -1.5])
  assert aabbox[1] == pytest.approx([0.5, 0.5, 1.5])


def test_capsule_rejects_non_uniform_radii():
  with pytest.raises(tl.TraitError):
    kb.Capsule(scale=(0.5, 0.7, 1.0))
```

If `test/test_core.py` does not already import `pytest` and `traitlets as tl`, add those imports.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest test/test_core.py -q -k "cylinder or capsule"
```

Expected: `AttributeError: module 'kubric' has no attribute 'Cylinder'`.

- [ ] **Step 3: Implement the primitives**

In `kubric/core/objects.py`, after `class Sphere`:

```python
class Cylinder(PhysicalObject):
  """A cylinder whose axis is the local Z axis.

  scale is (radius, radius, half_height).
  """

  @tl.default("bounds")
  def _get_bounds_default(self):
    return (-1, -1, -1), (1, 1, 1)

  @tl.validate("scale")
  def _valid_radial_scale(self, proposal):
    scale = proposal["value"]
    if not np.isclose(scale[0], scale[1]):
      raise tl.TraitError(
          f"cylinder requires equal X and Y radii ({scale})")
    return scale


class Capsule(PhysicalObject):
  """A capsule whose axis is the local Z axis.

  scale is (radius, radius, cylinder_half_height); the total local half-extent
  along Z is cylinder_half_height + radius.
  """

  @tl.default("bounds")
  def _get_bounds_default(self):
    return (-1, -1, -1), (1, 1, 1)

  @tl.validate("scale")
  def _valid_radial_scale(self, proposal):
    scale = proposal["value"]
    if not np.isclose(scale[0], scale[1]):
      raise tl.TraitError(
          f"capsule requires equal X and Y radii ({scale})")
    return scale

  @property
  def bbox_3d(self):
    radius = float(self.scale[0])
    extents = np.array(
        [radius, radius, float(self.scale[2]) + radius], dtype=np.float32)
    corners = itertools.product(*[(-e, e) for e in extents])
    obj_orientation = pyquat.Quaternion(*self.quaternion)
    rotated = [obj_orientation.rotate(np.array(x)) for x in corners]
    return np.array([self.position + x for x in rotated])
```

`Sphere` gets the same `_valid_radial_scale` validator only if `test/test_pybullet.py` already
relies on the existing assertion; do not change `Sphere` otherwise.

Export from `kubric/core/__init__.py` (it uses `from .objects import *`, so add `Cylinder` and
`Capsule` to `objects.__all__` if that module defines one; otherwise no change is needed) and add
to `kubric/__init__.py`:

```python
from kubric.core.objects import Cylinder
from kubric.core.objects import Capsule
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest test/test_core.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add kubric/core/objects.py kubric/__init__.py kubric/core/__init__.py test/test_core.py
git commit -m "feat(kubric): add local-Z cylinder and capsule physical objects"
```

---

## Task 4: PyBullet collision shapes for cylinder and capsule

**Files:**
- Modify: `kubric/simulator/pybullet.py`
- Modify: `test/test_pybullet.py`

**Interfaces:**
- Consumes: `kubric.Cylinder`, `kubric.Capsule` from Task 3.
- Produces: `PyBullet.add_asset` registrations returning a body id, using
  `pb.GEOM_CYLINDER(radius=scale[0], height=2 * scale[2])` and
  `pb.GEOM_CAPSULE(radius=scale[0], height=2 * scale[2])`.

- [ ] **Step 1: Write the failing simulator tests**

Append to `test/test_pybullet.py`, matching the file's existing fixture style:

```python
def test_pybullet_adds_cylinder_with_expected_extents():
  scene = kb.Scene(frame_start=0, frame_end=1)
  simulator = KubricSimulator(scene)
  cylinder = kb.Cylinder(scale=(0.25, 0.25, 0.75), position=(0, 0, 5), static=False)
  scene += cylinder
  body_id = cylinder.linked_objects[simulator]
  aabb_min, aabb_max = simulator._physics_client.getAABB(body_id)
  assert aabb_max[2] - aabb_min[2] == pytest.approx(1.5, abs=1e-3)
  assert aabb_max[0] - aabb_min[0] == pytest.approx(0.5, abs=1e-3)


def test_pybullet_adds_capsule_with_cap_extents():
  scene = kb.Scene(frame_start=0, frame_end=1)
  simulator = KubricSimulator(scene)
  capsule = kb.Capsule(scale=(0.25, 0.25, 0.5), position=(0, 0, 5), static=False)
  scene += capsule
  body_id = capsule.linked_objects[simulator]
  aabb_min, aabb_max = simulator._physics_client.getAABB(body_id)
  assert aabb_max[2] - aabb_min[2] == pytest.approx(1.5, abs=1e-3)


def test_pybullet_cylinder_falls_under_gravity():
  scene = kb.Scene(frame_start=0, frame_end=4, frame_rate=24, step_rate=240)
  simulator = KubricSimulator(scene)
  cylinder = kb.Cylinder(scale=(0.25, 0.25, 0.25), position=(0, 0, 2), mass=1.0)
  scene += cylinder
  simulator.run(frame_start=0, frame_end=4)
  assert cylinder.position[2] < 2.0
```

Use whatever import alias `test/test_pybullet.py` already uses for the simulator class.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest test/test_pybullet.py -q -k "cylinder or capsule"
```

Expected: failure because `add_asset` falls through to the base implementation.

- [ ] **Step 3: Implement the registrations**

In `kubric/simulator/pybullet.py`, after the `core.Sphere` registration, mirroring its structure
exactly (including `useMaximalCoordinates=True`, `contactProcessingThreshold=0`, and
`register_physical_object_setters`):

```python
  @add_asset.register(core.Cylinder)
  def _add_object(self, obj: core.Cylinder) -> Optional[int]:
    radius = obj.scale[0]
    assert radius == obj.scale[1], obj.scale  # only uniform radial scaling
    collision_idx = self._physics_client.createCollisionShape(
        pb.GEOM_CYLINDER, radius=radius, height=2 * obj.scale[2])
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    cylinder_idx = self._physics_client.createMultiBody(
        mass,
        collision_idx,
        visual_idx,
        obj.position,
        wxyz2xyzw(obj.quaternion),
        useMaximalCoordinates=True)
    self._physics_client.changeDynamics(
        cylinder_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, cylinder_idx, self._physics_client)

    return cylinder_idx

  @add_asset.register(core.Capsule)
  def _add_object(self, obj: core.Capsule) -> Optional[int]:
    radius = obj.scale[0]
    assert radius == obj.scale[1], obj.scale  # only uniform radial scaling
    collision_idx = self._physics_client.createCollisionShape(
        pb.GEOM_CAPSULE, radius=radius, height=2 * obj.scale[2])
    visual_idx = -1
    mass = 0 if obj.static else obj.mass
    capsule_idx = self._physics_client.createMultiBody(
        mass,
        collision_idx,
        visual_idx,
        obj.position,
        wxyz2xyzw(obj.quaternion),
        useMaximalCoordinates=True)
    self._physics_client.changeDynamics(
        capsule_idx, -1, contactProcessingThreshold=0)
    register_physical_object_setters(obj, capsule_idx, self._physics_client)

    return capsule_idx
```

Note for the implementer: PyBullet's `GEOM_CAPSULE` `height` is the *cylindrical* section length,
so `2 * scale[2]` is correct and the total extent is `2 * (scale[2] + radius)`, matching Task 3.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest test/test_pybullet.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add kubric/simulator/pybullet.py test/test_pybullet.py
git commit -m "feat(kubric): simulate cylinder and capsule collision proxies"
```

---

## Task 5: Blender meshes for cylinder and capsule

**Files:**
- Modify: `kubric/renderer/blender.py`
- Modify: `test/test_blender.py`

**Interfaces:**
- Consumes: `kubric.Cylinder`, `kubric.Capsule`.
- Produces: `Blender.add_asset` registrations creating deterministic local-Z meshes whose
  Blender `dimensions` equal `(2r, 2r, 2h)` for a cylinder and `(2r, 2r, 2(h + r))` for a capsule,
  with material, keyframe, and segmentation wiring identical to `Cube`/`Sphere`.

- [ ] **Step 1: Write the failing renderer tests**

Append to `test/test_blender.py` (this file already imports a real `bpy`; the `thesis` env provides it):

```python
def test_blender_cylinder_has_expected_dimensions():
  scene = kb.Scene(resolution=(16, 16))
  renderer = KubricBlender(scene)
  cylinder = kb.Cylinder(scale=(0.25, 0.25, 0.75), position=(0, 0, 0))
  scene += cylinder
  blender_obj = cylinder.linked_objects[renderer]
  assert tuple(round(v, 5) for v in blender_obj.dimensions) == (0.5, 0.5, 1.5)


def test_blender_capsule_dimensions_include_caps():
  scene = kb.Scene(resolution=(16, 16))
  renderer = KubricBlender(scene)
  capsule = kb.Capsule(scale=(0.25, 0.25, 0.5), position=(0, 0, 0))
  scene += capsule
  blender_obj = capsule.linked_objects[renderer]
  assert tuple(round(v, 5) for v in blender_obj.dimensions) == (0.5, 0.5, 1.5)


def test_blender_cylinder_tracks_material_and_segmentation():
  scene = kb.Scene(resolution=(16, 16))
  renderer = KubricBlender(scene)
  material = kb.PrincipledBSDFMaterial(color=kb.Color(1.0, 0.0, 0.0, 1.0))
  cylinder = kb.Cylinder(scale=(0.2, 0.2, 0.2), material=material, segmentation_id=7)
  scene += cylinder
  blender_obj = cylinder.linked_objects[renderer]
  assert blender_obj.active_material is not None
  assert blender_obj.pass_index == 7
```

Use the alias `test/test_blender.py` already uses for the renderer class.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest test/test_blender.py -q -k "cylinder or capsule"
```

Expected: failure — no registration for the new types.

- [ ] **Step 3: Implement the registrations**

In `kubric/renderer/blender.py` (4-space indentation in this file), after the `core.Sphere`
registration:

```python
    @add_asset.register(core.Cylinder)
    @blender_utils.prepare_blender_object
    def _add_asset(self, obj: core.Cylinder):
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1.0, depth=2.0)
        bpy.ops.object.shade_smooth()
        cylinder = bpy.context.active_object

        register_object3d_setters(obj, cylinder)
        obj.observe(
            AttributeSetter(
                cylinder, "active_material", converter=self._convert_to_blender_object
            ),
            "material",
        )
        obj.observe(AttributeSetter(cylinder, "scale"), "scale")
        obj.observe(KeyframeSetter(cylinder, "scale"), "scale", type="keyframe")
        return cylinder

    @add_asset.register(core.Capsule)
    @blender_utils.prepare_blender_object
    def _add_asset(self, obj: core.Capsule):
        capsule = blender_utils.build_capsule_mesh(bpy, obj.scale)
        register_object3d_setters(obj, capsule)
        obj.observe(
            AttributeSetter(
                capsule, "active_material", converter=self._convert_to_blender_object
            ),
            "material",
        )
        obj.observe(_CapsuleScaleSetter(capsule), "scale")
        obj.observe(
            _CapsuleScaleKeyframeSetter(capsule), "scale", type="keyframe"
        )
        return capsule
```

Because a capsule's Z extent is `scale[2] + scale[0]` rather than `scale[2]`, a plain
`AttributeSetter(capsule, "scale")` is wrong. Implement it as follows instead, which keeps the
mesh unit-sized and maps kubric scale onto Blender scale:

- Add `build_capsule_mesh(bpy, scale)` to `kubric/renderer/blender_utils.py`: create a UV sphere
  with `segments=32, ring_count=16, radius=1.0`, enter edit mode, translate the top ring hemisphere
  vertices by `+1.0` on Z and the bottom by `-1.0`, then bridge with a cylinder side — or, simpler
  and fully deterministic, build the mesh from `bmesh` primitives: a cylinder of radius 1 and depth
  2 plus two UV-sphere caps of radius 1 translated to `z = ±1`, joined with
  `bmesh.ops.remove_doubles(..., dist=1e-6)`. Return the created object.
  The resulting unit capsule has dimensions `(2, 2, 4)` and radius 1, cylinder half-height 1.
- Add module-private setter classes in `kubric/renderer/blender.py`:

```python
class _CapsuleScaleSetter:
    """Maps kubric (radius, radius, half_height) onto a unit-capsule mesh scale."""

    def __init__(self, blender_obj):
        self.blender_obj = blender_obj

    def _values(self, scale):
        radius = float(scale[0])
        return (radius, radius, float(scale[2]))

    def __call__(self, change):
        radius_x, radius_y, half_height = self._values(change.new)
        self.blender_obj.scale = (radius_x, radius_y, 1.0)
        self.blender_obj.modifiers["capsule_length"].strength = half_height
```

If a modifier-based length control proves fragile, the acceptable alternative — and the one to use
if the mesh is rebuilt per instance — is to *not* observe `scale` at all and instead bake the
realized `(radius, half_height)` into the mesh at creation time inside
`build_capsule_mesh(bpy, obj.scale)`, leaving `blender_obj.scale = (1, 1, 1)`. The intervention
render pipeline never animates object scale, so baking is sufficient and simpler. Prefer baking;
delete `_CapsuleScaleSetter`/`_CapsuleScaleKeyframeSetter` from the sketch above and register only
material and `Object3D` setters for `Capsule`. Cylinders keep the normal `scale` observer because
the unit cylinder maps linearly.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest test/test_blender.py -q
```

Expected: all pass. If Blender emits `Not freed memory blocks` warnings at exit, ignore them.

- [ ] **Step 5: Commit**

```powershell
git add kubric/renderer/blender.py kubric/renderer/blender_utils.py test/test_blender.py
git commit -m "feat(kubric): render deterministic cylinder and capsule meshes"
```

---

## Task 6: Extend `ObjectConfig` shapes and oriented placement bounds

**Files:**
- Modify: `interventions/schema.py`
- Modify: `interventions/dataset.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `schema.SUPPORTED_SHAPES == frozenset(("cube", "sphere", "cylinder", "capsule"))`
  - `schema.TARGET_SHAPES == frozenset(("cube", "sphere"))`
  - `def schema.half_extents(config: ObjectConfig) -> Tuple[float, float, float]` — local,
    unrotated half-extents: cube `size`; sphere `(r, r, r)`; cylinder `(r, r, h)`;
    capsule `(r, r, h + r)`.
  - `def schema.oriented_aabb(config: ObjectConfig) -> Tuple[Tuple[float, ...], Tuple[float, ...]]`
    — world-space min/max after applying `config.quaternion` to the local half-extent box.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing schema tests**

Append to `tests/test_schema.py`:

```python
import math


def test_supported_shapes_include_cylinder_and_capsule():
  assert schema.SUPPORTED_SHAPES == frozenset(
      ("cube", "sphere", "cylinder", "capsule"))
  assert schema.TARGET_SHAPES == frozenset(("cube", "sphere"))


@pytest.mark.parametrize(
    ("shape", "size", "expected"),
    (
        ("cube", (0.2, 0.3, 0.4), (0.2, 0.3, 0.4)),
        ("sphere", 0.25, (0.25, 0.25, 0.25)),
        ("cylinder", (0.2, 0.2, 0.5), (0.2, 0.2, 0.5)),
        ("capsule", (0.2, 0.2, 0.5), (0.2, 0.2, 0.7)),
    ),
)
def test_half_extents_follow_documented_size_semantics(shape, size, expected):
  config = schema.ObjectConfig(object_id="a", shape=shape, size=size)
  assert schema.half_extents(config) == pytest.approx(expected)


@pytest.mark.parametrize("shape", ("sphere", "cylinder", "capsule"))
def test_radial_shapes_require_equal_x_and_y(shape):
  with pytest.raises(ValueError, match="radi"):
    schema.ObjectConfig(object_id="a", shape=shape, size=(0.2, 0.3, 0.4))


def test_oriented_aabb_accounts_for_yaw():
  config = schema.ObjectConfig(
      object_id="a",
      shape="cube",
      size=(0.5, 0.1, 0.2),
      position=(1.0, 2.0, 3.0),
      quaternion=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)))
  lower, upper = schema.oriented_aabb(config)
  assert lower == pytest.approx((0.9, 1.5, 2.8))
  assert upper == pytest.approx((1.1, 2.5, 3.2))
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_schema.py -q
```

Expected: `AssertionError` on `SUPPORTED_SHAPES` / `AttributeError: half_extents`.

- [ ] **Step 3: Implement the schema changes**

- `SUPPORTED_SHAPES = frozenset(("cube", "sphere", "cylinder", "capsule"))` and add
  `TARGET_SHAPES = frozenset(("cube", "sphere"))`.
- In `ObjectConfig.__post_init__`, after normalizing `size` to a 3-tuple, add: if
  `shape in ("sphere", "cylinder", "capsule")` and `size[0] != size[1]`, raise
  `ValueError("{} requires equal X and Y radii".format(shape))`. For `sphere`, also require
  `size[1] == size[2]`.
- Add a class docstring note on `ObjectConfig.size` documenting the four size conventions verbatim
  from the spec.
- `half_extents(config)` returns the tuple described in Interfaces (pure, no NumPy).
- `oriented_aabb(config)` rotates the eight corners of the local half-extent box by
  `config.quaternion` (WXYZ) using a small local quaternion-rotation helper — do **not** import
  NumPy or pyquaternion into `interventions/schema.py`; implement
  `_rotate(quaternion, vector)` with plain arithmetic — then returns
  `(min per axis + position, max per axis + position)`.
- In `interventions/dataset.py`, replace the sphere/cube-specific extent math used by the
  non-overlap placement loop and the `target_out_of_bounds` / `target_static_clip` QC checks with
  `schema.oriented_aabb`. Keep the existing exact cube-sweep and sphere-sweep target logic
  untouched — targets remain cube or sphere.
- In `sample_instance_spec`, validate the configured `target.shape` against `schema.TARGET_SHAPES`
  and raise `ValueError("target shape must be cube or sphere")` otherwise.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_schema.py tests/test_dataset.py -q
```

Expected: all pass, including every pre-existing dataset test (physics bytes must not change).

- [ ] **Step 5: Prove physics-only determinism is unchanged**

Add to `tests/test_dataset.py`:

```python
def test_physics_only_instance_id_is_unchanged_by_shape_extension(tmp_path):
  ranges = dataset.load_ranges(_write_ranges(tmp_path))
  spec = dataset.sample_instance_spec(ranges, master_seed=1234, index=0)
  assert "visual_scene" not in spec.to_dict()
  assert spec.instance_id == dataset.sample_instance_spec(
      ranges, master_seed=1234, index=0).instance_id
```

Reuse the existing helper in that file that writes a ranges YAML into `tmp_path`; if it is named
differently, use the existing name rather than adding a new helper.

```powershell
& $py -m pytest tests/test_dataset.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add interventions/schema.py interventions/dataset.py tests/test_schema.py tests/test_dataset.py
git commit -m "feat(schema): support cylinder and capsule proxies with oriented AABB placement"
```

---

## Task 7: Material families and visual/physical coupling

**Files:**
- Create: `interventions/materials.py`
- Create: `tests/test_materials.py`
- Modify: `tests/test_module_documentation.py`
- Modify: `tests/test_offline_imports.py`

**Interfaces:**
- Consumes: `interventions.appearance.MATERIAL_FAMILIES`, `TEXTURE_KINDS`.
- Produces:
  - `COUPLING_MODES = frozenset(("coupled", "independent", "held_out"))`
  - `MASS_BOUNDS: Tuple[float, float] = (0.25, 4.0)`
  - `class FamilyPriors(_SchemaMixin)` with fields `family`, `density`, `friction`, `restitution`,
    `metallic`, `roughness`, `ior`, `transmission`, `specular`, `emission_strength`,
    `texture_kinds`. Every range field is a `(low, high)` tuple with `low <= high`.
  - `FAMILY_PRIORS: Mapping[str, FamilyPriors]` — one entry per family, values exactly as tabled
    in the spec.
  - `def sample_material(rng, family, color_rgba, texture) -> appearance.MaterialSpec`
  - `def coupled_physics(rng, family, proxy_volume) -> Mapping[str, float]` returning
    `{"effective_density": float, "unclamped_mass": float, "mass": float, "friction": float, "restitution": float}`
  - `def proxy_volume(shape: str, half_extents: Sequence[float]) -> float`
  - `def is_held_out(combination: Mapping[str, str], holdouts: Sequence[Mapping[str, str]]) -> bool`

- [ ] **Step 1: Write the failing material tests**

Create `tests/test_materials.py`:

```python
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
        ("capsule", (0.5, 0.5, 1.0), math.pi * 0.25 * 2.0 + 4.0 / 3.0 * math.pi * 0.125),
    ),
)
def test_proxy_volume_matches_closed_form(shape, half_extents, expected):
  assert materials.proxy_volume(shape, half_extents) == pytest.approx(expected)


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


def test_coupled_physics_is_deterministic_for_a_given_seed():
  first = materials.coupled_physics(np.random.default_rng(5), "wood", 0.01)
  second = materials.coupled_physics(np.random.default_rng(5), "wood", 0.01)
  assert first == second


def test_is_held_out_matches_on_all_declared_keys_only():
  holdouts = ({"material_family": "glass", "texture_kind": "checker"},)
  assert materials.is_held_out(
      {"material_family": "glass", "texture_kind": "checker", "shape": "cube"}, holdouts)
  assert not materials.is_held_out(
      {"material_family": "glass", "texture_kind": "noise", "shape": "cube"}, holdouts)


def test_family_declares_permitted_texture_kinds():
  for family, priors in materials.FAMILY_PRIORS.items():
    assert priors.texture_kinds
    assert set(priors.texture_kinds) <= appearance.TEXTURE_KINDS
  assert "wood" in materials.FAMILY_PRIORS["wood"].texture_kinds
  assert "checker" not in materials.FAMILY_PRIORS["glass"].texture_kinds
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_materials.py -q
```

Expected: `ModuleNotFoundError: No module named 'interventions.materials'`.

- [ ] **Step 3: Implement `interventions/materials.py`**

Docstring:

```python
"""Material families coupling appearance priors to simulated physical properties.

Purpose: hold the shipped per-family visual and physical priors and realize them.
Public API: COUPLING_MODES, MASS_BOUNDS, FamilyPriors, FAMILY_PRIORS, sample_material,
coupled_physics, proxy_volume, and is_held_out.
Dependencies: Python's standard library, NumPy generators supplied by the caller, and
interventions.appearance; no renderer or simulator import.
Trust boundary: priors are dataset-scale conventions, not physical measurements, and
clamping is recorded rather than hidden.
"""
```

- Populate `FAMILY_PRIORS` with exactly the two spec tables. `specular` is `(0.4, 0.6)` for every
  family; `emission_strength` is `(0.0, 0.0)` for every family (shipped config is non-emissive).
  `texture_kinds` per family: metal `("solid", "noise", "speckle")`; rubber
  `("solid", "noise", "speckle")`; plastic `("solid", "noise", "checker", "speckle")`; ceramic
  `("solid", "marble", "speckle")`; glass `("solid",)`; wood `("wood", "solid", "noise")`;
  stone `("marble", "noise", "speckle", "solid")`.
- `proxy_volume(shape, half_extents)`:
  cube `8 * hx * hy * hz`; sphere `4/3 * pi * r**3`; cylinder `pi * r**2 * (2 * hz)`;
  capsule `pi * r**2 * (2 * hz) + 4/3 * pi * r**3`, where for capsule `hz` is the *cylinder*
  half-height (i.e. `schema.half_extents` minus the cap radius on Z). Document that in a one-line
  comment. Raise `ValueError` for unknown shapes.
- `sample_material(rng, family, color_rgba, texture)`: draw each PBR value with
  `float(rng.uniform(low, high))` in this fixed order — `metallic`, `roughness`, `specular`, `ior`,
  `transmission` — then build `appearance.MaterialSpec(family=family, base_color=tuple(color_rgba),
  metallic=..., roughness=..., specular=..., ior=..., transmission=..., emission=(0.0, 0.0, 0.0, 1.0),
  texture=texture)`. A degenerate range (`low == high`) must return exactly `low`, so special-case it
  to avoid consuming a different number of RNG draws — always call `rng.uniform` for every field so
  the draw count is constant.
- `coupled_physics(rng, family, proxy_volume)`: draw `effective_density`, `friction`,
  `restitution` in that order; `unclamped_mass = effective_density * proxy_volume`;
  `mass = min(max(unclamped_mass, MASS_BOUNDS[0]), MASS_BOUNDS[1])`; return a plain dict of floats.
- `is_held_out(combination, holdouts)`: `any(all(combination.get(k) == v for k, v in h.items()) for h in holdouts)`.

Register the module in `tests/test_module_documentation.py` and `tests/test_offline_imports.py`.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_materials.py tests/test_module_documentation.py tests/test_offline_imports.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add interventions/materials.py tests/test_materials.py tests/test_module_documentation.py tests/test_offline_imports.py
git commit -m "feat(materials): add coupled visual and physical material families"
```

---

## Task 8: Domain-separated appearance sampling

**Files:**
- Create: `interventions/appearance_sampling.py`
- Create: `tests/test_appearance_sampling.py`
- Create: `configs/scene_ranges_visual.yaml`
- Modify: `tests/test_module_documentation.py`
- Modify: `tests/test_offline_imports.py`

**Interfaces:**
- Consumes: `interventions.dataset.derive_seed(master_seed, index, domain)`,
  `interventions.appearance`, `interventions.materials`, `interventions.schema`.
- Produces:
  - `APPEARANCE_DOMAINS = ("geometry", "physics", "appearance", "texture", "camera", "lighting", "background", "render")`
  - `def validate_appearance_ranges(ranges: Mapping[str, Any]) -> Mapping[str, Any]`
  - `def sample_visual_scene(ranges, scene_config, master_seed, index) -> appearance.VisualSceneSpec`
  - `def sample_object_geometry(ranges, rng) -> Mapping[str, Any]` returning
    `{"shape": str, "size": Tuple[float, float, float], "yaw": float}`
  - `def sample_color(ranges, rng) -> Tuple[float, float, float, float]`
  - `def sample_texture(ranges, rng, family) -> appearance.TextureSpec`
  - `def sample_camera(ranges, rng, scene_config) -> appearance.CameraRenderSpec`
  - `def sample_lights(ranges, rng) -> Tuple[appearance.LightSpec, ...]`
  - `def sample_background(ranges, rng) -> appearance.BackgroundSpec`

- [ ] **Step 1: Write the failing sampling tests**

Create `tests/test_appearance_sampling.py`:

```python
import copy

import pytest
import yaml

from interventions import appearance, appearance_sampling, dataset, materials, schema

_CONFIG = "configs/scene_ranges_visual.yaml"


def _ranges():
  return dataset.load_ranges(_CONFIG)


def _scene(object_ids=("obj_0", "obj_1", "obj_2")):
  return schema.SceneConfig(
      objects=tuple(
          schema.ObjectConfig(object_id=name, shape="cube", size=0.2)
          for name in object_ids),
      camera=schema.CameraConfig(
          position=(4.0, 4.0, 3.0), look_at=(0.0, 0.0, 0.0), focal_length=35.0),
      frame_range=(0, 2),
      frame_rate=24,
      step_rate=240)


def test_sample_visual_scene_is_deterministic_and_valid():
  ranges = _ranges()
  scene = _scene()
  first = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 0)
  second = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 0)
  assert first == second
  appearance.validate_scene_correspondence(first, scene)
  assert first.frame_steps == (0, 10)
  assert len(first.camera.positions) == 2
  assert {item.object_id for item in first.objects} == set(
      item.object_id for item in scene.objects)


def test_sample_visual_scene_varies_with_index():
  ranges = _ranges()
  scene = _scene()
  a = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 0)
  b = appearance_sampling.sample_visual_scene(ranges, scene, 4321, 1)
  assert appearance.visual_scene_hash(a) != appearance.visual_scene_hash(b)


def test_every_shipped_family_appears_in_a_deterministic_window():
  ranges = _ranges()
  scene = _scene(tuple("obj_{}".format(i) for i in range(6)))
  seen_families = set()
  seen_textures = set()
  for index in range(24):
    visual = appearance_sampling.sample_visual_scene(ranges, scene, 20260829, index)
    for item in visual.objects:
      seen_families.add(item.material.family)
      seen_textures.add(item.material.texture.kind)
  assert seen_families == set(appearance.MATERIAL_FAMILIES)
  assert seen_textures >= {"solid", "noise", "checker", "wood", "marble", "speckle"}


def test_geometry_sampler_covers_all_four_proxies():
  import numpy as np
  ranges = _ranges()
  shapes = set()
  for seed in range(64):
    rng = np.random.default_rng(seed)
    shapes.add(appearance_sampling.sample_object_geometry(ranges, rng)["shape"])
  assert shapes == {"cube", "sphere", "cylinder", "capsule"}


def test_sampled_sizes_respect_radial_equality():
  import numpy as np
  ranges = _ranges()
  for seed in range(32):
    rng = np.random.default_rng(seed)
    result = appearance_sampling.sample_object_geometry(ranges, rng)
    config = schema.ObjectConfig(
        object_id="a", shape=result["shape"], size=result["size"])
    assert schema.half_extents(config)[0] > 0.0


def test_camera_frames_full_scene_bounds():
  ranges = _ranges()
  scene = _scene()
  visual = appearance_sampling.sample_visual_scene(ranges, scene, 7, 3)
  for position in visual.camera.positions:
    assert position[2] > 0.0
    radius = sum(value * value for value in position) ** 0.5
    low, high = ranges["appearance"]["camera"]["radius"]
    assert low - 1e-6 <= radius <= high + 1e-6


def test_validate_appearance_ranges_rejects_unknown_family_and_bad_bounds():
  raw = yaml.safe_load(open(_CONFIG, encoding="utf-8"))
  bad = copy.deepcopy(raw)
  bad["appearance"]["materials"]["families"] = ["unobtanium"]
  with pytest.raises(ValueError, match="family"):
    appearance_sampling.validate_appearance_ranges(bad)
  bad = copy.deepcopy(raw)
  bad["appearance"]["camera"]["radius"] = [9.0, 2.0]
  with pytest.raises(ValueError, match="radius"):
    appearance_sampling.validate_appearance_ranges(bad)


def test_held_out_combinations_are_never_sampled():
  raw = yaml.safe_load(open(_CONFIG, encoding="utf-8"))
  raw["appearance"]["coupling"]["mode"] = "held_out"
  raw["appearance"]["coupling"]["held_out"] = [{"material_family": "glass"}]
  ranges = appearance_sampling.validate_appearance_ranges(raw)
  scene = _scene(tuple("obj_{}".format(i) for i in range(4)))
  for index in range(32):
    visual = appearance_sampling.sample_visual_scene(ranges, scene, 99, index)
    assert "glass" not in {item.material.family for item in visual.objects}


def test_seed_domain_independence_uses_frozen_ranges():
  raw = yaml.safe_load(open(_CONFIG, encoding="utf-8"))
  base = appearance_sampling.sample_visual_scene(
      appearance_sampling.validate_appearance_ranges(raw), _scene(), 4321, 0)
  mutated = copy.deepcopy(raw)
  mutated["appearance"]["background"]["color_value"] = [0.05, 0.35]
  changed = appearance_sampling.sample_visual_scene(
      appearance_sampling.validate_appearance_ranges(mutated), _scene(), 4321, 0)
  assert changed.background != base.background
  assert changed.objects == base.objects
  assert changed.camera == base.camera
  assert changed.lights == base.lights

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_appearance_sampling.py -q
```

Expected: `ModuleNotFoundError: No module named 'interventions.appearance_sampling'`.

- [ ] **Step 3: Write `configs/scene_ranges_visual.yaml`**

Start from `configs/scene_ranges.yaml` verbatim (so physics behavior is identical) and append:

```yaml
appearance:
  enabled: true
  schema_version: "1.0"
  geometry:
    shapes:
      cube: 0.35
      sphere: 0.35
      cylinder: 0.15
      capsule: 0.15
    size: [0.12, 0.28]
    aspect_ratio: [0.6, 1.8]      # half_height / radius for cylinder and capsule
    orientation: upright_yaw       # upright_yaw | free_so3
  sources:
    procedural: 1.0                # manifest-backed weights are added in Task 10
  colors:
    strategies:
      hsv: 0.6
      palette: 0.3
      neutral: 0.1
    hue: [0.0, 1.0]
    saturation: [0.25, 0.95]
    value: [0.25, 0.95]
    palette:
      - [0.85, 0.23, 0.20, 1.0]
      - [0.20, 0.45, 0.85, 1.0]
      - [0.95, 0.78, 0.20, 1.0]
      - [0.25, 0.65, 0.35, 1.0]
      - [0.55, 0.30, 0.75, 1.0]
    neutral_value: [0.15, 0.85]
  textures:
    scale: [1.0, 8.0]
    detail: [1.0, 6.0]
    roughness: [0.2, 0.8]
    distortion: [0.0, 0.4]
    rotation: [0.0, 1.0]
    color_count: [1, 3]
  materials:
    families: [metal, rubber, plastic, ceramic, glass, wood, stone]
    weights: [1.0, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0]
  coupling:
    mode: coupled                  # coupled | independent | held_out
    held_out: []
  camera:
    radius: [5.0, 8.0]
    elevation: [0.35, 1.15]        # radians above the ground plane
    azimuth: [0.0, 6.283185307179586]
    focal_length: [30.0, 50.0]
    sensor_width: 36.0
    clipping_range: [0.1, 200.0]
    motion: fixed                  # fixed | linear
    linear_azimuth_delta: [-0.15, 0.15]
    safety_margin: 1.25
  lighting:
    rig: three_point
    key:
      position: [[2.5, -3.0, 4.0], [4.0, -1.5, 6.0]]
      intensity: [90.0, 180.0]
      color_temperature: [4500.0, 6800.0]
      size: [1.0, 2.5]
    fill:
      position: [[-3.5, -2.0, 2.5], [-2.0, -0.5, 4.0]]
      intensity: [30.0, 80.0]
      color_temperature: [5200.0, 7200.0]
      size: [1.5, 3.0]
    rim:
      position: [[-1.0, 3.5, 3.0], [1.5, 5.0, 5.0]]
      intensity: [40.0, 110.0]
      color_temperature: [6000.0, 8000.0]
      size: [0.8, 2.0]
  background:
    kind: color                    # color | hdri
    color_value: [0.08, 0.35]
    color_saturation: [0.0, 0.15]
    strength: [0.8, 1.4]
    exposure: [-0.3, 0.3]
    rotation: [0.0, 1.0]
```

- [ ] **Step 4: Implement `interventions/appearance_sampling.py`**

Docstring:

```python
"""Deterministic sampling of one realized visual scene from validated ranges.

Purpose: turn YAML appearance ranges into a single immutable VisualSceneSpec.
Public API: APPEARANCE_DOMAINS, validate_appearance_ranges, sample_visual_scene,
sample_object_geometry, sample_color, sample_texture, sample_camera, sample_lights,
and sample_background.
Dependencies: NumPy, interventions.appearance, interventions.materials, and
interventions.schema; Kubric, Blender, and PyBullet are never imported.
Trust boundary: sampling realizes and records every value it draws; it does not
validate that an external asset exists or that a scene is physically feasible.
"""
```

Rules:

- `validate_appearance_ranges(ranges)` checks: every declared `materials.families` entry is in
  `appearance.MATERIAL_FAMILIES` (`ValueError("unknown material family ...")`); `weights` has the
  same length as `families` and all weights are `> 0`; every `[low, high]` pair has `low <= high`
  and raises `ValueError` naming the offending key (e.g. `"radius"`); geometry shape weights are
  non-negative, sum to a positive number, and only name `schema.SUPPORTED_SHAPES`;
  `coupling.mode in materials.COUPLING_MODES`; `orientation in {"upright_yaw", "free_so3"}`;
  `background.kind in appearance.BACKGROUND_KINDS`; `camera.motion in {"fixed", "linear"}`.
  Return the input frozen via the same freezing path `dataset.load_ranges` uses, so callers may
  pass either a raw dict or an already-frozen mapping.
- `sample_visual_scene(ranges, scene_config, master_seed, index)`:
  1. `validate_appearance_ranges(ranges)`.
  2. Build one `numpy.random.Generator` per domain:
     `rngs = {domain: np.random.default_rng(dataset.derive_seed(master_seed, index, domain)) for domain in APPEARANCE_DOMAINS}`.
     Never share a generator across domains — this is what makes the domain-independence test pass.
  3. For each `ObjectConfig` in `scene_config.objects`, sorted by `object_id`: draw the material
     family from `rngs["appearance"]` (weighted, rejecting held-out combinations by redrawing up to
     64 times and raising `ValueError("held-out combinations exhaust the family pool")` if none
     remain), the color from `rngs["appearance"]`, the texture from `rngs["texture"]` restricted to
     `materials.FAMILY_PRIORS[family].texture_kinds`, and the material via
     `materials.sample_material(rngs["appearance"], family, color, texture)`.
     Build `appearance.VisualObjectSpec(object_id=config.object_id, source_kind="procedural",
     asset=None, collision_proxy_id=config.object_id, scale=(1.0, 1.0, 1.0),
     origin_offset=(0.0, 0.0, 0.0), alignment_quaternion=(1.0, 0.0, 0.0, 0.0), material=material)`.
  4. Camera from `rngs["camera"]`, lights from `rngs["lighting"]`, background from
     `rngs["background"]`, `render_seed=dataset.derive_seed(master_seed, index, "render") % (2 ** 31)`.
  5. `frame_steps=appearance.frame_steps_for(scene_config)`.
  6. Return the `VisualSceneSpec`; `validate_scene_correspondence` is called by the caller.
- `sample_object_geometry(ranges, rng)` draws a shape by normalized weights, a radius/half-extent
  from `size`, and an `aspect_ratio` for cylinder/capsule, returning
  `{"shape": shape, "size": (r, r, h) | (r, r, r) | (sx, sy, sz), "yaw": float}` where cube sizes
  draw three independent values from `size` and `yaw` is `rng.uniform(0.0, 2 * math.pi)` for
  `upright_yaw`.
- `sample_camera` places the camera on the configured half-sphere shell, verifies that the
  projected `scene_config.scene_bounds` corners fit within the frustum with `safety_margin`
  (recompute per frame; if the check fails, redraw up to 32 times, then raise
  `ValueError("camera cannot frame the configured scene bounds")`), and emits one position and one
  look-at per entry in `frame_steps_for(scene_config)` — identical values for `motion: fixed`, and a
  linearly interpolated azimuth sweep for `motion: linear`.
- `sample_lights` returns exactly three `rect_area` lights with `light_id` `"key"`, `"fill"`,
  `"rim"`, converting the sampled colour temperature to linear RGB with a local
  `_kelvin_to_rgb(kelvin)` helper (Planckian approximation, clamped to `[0, 1]`).
- `sample_background` for `kind: color` draws a value and saturation and returns a
  `BackgroundSpec(kind="color", color=(v*(1-s)+..., ..., 1.0), hdri=None, ...)`.

Register the module in `tests/test_module_documentation.py` and `tests/test_offline_imports.py`.

- [ ] **Step 5: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_appearance_sampling.py tests/test_module_documentation.py tests/test_offline_imports.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add interventions/appearance_sampling.py configs/scene_ranges_visual.yaml tests/test_appearance_sampling.py tests/test_module_documentation.py tests/test_offline_imports.py
git commit -m "feat(appearance): sample visual scenes with domain-separated seeds"
```

---

## Task 9: Attach visual scenes to instances without changing legacy bytes

**Files:**
- Modify: `interventions/dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `appearance_sampling.sample_visual_scene`, `appearance.validate_scene_correspondence`,
  `appearance.visual_scene_hash`, `materials.coupled_physics`.
- Produces:
  - `InstanceSpec.visual_scene: Optional[appearance.VisualSceneSpec] = None` as the **last** field.
  - `InstanceSpec.to_dict()` omits `"visual_scene"` when it is `None` and includes it otherwise.
  - `instance_id` payload gains `"visual_scene"` only when present.
  - `sample_instance_spec` samples geometry/material-coupled physics and a visual scene when
    `ranges.get("appearance", {}).get("enabled")` is true.
  - `_publish_instance` writes `appearance.json` (canonical bytes) into the instance directory and
    lists it in `instance_manifest.json` when a visual scene is present.

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_dataset.py`:

```python
def test_instance_spec_omits_visual_scene_when_absent(tmp_path):
  ranges = dataset.load_ranges(_write_ranges(tmp_path))
  spec = dataset.sample_instance_spec(ranges, master_seed=77, index=0)
  assert spec.visual_scene is None
  assert "visual_scene" not in spec.to_dict()


def test_instance_spec_includes_and_hashes_visual_scene():
  ranges = dataset.load_ranges("configs/scene_ranges_visual.yaml")
  spec = dataset.sample_instance_spec(ranges, master_seed=77, index=0)
  assert spec.visual_scene is not None
  payload = spec.to_dict()
  assert "visual_scene" in payload
  appearance.validate_scene_correspondence(spec.visual_scene, spec.scene_config)

  stripped = dataclasses.replace(spec, visual_scene=None)
  assert stripped.instance_id != spec.instance_id


def test_visual_instances_use_coupled_material_physics():
  ranges = dataset.load_ranges("configs/scene_ranges_visual.yaml")
  spec = dataset.sample_instance_spec(ranges, master_seed=77, index=0)
  by_id = {item.object_id: item for item in spec.scene_config.objects}
  for visual_object in spec.visual_scene.objects:
    config = by_id[visual_object.object_id]
    if config.static:
      continue
    priors = materials.FAMILY_PRIORS[visual_object.material.family]
    assert priors.friction[0] - 1e-9 <= config.friction <= priors.friction[1] + 1e-9
    assert priors.restitution[0] - 1e-9 <= config.restitution <= priors.restitution[1] + 1e-9
    assert materials.MASS_BOUNDS[0] <= config.mass <= materials.MASS_BOUNDS[1]
    assert config.metadata["material_family"] == visual_object.material.family
    assert "effective_density" in config.metadata


def test_publishing_writes_and_hashes_appearance_json(tmp_path):
  ranges = dataset.load_ranges("configs/scene_ranges_visual.yaml")
  result = dataset.run_batch(
      ranges, tmp_path / "ds", master_seed=5, num_instances=1, max_attempts=6)
  assert result["status"] in ("complete", "capacity_exhausted")
  for instance_dir in (tmp_path / "ds" / "instances").iterdir():
    manifest = json.loads(
        (instance_dir / "instance_manifest.json").read_text(encoding="utf-8"))
    assert "appearance.json" in manifest["files"]
    payload = json.loads((instance_dir / "appearance.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == appearance.APPEARANCE_SCHEMA_VERSION
```

Add `import dataclasses`, `from interventions import appearance, materials` to the test module if
absent.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_dataset.py -q -k visual
```

Expected: `TypeError: InstanceSpec.__init__() got an unexpected keyword argument 'visual_scene'`.

- [ ] **Step 3: Implement the integration**

- Add `visual_scene: Optional[appearance.VisualSceneSpec] = None` as the last `InstanceSpec` field.
  Validate in `__post_init__`: when not `None`, it must be a `VisualSceneSpec` and
  `appearance.validate_scene_correspondence(visual_scene, scene_config)` must pass.
- Override `to_dict()` to build the existing canonical mapping and add `"visual_scene"` only when
  present. Do the same for the `identity_payload` used by the instance-id hash. Do not touch key
  ordering of the existing fields.
- In `sample_instance_spec`, after the current physics sampling and before building the
  `InstanceSpec`:
  - if the appearance section is absent or disabled, behave exactly as today;
  - otherwise, for each non-floor object, use `appearance_sampling.sample_object_geometry`
    (`geometry` domain) for shape/size/yaw, apply the yaw to `ObjectConfig.quaternion`, then, for
    `coupling.mode == "coupled"` or `"held_out"`, replace mass/friction/restitution with
    `materials.coupled_physics(rng_physics, family, materials.proxy_volume(shape, half_extents))`
    and record `metadata={"material_family": family, "effective_density": ..., "unclamped_mass": ...}`;
    for `"independent"`, keep the existing mass/friction/restitution ranges;
  - the target keeps `schema.TARGET_SHAPES` and its existing push-mass range;
  - finally call `appearance_sampling.sample_visual_scene(ranges, scene_config, master_seed, index)`.
  The family must be drawn once per object and shared between the physics coupling and the visual
  material — draw it inside `sample_visual_scene`'s `appearance` domain and return it, or (simpler
  and preferred) add `appearance_sampling.sample_object_families(ranges, master_seed, index, object_ids)`
  that both call, so the two consumers see identical values without sharing a generator.
- In `_publish_instance`, when `spec.visual_scene is not None`, write `appearance.json` with the
  canonical bytes of `spec.visual_scene.to_dict()` into the staging directory before validation so
  it is covered by the existing digest manifest.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_dataset.py -q
```

Expected: all pass, including every legacy physics-only test.

- [ ] **Step 5: Commit**

```powershell
git add interventions/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): attach optional visual scenes to sampled instances"
```

---

## Task 10: Digest-pinned asset catalog with a content-addressed cache

**Files:**
- Create: `interventions/asset_catalog.py`
- Create: `tests/test_asset_catalog.py`
- Modify: `interventions/appearance_sampling.py`
- Modify: `tests/test_module_documentation.py`
- Modify: `tests/test_offline_imports.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `interventions.appearance.AssetReference`.
- Produces:
  - `class CatalogEntry(_SchemaMixin)`: `asset_id`, `archive_uri`, `archive_sha256`,
    `collision_shape`, `collision_size`, `origin_offset`, `alignment_quaternion`, `uniform_scale`
  - `def load_manifest(uri: str, expected_sha256: str) -> Tuple[CatalogEntry, ...]`
  - `def select_assets(entries, rng, count) -> Tuple[CatalogEntry, ...]`
  - `def fetch_archive(entry: CatalogEntry, cache_root: PathLike) -> pathlib.Path`
  - `def extract_archive(archive: PathLike, destination: PathLike) -> pathlib.Path`
  - `class UnsafeArchiveError(ValueError)`

- [ ] **Step 1: Write the failing catalog tests**

Create `tests/test_asset_catalog.py`:

```python
import hashlib
import json
import pathlib
import tarfile

import numpy as np
import pytest

from interventions import asset_catalog


def _entry_payload(asset_id, archive_uri, archive_sha256):
  return {
      "asset_id": asset_id,
      "archive_uri": archive_uri,
      "archive_sha256": archive_sha256,
      "collision_shape": "cube",
      "collision_size": [0.2, 0.2, 0.2],
      "origin_offset": [0.0, 0.0, 0.0],
      "alignment_quaternion": [1.0, 0.0, 0.0, 0.0],
      "uniform_scale": 1.0,
  }


def _write_archive(directory, name="asset.tar.gz"):
  payload_dir = directory / "payload"
  payload_dir.mkdir(parents=True, exist_ok=True)
  (payload_dir / "model.obj").write_text("v 0 0 0\n", encoding="utf-8")
  archive = directory / name
  with tarfile.open(archive, "w:gz") as handle:
    handle.add(payload_dir / "model.obj", arcname="asset/model.obj")
  digest = hashlib.sha256(archive.read_bytes()).hexdigest()
  return archive, digest


def _write_manifest(directory, entries):
  manifest = directory / "manifest.json"
  payload = json.dumps({"assets": entries}, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
  manifest.write_bytes(payload)
  return manifest, hashlib.sha256(payload).hexdigest()


def test_load_manifest_verifies_digest(tmp_path):
  archive, archive_digest = _write_archive(tmp_path)
  manifest, digest = _write_manifest(
      tmp_path, [_entry_payload("a", archive.as_uri(), archive_digest)])
  entries = asset_catalog.load_manifest(manifest.as_uri(), digest)
  assert len(entries) == 1
  assert entries[0].asset_id == "a"
  with pytest.raises(ValueError, match="manifest sha256"):
    asset_catalog.load_manifest(manifest.as_uri(), "f" * 64)


def test_select_assets_is_deterministic_and_sorted(tmp_path):
  archive, archive_digest = _write_archive(tmp_path)
  entries_payload = [
      _entry_payload(name, archive.as_uri(), archive_digest)
      for name in ("c", "a", "b")]
  manifest, digest = _write_manifest(tmp_path, entries_payload)
  entries = asset_catalog.load_manifest(manifest.as_uri(), digest)
  assert [item.asset_id for item in entries] == ["a", "b", "c"]
  first = asset_catalog.select_assets(entries, np.random.default_rng(3), 2)
  second = asset_catalog.select_assets(entries, np.random.default_rng(3), 2)
  assert first == second


def test_fetch_archive_is_content_addressed_and_verified(tmp_path):
  archive, archive_digest = _write_archive(tmp_path)
  manifest, digest = _write_manifest(
      tmp_path, [_entry_payload("a", archive.as_uri(), archive_digest)])
  entry = asset_catalog.load_manifest(manifest.as_uri(), digest)[0]
  cache = tmp_path / "cache"
  cached = asset_catalog.fetch_archive(entry, cache)
  assert cached.exists()
  assert archive_digest in str(cached)
  assert asset_catalog.fetch_archive(entry, cache) == cached

  tampered = asset_catalog.CatalogEntry(
      asset_id="a",
      archive_uri=archive.as_uri(),
      archive_sha256="e" * 64,
      collision_shape="cube",
      collision_size=(0.2, 0.2, 0.2),
      origin_offset=(0.0, 0.0, 0.0),
      alignment_quaternion=(1.0, 0.0, 0.0, 0.0),
      uniform_scale=1.0)
  with pytest.raises(ValueError, match="archive sha256"):
    asset_catalog.fetch_archive(tampered, cache)


def test_missing_archive_never_falls_back(tmp_path):
  archive, archive_digest = _write_archive(tmp_path)
  manifest, digest = _write_manifest(
      tmp_path, [_entry_payload("a", archive.as_uri(), archive_digest)])
  entry = asset_catalog.load_manifest(manifest.as_uri(), digest)[0]
  archive.unlink()
  with pytest.raises(FileNotFoundError):
    asset_catalog.fetch_archive(entry, tmp_path / "cache")


@pytest.mark.parametrize("arcname", ("../escape.obj", "/abs.obj", "asset/../../escape.obj"))
def test_extract_archive_rejects_path_traversal(tmp_path, arcname):
  source = tmp_path / "evil.obj"
  source.write_text("x", encoding="utf-8")
  archive = tmp_path / "evil.tar.gz"
  with tarfile.open(archive, "w:gz") as handle:
    handle.add(source, arcname=arcname)
  with pytest.raises(asset_catalog.UnsafeArchiveError):
    asset_catalog.extract_archive(archive, tmp_path / "out")


def test_extract_archive_rejects_links_and_devices(tmp_path):
  archive = tmp_path / "links.tar.gz"
  with tarfile.open(archive, "w:gz") as handle:
    info = tarfile.TarInfo("asset/link.obj")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    handle.addfile(info)
  with pytest.raises(asset_catalog.UnsafeArchiveError):
    asset_catalog.extract_archive(archive, tmp_path / "out")


def test_extract_archive_accepts_a_well_formed_archive(tmp_path):
  archive, _ = _write_archive(tmp_path)
  root = asset_catalog.extract_archive(archive, tmp_path / "out")
  assert (root / "asset" / "model.obj").is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_asset_catalog.py -q
```

Expected: `ModuleNotFoundError: No module named 'interventions.asset_catalog'`.

- [ ] **Step 3: Implement `interventions/asset_catalog.py`**

Docstring:

```python
"""Digest-pinned external asset catalogs with a content-addressed local cache.

Purpose: resolve, verify, cache, and safely unpack manifest-backed visual assets.
Public API: CatalogEntry, UnsafeArchiveError, load_manifest, select_assets,
fetch_archive, and extract_archive.
Dependencies: Python's standard library only; NumPy generators are supplied by the
caller and no renderer or simulator is imported.
Trust boundary: digests detect corruption and drift and extraction rejects hostile
archive members, but a matching digest does not authenticate the asset publisher.
"""
```

- `load_manifest(uri, expected_sha256)` reads the bytes via `urllib.request.urlopen` for
  `file://`/`http(s)://` URIs, compares `hashlib.sha256(payload).hexdigest()` against
  `expected_sha256` and raises `ValueError("manifest sha256 mismatch")` on mismatch, parses JSON,
  and returns `CatalogEntry` values sorted by `asset_id`. Reject duplicate `asset_id`s.
- `CatalogEntry` validates `collision_shape in schema.SUPPORTED_SHAPES`, positive
  `collision_size`/`uniform_scale`, unit `alignment_quaternion`, and hex digests.
- `select_assets(entries, rng, count)` draws `count` distinct entries with
  `rng.choice(len(entries), size=count, replace=False)`, then returns them sorted by `asset_id` so
  the result is order-stable.
- `fetch_archive(entry, cache_root)` computes `cache_root / entry.archive_sha256[:2] / entry.archive_sha256`,
  returns it when it exists and its digest still matches, otherwise downloads to a `.part` file in
  the same directory, verifies the digest (raising `ValueError("archive sha256 mismatch")` and
  deleting the partial file on mismatch), then `os.replace`s it into place. Missing sources raise
  `FileNotFoundError`; never substitute another asset.
- `extract_archive(archive, destination)` opens the tar, and for every member raises
  `UnsafeArchiveError` when the member is absolute, contains `..` after normalization, is a symlink,
  hardlink, device, FIFO, or when the resolved path escapes `destination`. Requires a single
  top-level directory. Extracts into a staging directory and `os.replace`s it into `destination`.
- Extend `appearance_sampling` with an optional `sources` weight for `kubasic`/`gso`: when a
  non-procedural source is drawn, the caller supplies pre-resolved `CatalogEntry` values and
  `sample_visual_scene` builds `appearance.AssetReference` and sets
  `VisualObjectSpec.scale`/`origin_offset`/`alignment_quaternion` from the entry so the visual mesh
  fits inside its collision proxy. With the shipped `configs/scene_ranges_visual.yaml`
  (`procedural: 1.0`) this path is inert, which keeps the default offline.

Add `assets_cache/` to `.gitignore`. Register the module in the documentation and offline-import
tests.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_asset_catalog.py tests/test_appearance_sampling.py tests/test_module_documentation.py tests/test_offline_imports.py -q
```

Expected: all pass, with no network access.

- [ ] **Step 5: Commit**

```powershell
git add interventions/asset_catalog.py interventions/appearance_sampling.py tests/test_asset_catalog.py tests/test_module_documentation.py tests/test_offline_imports.py .gitignore
git commit -m "feat(assets): add digest-pinned catalogs with safe extraction and caching"
```

---

## Task 11: Variation-aware balancing and leak-free compositional splits

**Files:**
- Modify: `interventions/dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 9's `InstanceSpec.visual_scene`.
- Produces:
  - `CandidateSummary` gains, as trailing optional fields:
    `collider_family: Tuple[str, ...] = ()`, `visual_source_kinds: Tuple[str, ...] = ()`,
    `material_families: Tuple[str, ...] = ()`, `texture_families: Tuple[str, ...] = ()`,
    `group_keys: Tuple[str, ...] = ()`. `to_dict()` omits every empty one.
  - `def balance_axes(summary: CandidateSummary) -> Tuple[str, ...]` — the stratification key.
  - `def assign_grouped_splits(candidates, fractions=None, *, seed=0)` now unions candidates that
    share a `topology_signature` **or** any `group_keys` entry, and assigns whole components.
  - `def split_report(assignment, candidates) -> Mapping[str, Any]` — actual counts per split and
    `largest_component`.

- [ ] **Step 1: Write the failing balancing and split tests**

Append to `tests/test_dataset.py`:

```python
def _summary(instance_id, topology, **extra):
  values = {
      "instance_id": instance_id,
      "attempt_index": 0,
      "category": "contact_added",
      "hop_depth": 1,
      "hop_bucket": "1",
      "topology_signature": topology,
      "artifact_path": "instances/{}".format(instance_id),
  }
  values.update(extra)
  return dataset.CandidateSummary(**values)


def test_candidate_summary_omits_empty_variation_fields():
  summary = _summary("instance_a", "a" * 64)
  payload = summary.to_dict()
  for key in ("collider_family", "visual_source_kinds", "material_families",
              "texture_families", "group_keys"):
    assert key not in payload


def test_candidate_summary_serializes_present_variation_fields():
  summary = _summary(
      "instance_a", "a" * 64,
      collider_family=("capsule", "cube"),
      visual_source_kinds=("procedural",),
      material_families=("glass", "metal"),
      texture_families=("noise", "solid"),
      group_keys=("material:glass|texture:noise",))
  payload = summary.to_dict()
  assert payload["material_families"] == ["glass", "metal"]
  assert payload["group_keys"] == ["material:glass|texture:noise"]


def test_balance_axes_include_variation_dimensions():
  summary = _summary(
      "instance_a", "a" * 64,
      collider_family=("cube",),
      visual_source_kinds=("procedural",),
      material_families=("metal",),
      texture_families=("noise",))
  assert dataset.balance_axes(summary) == (
      "contact_added", "1", "cube", "procedural", "metal", "noise")


def test_group_keys_prevent_leakage_across_splits():
  shared = ("asset:Mug_001",)
  candidates = [
      _summary("instance_{:02d}".format(i), "{:064x}".format(i), group_keys=shared)
      for i in range(10)
  ]
  assignment = dataset.assign_grouped_splits(
      candidates, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
  assert len(set(assignment.values())) == 1


def test_topology_grouping_still_holds_with_variation_fields():
  candidates = [
      _summary("instance_a", "a" * 64, material_families=("metal",)),
      _summary("instance_b", "a" * 64, material_families=("glass",)),
      _summary("instance_c", "b" * 64, material_families=("wood",)),
  ]
  assignment = dataset.assign_grouped_splits(candidates, seed=1)
  assert assignment["instance_a"] == assignment["instance_b"]


def test_split_report_records_actual_counts_and_largest_component():
  candidates = [
      _summary("instance_a", "a" * 64),
      _summary("instance_b", "a" * 64),
      _summary("instance_c", "b" * 64),
  ]
  assignment = dataset.assign_grouped_splits(candidates, seed=1)
  report = dataset.split_report(assignment, candidates)
  assert sum(report["counts"].values()) == 3
  assert report["largest_component"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_dataset.py -q -k "variation or group or balance_axes or split_report"
```

Expected: `TypeError` on the new `CandidateSummary` keyword arguments.

- [ ] **Step 3: Implement the balancing and split changes**

- Add the five trailing optional tuple fields to `CandidateSummary`, each validated as a tuple of
  non-empty strings, deduplicated and sorted in `__post_init__`. Override `to_dict()` to build the
  existing seven-key mapping first, then add each new field **only when non-empty**. This keeps old
  journal bytes byte-identical.
- `balance_axes(summary)` returns `(category, hop_bucket)` plus, for each of `collider_family`,
  `visual_source_kinds`, `material_families`, `texture_families`, the `"|".join(values)` when
  non-empty. Missing axes are dropped so small runs still stratify.
- `select_balanced` uses `balance_axes` instead of the inline `(category, hop_bucket)` tuple; its
  round-robin order and seeded tie-break are unchanged.
- `assign_grouped_splits` builds a union-find over candidates keyed by `topology_signature` and by
  every entry in `group_keys`, then assigns whole components greedily, largest component first,
  to the split furthest below its target fraction. Return the same
  `MappingProxyType{instance_id: split}` sorted by `instance_id`.
- `split_report(assignment, candidates)` returns
  `{"counts": {split: int}, "largest_component": int, "fractions": {split: float}}`.
- In `run_batch`, populate the new `CandidateSummary` fields from `spec.visual_scene` when present:
  `collider_family` from the distinct `ObjectConfig.shape` values of non-floor objects,
  `visual_source_kinds`/`material_families`/`texture_families` from the visual objects, and
  `group_keys` from configured holdout keys — `"asset:<asset_id>"` for every external asset and
  `"material:<family>|texture:<kind>"` for every configured held-out combination present in the
  instance. Add `"split_report"` to the returned manifest mapping.
- Held-out combinations must never land in `train`: after assignment, move any component containing
  a held-out `group_keys` entry to `val` or `test` per the configured policy, and raise
  `ValueError("held-out partition is empty")` when the policy cannot be satisfied.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_dataset.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add interventions/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): balance on variation axes and split by connected components"
```

---

## Task 12: Render-only scene construction and physics replay

**Files:**
- Create: `interventions/rendering.py`
- Create: `tests/test_rendering.py`
- Modify: `tests/test_module_documentation.py`
- Modify: `tests/test_offline_imports.py`

**Interfaces:**
- Consumes: `interventions.appearance`, `interventions.schema`, `interventions.logging`
  (`read_paired_artifact`, the 13-wide state layout and its slices).
- Produces:
  - `def build_scene(visual, scene_config, profile)` — returns
    `(kb_scene, renderer, objects_by_id)`; imports `kubric` lazily.
  - `def apply_states(objects_by_id, object_ids, states, frame_index, step_index)` — sets
    `position`/`quaternion` for one output frame and keyframes them.
  - `def render_branch(visual, scene_config, profile, object_ids, states, frame_steps)` —
    returns the layer dict from `kubric.renderer.blender.Blender.render`.
  - `def segmentation_map(visual) -> Mapping[str, int]` — stable 1-based labels in `object_id` order.
  - `def build_material(kb, material: appearance.MaterialSpec)` — a
    `kubric.PrincipledBSDFMaterial` plus, for non-`solid` textures, the Blender node tree.
  - `def write_layers(layers, directory, profile) -> Mapping[str, Any]` — writes the encodings from
    the spec and returns per-file `{"sha256", "size", "shape", "dtype", "encoding"}`.

- [ ] **Step 1: Write the failing renderer unit tests**

Create `tests/test_rendering.py`. Everything here must run **without** importing `bpy`, following
the stub pattern already used in `tests/test_render_demo_branches_blender.py`:

```python
import json

import numpy as np
import pytest

from interventions import appearance, rendering, schema


def _profile(**overrides):
  values = {
      "name": "smoke",
      "resolution": (8, 8),
      "samples_per_pixel": 1,
      "adaptive_sampling": False,
      "use_denoising": False,
      "background_transparency": False,
      "layers": appearance.RENDER_LAYERS,
      "device": "CPU",
  }
  values.update(overrides)
  return appearance.RenderProfile(**values)


def test_segmentation_map_is_stable_and_one_based():
  visual = _visual_scene(("obj_1", "obj_0"))
  mapping = rendering.segmentation_map(visual)
  assert mapping == {"obj_0": 1, "obj_1": 2}
  assert rendering.segmentation_map(_visual_scene(("obj_0", "obj_1"))) == mapping


def test_apply_states_sets_pose_for_the_mapped_physics_step():
  class FakeObject:
    def __init__(self):
      self.position = (0.0, 0.0, 0.0)
      self.quaternion = (1.0, 0.0, 0.0, 0.0)
      self.keyframes = []

    def keyframe_insert(self, name, frame):
      self.keyframes.append((name, frame))

  objects = {"obj_0": FakeObject()}
  states = np.zeros((11, 1, 13), dtype=np.float64)
  states[:, :, 3] = 1.0
  states[10, 0, 0:3] = (1.0, 2.0, 3.0)
  rendering.apply_states(objects, ("obj_0",), states, frame_index=1, step_index=10)
  assert objects["obj_0"].position == pytest.approx((1.0, 2.0, 3.0))
  assert ("position", 1) in objects["obj_0"].keyframes
  assert ("quaternion", 1) in objects["obj_0"].keyframes


def test_apply_states_rejects_out_of_range_physics_step():
  objects = {"obj_0": object()}
  states = np.zeros((4, 1, 13), dtype=np.float64)
  with pytest.raises(IndexError):
    rendering.apply_states(objects, ("obj_0",), states, frame_index=0, step_index=99)


def test_write_layers_uses_the_documented_encodings(tmp_path):
  frames = 2
  layers = {
      "rgba": np.zeros((frames, 4, 4, 4), dtype=np.uint8),
      "segmentation": np.ones((frames, 4, 4, 1), dtype=np.uint32),
      "depth": np.ones((frames, 4, 4, 1), dtype=np.float32),
      "normal": np.zeros((frames, 4, 4, 3), dtype=np.uint16),
      "forward_flow": np.zeros((frames, 4, 4, 2), dtype=np.float32),
      "backward_flow": np.zeros((frames, 4, 4, 2), dtype=np.float32),
      "object_coordinates": np.zeros((frames, 4, 4, 3), dtype=np.uint16),
  }
  report = rendering.write_layers(layers, tmp_path, _profile())
  names = sorted(path.name for path in tmp_path.iterdir())
  assert "rgba_00000.png" in names
  assert "depth_00000.tiff" in names
  assert "segmentation_00000.png" in names
  assert "forward_flow_00000.png" in names
  assert "data_ranges.json" in names
  ranges = json.loads((tmp_path / "data_ranges.json").read_text(encoding="utf-8"))
  assert "forward_flow" in ranges and "backward_flow" in ranges
  for entry in report.values():
    assert len(entry["sha256"]) == 64
    assert entry["size"] > 0


def test_write_layers_rejects_nonfinite_values(tmp_path):
  layers = {"depth": np.full((1, 2, 2, 1), np.nan, dtype=np.float32)}
  with pytest.raises(ValueError, match="finite"):
    rendering.write_layers(layers, tmp_path, _profile(layers=("depth",)))


def test_build_material_maps_spec_onto_principled_values():
  recorded = {}

  class FakeMaterial:
    def __init__(self, **kwargs):
      recorded.update(kwargs)

  class FakeKB:
    PrincipledBSDFMaterial = FakeMaterial
    Color = tuple

  material = _material_spec(family="metal", metallic=0.9, roughness=0.2, ior=2.0)
  rendering.build_material(FakeKB, material)
  assert recorded["metallic"] == pytest.approx(0.9)
  assert recorded["roughness"] == pytest.approx(0.2)
  assert recorded["ior"] == pytest.approx(2.0)


def test_both_branches_share_visual_identity():
  visual = _visual_scene(("obj_0", "obj_1"))
  factual = rendering.scene_identity(visual, _profile())
  counterfactual = rendering.scene_identity(visual, _profile())
  assert factual == counterfactual
  assert factual["visual_scene_sha256"] == appearance.visual_scene_hash(visual)
```

Define `_visual_scene` and `_material_spec` local helpers in this test module by importing the ones
from `tests/test_appearance.py` is not allowed — copy the small builders instead, keeping them
minimal (procedural objects, one `rect_area` light, a colour background, `frame_steps=(0, 10)`).
Add `def scene_identity(visual, profile) -> Mapping[str, str]` to the produced interface list,
returning `{"visual_scene_sha256": ..., "render_profile_sha256": ...}`.

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_rendering.py -q
```

Expected: `ModuleNotFoundError: No module named 'interventions.rendering'`.

- [ ] **Step 3: Implement `interventions/rendering.py`**

Docstring:

```python
"""Render logged intervention branches onto a shared, immutable visual scene.

Purpose: build render-only Kubric scenes, replay logged states, and emit validated layers.
Public API: build_scene, build_material, apply_states, render_branch, segmentation_map,
scene_identity, and write_layers.
Dependencies: NumPy and interventions schemas eagerly; Kubric, Blender, and image writers
are imported lazily inside the rendering functions.
Trust boundary: rendering consumes already-validated physics and never modifies, re-runs,
or attests the simulation; renderer metadata is provenance, not authentication.
"""
```

- `build_scene(visual, scene_config, profile)` imports `kubric as kb` and
  `from kubric.renderer.blender import Blender` **inside the function**. It creates
  `kb.Scene(resolution=profile.resolution, frame_start=0, frame_end=len(visual.frame_steps) - 1,
  frame_rate=scene_config.frame_rate, step_rate=scene_config.frame_rate)` — the render scene has no
  physics substeps because states are replayed as keyframes. It instantiates
  `Blender(scene, scratch_dir=..., adaptive_sampling=profile.adaptive_sampling,
  use_denoising=profile.use_denoising, samples_per_pixel=profile.samples_per_pixel,
  background_transparency=profile.background_transparency)`, sets
  `renderer.blender_scene.cycles.seed = visual.render_seed`, adds the camera
  (`kb.PerspectiveCamera` with `focal_length`, `sensor_width`, and per-frame position/look-at
  keyframes), the lights, the background, and one object per `VisualObjectSpec` using the
  `ObjectConfig.shape` of the matching collision proxy: `kb.Cube`, `kb.Sphere`, `kb.Cylinder`, or
  `kb.Capsule` with `scale=schema.half_extents(config)` and
  `segmentation_id=segmentation_map(visual)[object_id]`.
- `build_material(kb, material)` returns `kb.PrincipledBSDFMaterial(color=kb.Color(*material.base_color),
  metallic=..., roughness=..., specular=..., ior=..., transmission=...)`. For non-`solid` texture
  kinds it additionally builds the Blender node tree in `build_scene` (a `ShaderNodeTexNoise`,
  `ShaderNodeTexChecker`, `ShaderNodeTexWave` for wood/marble, or `ShaderNodeTexVoronoi` for
  speckle, driven by `TextureSpec.seed/scale/detail/distortion/rotation`), assigning
  `image.colorspace_settings.name = reference.color_space` for every `ImageReference`.
- `apply_states(objects_by_id, object_ids, states, frame_index, step_index)` reads
  `states[step_index, i, 0:3]` and `states[step_index, i, 3:7]`, assigns `position`/`quaternion`,
  and calls `keyframe_insert("position", frame_index)` and `keyframe_insert("quaternion", frame_index)`.
  Raise `IndexError` when `step_index >= states.shape[0]`.
- `render_branch(...)` iterates `enumerate(frame_steps)`, calls `apply_states`, then
  `renderer.render(frames=range(0, len(frame_steps)), return_layers=profile.layers)`.
- `write_layers(layers, directory, profile)` validates that every requested layer is present, that
  float arrays are finite (`ValueError("... must be finite")`), then writes with
  `kubric.file_io`: `write_rgba_batch` (`rgba_{:05d}.png`), `write_palette_png` per frame
  (`segmentation_{:05d}.png`), `write_depth_batch` (`depth_{:05d}.tiff`), `write_normal_batch`
  (`normal_{:05d}.png`), `write_forward_flow_batch`/`write_backward_flow_batch`
  (`forward_flow_{:05d}.png` / `backward_flow_{:05d}.png`, which also write `data_ranges.json`),
  and `write_coordinates_batch` (`object_coordinates_{:05d}.png`). Return the per-file digest report.
  Import `kubric.file_io` lazily.

Register the module in the documentation and offline-import tests; the offline test must confirm
`interventions.rendering` imports while `kubric` and `bpy` are blocked.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_rendering.py tests/test_offline_imports.py tests/test_module_documentation.py -q
```

Expected: all pass.

- [ ] **Step 5: Add the real one-frame render smoke test**

Append to `tests/test_rendering.py`:

```python
def test_smoke_render_produces_all_layers(tmp_path):
  bpy = pytest.importorskip("bpy")
  del bpy
  visual = _visual_scene(("obj_0",))
  scene_config = _scene_config(("obj_0",))
  profile = _profile(resolution=(32, 32))
  states = np.zeros((11, 1, 13), dtype=np.float64)
  states[:, :, 3] = 1.0
  states[:, 0, 2] = 0.5
  layers = rendering.render_branch(
      visual, scene_config, profile, ("obj_0",), states, visual.frame_steps)
  assert set(layers) == set(profile.layers)
  assert layers["rgba"].shape == (2, 32, 32, 4)
  report = rendering.write_layers(layers, tmp_path, profile)
  assert report
```

```powershell
& $py -m pytest tests/test_rendering.py -q
```

Expected: pass. This is the first test that actually starts Cycles; it must finish in well under a
minute at 32×32 with one sample.

- [ ] **Step 6: Commit**

```powershell
git add interventions/rendering.py tests/test_rendering.py tests/test_module_documentation.py tests/test_offline_imports.py
git commit -m "feat(rendering): build render-only scenes and replay logged branch states"
```

---

## Task 13: Resumable render stage with journals and atomic publication

**Files:**
- Modify: `interventions/rendering.py`
- Modify: `interventions/dataset.py`
- Modify: `tests/test_rendering.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Task 12's rendering functions, `interventions.logging.read_paired_artifact`,
  `dataset._write_once`, `dataset._write_atomic`.
- Produces:
  - `def render_instance(root, instance_id, profile, *, branches=("factual", "counterfactual")) -> Mapping[str, Any]`
    — renders one instance/profile pair, writes `render_manifest.json`, returns the record.
  - `def render_selection(root, instance_ids, profiles, *, resume=True) -> Mapping[str, Any]`
    — returns `{"status": "complete" | "render_incomplete", "records": [...]}`.
  - Render journal at `<root>/renders/<profile>/<instance_id>.json` with
    `status` in `{"pending", "complete", "error"}`.
  - Per-instance output at `instances/<instance_id>/renders/<profile>/` exactly as laid out in the
    spec, with `render_manifest.json` at the profile root and one directory per branch.

- [ ] **Step 1: Write the failing pipeline tests**

Append to `tests/test_rendering.py`:

```python
def test_render_manifest_binds_identity_and_digests(tmp_path):
  pytest.importorskip("bpy")
  root = _tiny_dataset(tmp_path)
  instance_id = _only_instance_id(root)
  record = rendering.render_instance(root, instance_id, appearance.SMOKE_PROFILE)
  assert record["status"] == "complete"

  manifest_path = (root / "instances" / instance_id / "renders" / "smoke"
                   / "render_manifest.json")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  assert manifest["instance_id"] == instance_id
  assert manifest["profile"] == "smoke"
  assert len(manifest["visual_scene_sha256"]) == 64
  assert len(manifest["render_profile_sha256"]) == 64
  assert manifest["branches"] == ["counterfactual", "factual"]
  assert manifest["frame_steps"] == [0, 10]
  assert set(manifest["segmentation_map"]) == set(
      item.object_id for item in _visual_of(root, instance_id).objects)
  assert manifest["runtime"]["blender_version"]
  for branch in manifest["branches"]:
    for entry in manifest["outputs"][branch].values():
      assert len(entry["sha256"]) == 64
      assert entry["size"] > 0


def test_branches_share_visuals_and_pre_intervention_pixels(tmp_path):
  pytest.importorskip("bpy")
  import imageio.v2 as imageio_v2
  root = _tiny_dataset(tmp_path)
  instance_id = _only_instance_id(root)
  rendering.render_instance(root, instance_id, appearance.SMOKE_PROFILE)
  base = root / "instances" / instance_id / "renders" / "smoke"
  factual = imageio_v2.imread(base / "factual" / "rgba_00000.png")
  counterfactual = imageio_v2.imread(base / "counterfactual" / "rgba_00000.png")
  assert np.array_equal(factual, counterfactual)


def test_render_selection_resumes_without_rewriting_complete_output(tmp_path):
  pytest.importorskip("bpy")
  root = _tiny_dataset(tmp_path)
  instance_id = _only_instance_id(root)
  rendering.render_selection(root, (instance_id,), (appearance.SMOKE_PROFILE,))
  target = (root / "instances" / instance_id / "renders" / "smoke" / "factual"
            / "rgba_00000.png")
  before = target.read_bytes()
  mtime = target.stat().st_mtime_ns
  result = rendering.render_selection(
      root, (instance_id,), (appearance.SMOKE_PROFILE,))
  assert result["status"] == "complete"
  assert target.read_bytes() == before
  assert target.stat().st_mtime_ns == mtime


def test_failed_render_is_journaled_and_leaves_no_partial_output(tmp_path, monkeypatch):
  root = _tiny_dataset(tmp_path)
  instance_id = _only_instance_id(root)

  def explode(*args, **kwargs):
    raise RuntimeError("cycles exploded")

  monkeypatch.setattr(rendering, "render_branch", explode)
  result = rendering.render_selection(
      root, (instance_id,), (appearance.SMOKE_PROFILE,))
  assert result["status"] == "render_incomplete"
  journal = json.loads(
      (root / "renders" / "smoke" / "{}.json".format(instance_id)).read_text(
          encoding="utf-8"))
  assert journal["status"] == "error"
  assert "cycles exploded" in journal["message"]
  assert not (root / "instances" / instance_id / "renders" / "smoke").exists()
  assert (root / "instances" / instance_id / "spec.json").exists()


def test_rerendering_the_same_instance_reproduces_identical_bytes(tmp_path):
  pytest.importorskip("bpy")
  root = _tiny_dataset(tmp_path)
  instance_id = _only_instance_id(root)
  rendering.render_instance(root, instance_id, appearance.SMOKE_PROFILE)
  base = root / "instances" / instance_id / "renders" / "smoke"
  first = (base / "factual" / "rgba_00000.png").read_bytes()
  shutil.rmtree(base)
  (root / "renders" / "smoke" / "{}.json".format(instance_id)).unlink()
  rendering.render_instance(root, instance_id, appearance.SMOKE_PROFILE)
  assert (base / "factual" / "rgba_00000.png").read_bytes() == first
```

Add these module-level helpers to `tests/test_rendering.py`:

```python
import shutil

from interventions import dataset


def _tiny_dataset(tmp_path):
  root = tmp_path / "ds"
  ranges = dataset.load_ranges("configs/scene_ranges_visual.yaml")
  dataset.run_batch(ranges, root, master_seed=31, num_instances=1, max_attempts=8)
  return root


def _only_instance_id(root):
  manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
  return manifest["selected_ids"][0]


def _visual_of(root, instance_id):
  payload = json.loads(
      (root / "instances" / instance_id / "appearance.json").read_text(
          encoding="utf-8"))
  return rendering.visual_scene_from_dict(payload)
```

Add `def visual_scene_from_dict(payload) -> appearance.VisualSceneSpec` to
`interventions/appearance.py` (re-exported from `interventions/rendering.py`) that reconstructs
every frozen schema from its canonical dict and raises `ValueError` on an unknown
`schema_version`. Write its round-trip test in `tests/test_appearance.py`:

```python
def test_visual_scene_round_trips_through_canonical_json():
  original = _visual_scene()
  restored = appearance.visual_scene_from_dict(
      json.loads(json.dumps(original.to_dict())))
  assert restored == original
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_rendering.py -q -k "manifest or resume or journaled or rerender or round_trip"
```

Expected: `AttributeError: module 'interventions.rendering' has no attribute 'render_instance'`.

- [ ] **Step 3: Implement the render stage**

- `render_instance(root, instance_id, profile, *, branches=("factual", "counterfactual"))`:
  1. Read `spec.json` and `appearance.json`, verify the digests recorded in
     `instance_manifest.json`, and rebuild the `InstanceSpec` scene config and `VisualSceneSpec`.
  2. Read both branch logs with `interventions.logging.read_paired_artifact`; verify object ids and
     step counts, and that `visual.frame_steps` are all `< states.shape[0]`.
  3. Write a `"pending"` journal record with `_write_atomic` at
     `root / "renders" / profile.name / (instance_id + ".json")`.
  4. Render each branch into a staging directory `.render-<uuid>` created **under the destination
     parent** (`instances/<instance_id>/renders/`), so the final `os.replace` is same-volume.
  5. Validate every produced layer (`write_layers` already checks finiteness) and assemble
     `render_manifest.json` binding: `instance_id`, `profile`, `visual_scene_sha256`,
     `render_profile_sha256`, `pair_manifest_sha256` (from `instance_manifest.json`),
     `branches` (sorted), `segmentation_map`, `frame_steps`, `runtime`
     (`{"blender_version": bpy.app.version_string, "python": sys.version, "platform": platform.platform(), "device": profile.device}`),
     `asset_digests` (manifest/archive/image digests, `[]` for procedural), `outputs`
     (per branch, per file: `sha256`, `size`, `shape`, `dtype`, `encoding`), and `status`.
  6. `os.replace` the staging directory into `instances/<instance_id>/renders/<profile.name>`.
  7. Rewrite the journal record as `"complete"` with the manifest digest.
  On any exception: delete the staging directory, write the journal record as `"error"` with
  `error_type` and `message`, and re-raise a `RenderError` that `render_selection` catches.
- `render_selection(root, instance_ids, profiles, *, resume=True)` iterates instances in sorted
  order and profiles in name order. When `resume` and the journal says `"complete"` **and** the
  recorded manifest digest still matches the on-disk manifest, skip without touching bytes.
  Return `{"status": "complete"}` when every requested pair is complete, otherwise
  `{"status": "render_incomplete", ...}` with the per-pair records.
- Add `RenderError(RuntimeError)` to the module's public API and docstring.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_rendering.py tests/test_appearance.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add interventions/rendering.py interventions/appearance.py tests/test_rendering.py tests/test_appearance.py
git commit -m "feat(rendering): publish resumable, digest-bound render manifests atomically"
```

---

## Task 14: `render_dataset` CLI and `generate_dataset --render-profile`

**Files:**
- Create: `scripts/render_dataset.py`
- Create: `tests/test_render_dataset.py`
- Modify: `scripts/generate_dataset.py`
- Modify: `interventions/dataset.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_module_documentation.py`

**Interfaces:**
- Consumes: `rendering.render_selection`, `dataset.run_batch`.
- Produces:
  - `scripts/render_dataset.py` with `main(argv=None) -> int`, flags
    `--dataset` (required), `--profile` (repeatable, default `smoke`), `--all-candidates`
    (flag, default off — renders only `selected_ids`), `--no-resume`.
    Exit codes: `0` complete, `1` batch-level error, `3` `render_incomplete`.
  - `scripts/generate_dataset.py` gains a repeatable `--render-profile`. Without it, behavior and
    exit codes are byte-for-byte unchanged. With it, `run_batch` runs physics first, then renders.
  - `run_batch(..., render_profiles: Sequence[str] = (), render_all_candidates: bool = False)`
    adds `"render"` to the returned manifest and may return `status="render_incomplete"`.

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_render_dataset.py`:

```python
import importlib.util
import json
import pathlib
import sys

import pytest

from interventions import appearance, dataset

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_NAME = "render_dataset_script"
_SPEC = importlib.util.spec_from_file_location(
    _SCRIPT_NAME, _PROJECT_ROOT / "scripts" / "render_dataset.py")
render_dataset = importlib.util.module_from_spec(_SPEC)
sys.modules[_SCRIPT_NAME] = render_dataset
_SPEC.loader.exec_module(render_dataset)


def _dataset(tmp_path):
  root = tmp_path / "ds"
  ranges = dataset.load_ranges("configs/scene_ranges_visual.yaml")
  dataset.run_batch(ranges, root, master_seed=41, num_instances=1, max_attempts=8)
  return root


def test_cli_renders_selected_instances_and_returns_zero(tmp_path, capsys):
  pytest.importorskip("bpy")
  root = _dataset(tmp_path)
  code = render_dataset.main(["--dataset", str(root), "--profile", "smoke"])
  assert code == 0
  payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
  assert payload["status"] == "complete"
  manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
  for instance_id in manifest["selected_ids"]:
    assert (root / "instances" / instance_id / "renders" / "smoke"
            / "render_manifest.json").is_file()


def test_cli_returns_three_when_a_render_fails(tmp_path, capsys, monkeypatch):
  root = _dataset(tmp_path)
  monkeypatch.setattr(
      render_dataset.rendering, "render_instance",
      lambda *a, **k: (_ for _ in ()).throw(
          render_dataset.rendering.RenderError("boom")))
  code = render_dataset.main(["--dataset", str(root), "--profile", "smoke"])
  assert code == 3
  payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
  assert payload["status"] == "render_incomplete"


def test_cli_rejects_unknown_profile(tmp_path):
  root = _dataset(tmp_path)
  with pytest.raises(SystemExit):
    render_dataset.main(["--dataset", str(root), "--profile", "ultra"])
```

Append to `tests/test_dataset.py`:

```python
def test_run_batch_without_render_profiles_is_unchanged(tmp_path):
  ranges = dataset.load_ranges(_write_ranges(tmp_path))
  result = dataset.run_batch(
      ranges, tmp_path / "ds", master_seed=9, num_instances=1, max_attempts=6)
  assert "render" not in result
  assert result["status"] in ("complete", "capacity_exhausted")


def test_run_batch_with_render_profiles_reports_render_status(tmp_path):
  pytest.importorskip("bpy")
  ranges = dataset.load_ranges("configs/scene_ranges_visual.yaml")
  result = dataset.run_batch(
      ranges, tmp_path / "ds", master_seed=9, num_instances=1, max_attempts=8,
      render_profiles=("smoke",))
  assert result["render"]["status"] in ("complete", "render_incomplete")
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
& $py -m pytest tests/test_render_dataset.py -q
```

Expected: `FileNotFoundError` for `scripts/render_dataset.py`.

- [ ] **Step 3: Implement the CLIs**

`scripts/render_dataset.py` docstring (module + `main`, matching the contract enforced by
`tests/test_module_documentation.py`):

```python
"""Render an existing intervention dataset selection with the requested profiles.

Purpose: resolve a dataset selection and render every requested instance/profile pair.
Public API: main().
Dependencies: argparse and interventions.rendering; Kubric and Blender load lazily on
the rendering path.
Trust boundary: the CLI renders already-published physics and never regenerates,
repairs, or re-attests simulation artifacts.
"""
```

`main` builds the profile list from `appearance.PROFILES_BY_NAME` (an unknown name is an
`argparse` `choices` error → `SystemExit`),
reads `manifest.json` for `selected_ids` (or the accepted candidate ids when `--all-candidates`),
calls `rendering.render_selection`, prints one canonical JSON line per record plus a final summary
line, and returns `0`/`1`/`3`.

`scripts/generate_dataset.py` adds `parser.add_argument("--render-profile", action="append",
default=[], choices=("smoke", "production"))` and passes `render_profiles=tuple(args.render_profile)`
to `run_batch`. Exit-code mapping becomes: `0` complete, `2` capacity exhausted, `3`
`render_incomplete`, `1` batch-level error. The `--render-profile`-free path must not import
`rendering` at all.

In `run_batch`, after balancing and split assignment and after the physics manifest is written,
when `render_profiles` is non-empty, resolve each name through `appearance.PROFILES_BY_NAME`
(raising `ValueError` for an unknown name), import `interventions.rendering` lazily, and call
`render_selection` with the resolved `RenderProfile` objects. Store the result under `"render"`
and set the batch `status` to `"render_incomplete"` when rendering did not complete — never
downgrade a physics `capacity_exhausted` status silently; report both in the manifest.

Add `scripts/render_dataset.py` to `_MODULE_PATHS` and `_EXACT_MAIN_DOCSTRINGS` in
`tests/test_module_documentation.py` with the exact `main` docstring
`"Renders the requested dataset selection and returns zero, one, or three."`.

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
& $py -m pytest tests/test_render_dataset.py tests/test_dataset.py tests/test_module_documentation.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/render_dataset.py scripts/generate_dataset.py interventions/dataset.py tests/test_render_dataset.py tests/test_dataset.py tests/test_module_documentation.py
git commit -m "feat(scripts): add render_dataset CLI and optional render stage for generate_dataset"
```

---

## Task 15: Documentation and the measured smoke report

**Files:**
- Modify: `docs/trajectory_interventions.md`
- Modify: `README.md`
- Create: `notes/session-logs/2026-08-29-rendered-intervention-variety.md`

**Interfaces:**
- Consumes: everything above.
- Produces: documentation and one recorded measurement run. No new code.

- [ ] **Step 1: Run the full suite**

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
& $py -m pytest tests/ test/ -q
```

Expected: green, with any pre-existing baseline failures from Task 0 unchanged.

- [ ] **Step 2: Run the measured smoke batch**

```powershell
& $py -m scripts.generate_dataset --config configs/scene_ranges_visual.yaml --output output/smoke_variety --seed 20260829 --num-instances 8 --max-attempts 64 --render-profile smoke
```

Record: wall-clock time, physics acceptance rate (accepted attempts / attempts), render throughput
(instances per minute), on-disk bytes per layer (`Get-ChildItem -Recurse | Measure-Object Length -Sum`
grouped by layer prefix), render failures, the strata and split distributions from
`manifest.json`, and the environment identity from any `render_manifest.json`.

- [ ] **Step 3: Verify deterministic rerender**

```powershell
Copy-Item -Recurse output/smoke_variety output/smoke_variety_check
Remove-Item -Recurse output/smoke_variety_check/instances/*/renders
Remove-Item -Recurse output/smoke_variety_check/renders
& $py -m scripts.render_dataset --dataset output/smoke_variety_check --profile smoke
```

Compare the `outputs` digests in both `render_manifest.json` files; they must be identical. Record
the comparison result.

- [ ] **Step 4: Write the documentation**

Extend `docs/trajectory_interventions.md` with sections for: the appearance schema and its trust
boundary; the four `ObjectConfig.size` conventions; coupled/independent/held-out material
semantics; a YAML example for the procedural config and one for a manifest-backed config; the
smoke and production render commands using the `thesis` env (no Docker); the layer encodings and
the artifact layout tree from the spec; compositional split behavior; asset-cache and resume rules;
and the measured smoke evidence from Steps 2 and 3.

Add a short "Running without Docker" pointer in `README.md` to `docs/environment_thesis.md`.

- [ ] **Step 5: Write the session log**

Create `notes/session-logs/2026-08-29-rendered-intervention-variety.md` recording what was built,
the measured numbers, the environment identity, and any deviations from this plan.

- [ ] **Step 6: Verify nothing generated is staged**

```powershell
git status --short
```

Expected: no `output/`, `*.png`, `*.tiff`, `*.blend`, `*.npy`, or cache paths listed as staged.

- [ ] **Step 7: Commit**

```powershell
git add docs/trajectory_interventions.md README.md notes/session-logs/2026-08-29-rendered-intervention-variety.md
git commit -m "docs: document rendered intervention variety and record smoke evidence"
```

---

## Completion checklist

Mirrors the spec's completion criteria. Verify each before declaring the feature done.

- [ ] The deterministic sample window exercises every configured procedural geometry, material, and
      texture family (Task 8 tests).
- [ ] A local pinned manifest fixture exercises the asset-backed path (Task 10 tests).
- [ ] Coupled, independent, and held-out modes pass their exact tests (Tasks 7-9).
- [ ] Factual and counterfactual renders share identical visual specs, camera, lights, background,
      segmentation mapping, and pre-intervention pixels (Task 13 tests).
- [ ] Smoke and production profiles publish and validate all requested layers (Tasks 12-14).
- [ ] No configured topology, asset, or held-out combination leaks across splits (Task 11 tests).
- [ ] Interrupted and failed renders resume without changing valid physics or completed bytes
      (Task 13 tests).
- [ ] Existing core, intervention, demo, docs, and offline-import suites pass (Task 15 Step 1).
- [ ] A fresh measured smoke report records throughput, storage, failures, distributions,
      environment identity, and rerender hashes (Task 15 Steps 2-3).
- [ ] Generated datasets and media remain untracked (Task 15 Step 6).
