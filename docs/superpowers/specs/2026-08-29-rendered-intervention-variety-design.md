# Rendered Intervention Variety Design

**Status:** Approved in conversation on 2026-08-29

## Goal

Extend the trajectory-intervention dataset from physics-only cube/sphere pairs
into a deterministic, resumable, end-to-end dataset pipeline with:

- broader procedural and optional manifest-backed object appearance;
- cube, sphere, cylinder, and capsule collision proxies;
- typed colors, textures, materials, physical properties, cameras, lights, and
  backgrounds;
- factual and counterfactual renders with Kubric's full annotation set;
- configurable material/physics coupling and compositional holdouts; and
- fast smoke and configurable production render profiles.

The factual and counterfactual branches of an instance must replay the existing
logged physics and use exactly the same sampled visual scene. Rendering must
never modify or independently attest the simulation.

## Product Decisions

- Scope is the intervention dataset, not a migration of all historical MOVi
  workers.
- Appearance is a first-class, immutable dataset contract rather than loose
  renderer metadata.
- Procedural objects and textures work offline by default.
- KuBasic and GSO assets are optional and digest-pinned.
- Visual meshes may use validated primitive collision proxies. Asset geometry
  is not silently substituted into physics.
- Material families use coupled visual/physical priors by default. The YAML can
  select independent sampling or explicit held-out combinations.
- Camera, lighting, and background vary per instance but remain identical
  across its branches.
- Selected instances receive RGB(A), segmentation, depth, normals, forward and
  backward flow, and object-coordinate layers.
- Topology grouping remains mandatory. Configured asset and
  material/texture holdouts add further no-leakage constraints.
- Rendering provides smoke and production profiles over the same semantic
  instance.

## Non-Goals

- Rewriting every Kubric challenge worker to use the new sampler.
- Claiming bit-identical Cycles output across different Blender versions,
  hardware, drivers, or container images.
- Treating rendered pixels as evidence that physics or producer provenance is
  authentic.
- Silently repairing, replacing, or skipping missing external assets.
- Supporting arbitrary triangle-mesh collision in the first version.
- Allowing independently randomized factual/counterfactual appearance.
- Making cylinder or capsule an intervention target in the first version.
  Dynamic and non-target static objects may use them; targets remain cube or
  sphere so the existing swept-volume guarantees stay exact.

## Architecture

The implementation has four bounded layers.

### 1. Immutable visual schemas

`interventions/appearance.py` contains renderer-independent frozen values:

- `AssetReference`: manifest URI, manifest SHA-256, asset ID, archive SHA-256,
  and asset-native material policy.
- `TextureSpec`: texture kind, seed, colors, scale, detail, roughness,
  distortion, rotation, and an optional digest-pinned image reference.
- `MaterialSpec`: family and realized base color, metallic, roughness,
  specular, IOR, transmission, emission, and texture.
- `VisualObjectSpec`: object ID, source kind, optional asset reference,
  collision-proxy ID, asset alignment transform, and material.
- `CameraRenderSpec`: a realized per-output-frame camera pose path, look-at
  path, focal length, sensor width, and clipping range.
- `LightSpec`: light kind, pose, color, intensity, and shape parameters.
- `BackgroundSpec`: color or pinned HDRI identity, rotation, strength, and
  exposure.
- `VisualSceneSpec`: objects, camera, lights, background, render seed, output
  frame-to-physics-step mapping, and appearance schema version.
- `RenderProfile`: resolution, samples, adaptive sampling, denoising,
  transparency, output layers, and encoder settings.

Schemas validate finite numeric values, exact vector lengths, legal enum values,
unique object IDs, complete correspondence with `SceneConfig`, and JSON-safe
metadata. All containers become immutable and serialize canonically.

Manifest asset identity belongs to `VisualObjectSpec`, not `ObjectConfig`.
`ObjectConfig` remains the authoritative collision proxy and physical state.
This separation preserves old physics artifacts and avoids pretending a
render-only scanned mesh is the collision geometry.

### 2. Domain-separated sampling

`interventions/appearance_sampling.py` converts validated YAML ranges into one
fully realized `VisualSceneSpec`. It uses `derive_seed()` with independent
domains:

- `geometry`
- `physics`
- `appearance`
- `texture`
- `camera`
- `lighting`
- `background`
- `render`

Changing one domain must not reshuffle values in another. Sampling stores every
realized value; replay never consults mutable presets.

`InstanceSpec` gains an optional visual scene. Its canonical `to_dict()` omits
the field when absent, preserving existing physics-only artifact bytes and
resume behavior. When present, the visual scene participates in the semantic
instance hash. Render quality settings do not participate in that hash; they
produce a separate render-profile identity.

### 3. Physics geometry adapters

Kubric gains narrowly scoped `Cylinder` and `Capsule` physical objects:

- Blender creates deterministic local-Z visual meshes.
- PyBullet uses `GEOM_CYLINDER` and `GEOM_CAPSULE`.
- trait observation, materials, keyframes, segmentation, and physical
  properties follow the existing cube/sphere contract.

`ObjectConfig.size` has these documented meanings:

- cube: local XYZ half-extents;
- sphere: radius repeated on XYZ;
- cylinder: `(radius, radius, half_height)` with a local-Z axis;
- capsule: `(radius, radius, cylinder_half_height)` with total local-Z
  half-extent `cylinder_half_height + radius`.

Sphere, cylinder, and capsule validation requires equal X/Y radii. Initial
placement and QC use shape-specific oriented AABBs. The shipped sampler uses
upright objects with random yaw; arbitrary SO(3) orientation remains
configurable but is not enabled in the default ranges.

The intervention target remains cube or sphere. This avoids replacing the
existing exact cube sweep and sphere sweep with a conservative approximation.

### 4. Staged render pipeline

`interventions/rendering.py` reads validated pair artifacts and visual specs,
builds render-only scenes, replays logged states, and validates render output.
Heavy Kubric, Blender, and image dependencies are imported lazily.

`scripts/render_dataset.py` renders an existing dataset selection.
`scripts/generate_dataset.py` gains repeatable `--render-profile` arguments;
without them, its current physics-only behavior and exit codes remain
unchanged. With profiles, it runs physics first and then invokes the resumable
render stage over selected instances.

This separation keeps `run_batch()` backend-neutral, allows render retries
without rerunning physics, and makes smoke and production renders independently
resumable.

## Geometry and Asset Variety

### Procedural path

The default config is network-free and samples cube, sphere, cylinder, and
capsule proxies with configurable weights, size ranges, aspect ratios, and
upright/yaw orientation policies.

### Manifest-backed path

Optional visual objects come from configured KuBasic or GSO manifests.
Configuration must provide the expected manifest SHA-256. Batch preflight
loads the manifest, verifies the digest, deterministically chooses eligible
asset IDs, fetches archives through a content-addressed cache, and verifies each
archive SHA-256 before simulation begins.

Each external visual object records:

- manifest and archive identity;
- asset ID and source kind;
- proxy collider shape and dimensions;
- uniform scale;
- local origin offset;
- local alignment quaternion; and
- `native` or `override` material mode.

The realized transform fits the visual mesh within its collision proxy. Native
mode retains embedded GSO/KuBasic materials and records the referenced image
digests. Override mode replaces them with the sampled `MaterialSpec`.

There is no fallback from an external asset to a procedural object. Unavailable
or changed assets make preflight or the affected render fail explicitly.

## Materials, Colors, Textures, and Physics

### Color

Canonical colors are linear RGBA values in `[0, 1]`. Sampling strategies are:

- fixed color;
- named palette;
- uniform HSV with independently bounded hue, saturation, and value; and
- temperature-aware neutral palettes.

The config can weight strategies and families. Alpha below one is legal only
for material families that permit transmission.

Base-color and emission images are color data. Metallic, roughness, height, and
normal maps are non-color data. This intent is part of the visual spec and is
applied explicitly in Blender.

### Texture

The first version supports:

- `solid`
- `noise`
- `checker`
- `wood`
- `marble`
- `speckle`
- `image`

Procedural textures store seed, colors, scale, detail, roughness, distortion,
and rotation. Image textures require a digest-pinned reference and may provide
base-color, roughness, metallic, normal, and height maps. A material family
declares which texture kinds it permits.

### Material families

The shipped config includes:

- metal
- rubber
- plastic
- ceramic
- glass
- wood
- stone

Each family provides bounded priors for metallic, roughness, specular, IOR,
transmission, effective density, friction, and restitution. The initial
physical priors are:

| Family | Effective density | Friction | Restitution |
| --- | ---: | ---: | ---: |
| metal | 55–100 | 0.15–0.45 | 0.10–0.35 |
| rubber | 18–35 | 0.65–0.95 | 0.55–0.85 |
| plastic | 12–30 | 0.25–0.55 | 0.20–0.55 |
| ceramic | 30–65 | 0.30–0.60 | 0.10–0.35 |
| glass | 35–70 | 0.20–0.50 | 0.05–0.25 |
| wood | 10–28 | 0.35–0.70 | 0.15–0.45 |
| stone | 45–85 | 0.55–0.90 | 0.02–0.20 |

Effective density is a dataset-scale prior in kilograms per cubic metre, not a
claim of real-world density. Coupled mode computes
`mass = clamp(effective_density * proxy_volume, 0.25, 4.0)` and records the
unclamped density and realized mass. Target push mass is sampled from the same
configured `[0.25, 4.0]` envelope.

The initial visual priors are:

| Family | Metallic | Roughness | IOR | Transmission |
| --- | ---: | ---: | ---: | ---: |
| metal | 0.85–1.00 | 0.12–0.45 | 1.45–2.50 | 0 |
| rubber | 0 | 0.65–0.95 | 1.20–1.60 | 0 |
| plastic | 0 | 0.20–0.60 | 1.35–1.60 | 0 |
| ceramic | 0–0.05 | 0.15–0.45 | 1.45–1.65 | 0 |
| glass | 0 | 0.02–0.18 | 1.45–1.55 | 0.85–1.00 |
| wood | 0 | 0.35–0.75 | 1.35–1.55 | 0 |
| stone | 0–0.10 | 0.55–0.95 | 1.40–1.70 | 0 |

Specular and emission remain configurable bounded values. The shipped config
uses non-emissive objects.

### Coupling modes

- `coupled`: sample visual and physical properties from one material family.
- `independent`: sample visual material and the existing mass, friction, and
  restitution ranges independently.
- `held_out`: use coupled sampling while excluding configured geometry,
  material, texture, or color-family combinations from selected training data.

Regardless of mode, `ObjectConfig` stores realized physics and `MaterialSpec`
stores realized appearance.

## Camera, Lighting, and Background

Camera sampling uses the existing half-sphere-shell concept with configurable
radius, elevation, focal length, sensor width, and `fixed` or `linear` motion.
The realized camera pose and look-at values are stored for every output frame.
Sampling must frame the full configured scene bounds with a safety margin, so
the result does not depend on branch-specific outcomes.

Lighting samples a configured studio-rig template. Each light records kind,
position, look-at target, color, intensity, width/height or spot parameters.
The default uses key, fill, and rim area lights with bounded pose, color
temperature, and intensity jitter.

Backgrounds are either a bounded neutral color or a digest-pinned HDRI. HDRI
identity, rotation, strength, and exposure are sampled once per instance and
shared across branches.

## Frame Mapping and Render Outputs

Physics logs retain every Bullet step. Render output uses scene frame rate:

`physics_step = output_frame_index * step_rate / frame_rate`

The existing divisibility constraint makes this mapping integral. The mapping
is stored in `VisualSceneSpec` and the render manifest. A scene with
`frame_range=[0, 2]`, `frame_rate=24`, and `step_rate=240` therefore renders two
frames from physics steps 0 and 10.

Each requested branch publishes Kubric's established encodings:

- RGB/RGBA: PNG
- segmentation: lossless palette PNG
- depth: floating-point TIFF
- normals: PNG
- forward/backward flow: 16-bit PNG plus hash-bound range metadata
- object coordinates: PNG

Segmentation uses stable logical object IDs mapped to explicit integer labels.
Both branches use the same mapping.

The shipped profiles are:

- `smoke`: 64×64, one Cycles sample, adaptive sampling off, denoising off, CPU,
  full requested layers.
- `production`: 256×256, 64 Cycles samples, adaptive sampling on, denoising on,
  full requested layers.

Both profiles use the same scene, frame mapping, Blender render seed, and
semantic visual spec. Profile settings produce distinct render identities.

## Pipeline and Artifact Layout

The existing deterministic attempt, QC, balancing, and physics publication
order remains:

1. validate ranges and optional asset catalogs;
2. sample a complete semantic instance;
3. generate factual/counterfactual physics;
4. evaluate physics QC;
5. journal accepted/rejected/error attempts;
6. balance candidates and assign grouped splits;
7. render the selected IDs for each requested profile;
8. validate render artifacts; and
9. atomically publish the render manifest and final dataset status.

Rendering selected candidates is the default. A library option and CLI flag may
render every accepted candidate for cache-building experiments.

Each rendered selected instance adds:

```text
instances/<instance_id>/
  appearance.json
  renders/
    smoke/
      render_manifest.json
      factual/
        rgba_*.png
        segmentation_*.png
        depth_*.tiff
        normal_*.png
        forward_flow_*.png
        backward_flow_*.png
        object_coordinates_*.png
        data_ranges.json
      counterfactual/
        ...
    production/
      ...
```

`appearance.json` is canonical and hash-bound by the instance manifest.
`render_manifest.json` binds:

- instance and profile identity;
- visual-spec hash;
- source pair manifest/hash;
- branch names;
- object/segmentation mapping;
- frame-to-physics-step mapping;
- Blender/container/runtime and device metadata;
- manifest/archive/image digests;
- output layer shapes, dtypes, encodings, file sizes, and SHA-256 values; and
- render status.

Renderer/runtime metadata is provenance, not authentication. Deterministic hash
equality is required only within the same pinned runtime and device contract.

## Balancing and Compositional Splits

`CandidateSummary` gains optional variation fields and grouping keys. Empty
fields are omitted from canonical serialization so old journals retain their
bytes.

The default balancing axes are:

- existing primary outcome category;
- existing hop bucket;
- collider family;
- visual source kind;
- material family; and
- texture family.

Round-robin selection works over populated strata. Config may remove axes when
the requested sample count is too small to cover the cross-product.

Split assignment treats no-leakage requirements as an equivalence relation.
Candidates are connected when they share:

- the existing factual contact-topology signature; or
- any configured holdout key, initially external asset ID or
  material/texture-family combination.

Connected components, not individual candidates, are assigned wholesale to
train, validation, or test. This prevents an asset from leaking through a
different topology and preserves topology grouping. Fractions remain
best-effort and the manifest reports actual counts and the largest component.

Explicit `held_out` combinations are excluded from training and assigned to
validation/test according to the configured policy. Empty or impossible
holdout partitions fail validation rather than silently falling back.

## Failure, Resume, and Publication Safety

- Configuration, schema, digest, and asset-catalog errors fail before PyBullet
  or Blender opens.
- Physics candidate failures keep the existing candidate-local error journal.
- Render work has a separate immutable journal keyed by instance and profile.
  Status is `pending`, `complete`, or `error`.
- A selected render failure produces overall status `render_incomplete`; valid
  physics and prior renders remain available.
- Physics-only CLI exits remain `0` for complete, `2` for capacity exhausted,
  and `1` for a batch-level error.
- Rendered CLI additionally returns `3` for `render_incomplete`.
- Resume requires matching config, semantic instance, pair, asset, visual, and
  profile digests.
- Increasing the physics attempt budget may change the selected set. Verified
  renders remain content-addressed cache entries; they are not deleted or
  treated as selected unless the new manifest names them.
- Every render is staged under the destination parent, validated in full, and
  atomically renamed. Existing complete output survives interruption, disk
  errors, Blender failures, and validation failures.
- Archive extraction rejects absolute paths, parent traversal, symbolic links,
  hard links outside the archive root, device entries, and unexpected
  top-level layouts.
- Missing, corrupt, or changed assets never trigger substitution.

## Testing Strategy

Development follows red-green-refactor in these layers:

1. Pure frozen-schema tests for validation, canonical JSON, immutability,
   visual/physics object correspondence, and malformed values.
2. Legacy tests proving physics-only `InstanceSpec.to_dict()`, instance IDs,
   artifacts, journals, and resumes remain unchanged when appearance is absent.
3. Seed-domain tests proving deterministic exact values and that changing one
   domain does not alter another.
4. Fixed-seed coverage and bounds tests for every shipped shape, source,
   material, texture, color, camera, light, and background family.
5. Cylinder/capsule core, PyBullet collision, oriented-AABB, Blender dimension,
   material, keyframe, and segmentation tests.
6. Material tests for coupled effective-density mass, clamping, physical/PBR
   bounds, independent sampling, and held-out combinations.
7. Local fixture manifests and archives for digest pinning,
   content-addressed caching, native/override materials, alignment, and hostile
   archive rejection. Tests require no network.
8. Balancing tests for variation axes and split tests proving no configured
   topology, asset, or combination key crosses a split.
9. Renderer unit tests for node specifications, image color-space assignment,
   scene construction, state replay, frame mapping, and shared branch visuals.
10. Docker one-frame smoke renders for every procedural geometry,
    material/texture family, and the local manifest-backed fixture.
11. End-to-end tiny paired datasets verifying all layers, segmentation IDs,
    finite arrays, manifests, source hashes, atomic replacement, failure
    preservation, and resume.
12. Existing intervention, Kubric core, demo, documentation, and offline-import
    suites.
13. A measured smoke batch recording physics acceptance, render throughput,
    storage per layer, render failures, strata/split distributions, and
    deterministic rerender hashes under one pinned environment.

Distribution tests use deterministic fixtures and exact seeded expectations,
not probabilistic pass thresholds.

## Documentation

`docs/trajectory_interventions.md` gains:

- the appearance schema and trust boundary;
- geometry size conventions;
- material coupling semantics;
- YAML examples for procedural and manifest-backed scenes;
- smoke/production render commands;
- layer encodings and artifact layout;
- compositional split behavior;
- asset-cache and resume rules; and
- measured smoke evidence after implementation.

New modules follow the repository's structured module-docstring convention:
purpose, public API, dependencies/data flow, and trust boundary.

## Compatibility and Trust Boundaries

- Existing physics-only configs omit the new YAML sections and retain their
  current behavior.
- Existing cube/sphere schemas and artifacts remain readable.
- The deterministic forked-rack demo remains a separate fixed visual contract;
  it is regression-tested but not migrated to random appearance.
- New shape enum values are backward-compatible; target validation continues
  to accept only cube and sphere.
- Optional variation fields are omitted when empty so old canonical journal
  bytes remain stable.
- External visual meshes do not claim mesh-accurate collision.
- Pair hashes and render hashes detect internal drift but do not authenticate
  the producer.
- Cycles determinism is scoped to a pinned Blender/container/device contract.

## Completion Criteria

The feature is complete when:

- the shipped deterministic sample window exercises every configured
  procedural geometry, material, and texture family;
- a local pinned manifest fixture exercises the asset-backed path;
- coupled, independent, and held-out modes pass their exact tests;
- factual/counterfactual renders share identical visual specs, camera, lights,
  background, object/segmentation mapping, and pre-intervention poses;
- only logged branch physics can produce post-intervention pixel differences;
- smoke and production profiles publish and validate all requested layers;
- no configured topology, asset, or held-out combination leaks across splits;
- interrupted and failed renders resume without changing valid physics or
  completed render bytes;
- existing core, intervention, demo, docs, and offline-import tests pass;
- a fresh measured smoke report records throughput, storage, failures,
  distributions, environment identity, and rerender hashes; and
- generated datasets and media remain untracked.
