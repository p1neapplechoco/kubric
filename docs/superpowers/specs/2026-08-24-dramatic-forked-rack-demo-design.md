# Dramatic Forked-Rack Intervention Demo Design

**Status:** Approved in conversation on 2026-08-24

## Goal

Replace the current four-object billiards demo with a deterministic eleven-object
forked-rack scene whose three synchronized branches remain causally legible:

- `normal` produces a small two-to-three-edge chain;
- `trajectory_changed` produces a visibly larger seven-to-nine-edge chain that
  reaches at least six downstream balls;
- `target_removed` physically removes the target before the intervention step
  and produces no post-removal dynamic chain.

The final comparison remains realistic, inspectable, and generated entirely from
the public factual/counterfactual intervention pipeline plus the existing
demo-only removal runner. Module documentation is improved primarily through
structured Python docstrings, with only a short external module overview.

## Approved Product Decisions

- Visual metaphor: billiards chain reaction.
- Layout: forked rack.
- Total physics objects: eleven, including the floor and target.
- Contrast: normal has a small chain; changed has a substantially larger chain.
- Final duration: approximately eleven seconds.
- Documentation emphasis: module and public-symbol docstrings.
- Architecture: one pure-Python shared scene specification consumed by the
  generator, Blender renderer, and compositor.

## Scope

### In scope

- A fixed deterministic forked-rack physics fixture.
- Nine numbered lacquer balls, one wooden kinematic target, and one static floor.
- A shared, offline-safe demo specification with canonical version and digest.
- A 200-step replay bundle for all three branches.
- Dynamic object construction in the Blender renderer instead of four-object
  constants.
- Compact normal/changed chain overlays and an ending summary that remains
  readable with many affected objects.
- Structured module docstrings across `interventions/` and the main trajectory
  scripts, plus missing public class/function docstrings.
- Updated workflow documentation, tests, Docker smoke render, complete render,
  media validation, and visual spot checks.

### Out of scope

- A generic serialized scene-manifest standard.
- A public target-deletion intervention recipe.
- Changes under `kubric/`.
- Downloaded, scanned, or network-fetched visual assets.
- Large-scale dataset generation or any Milestone F training claim.
- Preservation of byte-identical output from the superseded four-object demo.

## Scene Specification

Create `scripts/trajectory_demo_spec.py` as an import-safe, pure-Python module.
It must not import Kubric, PyBullet, Blender, or the intervention runtime.

The module exposes frozen `DemoObjectSpec` and `DemoSceneSpec` values and one
canonical `FORKED_RACK_SPEC`. The canonical object order is:

1. `floor`
2. `breaker`
3. `rack_01`
4. `rack_02`
5. `rack_03`
6. `rack_04`
7. `rack_05`
8. `rack_06`
9. `side_01`
10. `side_02`
11. `target`

The nine balls form two groups:

- main group: `breaker` plus `rack_01` through `rack_06`;
- side group: `side_01` and `side_02`.

The factual target path reaches the side group and starts a small chain. The
fixed seed-0 `create_collision` perturbation bends toward `breaker`, which
transfers momentum into the six-ball rack. Numerical coordinates, masses, and
the final intervention magnitude are committed as fixed spec values after
test-driven calibration. No parameter search or random layout generation occurs
when the demo is run.

The scene uses:

- seed `0`;
- 200 Bullet steps;
- `frame_range=(0, 20)`, `frame_rate=24`, and `step_rate=240`;
- intervention window `[40, 160)`;
- zero gravity; target and balls use friction `0.02` and restitution `0.65`,
  while the static floor uses zero friction and restitution;
- a 200-sample `XYZ + WXYZ` commanded target path.

The spec provides canonical JSON-safe data and a SHA-256 digest. The digest
covers object order, collider geometry, physics properties, initial poses,
timing, path parameters, intervention parameters, visual roles, ball numbers,
and colors. It excludes generated replay states and output paths.

## Architecture and Data Flow

### Shared spec

`scripts/trajectory_demo_spec.py` is the single source of truth for object IDs,
geometry, grouping, timing, and visual roles. Generator and renderer adapters
convert its backend-neutral values to their local types. The compositor imports
only digest/group helpers from this pure module.

### Physics generator

`scripts/demo_collision_intervention.py` converts the shared spec to
`ObjectConfig`, `SceneConfig`, `Intervention`, and the factual path. It continues
to call public `generate_paired_instance()` and
`extract_pair_ground_truth()` for `normal` and `trajectory_changed`.

The removal branch remains a fresh matching Bullet world. It replays the exact
factual prefix, removes `target` before step 40 physics, retains the last finite
target row in the fixed-shape array, and makes the presence mask authoritative
from step 40 onward.

Outcome validation runs before any bundle write. The normal and changed result
must satisfy the behavior contract below; removal must contain no post-removal
target contact and no post-removal dynamic chain.

### Replay bundle

Each state array has shape `(200, 11, 13)` and each presence array has shape
`(200, 11)`. The existing filenames and three branch names remain unchanged.

`summary.json` gains one exact top-level `demo_spec` object:

```json
{
  "object_count": 11,
  "sha256": "<64 lowercase hex characters>",
  "source_frames": 200,
  "version": "forked_rack_v1"
}
```

The branch `contact_pairs` maps remain the canonical source for unique dynamic
pair counts. A “hit” in overlays means one unique non-floor contact pair, not a
raw PyBullet contact point or the number of sustained-contact steps.

Bundle publication retains same-parent temporary files and atomic leaf
replacement. Every consumer validates the exact object order and spec identity
before expensive work.

### Blender renderer

`scripts/render_demo_branches_blender.py` removes the fixed four-object collider
map. It iterates over `FORKED_RACK_SPEC.objects`, builds matching collider
parents, and attaches visuals by role:

- target: rounded procedural wood;
- balls: glossy billiard lacquer with deterministic color, stripe/solid style,
  and number badge;
- floor/table: procedural felt;
- rails/backdrop/lights: shared presentation-only assets outside the simulated
  collision area.

The camera pulls back enough to contain the complete fork and the largest
expected displacement in all branches. Camera, lighting, Cycles adaptive
sampling, denoising, and render seed remain identical across branches. Presence
keyframes hide the target and all collider-parented decoration in the removed
branch.

All requested branch videos are staged and verified before final publication.
Offline module import continues to work without Kubric, PyBullet, or Blender.

### FFmpeg compositor

`scripts/compose_intervention_demo.py` keeps its strict H.264, CFR-24, per-frame
PTS, source synchronization, path-alias, hardlink, and atomic-publication
checks. It additionally validates `demo_spec` against the shared module.

Event overlays become:

- normal: `SMALL CHAIN → SIDE 01` with its first normal-only target event;
- changed: `LARGE CHAIN → BREAKER` with the target-to-breaker changed event;
- removed: existing target-removal cue.

The ending hold uses compact, bounded text:

1. graph delta counts: added, removed, changed;
2. hard/soft affected object totals;
3. only the deterministic longest propagation path and its hop count.

The full list of affected objects and all propagation paths remain in
`summary.json`; they are not concatenated into an overflowing video line.

With 200 source frames, one second of initial hold, and two seconds of ending
hold, the final video contains 272 frames and lasts approximately 11.333333
seconds at 24 fps.

## Behavior Contract

All counts below refer to unique dynamic non-floor contact pairs.

### Normal branch

- contains between two and three unique dynamic pairs;
- contains `target|side_01`;
- reaches both side-group balls;
- contains no main-group contact;
- remains deterministic and finite for all 200 steps.

### Trajectory-changed branch

- contains between seven and nine unique dynamic pairs;
- contains `breaker|target`;
- reaches at least six distinct main-group balls through dynamic contacts;
- contains no side-group contact;
- has at least five more unique dynamic pairs than normal;
- differs from the factual commanded path only inside `[40, 160)`.

### Target-removed branch

- has an exact state prefix matching normal before step 40 for all objects;
- marks target present before step 40 and absent from step 40 onward;
- contains no target contact at or after step 40;
- contains no dynamic contact pair at or after step 40;
- retains finite fixed-shape states and `demo_only_removal_v1` metadata.

### Ground truth

- is recomputed from the public normal/changed pair before bundle publication;
- includes every state-divergent downstream object in either fork;
- contains a target-rooted propagation path for each hard-affected object;
- has no overlap between hard and soft affected sets.

## Validation and Failure Safety

The shared dataclasses reject duplicate/empty IDs, unsupported shapes or roles,
non-finite/non-positive geometry and masses, invalid group membership, a target
or floor count other than one, a ball count other than nine, non-integral timing,
and an intervention window outside the replay.

Generator validation rejects outcome drift before creating or replacing bundle
files. Bundle readers reject a missing, unexpected, malformed, or mismatched
`demo_spec`. Renderer validation rejects array/object-order/presence mismatches
before Blender initialization. Compositor validation rejects summary/spec/media
mismatches before FFmpeg execution and revalidates output identity immediately
before publication.

Existing trust boundaries remain explicit:

- public normal/changed pair: canonical public intervention pipeline;
- removed branch: visualization-only `demo_only_removal_v1`;
- paired artifacts: `caller_trusted_unattested_logs_v1` where applicable;
- spec SHA-256: drift/integrity signal, not producer authentication.

## Documentation Design

Documentation is code-adjacent first.

Each module in `interventions/*.py` and each main trajectory script receives a
structured module docstring covering:

- purpose and non-responsibilities;
- principal public API;
- input/output or data flow;
- important dependencies and deferred-import behavior;
- determinism, provenance, or trust-boundary notes where applicable.

Every exported public class/function and each script entry helper touched by
this work must have a concise behavior-oriented docstring. Private helper
docstrings are added only where invariants or transformations are not clear from
the name and types.

`docs/trajectory_interventions.md` gains a compact module-reference table and
the new eleven-object demo contract. It links readers to code docstrings instead
of duplicating detailed symbol documentation.

## Test and Verification Strategy

Development follows red-green-refactor in these layers:

1. Pure spec tests for exact IDs, roles, timing, canonical serialization,
   deterministic hash, immutability, and invalid values.
2. Real deterministic physics tests for the normal, changed, removal, ground
   truth, common-prefix, and contact-count contracts.
3. Bundle tests for `(200, 11, 13)` / `(200, 11)`, spec identity, deterministic
   bytes, stale-spec rejection, and failed-write preservation.
4. Offline renderer tests for dynamic object creation, material/decor roles,
   camera containment, presence visibility, replay preflight, and unchanged
   lazy imports.
5. Synthetic FFmpeg compositor tests for both chain cues, compact ending text,
   272 frames, 11.333333-second duration, stale-spec rejection, and existing
   source/output safety regressions.
6. AST-based documentation tests requiring structured module docstrings and
   docstrings on the declared public API surface without importing heavy
   backends.
7. Full extension and upstream Scene/PyBullet tests under the `thesis` Python
   with warnings treated as errors.
8. Docker one-frame smoke for all branches, followed by a complete Cycles
   render, FFprobe/decode validation, SHA-256 recording, and visual inspection
   before intervention, at the small chain, at the large chain, after removal,
   and during the ending summary.

Physics calibration is complete only when the committed fixed values pass the
behavior contract repeatedly with exact deterministic outputs. Render
calibration is complete only when all eleven objects remain visible and the two
forks are visually distinguishable without label overlap or clipping.

## Compatibility and Migration

The CLI modes, branch names, canonical output directory, state layout, and final
MP4 filename stay unchanged. The replay object count, frame count, summary
schema, media duration, and generated hashes intentionally change.

Old four-object bundles do not silently render with the new code. They fail
preflight because they lack `demo_spec` or carry an incompatible object order.
Users regenerate the bundle with `./run_demo.sh intervention-physics-only` or
the full workflow.

## Completion Criteria

The feature is complete when:

- all behavior, shape, spec-identity, docstring, renderer, and compositor tests
  pass under `thesis` with `-W error`;
- upstream Scene/PyBullet tests pass;
- no tracked file under `kubric/` changes;
- a complete 272-frame comparison MP4 is H.264/yuv420p, 1920x720, CFR 24 fps,
  approximately 11.333333 seconds, and decodes without errors;
- inspected frames demonstrate the approved small/large/no-chain contrast;
- updated docs state that removal remains demo-only and Milestone F remains
  incomplete;
- generated media stays untracked.
