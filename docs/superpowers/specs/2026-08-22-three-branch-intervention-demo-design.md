# Three-branch trajectory intervention demo

## Goal

Produce one realistic-looking comparison video for the trajectory intervention
module. The video shows the same deterministic scene in three synchronized
branches:

1. **Normal** — the kinematic target follows its factual path.
2. **Trajectory changed** — the same target changes course only inside the
   intervention window and creates a collision.
3. **Target removed** — the target follows the common prefix, then is removed
   from the physics world at the start of the same intervention window.

The demo is a presentation artifact, not a new dataset recipe. In particular,
`delete_target` will not be added to the public intervention schema, QC rules,
or artifact format.

## Physics and data flow

- Define one `SceneConfig`, target identifier, seed, factual path, and
  intervention window.
- Generate the normal and trajectory-changed branches through the public
  `generate_paired_instance()` pipeline. Extract the pair ground truth through
  the public graph API.
- Generate the removal branch in a demo-only runner using a fresh
  `KinematicSimulator` initialized from the same scene config and seed. It
  follows the factual path before the intervention and physically removes the
  target at the first intervention step. Remaining dynamic objects continue to
  simulate normally.
- Preserve fixed state-array dimensions for replay and write a separate Boolean
  presence mask. After removal, the target row retains its last finite pose but
  the mask is false; renderers must use the mask rather than treating that row
  as a live body.
- Persist branch state logs, presence masks, contact records, the public pair
  ground truth, and a compact summary. Assert that all three branches share an
  exact pre-intervention prefix for non-target bodies.

The third branch is explicitly marked `demo_only_removal_v1` so it cannot be
mistaken for a canonical paired dataset artifact.

## Visual design

Replay the exact recorded poses in Kubric's Blender renderer. Use only
procedural assets so the demo is deterministic and does not need external GSO
or HDRI downloads:

- a rounded, bevelled wooden pusher whose visible bounds match the target's box
  collider;
- glossy billiard-style balls whose radii match their sphere colliders;
- a dark felt tabletop, studio backdrop, soft area/key lights, contact shadows,
  depth of field, and Cycles denoising;
- a fixed camera shared by all branches.

Render one source MP4 per branch, then compose them with FFmpeg into a single
1920x720 H.264 video. The panels are synchronized and labelled `NORMAL`,
`TRAJECTORY CHANGED`, and `TARGET REMOVED`. A shared timeline marks the
intervention start. The changed branch flashes the contact and identifies the
affected object; the removed branch shows a short removal cue. The ending card
summarizes graph delta, hard/soft affected sets, and propagation path.

Target duration is 8-12 seconds at 24 fps. The output directory contains the
combined MP4, three source MP4s, logs, summary JSON, and optional `.blend`
files.

## Components

1. `scripts/demo_collision_intervention.py`
   - replace duplicated raw-PyBullet factual/counterfactual logic with the
     public intervention pipeline;
   - add the demo-only physical removal branch;
   - write deterministic replay inputs and summary metadata.
2. `scripts/render_demo_branches_blender.py`
   - build the procedural realistic scene;
   - honor presence masks and branch event metadata;
   - render synchronized source videos.
3. A small composition step, called by `run_demo.sh intervention`, that produces
   the labelled three-panel MP4 with FFmpeg.
4. Documentation and tests covering the command, outputs, and trust boundary.

## Error handling

- Fail before rendering if branch lengths, object order, step rate, prefix, or
  presence-mask shape disagree.
- Fail if the changed branch does not create the intended target contact or if
  its public ground truth does not mark the expected affected object.
- Fail if the removed target remains in any post-removal contact.
- Keep the raw logs when Blender rendering fails so physics results remain
  inspectable.
- Report missing Docker/Blender as an explicit incomplete demo, rather than
  silently claiming success with only schematic output.

## Verification

- Unit tests for deterministic three-branch generation, exact common prefix,
  removal timing, absence of post-removal target contacts, and expected graph
  ground truth.
- Offline renderer tests for state/presence validation, quaternion conversion,
  material/object construction helpers, and command composition.
- Run all extension tests under the `thesis` environment with warnings treated
  as errors, plus the upstream Scene/PyBullet tests.
- Render the final video, inspect representative frames before/during/after the
  intervention, and validate it with `ffprobe` (H.264, 1920x720, 24 fps,
  expected duration and frame count).

## Non-goals

- Adding scanned assets or network downloads.
- Adding `delete_target` to dataset generation.
- Claiming the demo-only removal branch has public `GroundTruth` semantics.
- Running Milestone F large-scale training or baseline evaluation.
