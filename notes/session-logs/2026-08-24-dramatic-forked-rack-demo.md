# Handoff: dramatic forked-rack three-branch demo

Verification completed: 2026-08-25 (Asia/Ho_Chi_Minh). The filename retains the
requested 2026-08-24 session date.

## Repository state

- Worktree:
  `/home/pineapple/Desktop/projects/kubric/.worktrees/dramatic-forked-rack-implementation`
- Branch: `feature/dramatic-forked-rack-implementation`
- Verified implementation HEAD before this handoff-log commit:
  `ae021dea92bfcdda89181316e9ca360f317053c7`
- The final documentation commit necessarily follows the implementation HEAD.
  Its hash is reported outside this committed log rather than attempting a
  self-referential hash.
- No file under `kubric/` was changed.
- Generated replay arrays and media remain ignored and uncommitted.

## Completed system and architecture

The demo is an inspectable, deterministic three-branch forked-rack collision:
`normal`, `trajectory_changed`, and `target_removed`. Its data flow and
module boundaries are:

1. `scripts/trajectory_demo_spec.py` defines the immutable, canonical scene
   payload and SHA-256 identity.
2. `scripts/demo_collision_intervention.py` creates the canonical
   normal/changed pair through the public physics and ground-truth pipeline,
   adds the narrowly scoped removal replay, and publishes a digest-bound replay
   bundle.
3. `scripts/render_demo_branches_blender.py` validates the spec, summary,
   replay arrays, and presence masks before rendering logged collider poses.
4. `scripts/compose_intervention_demo.py` validates summary, event, and source
   media contracts, then composes on an exact 24 fps output-frame lattice.

This is the trust path:

```text
immutable spec -> paired physics/replay bundle -> digest-bound Blender renderer
               -> exact-frame-lattice FFmpeg compositor
```

Module docstrings and `docs/trajectory_interventions.md` document the public
APIs, dependencies, invariants, exclusions, and trust boundary for every
`interventions` and demo module. Digests and manifests establish internal
identity and consistency; they are not producer signatures or external
attestation.

## Canonical fixture and physics evidence

- Spec version: `forked_rack_v1`
- Spec SHA-256:
  `792a36f2376cf7acf994819d885ec8bd0babf1d273dc5369892fd98a4017b977`
- Seed: `0`
- Source steps/frames: `200`
- Bullet step rate: `240` Hz
- Replay rate: `24` fps
- Intervention window: `[40, 160)`
- Exact object order: `breaker`, `floor`, `rack_01`, `rack_02`,
  `rack_03`, `rack_04`, `rack_05`, `rack_06`, `side_01`,
  `side_02`, `target`

All three state arrays have shape `(200, 11, 13)`; all three presence arrays
have shape `(200, 11)`. For the removed branch, target presence is
authoritative and false from step 40. Its retained target row is frozen at the
last finite pre-removal state.

Observed calibrated outcomes:

- `normal`: exactly 2 dynamic contact pairs, the small side chain
  `side_01|target` and `side_01|side_02`.
- `trajectory_changed`: exactly 9 dynamic contact pairs; the first main
  contact is at step 88, and cumulative contacts reach all seven main balls at
  step 160.
- `target_removed`: no target presence, contacts, or dynamic-body motion at
  or after step 40.
- Canonical-pair graph delta: `added=15`, `removed=4`, `changed=0`.
- Affected sets: `hard=9`, `soft=0`.

## Full workflow, render, and visual validation

The complete supported workflow was run as:

```bash
./run_demo.sh intervention
```

It exited 0 and rendered all three branches with 64 Cycles samples using Docker
image
`sha256:cc4fb8a65172cc1da81dd0ce04bfe47c2405db2e357d842c17583400079d1a80`.
Blender printed `Error: Not freed memory blocks: 3, total unfreed memory 0.003777 MB`
during shutdown. The render had already completed successfully; this diagnostic
was nonfatal.

Visual validation established:

- before intervention, the target, all nine balls, and floor/table are visible
  and unclipped in every panel;
- the common prefix is visually aligned;
- normal produces the intended small side chain;
- changed first contacts the main chain at step 88 and reaches all seven main
  balls at step 160, where six are visibly displaced;
- removed hides the target at step 40 only and has no post-removal dynamics;
- no teleport, collider/decor detachment, label overlap, or panel
  desynchronization was observed;
- maximum observed one-frame displacement is 3.07 cm;
- ending metadata is readable and remains inside the lower band.

A real visual QA pass found that rounded-time cue enables were one output frame
late. The correction chain is:

- `b965116` — `fix(demo): align cues to output frames`
- `d315435` — `fix(demo): enforce 24fps cue lattice`
- `ae021dea` — `test(demo): assert cue window endpoints`

The corrected inclusive decoded-frame windows are removal `64-81`, changed
`112-133`, and normal `117-138`; final metadata begins at frame `224`.
Decoded integration tests inspect the frame before, first, last, and frame after
each cue. The compositor was rerun at the final implementation HEAD with:

```bash
python -m scripts.compose_intervention_demo
```

No Blender rerender was needed because the commits after the successful full
render changed only compositor code and tests.

## Media contract

Each branch is H.264/yuv420p, 640x540, exact CFR `24/1`, 200 frames, and
8.333333 seconds. The composite is H.264/yuv420p, 1920x720, exact CFR `24/1`,
272 decoded frames, and 11.333333 seconds. Full `ffmpeg -xerror` decode of all
four videos exited 0.

## Artifact hashes

These hashes were checked against
`output/demo_collision_intervention/` at the implementation HEAD:

| Artifact | SHA-256 |
| --- | --- |
| `summary.json` | `24db6d3be0265e667bb98e957aa98f96c8db9a75c31ebfaa27287866ab8a5ec4` |
| `contacts.json` | `f32f77358fa9a45643f5b86cc79a4952dfa15905d0784f93e363d5eec9ed97ad` |
| `normal_states.npy` | `f247cf6fb3a16ead4d1d46e11e71ffd9760b6711056c4333319318234db99483` |
| `normal_presence.npy` | `34fa576e1e4a75f11d90ba6d33877282b8a4c59259a2457a2d40f4bfc9471ccb` |
| `trajectory_changed_states.npy` | `23b383fd93704889cf43213aa96d6dbd7167ade511fade282731f048bc5a96c2` |
| `trajectory_changed_presence.npy` | `34fa576e1e4a75f11d90ba6d33877282b8a4c59259a2457a2d40f4bfc9471ccb` |
| `target_removed_states.npy` | `8fa7ac8529c79e961c3e7ee2be19fa6e818441730413a5944db850a9a896a4c8` |
| `target_removed_presence.npy` | `00d23c66e23bc4b6a98f47d519a8db46be8e3b61df41a35349315073131b854c` |
| `normal_blender.mp4` | `9aa973fa211776adbed699ddb8dfbcc5370728bba105e6416c8f1d8fa2f09373` |
| `trajectory_changed_blender.mp4` | `ef052fb4f7b760f7067546c9e264238ea1bf7e2b7b816ae2fa25ba53cd2ddaef` |
| `target_removed_blender.mp4` | `1cd5187e8bbd505f82e2a5eee0076457e389e8e2f837441dc25b7d1c1ae01cd9` |
| `trajectory_intervention_demo.mp4` | `40323ab0ec96992305365326ce7eb9ad03d9f143c63c175a52b14996d94e54be` |

## Verification evidence

Fresh full implementation-head verification used the `thesis` Python with
warnings as errors and an isolated Matplotlib config:

```text
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest -q \
  tests test/test_scene.py test/test_pybullet.py

922 passed in 66.50s
```

Focused implementation-head evidence also includes:

- compositor suite: 83 passed;
- generator plus compositor suites: 122 passed;
- module-documentation plus offline-import suites: 45 passed.

The parent handoff will rerun compile, shell-syntax, and diff checks after the
documentation commit. No post-commit outcome for those checks is claimed here.

## Implementation landmarks

- `b32031d` through `2f93e57`: immutable forked-rack spec and validation.
- `300c5fd`, `37e57ba`: deterministic chain generator and object-order test.
- `00135fb`, `e667ab4`, `6fb6b83`: removal/contact and spec-digest binding.
- `e139900`, `e2499de`: eleven-object Blender renderer and hardening.
- `5a96fb7`, `8566edd`: chain annotations and compositor-summary binding.
- `880bf2d`, `d9ede21`: module contracts and enforcement.
- `b965116`, `d315435`, `ae021dea`: exact cue-frame correction and decoded
  endpoint coverage.

## Trust boundaries and remaining scope

- Only `normal` and `trajectory_changed` form the canonical paired artifact.
- `target_removed` uses `demo_only_removal_v1`. Its presence mask is
  authoritative, and the branch is neither a public dataset recipe nor
  training data.
- Spec digests, payload hashes, and manifests detect internal drift and
  corruption; they do not authenticate the producer.
- Generated media remains ignored and must not be staged.
- Milestone F remains incomplete. This demo establishes neither scale
  generation, baseline training, nor a training handoff.

## Next steps

1. Preserve the ignored artifact outside Git if a durable media archive is
   required, recording the hash table above.
2. Rerun the parent post-commit compile, shell-syntax, and diff gates.
3. If Milestone F is pursued, run a measured, visually audited dataset batch at
   the intended scale before defining baseline training and evaluation.
4. Do not make a training-readiness or Milestone F completion claim from this
   demo alone.
