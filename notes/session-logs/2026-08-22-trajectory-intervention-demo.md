# Handoff: trajectory interventions + realistic three-branch demo

Last verified: 2026-08-23 (Asia/Ho_Chi_Minh)

## Repository state

- Repository: `/home/pineapple/Desktop/projects/kubric`
- Branch: `feature/trajectory-interventions`
- Verified HEAD before this handoff commit: `55560d8f611c4ab2d89c467192b8cd4a9a9e7fa9`
- Base/main: `2cace2d1f1c1784ba60029207a92a825e90cd685`
- Conda Python: `/home/pineapple/miniconda3/envs/thesis/bin/python` (Python 3.11)
- Editable install: `kubric==0.0.0` points at this checkout; `import interventions`
  succeeds from `/tmp`.
- No tracked file under `kubric/` was modified.
- Generated replay/video output is untracked and must not be staged.

## What is complete

The branch implements Milestones A-E of the trajectory-intervention extension:

- validated backend-neutral scene/intervention schemas and trajectory builders;
- causal state/contact logging and temporal graph extraction;
- PyBullet kinematic simulator with explicit `push_mass` semantics;
- deterministic factual/counterfactual twin generation with exact common-prefix
  and provenance validation;
- immutable pair artifacts, public readers, QC, resumable batch generation,
  balancing, and topology-grouped splits;
- one deterministic realistic demo with three synchronized branches:
  `normal`, `trajectory_changed`, and `target_removed`.

The canonical pair uses public
`generate_paired_instance()` + `extract_pair_ground_truth()`. The third branch
uses a fresh matching Bullet world, replays the exact prefix, physically removes
the target before step 24 physics, retains its last finite state row, and sets
its presence mask false from step 24 onward.

## Important modules and entry points

- `interventions/schema.py`: `SceneConfig`, `ObjectConfig`, `Intervention`.
- `interventions/trajectory.py`: factual paths and bounded perturbations.
- `interventions/kinematic_simulator.py`: isolated Bullet execution.
- `interventions/logging.py`: immutable state/contact records.
- `interventions/twin_runner.py`: public deterministic pair runner.
- `interventions/graph_extraction.py`, `tagging.py`: oracle graph delta,
  affected sets, propagation paths, tags.
- `interventions/dataset.py`: QC, journaling, publication, balance/splits.
- `scripts/demo_collision_intervention.py`: deterministic 120-step demo and
  three-branch replay bundle.
- `scripts/render_demo_branches_blender.py`: exact replay through procedural
  Blender/Cycles scene; presence-aware target visibility.
- `scripts/compose_intervention_demo.py`: strict FFprobe preflight and FFmpeg
  1920x720 comparison compositor.
- `run_demo.sh intervention`: generate -> render three branches -> compose.
- `run_demo.sh intervention-physics-only`: replay bundle without Docker.

Key demo commits:

- `2a2ea35`, `a8a0a18`: public deterministic pair fixture.
- `b8e8614`, `40eb655`: real target-removal replay and immutable result.
- `fe7da53`, `61519a4`, `595cab0`: deterministic replay bundle.
- `79c8368`, `18435b4`, `239fd40`, `ebe902b`, `11b8716`: realistic,
  presence-aware Blender renderer and preflight hardening.
- `a815e5e`, `27bcea1`, `55560d8`: three-panel compositor and safety hardening.
- `dde86b4`: shell workflow, dependency declaration, and user docs.

## Demo fixture and observed physics

- Seed: `0`.
- 120 Bullet steps at 240 Hz; render rate 24 fps (5 seconds of source video).
- Intervention window: `[24, 96)`.
- Canonical object order: `floor`, `lower_ball`, `target`, `upper_ball`.
- `normal`: no dynamic-object contact.
- `trajectory_changed`: `target|upper_ball` contacts at steps 37, 38, 39.
- `target_removed`: target presence becomes false at step 24 and it has no
  contact at or after removal.
- Ground truth: `hard_affected=["upper_ball"]`, `soft_affected=[]`, propagation
  path `target -> upper_ball`.

Replay bundle hashes after deterministic regeneration:

```text
8315babd67c682b822d818a914dfc50d0e74296d4c09df29ae34f525ee25bbe2  normal_states.npy
697caddb61023bce048d300e3a8f43445d90b01f3ae2a4f96561b5be3a210c07  trajectory_changed_states.npy
ca033bf12f6e9bb936df086b7bde2cc9e3a74ebf413d578e50fb56ffa1abf013  target_removed_states.npy
fa1d48ff0d3fac815e2da27ed63e73795bd6b088a4d59e0a29031249dd68edb4  contacts.json
bbdc338b7fc57281c7c7b8a38337474444140543206e82ea813b43bce97ef41e  summary.json
```

## Rendered artifact

Canonical output:

`output/demo_collision_intervention/trajectory_intervention_demo.mp4`

Final media contract:

```text
codec=h264
pixel_format=yuv420p
resolution=1920x720
frame_rate=24/1
frames=192
duration=8.000000 seconds
size=165573 bytes
sha256=d5b8c77da2f4d70a2a9404d847390342ceb0f81bfda24ffc2eec9029a410bd2e
```

Branch render hashes:

```text
7d8a2e73b71816d890039e5b0b81b9b10f94d3393d6d2fb7c9b64dbeddea9dec  normal_blender.mp4
aa0efd18e2bad2e2aa7d3d97b334c8a5beb9800b5d46b6d50d2f34e83a9b1c8f  trajectory_changed_blender.mp4
a4e6fc4e4ea36be2dd50b8ffa2c477f1687efba2c1c76496254fac7960d2f4bd  target_removed_blender.mp4
```

Each source is H.264/yuv420p, 640x540, CFR 24 fps, 120 frames, and exactly
5.000000 seconds. Rendering used cached Docker image
`kubricdockerhub/kubruntudev` with image ID
`sha256:cc4fb8a65172cc1da81dd0ce04bfe47c2405db2e357d842c17583400079d1a80`.
Full-frame FFmpeg decode completed without errors.

Visual checkpoints were inspected at 1.500, 2.083, 2.583, and 6.500 seconds:

- all three panels have an identical visible prefix;
- the wooden target disappears only in `TARGET REMOVED` after 2.000 s;
- `TRAJECTORY CHANGED` visibly contacts the red upper ball at 2.542 s;
- normal remains collision-free and the changed upper ball has visibly moved by
  the ending hold;
- labels, event cues, timeline, graph delta, affected set, and propagation path
  are readable without overlap;
- no inspected checkpoint shows clipping, teleportation, or desynchronization.

The output directory may also contain legacy `factual*`/`counterfactual*`
artifacts from the superseded two-branch demo. They are not canonical inputs or
outputs of the current workflow and should be ignored or cleaned separately.

## Verification evidence

Executed against HEAD `55560d8` before adding this log:

```text
MPLCONFIGDIR=/tmp/kubric-mpl thesis/bin/python -W error -m pytest -q tests
695 passed in 54.88s

MPLCONFIGDIR=/tmp/kubric-mpl thesis/bin/python -W error -m pytest -q \
  test/test_scene.py test/test_pybullet.py
9 passed in 4.82s

tests/test_compose_intervention_demo.py
46 passed

three demo suites during Task 5 review
139 passed
```

Also passed:

- `python -W error -m compileall -q interventions scripts tests` with a
  task-specific pycache path;
- `bash -n run_demo.sh`;
- `git diff --check`;
- `./run_demo.sh intervention-physics-only` using the resolved `thesis` Python;
- exact FFprobe CFR/PTS validation and full FFmpeg decode of all final media;
- task-level spec and code-quality reviews for generator, removal, bundle,
  renderer, compositor, and shell/docs. The final compositor re-review was
  APPROVED with no Critical/Important finding.

The complete render and compose stages were executed successfully with the same
commands wired by `run_demo.sh`. The wrapper's physics-only path was executed
directly; the full wrapper was not rerun after the already-complete 64-sample
render solely to avoid repeating the long deterministic Cycles job.

## Safety and trust boundaries

- `target_removed` is visualization-only and marked
  `demo_only_removal_v1`; it is not a public dataset recipe and must not be
  presented as canonical training ground truth.
- Pair artifacts use `caller_trusted_unattested_logs_v1`: structural,
  provenance, contact, graph, and payload hashes are checked, but hashes are not
  producer signatures and do not attest external in-memory log origin.
- The compositor rejects symlink components, canonical aliases, hardlinks to
  source videos, non-H.264/non-CFR/non-24-fps sources, irregular per-frame PTS,
  mismatched source metadata, and revalidates the output target immediately
  before atomic publication.
- Demo visuals are procedural; no downloaded/scanned assets were introduced.

## Remaining work for the next agent

Milestone F is intentionally not complete. Do not claim training readiness from
this demo. A next agent should, on the exact commit it intends to use:

1. Run a fresh post-hardening batch at the intended scale and record config
   digest, seed, attempt budget, QC rejection distribution, category/hop
   distribution, unique topology count, throughput, and storage use.
2. Visually audit a stratified sample of accepted pairs, including changed,
   mixed, and state-only categories.
3. Decide retention/resume policy and whether single-worker throughput is
   adequate before scaling.
4. Only then define and run baseline training/evaluation and write the
   Milestone F training handoff.
5. Before merging, obtain one fresh holistic `main...HEAD` review if agent quota
   is available, then choose merge/PR/keep-branch explicitly. Do not stage
   `output/` artifacts.
