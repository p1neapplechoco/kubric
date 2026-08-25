# Milestone F smoke batch: measured generation evidence

Recorded: 2026-08-25 (Asia/Ho_Chi_Minh)

## Scope and provenance

- Worktree:
  `/home/pineapple/Desktop/projects/kubric/.worktrees/dramatic-forked-rack-implementation`
- Branch: `feature/dramatic-forked-rack-implementation`
- Commit: `421fddc1563e08002a3d70c9526cbc2b87d7b5d0`
- Config: `configs/scene_ranges.yaml`
- Config SHA-256:
  `7f86690ea4ec8236bef6c31e6a079de021a749f94b95b189dfd2b40fc5740700`
- Master seed: `1701`
- Output:
  `outputs/milestone-f-smoke-20260825-seed1701`
- The output directory is ignored and is not a Git deliverable.

## Commands and completion

The initial deterministic budget used:

```bash
python -W error scripts/generate_dataset.py \
  --config configs/scene_ranges.yaml \
  --output outputs/milestone-f-smoke-20260825-seed1701 \
  --seed 1701 \
  --num-instances 50 \
  --max-attempts 120
```

It completed all 120 attempts in 22 seconds and returned the documented
`capacity_exhausted` exit code `2`: 37 attempts passed QC, leaving a shortfall
of 13.

The same run was resumed without resampling:

```bash
python -W error scripts/generate_dataset.py \
  --config configs/scene_ranges.yaml \
  --output outputs/milestone-f-smoke-20260825-seed1701 \
  --seed 1701 \
  --num-instances 50 \
  --max-attempts 200 \
  --resume
```

The additional 80 attempts completed in 19 seconds. The final status was
`complete` with exit code `0`. Total measured generation time was 41 seconds,
or approximately 4.88 attempts/second and 1.71 accepted attempts/second.

## Final yield and QC

- Attempts: 200.
- Accepted candidate pool: 70.
- Rejected: 130.
- Acceptance rate: 35.0%.
- Balanced selected set: 50.
- Accepted but not selected: 20.
- Candidate-local runtime errors: 0.
- Rejection-reason occurrences:
  - `recipe_outcome_mismatch`: 108;
  - `empty_affected`: 102;
  - `angular_velocity_ceiling`: 1.
- Rejection combinations:
  - both dominant reasons: 81;
  - recipe mismatch only: 27;
  - empty affected only: 21;
  - angular velocity ceiling only: 1.

Recipe acceptance was:

- `break_contact`: 11/41, 26.8%;
- `create_collision`: 15/47, 31.9%;
- `maintain_contact`: 16/39, 41.0%;
- `remove_collision`: 14/38, 36.8%;
- `retime`: 14/35, 40.0%.

Acceptance increased with sampled dynamic-object count:

- 2 objects: 6/38, 15.8%;
- 3 objects: 16/54, 29.6%;
- 4 objects: 17/54, 31.5%;
- 5 objects: 31/54, 57.4%.

This is descriptive smoke evidence, not a controlled causal estimate.

## Balance, topology, and splits

The 70-candidate accepted pool contained:

- categories: 67 `mixed_contact_delta`, 3 `contact_changed`;
- hops: 47 at `1`, 19 at `2`, and 4 at `3+`;
- 50 unique topology signatures.

The balanced 50-instance selection contained:

- categories: 47 `mixed_contact_delta`, 3 `contact_changed`;
- hops: 27 at `1`, 19 at `2`, and 4 at `3+`;
- 41 unique topology signatures;
- seven duplicated topology groups, with a largest group size of three;
- splits: 40 train, 5 validation, and 5 test;
- zero topology groups crossing split boundaries.

No `state_only`, `contact_added`, `contact_removed`, or `null_effect` candidate
was accepted. Category coverage is therefore insufficient for a training
handoff despite the completed requested count.

## Integrity and storage

The public `read_paired_artifact()` reader successfully validated all 70
accepted artifacts, including all 50 selected instances.

- Files: 1,253 before adding audit plots.
- Dataset bytes before audit plots: 19,413,707 bytes.
- Accepted-instance payloads: 18,719,771 bytes.
- Attempt journals: 668,729 bytes.
- Approximate total bytes per accepted candidate: 277,339.
- `run.json` SHA-256:
  `c4465231697852b4230557ad8311906c2c9065a0c73ab65636245090e88d5794`
- `manifest.json` SHA-256:
  `95e66d06546978f28468149fd52c893090f42930eedd71824fea0a635ec8c9a0`

## Stratified trajectory spot-check

Six selected examples were plotted as factual/counterfactual top-down paths
plus per-object position deviation:

- one `contact_changed`, hop-1 `maintain_contact`;
- one mixed hop-1 `create_collision`;
- one mixed hop-2 `break_contact`;
- one mixed hop-2 `retime`;
- one mixed hop-3+ `break_contact`;
- one mixed hop-3+ `remove_collision`.

The plots are under
`outputs/milestone-f-smoke-20260825-seed1701/audit/`.
The inspected samples retain aligned common prefixes, begin deviation within
their intervention windows, and show downstream displacement consistent with
their hop labels. No non-finite trajectory, prefix jump, or unexplained
pre-intervention divergence was observed. A `state_only` example could not be
audited because the batch produced none.

## Status and next steps

This run supplies fresh post-hardening throughput, storage, QC, topology, split,
and visual spot-check evidence. It does not complete Milestone F and does not
establish training readiness.

Before baseline training:

1. Diagnose why the current non-null ranges collapse almost entirely to mixed
   contact deltas and produce no state-only examples.
2. Define explicit category coverage targets and either tune sampling/QC or run
   purpose-built strata rather than relying on aggregate round-robin balance.
3. Repeat the measured batch and visual audit after any range or QC change.
4. Decide output retention and whether the measured single-worker throughput is
   sufficient at the intended dataset scale.
