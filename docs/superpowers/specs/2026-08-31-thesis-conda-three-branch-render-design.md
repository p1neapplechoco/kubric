# Thesis Conda Three-Branch Render Design

**Status:** Approved in conversation on 2026-08-31

## Goal

Produce a Blender-rendered intervention demo entirely outside Docker, using the
`thesis` Conda environment, with canonical full-length outputs for all three
branches:

- `normal_blender.mp4`
- `trajectory_changed_blender.mp4`
- `target_removed_blender.mp4`

Then regenerate the synchronized comparison video
`trajectory_intervention_demo.mp4`. If and only if the run succeeds, rewrite
`render_no_docker.md` into a thesis-conda-only guide that removes Docker instructions.

## Scope

- Use existing CLI modules in `scripts/` without introducing a new wrapper.
- Keep canonical branch semantics and synchronization guarantees unchanged.
- Keep full-length rendering (no smoke truncation).
- Update docs tightly coupled to this workflow (`render_no_docker.md`, and nearby stale
  wording only when directly related).

## Non-Goals

- No Docker-based fallback path.
- No broad refactor of rendering internals.
- No change to canonical intervention physics contracts.

## Execution Design

### 1) Environment gate

Run only with:

`C:\Users\uya7hc\.conda\envs\thesis\python.exe`

Confirm required imports for Kubric + Blender path (`bpy`, render module, and
intervention scripts) before expensive rendering.

### 2) Canonical replay generation

Generate deterministic replay arrays and summary with:

- `python -m scripts.demo_collision_intervention`

This produces synchronized replay data for `normal`, `trajectory_changed`, and
`target_removed`.

### 3) Branch rendering

Render all canonical branches with:

- `python -m scripts.render_demo_branches_blender --branches normal trajectory_changed target_removed`

Use default full-frame settings to satisfy the required full demo output.

### 4) Demo composition

Compose the side-by-side demo using:

- `python -m scripts.compose_intervention_demo`

### 5) Success criteria

The run succeeds only when all four outputs exist and are non-empty in
`output/demo_collision_intervention/`:

- `normal_blender.mp4`
- `trajectory_changed_blender.mp4`
- `target_removed_blender.mp4`
- `trajectory_intervention_demo.mp4`

## Documentation Rewrite Design (`render_no_docker.md`)

After successful run, rewrite `render_no_docker.md` to:

- center on thesis Conda execution on Windows;
- remove Docker installation/runtime guidance;
- include exact end-to-end commands for replay generation, branch rendering, and
  composition;
- include troubleshooting aligned to non-Docker execution (missing `bpy`,
  missing `imageio-ffmpeg`, ffmpeg probing, path/setup pitfalls).

## Error Handling

- Preserve existing fail-fast behavior from script preflight checks
  (summary/schema/shape/presence/synchronization validation).
- Surface errors explicitly from CLI commands; no silent fallback to Docker.
- Do not publish docs claiming success unless output verification passes.

## Verification Plan

- Command-level verification of successful end-to-end run in `thesis`.
- File-level output verification (all expected MP4 artifacts).
- Targeted regression checks for touched rendering workflow/tests:
  `tests/test_render_demo_branches_blender.py` (and any directly coupled tests
  if code edits require them).
