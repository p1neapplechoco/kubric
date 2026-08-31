# Thesis Conda Three-Branch Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the canonical three-branch Blender intervention demo fully inside the `thesis` Conda environment (no Docker), then rewrite `render_no_docker.md` into a conda-only guide after verified success.

**Architecture:** Reuse existing deterministic CLIs in sequence: replay generation, branch rendering, then composition. Keep rendering contracts unchanged and gate documentation rewrite on verified artifact success. Validate with targeted tests that cover replay preflight and Blender branch rendering behavior.

**Tech Stack:** Python 3.11 (`C:\Users\uya7hc\.conda\envs\thesis\python.exe`), Kubric, PyBullet, `bpy`, NumPy, imageio-ffmpeg, pytest.

## Global Constraints

- Use only `C:\Users\uya7hc\.conda\envs\thesis\python.exe` for Python and pytest commands.
- Do not use Docker anywhere in execution or documentation.
- Render all three canonical branches (`normal`, `trajectory_changed`, `target_removed`) at full length (no smoke truncation).
- Regenerate the synchronized combined demo `trajectory_intervention_demo.mp4`.
- Rewrite `render_no_docker.md` only after successful end-to-end output verification.
- Preserve existing branch semantics, preflight validation, and fail-fast error behavior.

---

### Task 1: Verify thesis environment and rendering prerequisites

**Files:**
- Modify: none
- Test: command-level import checks

**Interfaces:**
- Consumes: existing environment doc and interpreter path.
- Produces: verified executable prerequisites for Tasks 2-4 (`bpy`, Kubric renderer, `imageio_ffmpeg` imports succeed).

- [ ] **Step 1: Clear session Python path and define thesis interpreter**

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
```

- [ ] **Step 2: Run prerequisite import check**

Run:

```powershell
& $py -c "import kubric as kb; import kubric.renderer.blender; import imageio_ffmpeg; import bpy; print('ok')"
```

Expected: prints `ok` and exits with code 0.

- [ ] **Step 3: Validate working output directory exists**

Run:

```powershell
& $py -c "from pathlib import Path; p=Path('output/demo_collision_intervention'); p.mkdir(parents=True, exist_ok=True); print(p.resolve())"
```

Expected: absolute path printed and directory exists.

- [ ] **Step 4: Commit**

No repository changes are expected in this task; do not create an empty commit.

### Task 2: Generate canonical replay bundle in thesis env

**Files:**
- Modify: none
- Test: replay artifact existence checks

**Interfaces:**
- Consumes: `scripts/demo_collision_intervention.py::main()`.
- Produces: canonical replay inputs for Task 3:
  - `output/demo_collision_intervention/summary.json`
  - `normal_states.npy`, `trajectory_changed_states.npy`, `target_removed_states.npy`
  - corresponding `*_presence.npy` files.

- [ ] **Step 1: Run replay generation CLI**

Run:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
& $py -m scripts.demo_collision_intervention --output output/demo_collision_intervention
```

Expected: exit code 0 and canonical replay files written.

- [ ] **Step 2: Verify required replay files are present and non-empty**

Run:

```powershell
& $py -c "from pathlib import Path; root=Path('output/demo_collision_intervention'); names=['summary.json','normal_states.npy','normal_presence.npy','trajectory_changed_states.npy','trajectory_changed_presence.npy','target_removed_states.npy','target_removed_presence.npy']; missing=[n for n in names if not (root/n).is_file() or (root/n).stat().st_size<1]; print('missing', missing); raise SystemExit(1 if missing else 0)"
```

Expected: prints `missing []` and exits with code 0.

- [ ] **Step 3: Commit**

No repository changes are expected in this task; do not create an empty commit.

### Task 3: Render all three branches and compose synchronized demo

**Files:**
- Modify: none
- Test: output MP4 verification checks

**Interfaces:**
- Consumes: `scripts/render_demo_branches_blender.py::main()` and `scripts/compose_intervention_demo.py::main()`.
- Produces: four verified MP4 artifacts:
  - `normal_blender.mp4`
  - `trajectory_changed_blender.mp4`
  - `target_removed_blender.mp4`
  - `trajectory_intervention_demo.mp4`

- [ ] **Step 1: Render canonical branches in thesis env**

Run:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
& $py -m scripts.render_demo_branches_blender --states-dir output/demo_collision_intervention --branches normal trajectory_changed target_removed
```

Expected: exit code 0 and JSON render summaries printed for all requested branches.

- [ ] **Step 2: Compose the synchronized comparison video**

Run:

```powershell
& $py -m scripts.compose_intervention_demo --states-dir output/demo_collision_intervention --output output/demo_collision_intervention/trajectory_intervention_demo.mp4
```

Expected: exit code 0 and composed demo MP4 created.

- [ ] **Step 3: Verify all MP4 artifacts are present and non-empty**

Run:

```powershell
& $py -c "from pathlib import Path; root=Path('output/demo_collision_intervention'); names=['normal_blender.mp4','trajectory_changed_blender.mp4','target_removed_blender.mp4','trajectory_intervention_demo.mp4']; missing=[n for n in names if not (root/n).is_file() or (root/n).stat().st_size<1]; print('missing', missing); raise SystemExit(1 if missing else 0)"
```

Expected: prints `missing []` and exits with code 0.

- [ ] **Step 4: Commit**

No repository changes are expected in this task; do not create an empty commit.

### Task 4: Rewrite render_no_docker.md for thesis-conda-only operation

**Files:**
- Modify: `render_no_docker.md`
- Optional modify (only if wording is touched while syncing docs): `docs/trajectory_interventions.md`, `scripts/render_demo_branches_blender.py` comments
- Test: documentation consistency by command spot-check

**Interfaces:**
- Consumes: successful artifacts and validated command sequence from Tasks 2-3.
- Produces: Docker-free render guide with exact thesis commands for still/video/three-branch demo usage.

- [ ] **Step 1: Write failing documentation expectation checklist**

Use this checklist (current file fails it because Docker commands are present):

```text
1) No docker build/run command appears anywhere in render_no_docker.md.
2) Thesis interpreter path is explicit for Windows.
3) Three-branch canonical demo flow includes replay generation, branch rendering, and composition.
4) Success criteria lists all four expected MP4 outputs.
```

- [ ] **Step 2: Rewrite render_no_docker.md to pass the checklist**

Implement these content changes:

```markdown
- Replace Blender installation + Docker sections with thesis Conda setup/verification.
- Add command blocks that use:
  C:\Users\uya7hc\.conda\envs\thesis\python.exe
- Add an end-to-end "Three-branch intervention demo" section:
  1) scripts.demo_collision_intervention
  2) scripts.render_demo_branches_blender (normal, trajectory_changed, target_removed)
  3) scripts.compose_intervention_demo
- Include explicit artifact verification list for all four MP4 files.
```

- [ ] **Step 3: Re-open render_no_docker.md and verify checklist items are all satisfied**

Run a content check:

```powershell
git --no-pager grep -n "docker" -- render_no_docker.md
```

Expected: no matches.

- [ ] **Step 4: Commit**

```powershell
git --no-pager add render_no_docker.md
git --no-pager commit -m "docs: rewrite render guide for thesis conda workflow" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 5: Run targeted validation and publish execution summary

**Files:**
- Modify: none (unless tests force direct fix tied to doc-coupled comment updates)
- Test: `tests/test_render_demo_branches_blender.py`

**Interfaces:**
- Consumes: updated docs and existing renderer behavior.
- Produces: validated confidence that three-branch render flow and preflight contracts remain intact.

- [ ] **Step 1: Run targeted renderer test suite**

Run:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
& $py -m pytest tests/test_render_demo_branches_blender.py -q
```

Expected: tests pass.

- [ ] **Step 2: Capture final artifact manifest**

Run:

```powershell
& $py -c "from pathlib import Path; root=Path('output/demo_collision_intervention'); names=['normal_blender.mp4','trajectory_changed_blender.mp4','target_removed_blender.mp4','trajectory_intervention_demo.mp4']; print('\\n'.join(f'{n}: {(root/n).stat().st_size}' for n in names))"
```

Expected: all four files listed with positive byte sizes.

- [ ] **Step 3: Commit**

No repository changes are expected in this task; do not create an empty commit.

- [ ] **Step 4: Report completion**

Report:

```text
- exact commands executed
- pass/fail status per step
- final artifact paths and sizes
- render_no_docker.md rewrite confirmation
```
