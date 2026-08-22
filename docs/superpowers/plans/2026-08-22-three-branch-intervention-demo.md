# Three-Branch Intervention Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and render one realistic three-panel video comparing a normal target path, a changed trajectory, and physical target removal from the same deterministic scene.

**Architecture:** The demo generator uses the public paired-runner for the normal and changed branches, then a narrowly scoped demo runner replays the common prefix and removes the target from a fresh matching physics world. A Blender replay script renders exact logged states with procedural realistic materials; a separate FFmpeg compositor labels and combines the three synchronized branch videos. The public intervention schema and dataset recipes remain unchanged.

**Tech Stack:** Python 3.11 (`thesis` Conda env), NumPy, Kubric, PyBullet, Pillow, imageio/imageio-ffmpeg, Blender/Cycles in `kubricdockerhub/kubruntudev`, FFmpeg, pytest.

---

## File map

- Modify `scripts/demo_collision_intervention.py`: deterministic scene/path setup, public pair generation, demo-only removal simulation, replay bundle and summary output.
- Modify `scripts/render_demo_branches_blender.py`: validate replay inputs and render procedural realistic branch videos with presence-aware target visibility.
- Create `scripts/compose_intervention_demo.py`: validate source videos and invoke a deterministic FFmpeg three-panel composition.
- Modify `run_demo.sh`: run generation, require or explicitly report Blender, then compose the final video.
- Modify `requirements_full.txt`: retain the direct video encoder dependency.
- Modify `docs/trajectory_interventions.md`: document the three branches, trust boundary, command, and outputs.
- Modify `tests/test_demo_collision_intervention.py`: physics/data-contract tests.
- Modify `tests/test_render_demo_branches_blender.py`: offline replay/renderer helper tests.
- Create `tests/test_compose_intervention_demo.py`: FFmpeg filter/CLI tests.
- Create `notes/session-logs/2026-08-22-trajectory-intervention-demo.md`: durable next-agent handoff after verification.

### Task 1: Replace the raw demo with the public paired pipeline

**Files:**
- Modify: `scripts/demo_collision_intervention.py`
- Modify: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Write failing public-pipeline tests**

Add tests that import the demo script and exercise the real deterministic fixture:

```python
def test_generate_demo_uses_public_pair_and_expected_ground_truth():
  result = demo.generate_demo(seed=0)
  assert result.normal.branch == "factual"
  assert result.changed.branch == "counterfactual"
  assert result.ground_truth.hard_affected == ("upper_ball",)
  assert result.ground_truth.soft_affected == ("upper_ball",)
  assert not demo.dynamic_contacts(result.normal.contacts)
  assert any(
      {record.object_a, record.object_b} == {"target", "upper_ball"}
      for record in result.changed.contacts
  )


def test_changed_branch_diverges_only_inside_intervention_window():
  result = demo.generate_demo(seed=0)
  start, end = result.intervention_window
  np.testing.assert_array_equal(
      result.normal.commanded_path[:start], result.changed.commanded_path[:start]
  )
  np.testing.assert_array_equal(
      result.normal.commanded_path[end:], result.changed.commanded_path[end:]
  )
  assert np.any(
      result.normal.commanded_path[start:end]
      != result.changed.commanded_path[start:end]
  )
```

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest \
  -q tests/test_demo_collision_intervention.py \
  -k 'public_pair or changed_branch'
```

Expected: failure because `generate_demo()` and the structured demo result do not exist; the current script runs duplicated raw PyBullet logic.

- [ ] **Step 3: Implement the deterministic fixture and public pair call**

Define immutable demo containers and a fixture builder. The scene must use one floor, one static cube target, and two dynamic spheres, with 120 Bullet steps:

```python
@dataclass(frozen=True)
class DemoResult:
  scene_config: SceneConfig
  intervention: Intervention
  normal: SimulationLog
  changed: SimulationLog
  removed: "RemovedBranch"
  ground_truth: GroundTruth


def build_demo_inputs():
  floor = ObjectConfig(
      "floor", "cube", size=(4.0, 4.0, 0.25), mass=1.0,
      position=(0.0, 0.0, -0.25), static=True, friction=0.0, restitution=0.0,
  )
  target = ObjectConfig(
      "target", "cube", size=0.18, mass=2.0,
      position=(-1.0, 0.0, 0.18), static=True,
      friction=0.0, restitution=0.0,
  )
  upper_ball = ObjectConfig(
      "upper_ball", "sphere", size=0.26, mass=1.0,
      position=(0.0, 0.45, 0.26), friction=0.0, restitution=0.0,
  )
  lower_ball = ObjectConfig(
      "lower_ball", "sphere", size=0.26, mass=1.0,
      position=(0.0, -0.45, 0.26), friction=0.0, restitution=0.0,
  )
  scene = SceneConfig(
      objects=(floor, target, upper_ball, lower_ball),
      seed=0,
      scene_bounds=((-4.5, -4.5, -1.0), (4.5, 4.5, 2.0)),
      gravity=(0.0, 0.0, 0.0),
      frame_range=(0, 12),
      frame_rate=24,
      step_rate=240,
  )
  intervention = Intervention(
      target_id="target",
      recipe="create_collision",
      magnitude=0.35,
      time_window=(24, 96),
      push_mass=2.0,
  )
  factual_path = np.zeros((120, 7), dtype=np.float64)
  factual_path[:, 0] = np.linspace(-1.0, 1.0, 120)
  factual_path[:, 2] = 0.18
  factual_path[:, 3] = 1.0  # identity quaternion in WXYZ order
  return scene, intervention, factual_path


def generate_demo(seed=0):
  scene, intervention, factual_path = build_demo_inputs()
  normal, changed = generate_paired_instance(
      scene, "target", intervention, seed, factual_path=factual_path
  )
  truth = extract_pair_ground_truth(scene, intervention, normal, changed)
  removed = _run_removed_branch(
      scene, factual_path, int(intervention.time_window[0]), intervention.push_mass
  )
  _validate_demo_outcomes(normal, changed, removed, truth, intervention)
  return DemoResult(scene, intervention, normal, changed, removed, truth)
```

Use only supported public classes/functions for the pair. Assert the intended target contact and affected set before returning.

- [ ] **Step 4: Run the focused tests and capture GREEN**

Run the command from Step 2. Expected: all selected tests pass with no warning.

- [ ] **Step 5: Commit the public-pipeline conversion**

Stage only the two task files and commit:

```bash
git add scripts/demo_collision_intervention.py tests/test_demo_collision_intervention.py
git commit -m "refactor(demo): use intervention pipeline"
```

### Task 2: Add the physical target-removal branch

**Files:**
- Modify: `scripts/demo_collision_intervention.py`
- Modify: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Write failing removal-contract tests**

```python
def test_removed_branch_has_exact_prefix_and_presence_mask():
  result = demo.generate_demo(seed=0)
  start, _ = result.intervention_window
  target = result.normal.object_ids.index("target")
  non_target = [i for i, name in enumerate(result.normal.object_ids) if name != "target"]
  np.testing.assert_array_equal(
      result.removed.states[:start, non_target],
      result.normal.states[:start, non_target],
  )
  assert result.removed.presence[:start, target].all()
  assert not result.removed.presence[start:, target].any()
  assert result.removed.metadata["trust_model"] == "demo_only_removal_v1"


def test_removed_target_has_no_post_removal_contacts():
  result = demo.generate_demo(seed=0)
  start, _ = result.intervention_window
  assert all(
      record.step < start
      for record in result.removed.contacts
      if "target" in (record.object_a, record.object_b)
  )
```

- [ ] **Step 2: Run the two tests and capture RED**

Expected: `RemovedBranch`/presence data or a real removal runner is missing.

- [ ] **Step 3: Implement the demo-only runner**

Add an immutable container with fixed object order and a Boolean mask:

```python
@dataclass(frozen=True)
class RemovedBranch:
  branch: str
  object_ids: tuple[str, ...]
  steps: tuple[int, ...]
  states: np.ndarray
  presence: np.ndarray
  contacts: tuple[ContactRecord, ...]
  metadata: Mapping[str, object]
```

Create a fresh scene/simulator from the same `SceneConfig`. Run the factual
prefix through `KinematicSimulator.run_with_intervention()`. At `start`, call
`scene.remove(target)` so the Bullet body and observers are removed. For each
remaining step, call `step_passive()`, snapshot live bodies in the original
logical order, retain the target's last finite row, and set its presence false.
Log contacts only among live known bodies. Close the simulator in a context
manager on every path.

- [ ] **Step 4: Add deterministic and validation tests**

```python
def test_removed_branch_is_deterministic():
  first = demo.generate_demo(seed=3).removed
  second = demo.generate_demo(seed=3).removed
  np.testing.assert_array_equal(first.states, second.states)
  np.testing.assert_array_equal(first.presence, second.presence)
  assert first.contacts == second.contacts


def test_removed_branch_rejects_misaligned_prefix(monkeypatch):
  original = demo._run_removed_branch

  def corrupt_prefix(*args, **kwargs):
    branch = original(*args, **kwargs)
    states = np.array(branch.states, copy=True)
    states[0, 0, 0] += 0.01
    return dataclasses.replace(branch, states=states)

  monkeypatch.setattr(demo, "_run_removed_branch", corrupt_prefix)
  with pytest.raises(RuntimeError, match="prefix"):
    demo.generate_demo(seed=0)
```

- [ ] **Step 5: Run the full demo physics tests and commit**

Expected: all tests in `tests/test_demo_collision_intervention.py` pass under
`-W error`. Commit:

```bash
git add scripts/demo_collision_intervention.py tests/test_demo_collision_intervention.py
git commit -m "feat(demo): simulate target removal"
```

### Task 3: Persist a replay bundle with exact event metadata

**Files:**
- Modify: `scripts/demo_collision_intervention.py`
- Modify: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Write failing output tests**

```python
def test_write_demo_bundle_roundtrips_all_branches(tmp_path):
  result = demo.generate_demo(seed=0)
  demo.write_demo_bundle(tmp_path, result)
  for branch in ("normal", "trajectory_changed", "target_removed"):
    states = np.load(tmp_path / f"{branch}_states.npy")
    presence = np.load(tmp_path / f"{branch}_presence.npy")
    assert states.shape == (120, 4, 13)
    assert presence.shape == (120, 4)
  summary = json.loads((tmp_path / "summary.json").read_text())
  assert summary["intervention_start"] == 24
  assert summary["branches"]["target_removed"]["removed_step"] == 24
  assert summary["ground_truth"]["hard_affected"] == ["upper_ball"]
```

- [ ] **Step 2: Run the test and capture RED**

Expected: the current writer uses factual/counterfactual names and has no
presence/event bundle.

- [ ] **Step 3: Implement canonical writes**

Write `.npy` files atomically through temporary siblings plus `os.replace`.
Write canonical sorted JSON containing object order, step rate, intervention
window, per-branch contact steps, removal event, graph delta, affected sets,
propagation paths, and `demo_only_removal_v1`. Normal/changed masks are all true;
the removal mask is false for the target from step 24 onward.

- [ ] **Step 4: Fix the existing ImageIO reader warning**

Update tests to use a reader context manager or explicitly close the iterator:

```python
reader = imageio.get_reader(output)
try:
  frames = [frame for frame in reader]
finally:
  reader.close()
```

This removes the current `ResourceWarning` failure under `-W error`.

- [ ] **Step 5: Run all demo-generator tests and commit**

```bash
git add scripts/demo_collision_intervention.py tests/test_demo_collision_intervention.py
git commit -m "feat(demo): persist three-branch replay"
```

### Task 4: Render procedural realistic branch videos

**Files:**
- Modify: `scripts/render_demo_branches_blender.py`
- Modify: `tests/test_render_demo_branches_blender.py`

- [ ] **Step 1: Write failing offline replay tests**

```python
def test_load_replay_requires_matching_presence(tmp_path):
  states = _synthetic_states()
  np.save(tmp_path / "normal_states.npy", states)
  np.save(tmp_path / "normal_presence.npy", np.ones((4, 4), dtype=bool))
  with pytest.raises(ValueError, match="presence"):
    render_script._load_replay(tmp_path, "normal")


def test_pose_at_reads_simulation_log_wxyz_without_reordering():
  states = _synthetic_states(num_steps=1)
  states[0, 1, 3:7] = (0.4, 0.1, 0.2, 0.3)
  position, quaternion = render_script._pose_at(states, 1, 0)
  assert position == (-0.5, 0.0, 0.0)
  assert quaternion == (0.4, 0.1, 0.2, 0.3)


def test_material_specs_are_deterministic_and_realistic():
  specs = render_script._material_specs()
  assert specs["target"]["material"] == "wood"
  assert specs["upper_ball"]["roughness"] < 0.25
  assert specs["floor"]["material"] == "felt"
```

- [ ] **Step 2: Run the renderer tests and capture RED**

Expected: missing replay/presence helpers and old XYZW conversion.

- [ ] **Step 3: Implement replay validation and visibility keyframes**

Load `states`, `presence`, and `summary.json`; require finite WXYZ states,
Boolean mask shape equality, exact branch names, common object order, and equal
frame count. Keyframe `hide_render`/`hide_viewport` on the target at the first
false presence frame. Do not move hidden objects from retained rows.

- [ ] **Step 4: Implement the procedural scene**

Use Kubric primitives for collider-aligned transforms, then `bpy` node materials
and modifiers for appearance:

- target cube: bevel modifier plus noise-driven wood color/roughness;
- balls: glossy colored lacquer with a white band and small number decal made
  from procedural geometry/text parented to the sphere;
- floor: dark-green felt noise/bump material and a raised wooden border outside
  the simulated contact area;
- world/studio: neutral backdrop, two area lights and one rim light, Cycles
  denoising, transparent=False, shared camera and focal length.

All decorative children follow their parent pose and visibility. Do not change
collider-sized parent scales.

- [ ] **Step 5: Run renderer tests and a one-frame Docker smoke**

Run offline tests first. Then render frame 1 of all branches at 320x180 and 8
samples inside the cached Kubric Docker image. Expected: three readable MP4s or
PNGs, no missing target before step 24, and no backend exception.

- [ ] **Step 6: Commit the renderer**

```bash
git add scripts/render_demo_branches_blender.py tests/test_render_demo_branches_blender.py
git commit -m "feat(demo): render realistic branches"
```

### Task 5: Compose the synchronized three-panel MP4

**Files:**
- Create: `scripts/compose_intervention_demo.py`
- Create: `tests/test_compose_intervention_demo.py`

- [ ] **Step 1: Write failing compositor tests**

```python
def test_filter_builds_three_panel_1920x720_video():
  graph = compose._filter_graph(
      intervention_time=1.0,
      contact_time=1.42,
      duration=5.0,
      font="/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
  )
  assert "hstack=inputs=3" in graph
  assert "pad=1920:720" in graph
  assert "NORMAL" in graph
  assert "TRAJECTORY CHANGED" in graph
  assert "TARGET REMOVED" in graph
  assert "INTERVENTION" in graph


def test_probe_rejects_mismatched_sources(monkeypatch, tmp_path):
  probes = iter((
      {"width": 640, "height": 540, "fps": 24.0, "frames": 120},
      {"width": 640, "height": 540, "fps": 24.0, "frames": 119},
      {"width": 640, "height": 540, "fps": 24.0, "frames": 120},
  ))
  monkeypatch.setattr(compose, "_probe", lambda path: next(probes))
  with pytest.raises(ValueError, match="synchronized"):
    compose.compose(tmp_path)
```

- [ ] **Step 2: Run the tests and capture RED**

Expected: module does not exist.

- [ ] **Step 3: Implement probe, filter graph, and command execution**

Use `ffprobe -of json` to require H.264-compatible equal size/fps/frame count.
Build one filter graph that scales each panel to 640x540, applies a one-second
start hold and two-second end hold, hstacks the panels, pads to 1920x720, and
draws labels/timeline/events from summary JSON. Encode with `libx264`,
`yuv420p`, 24 fps, `-movflags +faststart`, and write through a temporary output
renamed only after successful encoding.

- [ ] **Step 4: Run unit and synthetic integration tests**

Create three six-frame synthetic MP4s in the test, call `compose()`, and verify
with `ffprobe` that output is 1920x720, 24 fps, and contains the padded duration.

- [ ] **Step 5: Commit the compositor**

```bash
git add scripts/compose_intervention_demo.py tests/test_compose_intervention_demo.py
git commit -m "feat(demo): compose comparison video"
```

### Task 6: Wire the command and document the outputs

**Files:**
- Modify: `run_demo.sh`
- Modify: `requirements_full.txt`
- Modify: `docs/trajectory_interventions.md`
- Test: `tests/test_demo_collision_intervention.py`

- [ ] **Step 1: Add failing command/documentation assertions**

Test that `run_demo.sh intervention` invokes generation, Blender rendering, and
composition in that order, and that missing Docker/Blender returns a nonzero
status unless `--physics-only` is explicitly requested. Assert docs name the
three branches and final `trajectory_intervention_demo.mp4`.

- [ ] **Step 2: Run tests and capture RED**

Expected: existing shell script renders two branch names, silently skips
Blender, and never composes one video.

- [ ] **Step 3: Update the shell workflow**

Make `intervention` run:

```text
/home/pineapple/miniconda3/envs/thesis/bin/python scripts/demo_collision_intervention.py
docker run --rm --volume "$(pwd):/workspace" kubricdockerhub/kubruntudev \
  python3 /workspace/scripts/render_demo_branches_blender.py
/home/pineapple/miniconda3/envs/thesis/bin/python scripts/compose_intervention_demo.py
```

Print exact output paths and sizes. Keep `hello`/`sim` behavior unchanged.
Do not install dependencies from the network when the required package is
already present. Add an explicit `intervention-physics-only` mode for logs and
summary without Blender.

- [ ] **Step 4: Update user documentation**

Document the three-branch semantics, procedural-assets choice, Docker
requirement, deterministic seed, demo-only removal trust boundary, output tree,
and a sample `ffprobe` command. State that this demo completes visual validation
for Milestone E but not Milestone F training.

- [ ] **Step 5: Run shell and documentation checks, then commit**

```bash
bash -n run_demo.sh
/home/pineapple/miniconda3/envs/thesis/bin/python scripts/demo_collision_intervention.py --help
/home/pineapple/miniconda3/envs/thesis/bin/python scripts/render_demo_branches_blender.py --help
/home/pineapple/miniconda3/envs/thesis/bin/python scripts/compose_intervention_demo.py --help
git diff --check
git add run_demo.sh requirements_full.txt docs/trajectory_interventions.md tests/test_demo_collision_intervention.py
git commit -m "docs(demo): add three-branch workflow"
```

### Task 7: Render, inspect, and verify the final artifact

**Files:**
- Output only: `output/demo_collision_intervention/`

- [ ] **Step 1: Run focused and full automated gates**

```bash
MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest -q \
  tests/test_demo_collision_intervention.py \
  tests/test_render_demo_branches_blender.py \
  tests/test_compose_intervention_demo.py

MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest -q tests

MPLCONFIGDIR=/tmp/kubric-mpl \
  /home/pineapple/miniconda3/envs/thesis/bin/python -W error -m pytest -q \
  test/test_scene.py test/test_pybullet.py
```

Expected baseline before new work: committed core `556 passed`; current dirty
demo baseline has one ImageIO ResourceWarning failure. Final expectation: every
suite exits zero with no warning.

- [ ] **Step 2: Render the complete artifact**

Run `./run_demo.sh intervention`. Preserve generator logs and Docker output.
The command must create:

```text
output/demo_collision_intervention/
  normal_states.npy
  trajectory_changed_states.npy
  target_removed_states.npy
  *_presence.npy
  contacts.json
  summary.json
  normal_blender.mp4
  trajectory_changed_blender.mp4
  target_removed_blender.mp4
  trajectory_intervention_demo.mp4
```

- [ ] **Step 3: Validate media metadata**

Use `ffprobe` to assert the final output is H.264/yuv420p, 1920x720, 24 fps,
8-12 seconds, with no decode errors. Record exact duration, frame count, size,
and SHA-256.

- [ ] **Step 4: Perform visual spot checks**

Extract frames before intervention, immediately after intervention, at changed
branch contact, and at the ending summary. Inspect them for:

- identical scene/object prefix across all panels;
- target visible before step 24 in all panels;
- changed target visibly swerving and striking the upper ball;
- target absent only in the removal panel from step 24 onward;
- no object teleport, clipping, label overlap, or branch desynchronization;
- readable affected-set and graph-delta summary.

- [ ] **Step 5: Record verification evidence**

Update the docs demo section with final file metadata and visual-check result.
Do not commit generated MP4/log outputs unless repository policy explicitly
tracks `output/` artifacts.

### Task 8: Create the next-agent handoff and finish the branch

**Files:**
- Create: `notes/session-logs/2026-08-22-trajectory-intervention-demo.md`

- [ ] **Step 1: Write the handoff**

Record branch/HEAD, completed milestones A-E, demo architecture and output path,
all fresh test counts, media metadata/hash, environment (`thesis`), trust-model
boundaries, untouched `kubric/` invariant, and remaining Milestone F work.

- [ ] **Step 2: Verify repository scope and cleanliness**

```bash
git status --short --branch
git diff main...HEAD --name-only
git diff --check
git log --oneline --decorate -20
```

Confirm no modified path under `kubric/` and no generated media was staged.

- [ ] **Step 3: Commit the handoff**

```bash
git add notes/session-logs/2026-08-22-trajectory-intervention-demo.md docs/trajectory_interventions.md
git commit -m "docs(interventions): record demo handoff"
```

- [ ] **Step 4: Use `finishing-a-development-branch`**

Rerun the required verification, identify the merge base, and offer the user
the structured choices to merge locally, push/create a PR, keep the branch, or
discard it. Do not merge, push, or delete without the user's explicit choice.
