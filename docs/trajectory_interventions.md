# Trajectory interventions

This extension generates paired factual and counterfactual PyBullet rollouts for
trajectory-intervention experiments. It records an oracle temporal contact-graph
delta, affected object sets, and deterministic temporal propagation walks. All
extension code lives in `interventions/`, `scripts/`, and `configs/`; it imports
or subclasses Kubric without monkey-patching or changing the vendored `kubric/`
package.

## Architecture and conventions

Each module declares the same four-part contract in its module docstring. The
table below is the human-facing map of those boundaries.

| Module | Purpose, public API, and dependencies | Trust boundary |
| --- | --- | --- |
| `interventions/schema.py` | Standard-library-only, backend-neutral validation and deterministic JSON conversion through the config/ground-truth dataclasses, `to_jsonable()`, and `derive_seed()`. | Shape and JSON safety are validated; physical feasibility, execution, and origin are not. |
| `interventions/trajectory.py` | NumPy/SciPy construction, validation, comparison, and named perturbation recipes through `build_path()`, `validate_path()`, `max_position_deviation()`, and `perturb_path()`. | Recipes are heuristic candidates until a physics rollout and QC establish the requested effect. |
| `interventions/logging.py` | Immutable NumPy state/contact logs, stable state-vector slices, serialization, hashing, and publication; it imports no Bullet or Kubric backend. | Immutability and hashes protect internal consistency, not simulator or producer origin. |
| `interventions/graph_extraction.py` | Pure aggregation, temporal graphs, graph deltas, reachability, affected sets, and packaged ground truth over supplied logs/states. | The result is not causal proof beyond the completeness and authenticity of its inputs. |
| `interventions/tagging.py` | `derive_tags()` creates deterministic metadata from validated ground truth and explicit role/stability inputs. | Tags summarize supplied metadata and do not independently verify physics or causality. |
| `interventions/appearance.py` | Standard-library-only immutable appearance schemas (`VisualSceneSpec` and its nested material/texture/light/camera/background records), `visual_scene_hash()`, `visual_scene_from_payload()`, `validate_scene_correspondence()`, and `frame_steps_for()`. | Values, enums, and JSON safety are validated; it does not prove an external asset exists or that a scene renders. |
| `interventions/materials.py` | Shipped per-family visual and physical priors, `sample_material()`, and `coupled_physics()` linking a material family to mass and friction. | Priors are dataset-scale conventions, not physical measurements; clamping is recorded rather than hidden. |
| `interventions/appearance_sampling.py` | `sample_visual_scene()` draws one `VisualSceneSpec` for a `SceneConfig` over the `APPEARANCE_DOMAINS` seed streams; NumPy plus the two modules above, never a renderer. | Sampling records every value it draws; it does not verify asset existence or physical feasibility. |
| `interventions/kinematic_simulator.py` | `KinematicSimulator` wraps Kubric/PyBullet for mass-carrying prescribed paths; the package exposes this backend lazily. | Private Bullet snapshots remain bound to the creating simulator, physics client, and backend lifetime. |
| `interventions/twin_runner.py` | Creates fresh worlds for the canonical factual/counterfactual pair, checks prefixes/provenance, derives truth, and reads/writes paired artifacts, including the pair's optional shared `VisualSceneSpec`. | Canonically generated pairs record provenance; caller-supplied logs remain `caller_trusted_unattested_logs_v1`. |
| `interventions/dataset.py` | Deterministic attempts, QC, journals, balancing, grouped splits, atomic publication, and resume through `run_batch()` and supporting APIs; `sample_instance_appearance()` draws the one visual scene an accepted instance publishes. | Resume requires the same run contract; journals/hashes protect consistency but do not authenticate the producer. |
| `interventions/__init__.py` | Stable public exports for schemas, trajectories, logs, graph extraction, tags, and lazy simulator/twin-runner entry points. | Re-exporting a value does not strengthen its provenance or attestation. |
| `scripts/__init__.py` | Package marker for module-based CLI and demo entry points; it intentionally exports no callable API. | Import performs no validation, simulation, rendering, composition, or publication. |
| `scripts/generate_dataset.py` | `main()` parses one resumable batch request and emits the stable `run_batch()` JSON status. | Sampling, QC, journaling, and publication trust remain those of `dataset.py`. |
| `scripts/generate_instance.py` | `main()` samples, executes, evaluates, optionally publishes, and reports one inspectable attempt. | Attempt QC/integrity does not authenticate the machine or producer. |
| `scripts/trajectory_demo_spec.py` | Standard-library-only immutable eleven-object contract, canonical payload, and SHA-256 identity shared by physics, replay, Blender, and FFmpeg. | The digest detects specification drift; it is not a signature or producer attestation. |
| `scripts/demo_collision_intervention.py` | Builds the canonical inputs and atomically publishes three digest-bound replay branches. | Normal/changed use the public paired pipeline; removed is `demo_only_removal_v1`, not a dataset recipe or attested pair. |
| `scripts/render_demo_branches_blender.py` | Preflights spec/summary/array contracts and renders logged colliders with procedural Blender appearance. | Pixels and decoration do not change or independently attest the logged physics. |
| `scripts/compose_intervention_demo.py` | Validates exact summary/event/media contracts and atomically composes the synchronized comparison with FFmpeg. | Composition neither reruns physics nor attests the source producer. |

Unless a field says otherwise, positions and sizes are in metres, time is in
seconds, mass is in kilograms, and velocities use metres/second or
radians/second. Pose arrays are `XYZ + WXYZ`; quaternions are always WXYZ, not
PyBullet's native XYZW order. A cube's `size` is its half-extent and a sphere's
`size` is its radius. `SceneConfig.frame_range`, intervention `time_window`, and
temporal graph intervals are half-open: `[start, end)`. Intervention windows use
Bullet step indices. The number of rollout steps is
`(frame_end - frame_start) * step_rate / frame_rate`.

## Isolated `thesis` environment

Do not install into Conda `base` or the system Python. Reuse the `thesis`
environment when it exists; otherwise create it with a supported Python:

```bash
conda create --name thesis python=3.11
conda activate thesis
python -m pip install -e .
python -m pip install -r requirements_full.txt
python -c "import pybullet, scipy, yaml; print('trajectory dependencies OK')"
```

`requirements.txt` contains the direct runtime requirements for YAML range
loading and spline trajectories. `requirements_full.txt` adds the local Kubric
stack packages used here, including PyBullet and visualization/rendering tools.
An editable install keeps development pointed at this checkout; it does not
modify `kubric/` sources.

The repository config is `configs/scene_ranges.yaml`. A wheel installs the same
canonical file at
`<environment-prefix>/share/kubric/configs/scene_ranges.yaml`; for `thesis` this
normally means `$CONDA_PREFIX/share/kubric/configs/scene_ranges.yaml`. The CLIs
never guess a config location: pass either path explicitly with `--config`.

## Generate one inspectable pair

From the repository root with `thesis` active:

```bash
python scripts/generate_instance.py \
  --config configs/scene_ranges.yaml \
  --output outputs/intervention-debug \
  --seed 1701 \
  --attempt-index 0
```

The command prints one JSON object. `status` is `accepted` or `rejected`,
`artifact_path` identifies the published pair, and `qc` contains stable reason
codes and metrics. Both accepted and rejected candidates return exit code 0
because the requested simulation completed and remains useful for inspection;
an exception returns machine-readable `status: error` and exit code 1.

## Run the three-branch collision demo

The inspectable forked-rack demo compares three synchronized outcomes from one
immutable scene contract. Its eleven canonical objects fall into these semantic
groups:

| Semantic group | Canonical object IDs | Role |
| --- | --- | --- |
| Main balls | `breaker`, `rack_01`, `rack_02`, `rack_03`, `rack_04`, `rack_05`, `rack_06` | Seven numbered dynamic balls used by the large chain. |
| Side balls | `side_01`, `side_02` | Two numbered dynamic balls used by the small chain. |
| Target | `target` | Mass-carrying kinematic wooden striker. |
| Environment | `floor` | Static simulated support surface. |

The specification fixes seed `0`, 200 Bullet steps at 240 Hz, 24 replay frames
per second, and the half-open intervention window `[40, 160)`. The normal and
changed rollouts come from the public paired runner and ground-truth extractor;
the demo does not reimplement that pair pipeline.

The branch envelopes and calibrated deterministic outcomes are:

| Branch | Required acceptance envelope | Calibrated result |
| --- | --- | --- |
| `normal` | 2–3 unique dynamic contact pairs, including the side chain and no main-ball endpoint. | Exactly 2 pairs: `side_01|target` and `side_01|side_02`. |
| `trajectory_changed` | 7–9 pairs, at least five more than normal, `breaker|target`, at least six main balls reached, and no side-ball endpoint. | Exactly 9 pairs and all 7 main balls reached. |
| `target_removed` | Exact shared prefix, then no contact or dynamic-body motion at or after removal. | Target removed before step 40 physics; presence is false from step 40 and no post-removal chain occurs. |

For the canonical normal/changed pair, the graph delta contains exactly 15
`added`, 4 `removed`, and 0 `changed` temporal edges. The affected sets are
`hard=9` (all nine balls) and `soft=0`. Target-removal is deliberately excluded
from this pair ground truth.

Every branch stores `states` with shape `[200, 11, 13]` (XYZ + WXYZ + linear and
angular velocity) and `presence` with shape `[200, 11]`. The removed branch keeps
the last finite target state row after removal, but its presence mask is
authoritative. `summary.json` embeds the canonical demo-spec summary and SHA-256;
the generator, renderer, and compositor require exact digest-bound identity.
Legacy four-object bundles fail preflight and must be regenerated.

Run the complete workflow from the repository root:

```bash
./run_demo.sh intervention
```

The command requires the Conda `thesis` environment, a running Docker daemon,
and the cached `kubricdockerhub/kubruntudev` image containing Blender. It runs
the generator and compositor with module invocations in `thesis`, renders all
three branch replays through Blender/Cycles, and fails nonzero if Docker or the
Blender render is unavailable. For physics logs and replay arrays without any
Docker or composition requirement, use:

```bash
./run_demo.sh intervention-physics-only
```

`imageio-ffmpeg` is a direct dependency in `requirements_full.txt`. The Docker
workflow first checks whether it is already importable and installs it inside
the ephemeral container only when the cached image lacks it.

### Procedural realistic scene

The renderer uses the logged collider poses exactly; appearance is added only
as collider-parented decoration. The target has rounded, noise-textured wood;
all nine balls use glossy billiard lacquer and number decals, while striped balls
additionally receive a white band (currently only `side_02`). The table has
procedural dark-green felt plus wooden rails outside the simulated contact area.
A shared studio-light rig, neutral world, camera, depth of field, Cycles adaptive
sampling, and denoising are identical across all branches.

The complete output tree is:

```text
output/demo_collision_intervention/
  normal_states.npy
  normal_presence.npy
  trajectory_changed_states.npy
  trajectory_changed_presence.npy
  target_removed_states.npy
  target_removed_presence.npy
  contacts.json
  summary.json
  normal_blender.mp4
  trajectory_changed_blender.mp4
  target_removed_blender.mp4
  trajectory_intervention_demo.mp4
```

Each branch video has 200 frames. The compositor adds a one-second opening hold
and a two-second ending hold, so `trajectory_intervention_demo.mp4` has exactly
272 frames / 11.333333 seconds. It is a synchronized, labelled three-panel
H.264/yuv420p video at 1920x720 and 24 fps. Inspect its media contract with:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of json output/demo_collision_intervention/trajectory_intervention_demo.mp4
```

### Verified forked-rack artifact

Verification completed on 2026-08-25 (Asia/Ho_Chi_Minh) on branch
`feature/dramatic-forked-rack-implementation` at implementation HEAD
`ae021dea92bfcdda89181316e9ca360f317053c7`. The documentation handoff commit
follows that implementation hash; recording its own final hash here would be
self-referential. The canonical `forked_rack_v1` seed-0 specification has
SHA-256
`792a36f2376cf7acf994819d885ec8bd0babf1d273dc5369892fd98a4017b977`.
Its exact replay order is `breaker`, `floor`, `rack_01` through
`rack_06`, `side_01`, `side_02`, `target`.

The supported `./run_demo.sh intervention` workflow exited 0 and produced
three 200-frame, 640x540, H.264/yuv420p, CFR-24 branch videos at 64 Cycles
samples. It used Docker image
`sha256:cc4fb8a65172cc1da81dd0ce04bfe47c2405db2e357d842c17583400079d1a80`.
Blender's shutdown report of `3 blocks / 0.003777 MB unfreed` was nonfatal.
The 1920x720 composite is H.264/yuv420p, CFR 24/1, 272 decoded frames, and
11.333333 seconds; all four videos passed full `ffmpeg -xerror` decoding.

Visual review found the complete target, nine balls, and table/floor visible
and unclipped before the intervention; an aligned prefix; the expected small
normal chain; the changed branch reaching all seven main balls by step 160;
and target removal only at step 40 with no later contact or motion. No teleport,
detached decoration, label overlap, or panel desynchronization was observed.
The maximum observed one-frame displacement was 3.07 cm, and the ending
metadata remained readable inside its lower band. A visual QA pass caught
rounded-time cues one frame late; commits `b965116`, `d315435`, and
`ae021dea` moved them to the exact decoded-frame lattice: removal frames
64-81, changed-chain frames 112-133, normal-chain frames 117-138, and final
metadata beginning at frame 224. Decoded integration tests cover the frame
before, first, last, and frame after each cue. The final compositor was rerun
with `python -m scripts.compose_intervention_demo`; no Blender rerender was
needed because the later commits changed only compositor code and tests. A
fresh implementation-head suite reported 922 passed in 66.50 seconds.

| Artifact | SHA-256 |
| --- | --- |
| `summary.json` | `24db6d3be0265e667bb98e957aa98f96c8db9a75c31ebfaa27287866ab8a5ec4` |
| `contacts.json` | `f32f77358fa9a45643f5b86cc79a4952dfa15905d0784f93e363d5eec9ed97ad` |
| `normal_blender.mp4` | `9aa973fa211776adbed699ddb8dfbcc5370728bba105e6416c8f1d8fa2f09373` |
| `trajectory_changed_blender.mp4` | `ef052fb4f7b760f7067546c9e264238ea1bf7e2b7b816ae2fa25ba53cd2ddaef` |
| `target_removed_blender.mp4` | `1cd5187e8bbd505f82e2a5eee0076457e389e8e2f837441dc25b7d1c1ae01cd9` |
| `trajectory_intervention_demo.mp4` | `40323ab0ec96992305365326ce7eb9ad03d9f143c63c175a52b14996d94e54be` |

### Demo-only removal trust boundary

Only `normal` and `trajectory_changed` are canonical public paired rollouts.
`target_removed` retains the existing, narrowly scoped visualization semantics
marked `demo_only_removal_v1`: a fresh matching Bullet world replays the exact
prefix, physically removes the target before step 40 physics, and retains its
last finite pose while setting presence false. It is not a public dataset recipe,
not covered by paired-artifact attestation, and must not be presented as training
data.

This three-branch artifact completes Milestone E visual validation. Milestone F
remains unchanged and incomplete: no large-scale generation, baseline training,
or training handoff claim follows from this demo.

## Generate or resume a batch

```bash
python scripts/generate_dataset.py \
  --config configs/scene_ranges.yaml \
  --output outputs/intervention-smoke \
  --seed 1701 \
  --num-instances 50 \
  --max-attempts 120
```

Batch exit codes are:

- `0`: `complete`; at least `--num-instances` accepted candidates were selected.
- `2`: `capacity_exhausted`; the attempt budget finished before selection reached
  the requested size. Valid accepted artifacts remain available.
- `1`: a batch-level error, emitted as machine-readable JSON. Candidate-local
  failures are journaled and do not by themselves terminate the batch.

Resume the same run after interruption or with a larger attempt budget:

```bash
python scripts/generate_dataset.py \
  --config configs/scene_ranges.yaml \
  --output outputs/intervention-smoke \
  --seed 1701 \
  --num-instances 50 \
  --max-attempts 200 \
  --resume
```

On resume, output must already exist and the config snapshot, seed, and requested
instance count must match. `--max-attempts` may increase. Attempt indices and
derived seeds are deterministic, so completed attempts are validated and reused
rather than resampled. The batch implementation currently accepts only
`workers=1`.

## Recipes, controls, and QC

The five recipe names describe desired physical outcomes, not guarantees made by
the path perturbation alone:

- `remove_collision` creates a clearance candidate; QC requires a removed
  target contact.
- `create_collision` creates an approach candidate; QC requires an added target
  contact.
- `retime` warps progress inside the intervention window; QC requires a non-zero
  command change there.
- `break_contact` creates a lift candidate; QC requires a removed target contact.
- `maintain_contact` creates a lateral candidate; QC requires at least one
  dynamic peer to contact the target in both branches during the window.

The shipped config samples `expected_effects: [non_null]` and a strictly positive
magnitude. Null controls are never inferred from a failed intervention. To
request them, configure `expected_effects: [null]` and `magnitude: [0.0, 0.0]`
explicitly; schema validation rejects a null expectation with non-zero magnitude.

QC rejects malformed or misaligned branches, non-finite states, excessive linear
or angular velocity, unequal pre-intervention state/contact prefixes, a target
outside scene bounds or clipping static geometry, and outcomes inconsistent with
the expected effect or recipe. Stable reason codes include
`branch_misaligned`, `nonfinite_state`, `linear_velocity_ceiling`,
`angular_velocity_ceiling`, `twin_prefix_mismatch`, `target_out_of_bounds`,
`target_static_clip`, `empty_affected`, `expected_null_mismatch`, and
`recipe_outcome_mismatch`.

## Categories, balancing, hops, and splits

Each accepted candidate gets one primary category:

- `contact_added`, `contact_removed`, or `contact_changed` when exactly one graph
  delta bucket is populated;
- `mixed_contact_delta` when more than one bucket is populated;
- `state_only` when states differ without a contact delta;
- `null_effect` when neither states nor contacts differ.

Hop depth is the maximum number of graph edges in any oracle propagation path.
Balancing round-robins deterministic `(category, hop_bucket)` strata, where the
buckets are `0`, `1`, `2`, and `3+`.

The topology signature is an ID-invariant canonical hash of the factual rollout's
unweighted union contact topology. Node labels preserve target/environment/dynamic
role, shape, and static status; timing, force, and contact multiplicity are
excluded. Every candidate with the same signature is assigned to the same
train/validation/test split. Fractions are therefore targets rather than exact
counts, especially for small datasets or large topology groups.

## Shared visual scenes

A pair is a controlled comparison: the two branches must differ in the intervened
physics and in nothing else. Appearance is therefore sampled **once per accepted
instance** and published **once for the pair**, so the factual and counterfactual
branches cannot disagree about how the scene looks — there is one record, and both
branches read it.

Appearance is opt-in. A range config without an `appearance:` section publishes
physics-only pairs that are byte-identical to those produced before visual
sampling existed; `configs/scene_ranges.yaml` is such a config.
`configs/scene_ranges_visual.yaml` adds the section and exercises the visual path.

```python
from interventions import read_paired_artifact
from interventions.appearance import visual_scene_hash, visual_scene_from_payload

factual, counterfactual, truth, provenance = read_paired_artifact(instance_dir)
print(provenance["trust_model"])        # caller_trusted_unattested_logs_v1
print(provenance["visual_scene_hash"])  # digest of the one shared appearance
scene = visual_scene_from_payload(provenance["visual_scene"])
assert visual_scene_hash(scene) == provenance["visual_scene_hash"]
```

### Where it is sampled

`run_batch()` calls `sample_instance_appearance(ranges, spec, seed, index)` only
**after** `evaluate_qc()` accepts the candidate. Appearance work is never spent on
a rejected rollout, and — because it is drawn after the physics is already
decided — it cannot influence which candidates pass QC. The function returns
`None` when the config has no `appearance:` section.

Seeds come from `derive_seed(master_seed, attempt_index, domain)`, which now lives
in `interventions/schema.py` (`interventions.dataset` re-exports it, so existing
imports keep working). `sample_visual_scene()` opens one generator per entry in
`APPEARANCE_DOMAINS` — `geometry`, `physics`, `appearance`, `texture`, `camera`,
`lighting`, `background`, and `render`. Those domain names are disjoint from the
physics domains (`sampling`, `environment`, `scene`, `instance`), so the SHA-256
separation guarantees adding appearance perturbs no draw the physics sampler
already made: the same seed and config produce the same instance ids and the same
rollouts with or without an `appearance:` section.

Because the seed is a pure function of the master seed and the attempt index,
appearance is reproducible and resumable exactly like every other domain.
Regenerating a batch republishes the same `visual_scene_hash` for every instance.

### What lands in `pair.json`

Two keys are added beside `scene_config`, and only as a pair:

- `visual_scene` — the full `VisualSceneSpec.to_dict()` payload.
- `visual_scene_hash` — the SHA-256 of that payload in canonical JSON form.

`read_paired_artifact()` rejects a half-written record (one key without the
other), a payload whose recomputed hash disagrees with the stored one, a payload
that fails schema re-validation, and a payload that does not describe the
simulated scene — `validate_scene_correspondence()` requires the visual scene to
cover exactly the simulated object ids. Nothing appearance-related is stored per
branch, so there is no per-branch field that could drift.

**The trust model is unchanged.** Pair artifacts still report
`caller_trusted_unattested_logs_v1`. Visual provenance is one more internally
consistent, hash-checked payload; it attests no more about origin than the logs
it accompanies do.

### Static `environment` obstacles

`objects.static_fraction` is an optional `[low, high]` range that promotes a share
of the sampled free objects to static obstacles: `mass=0.0`, `static=True`,
`metadata={"role": "environment"}`. They give the renderer fixed scene furniture
and give the contact graph immovable nodes, and the topology signature already
distinguishes the `environment` role from `dynamic` and `target`.

Only objects clear of the corridor the target sweeps are eligible — the check
reserves the target's half-extents plus the largest configured intervention
magnitude, so a counterfactual path cannot be blocked by a body it can never move.
The realized share therefore often falls below the drawn fraction, and a scene
with no eligible object gets none.

Designation draws from its own `environment` seed domain, so a config without
`static_fraction` leaves both the result and the main sampling stream exactly as
they were. Enabling it *does* change the sampled scene, and hence the instance ids,
of every dataset generated from that config. That is why it is enabled in
`configs/scene_ranges_visual.yaml` and left commented out in
`configs/scene_ranges.yaml`, whose physics-only instance ids are pinned by test.

For rendered evidence that both branches read the one shared record — three
branches under a fixed camera, and a six-instance appearance gallery — see the
[shared visual-scene demo](shared_visual_scene_demo.md).

## Artifact layout and public reader

A dataset root is an append-only/resumable journal with this shape (hash directory
names are abbreviated):

```text
output/
  config.yaml                 # canonical snapshot used by this run
  run.json                    # config digest, seed, requested count
  manifest.json               # candidates, balanced selection, grouped splits
  attempts/00000000.json      # accepted/rejected/error journal record
  errors/00000000.json        # traceback for a candidate-local exception
  instances/<instance_id>/
    spec.json
    instance_manifest.json    # hashes every instance payload
    manifest.json             # atomic pointer to one paired generation
    generations/<pair_sha256>/
      pair.json               # scene, intervention, seed, tags, thresholds,
                              # and the pair's one shared visual scene + hash
      ground_truth.json
      factual/
        manifest.json
        generations/<log_sha256>/
          contacts.jsonl
          metadata.json
          states.npy
      counterfactual/
        manifest.json
        generations/<log_sha256>/
          contacts.jsonl
          metadata.json
          states.npy
```

Use the public reader rather than opening payloads independently. It validates
manifests, schemas, provenance, branch alignment, twin consistency, ground truth,
and tags:

```python
from interventions import read_paired_artifact

factual, counterfactual, truth, provenance = read_paired_artifact(
    "outputs/intervention-smoke/instances/<instance_id>"
)
print(provenance["trust_model"])
print(truth.hard_affected, truth.soft_affected)
print(truth.graph_delta.to_dict())
print(truth.propagation_path)
```

`manifest.json` at the dataset root lists all accepted `candidates`, the balanced
`selected_ids`, and an instance-to-split mapping in `splits`. Rejected attempts
have no selected pair artifact; their complete sampled spec and QC result remain
in the attempt journal.

## Trust boundary

Pair artifacts report the exact trust model
`caller_trusted_unattested_logs_v1`. `write_paired_artifact()` accepts caller-held
`SimulationLog` objects and verifies structure, scene/provenance consistency,
contact evidence, paths, and twin prefixes, but it cannot attest that those
in-memory logs came from the simulator. Manifest hashes provide internal
integrity and detect payload changes relative to the published pointer; they are
not signatures and do not authenticate origin or prevent a producer from
constructing and publishing a different internally consistent artifact.

Recording a shared `VisualSceneSpec` does not change this. The reader verifies
that the stored appearance hashes to its stored digest, re-validates against the
appearance schema, and describes exactly the simulated objects; it cannot attest
that any renderer was actually given that appearance. Trust in pixels is
established by the renderer reporting the digest of what it fed to Blender, not by
the artifact.

For canonical oracle generation, use `generate_paired_instance()` directly or
the dataset `run_batch()` path (the CLIs use the latter pipeline). Treat direct
publication of externally supplied logs as caller-trusted data even when the
public reader accepts it.

## Milestone and scaling status

The current implementation reaches Milestone E: it supports a deterministic
single instance, a resumable small batch, QC, balancing, topology-grouped splits,
and inspectable pair artifacts. Milestone F is not complete: no large-scale batch
claim or baseline-training handoff is made here. Generation is deliberately
single-worker, and throughput, storage retention, category yield, and split
quality must be measured again at the intended scale before baseline training.

### Smoke-results status (not a final claim)

This document intentionally publishes no post-hardening smoke counts or pass-rate
numbers. Release handoff should add results only from a fresh run against the
exact reviewed commit and record its config digest, seed, attempt budget, QC
reason distribution, category/hop distribution, and visual spot-check outcome.
