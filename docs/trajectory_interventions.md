# Trajectory interventions

This extension generates paired factual and counterfactual PyBullet rollouts for
trajectory-intervention experiments. It records an oracle temporal contact-graph
delta, affected object sets, and deterministic temporal propagation walks. All
extension code lives in `interventions/`, `scripts/`, and `configs/`; it imports
or subclasses Kubric without monkey-patching or changing the vendored `kubric/`
package.

## Architecture and conventions

- `schema.py` and `trajectory.py` define validated, backend-neutral inputs and
  path construction/perturbation.
- `kinematic_simulator.py` wraps Kubric's PyBullet simulator. The target remains
  static in Kubric bookkeeping and receives a positive `push_mass` only during
  each physics step.
- `logging.py` records immutable state/contact generations.
- `twin_runner.py` rebuilds both branches from one `SceneConfig` and seed, using
  a fresh physics client per branch, and validates their common prefix.
- `graph_extraction.py` and `tagging.py` derive temporal graph deltas, affected
  sets, propagation paths, and deterministic tags.
- `dataset.py` performs deterministic sampling, QC, balancing, grouped splits,
  resumable journaling, and artifact publication.

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

The inspectable demo compares three synchronized outcomes from one fixed scene:

- `normal`: the factual target follows its straight path between the balls and
  makes no dynamic-object contact;
- `trajectory_changed`: the public `create_collision` intervention changes the
  target path only inside steps `[24, 96)` and makes it strike `upper_ball`;
- `target_removed`: a fresh matching physics world replays the exact common
  prefix, then physically removes the target before step 24 physics. Its last
  finite target pose stays in the replay array while its presence mask is false.

The fixture always uses seed `0`, 120 Bullet steps at 240 Hz, and 24 rendered
frames per second. The normal/changed pair comes from the public paired runner
and ground-truth extractor; this is not a second implementation of those APIs.

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
as collider-parented decoration. The target has rounded, noise-textured wood,
the two balls use glossy billiard lacquer with bands and number decals, and the
table has procedural dark-green felt plus wooden rails outside the simulated
contact area. A shared studio-light rig, neutral world, camera, depth of field,
Cycles adaptive sampling, and denoising are identical across all branches.

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

The final `trajectory_intervention_demo.mp4` is a synchronized, labelled
three-panel H.264/yuv420p video at 1920x720 and 24 fps. Inspect its media contract
with:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of json output/demo_collision_intervention/trajectory_intervention_demo.mp4
```

### Demo-only removal trust boundary

Only `normal` and `trajectory_changed` are canonical public paired rollouts.
`target_removed` is narrowly scoped visualization data marked
`demo_only_removal_v1`: it uses real removal from a fresh Bullet world, but is
not a public dataset recipe, not covered by the paired-artifact attestation
model, and must not be presented as training data. The presence mask, rather
than the retained finite pose row, is authoritative after the removal step.

This three-branch artifact completes Milestone E visual validation. It does not
complete Milestone F: no large-scale generation, baseline training, or training
handoff claim follows from this demo.

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
      pair.json               # scene, intervention, seed, tags, thresholds
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
