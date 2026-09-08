# Velocity-intervention dataset

A dataset of three-branch counterfactual physics videos. Each **instance** is
one scene rendered three times from a single shared appearance record; the only
thing that differs between branches is the intervention on the subject's
initial velocity.

This pipeline is separate from the kinematic-drag pipeline in
[`interventions/dataset.py`](../interventions/dataset.py), and the difference is
the point. There, a target is driven along a prescribed `factual_path` and the
counterfactual perturbs that path. Here every body is a **free dynamic body**
for the whole rollout, the only thing ever set is an initial linear velocity on
one subject at `t=0`, and the counterfactual **replaces** that initial velocity
rather than steering the body afterwards. Nothing is kinematically driven, so no
branch contains motion that Bullet did not produce.

| | |
| --- | --- |
| Config | [`configs/scene_ranges_velocity.yaml`](../configs/scene_ranges_velocity.yaml) |
| Sampling and physics | [`interventions/velocity_scenes.py`](../interventions/velocity_scenes.py) |
| Rendering | [`scripts/render_velocity_scene.py`](../scripts/render_velocity_scene.py) |
| Batch driver | [`scripts/generate_velocity_dataset.py`](../scripts/generate_velocity_dataset.py) |
| Publication | [`scripts/upload_velocity_dataset.py`](../scripts/upload_velocity_dataset.py) |
| Notebook | [`notebooks/velocity_dataset_molab.py`](../notebooks/velocity_dataset_molab.py) |
| Trust model | `velocity_intervention_v1` |

---

## The three branches

| Branch | What it is |
| --- | --- |
| `factual` | The subject is given one initial linear velocity at `t=0` and released. |
| `counterfactual` | The same scene, but the subject's initial velocity is **replaced**: the factual velocity is discarded and a new speed and turned direction are applied once, also at `t=0`. |
| `subject_removed` | The same scene with the subject absent from the start. |

All three share **one** `VisualSceneSpec` — the same bodies, sizes, colours,
materials, textures, three-point light rig, background, and camera. That record
is SHA-256 hashed and published per instance as `visual_scene_hash`, so any
pixel difference between branches is attributable to the intervention rather
than to re-sampled appearance.

`subject_removed` differs from the `target_removed` branch of the
[three-branch collision demo](trajectory_interventions.md#demo-only-removal-trust-boundary),
which carries `demo_only_removal_v1` and is presentation-only. Here removal is
a first-class branch: the subject is absent from `t=0` rather than deleted
mid-clip, so there is no copied prefix and no re-warm-started contact solver.
Every branch is a clean rollout from its own initial conditions.

## Scene configuration

The config implements this specification:

| Requirement | Where it lives | How |
| --- | --- | --- |
| Mass fixed | `objects.mass` | A scalar, not a range. The material sampler may move friction and restitution; it never touches mass. |
| One direction, initial velocity only | `subject.speed`, `subject.azimuth` | Velocity is `(v·cosθ, v·sinθ, 0)` applied once at `t=0`; angular velocity is zero and no force is applied afterwards. |
| Material sampled | `appearance.materials` | Per body from `{metal, rubber, plastic, ceramic, wood, stone}`, with friction and restitution coupled to the family. |
| 4–6 objects, some struck and some not | `objects.count`, `objects.in_path_count` | 4–6 movable bodies including the subject; 2–3 planted in the subject's corridor, the rest bystanders held clear of it. |
| Sliding and rolling | `objects.shapes`, `objects.require_both_shapes` | Cubes slide, spheres roll; a scene with only one shape is rejected. |
| Camera fixed per clip, varied between clips | `appearance.camera.motion: fixed` | One viewpoint held for the whole clip, resampled per instance. |

Glass is excluded from the material families on purpose: a transmissive body is
hard to segment cleanly, and the mask is a published output.

### Why `along_path` tops out at the stopping distance

`objects.along_path` places struck bodies along the subject's travel direction,
stratified across the `in_path_count` slots. Its upper bound is the subject's
stopping distance (`v²/2μg`, roughly 0.7–2.0 m for the sampled speeds and
frictions) rather than something beyond it.

Measured over 24 seeds, an upper bound of 3.0 m puts the far slot past where the
subject can still reach, making it a bystander wearing an in-path label: mean
1.50 bodies struck, and only 50% of scenes with more than one. At 2.0 m that
becomes 1.82 and 77%, at a cost of 2 placement rejections in 24 rather than 0.
Bodies that are deliberately never touched are what `bystander_*` is for; a slot
that claims to be in the corridor should usually be reached.

## Camera framing

The camera is drawn, then **tested against the real frustum of the sampled
lens** over the whole rollout, and re-framed once the rollout exists.

This is not what `appearance_sampling._camera_fits` does. That function compares
the scene bounds against a fixed half-angle proxy with a 1.25 safety margin —
it ignores both the sampled focal length and the output aspect ratio. At the
focal lengths this dataset uses, the actual lens gives a horizontal `tan(½ FOV)`
of 0.56 down to 0.375, while framing the action disc needs about 0.5, so the
proxy accepts cameras that cannot frame the scene at all.

The symptom was silent and expensive: a rendered instance whose segmentation
labels ran `[0,1,2,3,5,6,7]`, with label 4 missing entirely — `object_2` had
zero visible pixels on all 48 frames, a `[-1,-1,-1,-1]` bounding box, and a
projected image position 14 px below the bottom of the image. A body was
simulated, logged, graphed, and tracked, but never photographed.

Three things fix it:

1. **`_frames_all`** tests the actual half-angles of the actual lens
   (`half_x = (sensor_width/2)/focal_length`, corrected for aspect) against the
   points that actually have to be visible.
2. **`_corner_targets`** frames each body's bounding-box *corners*, not its
   centre. Framing centres keeps a body present in the image; framing corners
   keeps it whole. The floor is excluded on purpose — it is a 10 m backdrop no
   reasonable lens frames from this distance, and cropping it costs nothing
   while cropping a body costs an annotation.
3. **`refit_camera`** re-runs acceptance against the trajectories the rollout
   actually produced, unioning targets across all three branches. It draws from
   a fresh generator over the same `camera` seed domain, so the candidate
   viewpoints are the ones `sample_instance` would have drawn; only the
   acceptance test has more information. Every other seed domain is untouched,
   so appearance, physics, and velocities are unchanged.

The config ranges moved with the test: `radius` 6.0–8.5 → **7.5–11.0 m**,
`focal_length` 32–48 → **22–32 mm**, plus `frame_margin: 0.06`. If 512 draws
all fail, the sampler keeps the last direction, opens the lens as wide as the
config allows, and dollies back in 48 steps; only if that also fails does it
raise.

Verified over 12 instances × 3 branches: `branches_out_of_frame 0.0`, 12
accepted and 0 rejected, achieved radius 8.0–10.6 m and focal length 24–32 mm.
Confirmed in pixels afterwards — every body in every branch has non-zero visible
pixels and a valid box on every frame.

## GPU

Setting `KUBRIC_USE_GPU=true` is **not** sufficient, and the failure is silent.
Kubric reads it and sets `scene.cycles.device = "GPU"`
([`kubric/renderer/blender.py:127`](../kubric/renderer/blender.py)) but never
sets `preferences.addons["cycles"].preferences.compute_device_type`, which is
what actually selects the CUDA/OPTIX backend. A fresh pip `bpy` has no
preferences file to supply one, so it stays `NONE`, the device list is empty,
and Cycles falls back to the CPU without warning.

`render_velocity_scene.enable_gpu()` handles this. Two details matter:

- It runs **after** the renderer is constructed. Blender's constructor calls
  `clear_and_reset_blender_scene` → `read_factory_settings`, which discards any
  preferences set beforehand.
- It enables only devices whose `type` matches the chosen backend, which
  excludes the CPU entry Blender lists alongside the GPU under `OPTIX` and
  `CUDA`.

It returns the device record that lands in `render.json`, and when no backend is
available it falls back to the CPU **with a stated reason** rather than
silently. See [Cycles device selection](shared_visual_scene_demo.md#cycles-device-selection)
for the benchmark: OPTIX is about 3.5× faster than the CPU on this workload.

## Outputs

```
<output>/
  manifest.json                    the whole batch: settings, splits, rejections
  split_index.json                 written at upload time; one row per instance
  README.md                        written at upload time from the manifest
  instances/<instance_id>/
    instance.json                  spec, ground truth, QC, motion modes, split
    render.json                    render profile, device record, per-branch timings
    factual/ counterfactual/ subject_removed/
      video.mp4                    H.264, one frame per rendered step
      segmentation.npz             the mask
      segmentation_preview.mp4     colourised mask, for looking at
      tracking.npz                 states, projections, boxes, camera
      graph.json                   temporal contact graph
      depth.npz                    float32 metres
```

### Mask

`segmentation.npz` holds `segmentation` `[T,H,W,1]` uint16. **Label 0 is
background and label `i+1` is row `i` of that branch's tracking arrays**, so a
mask value indexes directly into `states`, `bboxes`, and `object_ids` with no
lookup table. This works because `asset.segmentation_id` is honoured when
cryptomatte hashes are remapped
([`kubric/renderer/blender_utils.py:314`](../kubric/renderer/blender_utils.py));
without it, labels would be asset index + 1, which is a different ordering.

### Tracking

`tracking.npz` holds `object_ids`, `frame_steps`, `states` `[T,N,13]`
(position, quaternion wxyz, linear velocity, angular velocity),
`image_positions` `[T,N,2]` in pixels, `in_front_of_camera`, `bboxes` `[T,N,4]`
as inclusive `(min_row, min_col, max_row, max_col)` with `-1` where the body is
not visible, `visible_pixels`, `visible`, and the per-frame camera
`matrix_world` and `intrinsics`.

Boxes and visibility come from the rendered mask, and are omitted rather than
guessed when the segmentation layer was not requested.

Every array loads **without `allow_pickle`**. String arrays are fixed-width
unicode rather than `dtype=object`, because an object array is pickled inside
the npz and a reader then needs `allow_pickle=True` to reach the numbers next to
it — which is arbitrary code execution on a file downloaded from a dataset host.

### Graph

`graph.json` is the temporal contact graph: one node per body, and one edge per
contact pair carrying its start step, end step, total impulse, and peak force.

### Ground truth

`instance.json` carries, for both the counterfactual and the removal branch:

- `hard_affected` — bodies diverging from the factual branch by more than the
  configured position/velocity/quaternion epsilons;
- `soft_affected` — bodies diverging only below those thresholds;
- `propagation_path` — for each affected body, the contact chain through which
  the intervention reached it, starting at the subject;
- `graph_delta` — the contact-graph difference, as `added`, `removed`, and
  `changed` edges.

## Running it

### Environment

Everything runs in the isolated `thesis` Conda environment — never Conda `base`,
never the system Python. See [`docs/environment_thesis.md`](environment_thesis.md)
for how it is created, and [`render_no_docker.md`](../render_no_docker.md) for
the command prelude.

```bash
export PYTHONPATH=            # the agent terminal is a persistent session
export MPLCONFIGDIR="$TEMP/kubric-mpl"
py="C:/Users/uya7hc/.conda/envs/thesis/python.exe"
```

The notebook and uploader need two packages the render path does not:

```bash
"$py" -m pip install marimo huggingface_hub
```

### One instance

```bash
"$py" -m scripts.render_velocity_scene --seed 20260908 --index 0 \
  --output output/velocity_smoke
```

Useful flags: `--branches factual`, `--resolution 256 192`,
`--samples-per-pixel 8`, `--device CPU`, `--save-blend`, `--allow-rejected`.

### A batch

```bash
"$py" -m scripts.generate_velocity_dataset \
  --output output/velocity_dataset --seed 20260908 --num-instances 50 --resume
```

`--no-render` runs the sampling and physics half only, publishing
`instance.json` per instance. It needs no Blender and runs anywhere, which makes
it the fast way to check acceptance rates and split balance before committing
GPU time.

Placement and camera failures are ordinary outcomes of a rejection sampler, not
faults: they are recorded in the manifest's `rejections` list and the batch
continues. `--max-attempts` defaults to 4× `--num-instances`.

**Resume** is decided before anything expensive runs. `instance_id` is a pure
function of `(seed, index)`, so an already-published index costs a `stat()`
rather than a three-branch rollout that would be thrown away — resuming a
20-instance batch takes 0.8 s instead of 37 s. Completion is judged on the
artifacts themselves, not on a journal entry, because a run killed mid-render
leaves a journal claiming success for a branch whose mp4 is truncated.

**Splits** are assigned by hashing `(split.seed, instance_id)`, not by position
in the accepted sequence. A resumed run, a longer run, or a run with a different
rejection pattern puts a given instance in the same split, so a batch can be
extended without leaking a held-out scene into training. Over 20 000 synthetic
ids the bucketing lands at 0.803 / 0.098 / 0.099 against the configured
0.8 / 0.1 / 0.1.

### Publishing

```bash
export HF_TOKEN=...            # or let the script prompt, with echo off
"$py" -m scripts.upload_velocity_dataset \
  --dataset-dir output/velocity_dataset \
  --repo-id username/kubric-velocity-interventions --dry-run
```

There is deliberately **no `--token` flag**: an argument lands in the shell
history and in the process table. The token is read from `HF_TOKEN`,
`HUGGINGFACE_TOKEN`, or `HUGGING_FACE_HUB_TOKEN`, or from a no-echo prompt, and
is never written to disk, printed, or placed in the card or commit message.

Repos are created **private** unless `--public` is passed. Uploading publishes
to a third party, and a public dataset can be mirrored or indexed before it can
be deleted. `--dry-run` writes `README.md` and `split_index.json` locally and
reports exactly what would be sent, without sending it.

The dataset card is generated from the manifest, so it cannot drift from the
data it describes.

## The notebook

[`notebooks/velocity_dataset_molab.py`](../notebooks/velocity_dataset_molab.py)
is a marimo notebook for <https://molab.marimo.io>: it probes the environment,
clones the fork, picks a rendering backend, generates the batch, previews the
branches side by side, and uploads.

### Docker on molab

The specification asked for Blender in Docker on the GPU. The notebook does
that whenever a Docker daemon is reachable and `nvidia-container-toolkit` is
installed:

```bash
docker run --rm --gpus all -v <repo>:/workspace -w /workspace \
  -e HOME=/tmp -e KUBRIC_USE_GPU=true --user root \
  kubricdockerhub/kubruntudev:latest \
  python3 -m scripts.generate_velocity_dataset ...
```

**On molab it will not be reachable.** Hosted notebook sandboxes do not expose a
Docker daemon to the notebook process, so the Docker path is unavailable there
and the notebook falls back to running Blender in-process through the pip `bpy`
wheel. The backend selector defaults to `auto`, which probes `docker info`
rather than `docker --version` — the CLI can be present with no daemon behind
it, which is exactly this case.

Both backends execute the same `scripts/generate_velocity_dataset.py` with the
same arguments and produce the same outputs; the only difference is where
Blender lives. GPU selection is explicit in both.

The Docker path is written but **has not been executed** — there is no Docker
daemon on the machine this was developed on. The in-process path is the one that
produced every number in this document.

## Measured

On an RTX 3500 Ada Generation Laptop GPU via OPTIX, 48 frames at 24 fps
(2.0 s of a 240 Hz simulation):

| Stage | Cost |
| --- | --- |
| Sample + simulate three branches | ~2 s per instance |
| Render one branch, 512×384, 48 spp | ~83 s |
| Render one instance, all three branches | ~4.2 min |
| Render one branch, 256×192, 8 spp | ~39 s |
| Resume a published 20-instance batch | 0.8 s total, against 37 s before the id was made computable |

Acceptance over 24 seeds is 22/24, the two failures being placement rejections
that the batch driver records and steps past.

## Trust boundary

Pixels do not attest physics. Every annotation is derived from the logged states
and contacts, or from Blender's own segmentation pass; nothing is re-simulated
at render time. The digests in `render.json` are claims about the **inputs**
handed to Blender, not about the encoded frames.

The manifest records rejected attempts rather than hiding them, so the
acceptance rate of a published batch is auditable from the batch itself.
