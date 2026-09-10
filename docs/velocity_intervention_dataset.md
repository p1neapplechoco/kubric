# Velocity-intervention dataset (free rigid bodies, three branches, GPU render)

This pipeline generates table-top scenes of **4–6 fixed-mass rigid bodies** where exactly one
body (the *subject*) receives an **initial velocity along a single heading** and nothing else
is actuated. Every scene is realised as three branches that share one visual scene and one
static camera, and is rendered with Blender/Cycles on the GPU to video, masks, contact graphs
and tracking data. It reuses the repository's existing components:

| concern | component |
| --- | --- |
| scene / object schema, seeds | `interventions/schema.py` (`SceneConfig`, `ObjectConfig`, `CameraConfig`, `derive_seed`) |
| appearance (materials, textures, lights, background) | `interventions/appearance.py`, `appearance_sampling.py`, `materials.py` |
| physics | `kubric.simulator.PyBullet` (kubric primitives, rolling/spinning friction via `changeDynamics`) |
| contact + state logging | `interventions/logging.py` (`ContactLogger`, `SimulationLog`, atomic artifact writer) |
| temporal graphs, deltas, affected objects | `interventions/graph_extraction.py` |
| rendering | `kubric.renderer.Blender` (Cycles) with explicit GPU device selection |

## Scene configuration (`configs/velocity_intervention.yaml`)

| requirement | how it is realised |
| --- | --- |
| mass fixed | `objects.mass: 1.0` for every dynamic body (material family only drives friction/restitution) |
| one initial velocity, one heading | subject `linear_velocity = speed · (cos θ, sin θ, 0)` applied once before step 0; no forces afterwards |
| material sampling | `appearance_sampling.sample_visual_scene` picks a material family, colour and procedural texture per object; `materials.coupled_physics` derives friction & restitution from the family |
| 4–6 bodies, some struck, some not | roles: `subject`, `interactive` (placed inside the subject's corridor), `bystander` (outside the corridor); QC requires ≥1 struck and ≥1 untouched body in the factual branch |
| sliding **and** rolling | spheres roll, cubes/cylinders slide; a sphere launched without spin slides then rolls. `motion_summary` labels each body from contact-point slip; QC requires both regimes |
| camera fixed in a clip, varied between clips | one pose per scene (`radius`, `elevation`, `azimuth`, `focal_length` sampled once, widened until every body fits), repeated for all frames of all three branches |

### Branches

| branch | scene | subject velocity |
| --- | --- | --- |
| `factual` | `spec.scene` | sampled `factual_velocity` |
| `counterfactual` | same bodies, same poses | `factual_velocity` **discarded**, new `counterfactual_velocity` applied at step 0 (new speed + heading) |
| `subject_removed` | subject absent | — |

Ground truth (`ground_truth.json`) comes from `graph_extraction.extract_ground_truth(factual,
counterfactual, subject, 0, exclude_nodes=(floor,))`: added/removed/changed contact episodes,
`hard_affected` (causally reachable through contact chains), `soft_affected` (state diverges
without a contact path) and propagation paths.

## Artifacts

```
<output>/
  config.yaml                 ranges used
  manifest.jsonl              one row per instance (split, roles, velocities, QC + GT summary, render device)
  dataset_summary.json
  README.md                   dataset card (written by the publisher)
  instances/<index>/
    instance.json             VelocityInstanceSpec (scene, physics, visual scene, camera, roles)
    qc.json                   QCReport + rejected attempts
    ground_truth.json
    <branch>/
      sim_log/                immutable per-step states [T,N,13] + contacts (interventions.logging format)
      graph.json              contact episodes between bodies (+ floor episodes listed separately), in steps and frames
      video.mp4               RGB, 48 frames @ 24 fps (H.264)
      mask.mp4                colourised instance masks
      depth.mp4               colourised depth map video (viridis colormap)
      flow.mp4                colourised optical flow video (Middlebury color wheel)
      segmentation.npz        uint8 [T,H,W]; 0 = floor/background, k = index of object_ids[k]
      depth.npz               float16 [T,H,W] clipped at the camera far plane
      forward_flow.npz        float16 [T,H,W,2] optical flow vectors
      tracking.npz            positions, quaternions (wxyz), velocities, 2D image positions, boxes (yxyx, normalised),
                              visible pixels, presence flags — indexed by the dataset-wide object order
      render_info.json        device actually used (GPU/OptiX, CUDA, … or CPU), resolution, spp, timing
```

Segmentation ids and `tracking.npz` object order are identical across the three branches (the
removed subject keeps its slot with `present=False`), so masks and tracks line up between branches.

## Running

Physics only (any Python ≥3.9 with the `thesis` dependencies):

```bash
python scripts/build_velocity_dataset.py --output out --seed 0 --count 8 --no-render
```

Physics + GPU render in one process (needs `bpy==4.2.0`, Python 3.11 — `requirements_render.txt`):

```bash
python scripts/build_velocity_dataset.py --output out --seed 0 --count 8 --require-gpu
```

Render with a different interpreter or inside Docker:

```bash
python scripts/build_velocity_dataset.py --output out --count 8 \
  --render-command "/path/to/.venv-render/bin/python scripts/render_velocity_intervention.py"
```

Re-render one branch: delete its `render_info.json` and rerun. Regenerate a scene: delete
`instances/<index>`.

Publish:

```bash
HF_TOKEN=hf_xxx python scripts/publish_velocity_dataset.py --output out --repo-id user/kubric-velocity-intervention
```

### GPU rendering

`KUBRIC_USE_GPU=true` only flips `cycles.device`; Cycles also needs a `compute_device_type` and
enabled devices. `scripts/render_velocity_intervention.py::enable_gpu` probes `OPTIX → CUDA →
HIP → ONEAPI → METAL`, enables the first backend with a device, switches the OptiX denoiser on,
and records the result in `render_info.json`. `--require-gpu` turns a silent CPU fallback into an
error.

### Docker

`docker/KubricGPU.Dockerfile` builds a CUDA 12.4 + Python 3.11 image with `bpy==4.2.0`
(Blender 4.2 LTS with CUDA/OptiX kernels) and the pinned `requirements_render.txt`; the stock
`kubricdockerhub/*` images ship Blender 2.93 without GPU kernels. Run with
`docker run --gpus all …` (NVIDIA Container Toolkit required).

### Marimo / molab

`notebooks/velocity_intervention_molab.py` drives everything from https://molab.marimo.io:
clone → `apt` Docker install (when permitted) → GPU image or native `uv` env with the same pins →
smoke test → generation → preview → Hugging Face upload with an access token.

## Tests

`tests/test_velocity_intervention.py` covers determinism, the configuration invariants (fixed
mass, single subject velocity, roles, shapes), material/physics coupling, the static camera,
branch construction, simulation shapes, QC (mixed motion, struck/untouched, quiet removed
branch, non-null graph delta) and artifact round trips. Rendering is exercised manually because
it needs `bpy`.
