# Rendering Videos with Blender via Kubric

This guide explains how to render a proper video (multi-frame animation) using Kubric's
Blender renderer. It is written for agents and developers working in this repository.

---

## Prerequisites

### 1. The `thesis` Conda environment (Windows, Docker-free)

All rendering in this repository runs inside a single Conda environment named
`thesis`. There is no Docker dependency anywhere in this workflow: Blender is
provided by the pip-installed `bpy` module (Blender-as-a-Python-module),
running headlessly by construction because it is imported directly rather than
launched as a separate application.

```
C:\Users\uya7hc\.conda\envs\thesis\python.exe
```

See [`docs/environment_thesis.md`](docs/environment_thesis.md) for how the
environment is created and verified, and for the platform notes that apply on
Windows (long-path support, publication retries, etc.).

**Check that the interpreter and `bpy` are available:**

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" -c "import kubric as kb; import kubric.renderer.blender; import imageio_ffmpeg; import bpy; print('ok')"
```

Expected: prints `ok` and exits with code 0. If any import fails, install the
missing package into `thesis` — never into Conda `base` or the system Python.

### 2. Command prelude

The agent terminal is a persistent session, so clear `PYTHONPATH` before
running anything and give Matplotlib a writable config directory:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
```

### 3. GPU acceleration (optional)

Set the environment variable to enable GPU rendering with CUDA/OptiX:

```powershell
$env:KUBRIC_USE_GPU = "true"
```

When disabled (default), rendering runs on CPU via Blender's Cycles engine.

### 4. `ffmpeg`/`ffprobe` on `PATH` (only for composition/verification)

The pip-installed `bpy` wheel does not ship a functional native FFmpeg movie
muxer, so per-frame rendering always falls back to encoding through the
`imageio-ffmpeg` package, which is self-contained and needs nothing extra on
`PATH`. However, [`scripts/compose_intervention_demo.py`](scripts/compose_intervention_demo.py)
and the optional `ffprobe`-based verification in
[`scripts/render_demo_branches_blender.py`](scripts/render_demo_branches_blender.py)
call the real `ffmpeg`/`ffprobe` binaries (needed for `drawtext` overlays),
which `thesis` does not include. Point `PATH` at any Conda environment that has
them, for example:

```powershell
$env:PATH = "C:\Users\uya7hc\.conda\envs\kubric-demo\Library\bin;$env:PATH"
```

---

## Core Concepts

| Concept | Class | Description |
|---|---|---|
| Scene | `kb.Scene` | Holds resolution, frame range, FPS, camera, and all objects |
| Renderer | `KubricRenderer` (= `kubric.renderer.blender.Blender`) | Blender/Cycles backend, observes the scene |
| Objects | `kb.Cube`, `kb.Sphere`, … | 3-D assets added to the scene |
| Lights | `kb.DirectionalLight`, `kb.PointLight`, … | Light sources |
| Camera | `kb.PerspectiveCamera` | Point-of-view for rendering |

---

## Minimal Working Example — Still Frame

```python
import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

scene = kb.Scene(resolution=(512, 512))
renderer = KubricRenderer(scene, scratch_dir="output_tmp")

scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1))
scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                              look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                               look_at=(0, 0, 1))

frame = renderer.render_still()            # renders scene.frame_start by default

kb.write_png(frame["rgba"], "output/frame.png")
```

Save this as `examples/still_frame_demo.py` and run it with the `thesis`
interpreter (no `blender --background` wrapper needed, since `bpy` is imported
directly):

```powershell
& $py examples/still_frame_demo.py
```

---

## Rendering a Video (Multi-Frame Animation)

### Step 1 — Define the scene timeline

```python
scene = kb.Scene(resolution=(512, 512))
scene.frame_start = 1
scene.frame_end   = 60   # 60 frames
scene.frame_rate  = 24   # 24 fps → 2.5-second clip
```

### Step 2 — Add objects, lights, and camera

```python
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                              look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                               look_at=(0, 0, 1))
scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
```

### Step 3 — Create the renderer

> ⚠️ **Important:** The renderer must be created **before** inserting any keyframes.
> Kubric uses a traitlets observer system — the renderer subscribes to scene events when it is
> instantiated. Keyframes inserted before the renderer exists are never forwarded to Blender's
> animation curves and will have no effect at render time.

```python
from kubric.renderer.blender import Blender as KubricRenderer

renderer = KubricRenderer(
    scene,
    scratch_dir="output_tmp",      # temp dir for per-frame EXR/PNG files
    samples_per_pixel=128,         # ray-tracing quality (higher = slower)
    use_denoising=True,            # Blender denoiser (recommended)
    adaptive_sampling=False,       # speed-quality trade-off
)
```

**Key renderer parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `samples_per_pixel` | 128 | Rays per pixel — higher improves quality |
| `use_denoising` | `True` | Blender NLM/OptiX denoiser |
| `adaptive_sampling` | `False` | Auto-adjusts samples per tile |
| `background_transparency` | `False` | Renders alpha channel in background |
| `motion_blur` | `None` | Enable motion blur (pass shutter duration in frames) |
| `custom_scene` | `None` | Path to a `.blend` file to load as the base scene |

### Step 4 — Keyframe animated properties

Always keep a direct Python reference to the object you want to animate — `scene.assets`
returns a tuple, not a dict, so there is no name-based lookup.

```python
ball = kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
scene += ball

# renderer must already exist at this point (see Step 3)

ball.position = (0, 0, 1.0)
ball.keyframe_insert("position", frame=1)

ball.position = (0, 0, 3.0)
ball.keyframe_insert("position", frame=60)
```

Camera paths are keyframed the same way:

```python
scene.camera.position = (3, -1, 4)
scene.camera.look_at((0, 0, 1))
scene.camera.keyframe_insert("position", frame=1)
scene.camera.keyframe_insert("quaternion", frame=1)

scene.camera.position = (-3, -1, 4)
scene.camera.look_at((0, 0, 2))
scene.camera.keyframe_insert("position", frame=60)
scene.camera.keyframe_insert("quaternion", frame=60)
```

### Step 5 — Render all frames

```python
frames_dict = renderer.render(
    frames=range(scene.frame_start, scene.frame_end + 1),
    return_layers=["rgba", "depth", "segmentation"],
)
# frames_dict["rgba"].shape == (60, 512, 512, 4)
```

**Available return layers:**

| Layer | Shape | Description |
|---|---|---|
| `rgba` | `(T, H, W, 4)` | Color + alpha |
| `depth` | `(T, H, W, 1)` | Depth map (float32) |
| `normal` | `(T, H, W, 3)` | Surface normals (uint16) |
| `segmentation` | `(T, H, W, 1)` | Object instance IDs (int) |
| `forward_flow` | `(T, H, W, 2)` | Optical flow t→t+1 |
| `backward_flow` | `(T, H, W, 2)` | Optical flow t→t-1 |
| `object_coordinates` | `(T, H, W, 3)` | Per-object 3-D coords (uint16) |
| `uv` | `(T, H, W, 2)` | UV coordinates |

### Step 6 — Save output

```python
import os
os.makedirs("output", exist_ok=True)

# Save every RGBA frame as PNG
kb.write_image_dict(frames_dict, "output")

# Also save the Blender scene for inspection / re-rendering
renderer.save_state("output/scene.blend")
```

`kb.write_image_dict` writes one PNG per frame per layer under the target directory:
`output/rgba_00001.png`, `output/rgba_00002.png`, …

---

## Assembling Frames into a Video File

Kubric does **not** encode a video file directly. Use `ffmpeg` after rendering
(add a Conda environment's `Library\bin` to `PATH` as shown in
[Prerequisites](#4-ffmpegffprobe-on-path-only-for-compositionverification) if
`thesis` itself has no `ffmpeg` binary):

```powershell
# H.264 MP4 at 24 fps from RGBA PNGs
ffmpeg -framerate 24 `
       -i output/rgba_%05d.png `
       -c:v libx264 `
       -pix_fmt yuv420p `
       -crf 18 `
       output/video.mp4
```

For lossless encoding:

```powershell
ffmpeg -framerate 24 `
       -i output/rgba_%05d.png `
       -c:v ffv1 `
       output/video.mkv
```

---

## Using a Custom `.blend` Scene

Pass a pre-built `.blend` file as the base environment. Kubric's physics / object model
operates on top of it; the `.blend` content only affects the rendered output.

```python
renderer = KubricRenderer(
    scene,
    scratch_dir="output_tmp",
    custom_scene="path/to/my_environment.blend",
)
```

> **Note:** The custom scene is not accessible from the Kubric object graph and is not
> simulated by PyBullet. Use it for static backgrounds, props, and lighting rigs only.

---

## Three-Branch Intervention Demo (Blender, `thesis` Conda, no Docker)

This is the canonical end-to-end demo in this repository: three synchronized
Blender renders of one eleven-object collision scene — `normal`,
`trajectory_changed`, and `target_removed` — plus a composed side-by-side
comparison video. Everything runs through the `thesis` interpreter; nothing
here uses Docker.

| Branch | Meaning |
| --- | --- |
| `normal` | The factual rollout with the small side-chain interaction. |
| `trajectory_changed` | The counterfactual rollout with the altered target trajectory and large rack-chain propagation. |
| `target_removed` | A demo-only visualization in which the target is removed at step 40; it is not an attested dataset recipe. |

### Step 1 — Generate the canonical replay bundle

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
& $py -m scripts.demo_collision_intervention --output output/demo_collision_intervention
```

This writes `summary.json` plus `normal_states.npy`, `trajectory_changed_states.npy`,
`target_removed_states.npy`, and their matching `*_presence.npy` files.

### Step 2 — Render all three branches

```powershell
& $py -m scripts.render_demo_branches_blender `
    --states-dir output/demo_collision_intervention `
    --branches normal trajectory_changed target_removed
```

This renders the full default frame count (no `--max-frames` truncation) for
every requested branch and atomically publishes:

- `output/demo_collision_intervention/normal_blender.mp4`
- `output/demo_collision_intervention/trajectory_changed_blender.mp4`
- `output/demo_collision_intervention/target_removed_blender.mp4`

Pip-distributed `bpy` wheels typically list `FFMPEG` as an image-format enum
value without a working muxer behind it. The renderer probes this once per
branch with a throwaway one-frame render and transparently falls back to the
portable `imageio-ffmpeg` encoder when the native muxer does not actually
produce a file — no flags or environment variables are needed to trigger this.

### Step 3 — Compose the synchronized comparison video

```powershell
$env:PATH = "C:\Users\uya7hc\.conda\envs\kubric-demo\Library\bin;$env:PATH"
& $py -m scripts.compose_intervention_demo `
    --states-dir output/demo_collision_intervention `
    --output output/demo_collision_intervention/trajectory_intervention_demo.mp4
```

Unlike branch rendering, composition calls the real `ffmpeg`/`ffprobe`
binaries directly (for the `drawtext` label overlays), so a `PATH` entry with
those binaries is required; `imageio-ffmpeg`'s bundled binary is not enough.

### Step 4 — Verify all four outputs

```powershell
& $py -c "from pathlib import Path; root=Path('output/demo_collision_intervention'); names=['normal_blender.mp4','trajectory_changed_blender.mp4','target_removed_blender.mp4','trajectory_intervention_demo.mp4']; missing=[n for n in names if not (root/n).is_file() or (root/n).stat().st_size<1]; print('missing', missing); raise SystemExit(1 if missing else 0)"
```

Expected: prints `missing []` and exits with code 0.

For the full physics contract, replay metadata, contact graph, and trust
boundary behind this demo, see
[Trajectory interventions](docs/trajectory_interventions.md) and
[Object-shape demos](docs/object_shape_demos.md).

---

## Complete Video Rendering Script

```python
import logging
import os
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

# --- Scene setup
scene = kb.Scene(resolution=(512, 512))
scene.frame_start = 1
scene.frame_end   = 60
scene.frame_rate  = 24

# --- Populate scene
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1))
ball = kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
scene += ball
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                              look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                               look_at=(0, 0, 1))

# --- Renderer must be created BEFORE keyframes
renderer = KubricRenderer(
    scene,
    scratch_dir="output_tmp",
    samples_per_pixel=64,
    use_denoising=True,
)

# --- Animate the ball (renderer is already attached)
ball.position = (0, 0, 1.0)
ball.keyframe_insert("position", frame=1)
ball.position = (0, 0, 3.0)
ball.keyframe_insert("position", frame=60)

# --- Render all frames
frames_dict = renderer.render(
    frames=range(scene.frame_start, scene.frame_end + 1),
    return_layers=["rgba"],
)

# --- Save output
os.makedirs("output", exist_ok=True)
kb.write_image_dict(frames_dict, "output")
renderer.save_state("output/scene.blend")

# --- Assemble video with ffmpeg (requires ffmpeg on PATH)
os.system(
    "ffmpeg -framerate 24 -i output/rgba_%05d.png "
    "-c:v libx264 -pix_fmt yuv420p -crf 18 output/video.mp4"
)
```

Save this as `examples/complete_video_demo.py` and run it with `& $py
examples/complete_video_demo.py` (see the [command prelude](#2-command-prelude)
for `$py`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: bpy` | Not running inside the `thesis` interpreter | Confirm you invoked `C:\Users\uya7hc\.conda\envs\thesis\python.exe`, not `base` or the system Python; `pip install bpy==4.2.0` inside `thesis` if missing (see `docs/environment_thesis.md`) |
| Blender animation render "Finished" but the MP4 file never appears | Pip `bpy` wheel lists `FFMPEG` in the format enum without a working native muxer | Already handled automatically by `render_demo_branches_blender.py`'s one-frame capability probe, which falls back to `imageio-ffmpeg`; for custom scripts, verify the output file after `bpy.ops.render.render(animation=True)` and fall back to `renderer.render()` + `imageio.mimwrite` if it is missing |
| `RuntimeError: ffprobe was not found on PATH` / `RuntimeError: ffmpeg was not found on PATH` | `compose_intervention_demo.py` needs the real binaries for `drawtext`, and `thesis` does not include them | Add a Conda environment's `Library\bin` to `PATH`, e.g. `$env:PATH = "C:\Users\uya7hc\.conda\envs\kubric-demo\Library\bin;$env:PATH"` |
| Purple / missing textures | Texture paths unresolved | Pass `ignore_missing_textures=True` to `render()` or check asset paths |
| Black frames | No lights in scene | Add at least one `DirectionalLight` or `PointLight` |
| Animation keyframes have no effect (all frames identical) | Renderer created after `keyframe_insert` calls | Create `KubricRenderer` **before** any `keyframe_insert` — it must be attached to observe keyframe events |
| `AttributeError: 'Scene' object has no attribute 'objects'` | `scene.objects` does not exist | Keep a direct Python variable reference to each asset; use `scene.assets` (returns a tuple) to iterate all assets |
| Very slow rendering | High `samples_per_pixel` on CPU | Lower SPP, enable `adaptive_sampling=True`, or set `KUBRIC_USE_GPU=true` |
| `scene.blend1` created instead of `scene.blend` | Blender auto-backup | Kubric deletes the old `.blend` before saving — ensure no file lock |
| Frames out of order in ffmpeg | Wrong glob pattern | Use zero-padded filenames: `%05d` for 5-digit frame numbers |
