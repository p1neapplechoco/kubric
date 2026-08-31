# Rendering Videos with Blender via Kubric

This guide explains how to render a proper video (multi-frame animation) using Kubric's
Blender renderer. It is written for agents and developers working in this repository.

---

## Prerequisites

### 1. Blender must be installed

Kubric's renderer is a Python wrapper around Blender's Python API (`bpy`). Blender must be
installed and its Python environment must be the one running your script.

**Check whether Blender is available:**

```bash
blender --version
```

If Blender is not installed, download it from [blender.org](https://www.blender.org/download/)
and ensure the `blender` binary is on your `PATH`.

### 2. Python dependencies

Install Kubric and its dependencies inside Blender's Python:

```bash
# From repo root — installs kubric into the active Python environment
pip install -e .
```

Or use the provided Docker image (recommended for CI / headless rendering):

```bash
docker build -f docker/Dockerfile -t kubric .
```

### 3. GPU acceleration (optional)

Set the environment variable to enable GPU rendering with CUDA/OptiX:

```bash
export KUBRIC_USE_GPU=true
```

When disabled (default), rendering runs on CPU via Blender's Cycles engine.

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

### Step 3 — Keyframe animated properties

Use `keyframe_insert` to animate any property over time:

```python
# Frame 1: ball at z=1
scene.objects["ball"].position = (0, 0, 1.0)
scene.objects["ball"].keyframe_insert("position", frame=1)

# Frame 60: ball at z=3
scene.objects["ball"].position = (0, 0, 3.0)
scene.objects["ball"].keyframe_insert("position", frame=60)
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

### Step 4 — Create the renderer

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

Kubric does **not** encode a video file directly. Use `ffmpeg` after rendering:

```bash
# H.264 MP4 at 24 fps from RGBA PNGs
ffmpeg -framerate 24 \
       -i output/rgba_%05d.png \
       -c:v libx264 \
       -pix_fmt yuv420p \
       -crf 18 \
       output/video.mp4
```

For lossless encoding:

```bash
ffmpeg -framerate 24 \
       -i output/rgba_%05d.png \
       -c:v ffv1 \
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

## Headless / Docker Rendering

For server or CI rendering without a display:

```bash
# Run a script headlessly inside the kubric Docker image
docker run --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  kubric \
  python examples/helloworld.py
```

Set `KUBRIC_USE_GPU=true` and pass `--gpus all` to `docker run` for GPU rendering.

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
scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.0))
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                              look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                               look_at=(0, 0, 1))

# --- Animate the ball
scene.objects["ball"].position = (0, 0, 1.0)
scene.objects["ball"].keyframe_insert("position", frame=1)
scene.objects["ball"].position = (0, 0, 3.0)
scene.objects["ball"].keyframe_insert("position", frame=60)

# --- Renderer
renderer = KubricRenderer(
    scene,
    scratch_dir="output_tmp",
    samples_per_pixel=64,
    use_denoising=True,
)

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

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: bpy` | Not running inside Blender Python | Run via `blender --background --python script.py` or use the Docker image |
| Purple / missing textures | Texture paths unresolved | Pass `ignore_missing_textures=True` to `render()` or check asset paths |
| Black frames | No lights in scene | Add at least one `DirectionalLight` or `PointLight` |
| Very slow rendering | High `samples_per_pixel` on CPU | Lower SPP, enable `adaptive_sampling=True`, or set `KUBRIC_USE_GPU=true` |
| `scene.blend1` created instead of `scene.blend` | Blender auto-backup | Kubric deletes the old `.blend` before saving — ensure no file lock |
| Frames out of order in ffmpeg | Wrong glob pattern | Use zero-padded filenames: `%05d` for 5-digit frame numbers |
