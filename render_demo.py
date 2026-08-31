"""render_demo.py — renders a 24-frame animation of a bouncing ball and saves PNGs."""
import logging
import os
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "render_demo")
SCRATCH_DIR = os.path.join(SCRIPT_DIR, "output", "render_demo_tmp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Scene
scene = kb.Scene(resolution=(512, 512))
scene.frame_start = 1
scene.frame_end   = 24
scene.frame_rate  = 24

# --- Objects
scene += kb.Cube(name="floor", scale=(5, 5, 0.1), position=(0, 0, -0.1))
ball = kb.Sphere(name="ball", scale=0.5, position=(0, 0, 0.5))
scene += ball
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                              look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(4, -3, 3),
                               look_at=(0, 0, 1))

# --- Renderer must be created before keyframes so it observes keyframe events
renderer = KubricRenderer(
    scene,
    scratch_dir=SCRATCH_DIR,
    samples_per_pixel=32,
    use_denoising=True,
)

# --- Animate ball: simple up-down bounce (renderer must already be attached)
for frame, z in [(1, 0.5), (6, 2.5), (12, 0.5), (18, 2.0), (24, 0.5)]:
    ball.position = (0, 0, z)
    ball.keyframe_insert("position", frame=frame)

# --- Render all frames
logging.info("Rendering %d frames...", scene.frame_end - scene.frame_start + 1)
frames_dict = renderer.render(
    frames=range(scene.frame_start, scene.frame_end + 1),
    return_layers=["rgba"],
)

# --- Save PNGs
kb.write_image_dict(frames_dict, OUTPUT_DIR)
renderer.save_state(f"{OUTPUT_DIR}/demo.blend")
logging.info("Done! Frames saved to %s", OUTPUT_DIR)
