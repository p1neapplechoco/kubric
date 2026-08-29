"""Render one gallery scene containing all supported primitive object shapes.

Purpose: provide a minimal visual smoke demo for cube, sphere, cylinder, and capsule.
Public API: the module executes the gallery when launched as a script.
Dependencies: Kubric, its Blender renderer, and the thesis environment; no Docker.
Trust boundary: this is a visual primitive demonstration, not a physics benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path

import kubric as kb
from kubric.renderer import Blender


logging.basicConfig(level="INFO")

scene = kb.Scene(resolution=(640, 480), frame_start=0, frame_end=1, frame_rate=24)
scene += kb.Cube(
    name="cube",
    scale=(0.55, 0.55, 0.55),
    position=(-1.8, 0.0, 0.55),
    material=kb.FlatMaterial(color=(0.85, 0.20, 0.12, 1.0)),
)
scene += kb.Sphere(
    name="sphere",
    scale=0.55,
    position=(-0.6, 0.0, 0.55),
    material=kb.FlatMaterial(color=(0.12, 0.42, 0.85, 1.0)),
)
scene += kb.Cylinder(
    name="cylinder",
    scale=(0.48, 0.48, 0.70),
    position=(0.65, 0.0, 0.70),
    material=kb.FlatMaterial(color=(0.95, 0.68, 0.10, 1.0)),
)
scene += kb.Capsule(
    name="capsule",
    scale=(0.42, 0.42, 0.62),
    position=(1.85, 0.0, 0.84),
    material=kb.FlatMaterial(color=(0.18, 0.68, 0.34, 1.0)),
)
scene += kb.Cube(
    name="floor",
    scale=(3.6, 1.8, 0.08),
    position=(0.0, 0.0, -0.08),
    material=kb.FlatMaterial(color=(0.08, 0.09, 0.11, 1.0)),
    static=True,
)
scene += kb.DirectionalLight(
    name="sun",
    position=(-3.0, -4.0, 6.0),
    look_at=(0.0, 0.0, 0.5),
    intensity=4.0,
)
scene += kb.PerspectiveCamera(
    name="camera",
    position=(0.0, -7.0, 3.4),
    look_at=(0.0, 0.0, 0.65),
    focal_length=52.0,
)

renderer = Blender(
    scene,
    samples_per_pixel=16,
    adaptive_sampling=False,
    use_denoising=False,
)
logging.info("Rendering supported object shapes: cube, sphere, cylinder, capsule")
output_dir = Path("output/object_shapes")
data_stack = renderer.render_still(
    return_layers=("rgba", "segmentation")
)
kb.write_image_dict(
    {key: value[None, ...] for key, value in data_stack.items()}, output_dir
)
logging.info("Wrote shape gallery layers to %s", output_dir)
