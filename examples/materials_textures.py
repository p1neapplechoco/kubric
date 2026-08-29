"""Render a gallery of physically based material controls.

Purpose: demonstrate material families and texture-like surface variation using Kubric's Principled BSDF material.
Public API: the module executes the gallery when launched as a script.
Dependencies: Kubric, its Blender renderer, and the thesis environment; no Docker.
Trust boundary: this gallery demonstrates renderer parameters, not calibrated physical measurements.
"""

from __future__ import annotations

import logging
from pathlib import Path

import kubric as kb
from kubric.renderer import Blender


logging.basicConfig(level="INFO")

scene = kb.Scene(resolution=(768, 512), frame_start=0, frame_end=1, frame_rate=24)
materials = (
    ("metal", (0.65, 0.68, 0.72, 1.0), 1.0, 0.22, 0.0, 1.45),
    ("rubber", (0.04, 0.05, 0.06, 1.0), 0.0, 0.82, 0.0, 1.45),
    ("plastic", (0.10, 0.38, 0.82, 1.0), 0.0, 0.30, 0.0, 1.45),
    ("ceramic", (0.88, 0.84, 0.72, 1.0), 0.0, 0.20, 0.0, 1.52),
    ("glass", (0.55, 0.72, 0.86, 0.42), 0.0, 0.06, 0.92, 1.50),
    ("wood", (0.42, 0.12, 0.035, 1.0), 0.0, 0.58, 0.0, 1.45),
    ("stone", (0.32, 0.34, 0.36, 1.0), 0.0, 0.76, 0.0, 1.50),
)

for index, (_, color, metallic, roughness, transmission, ior) in enumerate(materials):
  column = index % 4
  row = index // 4
  scene += kb.Sphere(
      name="material_{}".format(index),
      scale=0.52,
      position=(-2.25 + column * 1.5, 0.25 - row * 1.45, 0.58),
      material=kb.PrincipledBSDFMaterial(
          color=color,
          metallic=metallic,
          roughness=roughness,
          transmission=transmission,
          ior=ior,
      ),
  )

scene += kb.Cube(
    name="floor",
    scale=(3.5, 1.65, 0.08),
    position=(0.0, 0.0, -0.08),
    material=kb.PrincipledBSDFMaterial(color=(0.035, 0.04, 0.05, 1.0), roughness=0.72),
    static=True,
)
scene += kb.DirectionalLight(
    name="key",
    position=(-3.0, -4.0, 6.0),
    look_at=(0.0, 0.0, 0.5),
    intensity=4.0,
)
scene += kb.DirectionalLight(
    name="fill",
    position=(4.0, 1.0, 3.0),
    look_at=(0.0, 0.0, 0.5),
    intensity=1.5,
)
scene += kb.PerspectiveCamera(
    name="camera",
    position=(0.0, -13.5, 5.0),
    look_at=(0.0, 0.0, 0.55),
    focal_length=55.0,
)

renderer = Blender(scene, samples_per_pixel=16, adaptive_sampling=False, use_denoising=False)
logging.info("Rendering material families: %s", ", ".join(item[0] for item in materials))
data_stack = renderer.render_still(return_layers=("rgba", "segmentation"))
output_dir = Path("output/materials_textures")
kb.write_image_dict(
    {key: value[None, ...] for key, value in data_stack.items()}, output_dir
)
logging.info("Wrote material gallery layers to %s", output_dir)
