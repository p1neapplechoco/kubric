"""Render every supported primitive with every shipped material family.

Purpose: create a visual 4-by-7 matrix and individual crops for shape/material combinations.
Public API: the module executes the matrix render when launched as a script.
Dependencies: Kubric, Blender, Pillow, and the thesis environment; no Docker.
Trust boundary: colors and Principled BSDF settings are demonstration priors, not measured materials.
"""

from __future__ import annotations

import logging
from pathlib import Path

import kubric as kb
from PIL import Image
from kubric.renderer import Blender


logging.basicConfig(level="INFO")

SHAPES = ("cube", "sphere", "cylinder", "capsule")
MATERIALS = (
    ("metal", (0.65, 0.68, 0.72, 1.0), 1.0, 0.22, 0.0, 1.45),
    ("rubber", (0.04, 0.05, 0.06, 1.0), 0.0, 0.82, 0.0, 1.45),
    ("plastic", (0.10, 0.38, 0.82, 1.0), 0.0, 0.30, 0.0, 1.45),
    ("ceramic", (0.88, 0.84, 0.72, 1.0), 0.0, 0.20, 0.0, 1.52),
    ("glass", (0.55, 0.72, 0.86, 0.42), 0.0, 0.06, 0.92, 1.50),
    ("wood", (0.42, 0.12, 0.035, 1.0), 0.0, 0.58, 0.0, 1.45),
    ("stone", (0.32, 0.34, 0.36, 1.0), 0.0, 0.76, 0.0, 1.50),
)


def _material(values):
  _, color, metallic, roughness, transmission, ior = values
  return kb.PrincipledBSDFMaterial(
      color=color,
      metallic=metallic,
      roughness=roughness,
      transmission=transmission,
      ior=ior,
  )


def _shape(name, shape, position, material):
  if shape == "cube":
    return kb.Cube(name=name, scale=(0.38, 0.38, 0.38), position=position, material=material)
  if shape == "sphere":
    return kb.Sphere(name=name, scale=0.40, position=position, material=material)
  if shape == "cylinder":
    return kb.Cylinder(name=name, scale=(0.34, 0.34, 0.48), position=position, material=material)
  if shape == "capsule":
    return kb.Capsule(name=name, scale=(0.32, 0.32, 0.43), position=position, material=material)
  raise ValueError("unknown shape: {!r}".format(shape))


scene = kb.Scene(resolution=(1400, 900), frame_start=0, frame_end=1, frame_rate=24)
for row, shape in enumerate(SHAPES):
  for column, material_values in enumerate(MATERIALS):
    x = -4.5 + column * 1.5
    z = 3.1 - row * 1.45
    scene += _shape(
        "{}_{}".format(shape, material_values[0]),
        shape,
        (x, 0.0, z),
        _material(material_values),
    )

scene += kb.DirectionalLight(
    name="key",
    position=(-4.0, -6.0, 8.0),
    look_at=(0.0, 0.0, 1.0),
    intensity=4.0,
)
scene += kb.DirectionalLight(
    name="fill",
    position=(5.0, 1.0, 4.0),
    look_at=(0.0, 0.0, 1.0),
    intensity=1.5,
)
scene += kb.OrthographicCamera(
    name="camera",
    position=(0.0, -12.0, 1.6),
    look_at=(0.0, 0.0, 1.6),
    orthographic_scale=6.6,
)

renderer = Blender(scene, samples_per_pixel=16, adaptive_sampling=False, use_denoising=False)
logging.info("Rendering %d shapes x %d material families", len(SHAPES), len(MATERIALS))
data_stack = renderer.render_still(return_layers=("rgba", "segmentation"))
output_dir = Path("output/object_material_matrix")
output_dir.mkdir(parents=True, exist_ok=True)
kb.write_image_dict(
    {key: value[None, ...] for key, value in data_stack.items()}, output_dir
)

image = Image.open(output_dir / "rgba_00000.png").convert("RGBA")
cell_width = image.width // len(MATERIALS)
cell_height = image.height // len(SHAPES)
for row, shape in enumerate(SHAPES):
  for column, material_values in enumerate(MATERIALS):
    material_name = material_values[0]
    left = column * cell_width
    top = row * cell_height
    crop = image.crop((left, top, left + cell_width, top + cell_height))
    crop.save(output_dir / "{}_{}.png".format(shape, material_name))
logging.info("Wrote contact sheet and %d named combination images to %s", len(SHAPES) * len(MATERIALS), output_dir)
