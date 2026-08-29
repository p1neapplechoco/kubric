# Object-shape demos

This repository has two different demo families:

- The intervention demo compares three outcomes of one eleven-object scene:
  `normal`, `trajectory_changed`, and `target_removed`.
- The object-shape demo is a visual gallery of the four supported primitive
  proxies: `cube`, `sphere`, `cylinder`, and `capsule`.

## Shape gallery

Run the gallery from the repository root with the isolated `thesis` Conda
interpreter. This uses the native Blender Python module and does not use Docker.

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" examples/object_shapes.py
```

The script creates a 640x480 Blender scene containing one instance of each
shape and writes `output/object_shapes/rgba_00000.png` plus
`output/object_shapes/segmentation_00000.png`. To save an interactive `.blend`
file or additional render layers, use the same scene setup and call the
renderer methods shown in `examples/bouncing_balls.py`. The four shapes are
also covered by the geometry, PyBullet, and Blender tests.

### Size semantics

`ObjectConfig.size` is a local half-extent:

| Shape | Meaning of `size` |
| --- | --- |
| `cube` | `(half_x, half_y, half_z)` |
| `sphere` | `(radius, radius, radius)` |
| `cylinder` | `(radius, radius, half_height)` |
| `capsule` | `(radius, radius, cylinder_half_height)`; the hemispherical caps add one radius to the total Z half-extent |

Cylinders and capsules use local Z as their longitudinal axis. The schema
requires equal X/Y radii for radial shapes. Intervention targets remain limited
to cubes and spheres; cylinders and capsules are available as dynamic and
non-target static proxies.

## Three-branch intervention demo

The rendered intervention files are in:

```text
output/demo_collision_intervention/
  normal_blender.mp4
  trajectory_changed_blender.mp4
  target_removed_blender.mp4
  trajectory_intervention_demo.mp4
```

The combined `trajectory_intervention_demo.mp4` is the labelled three-panel
comparison. The first three files are the individual branches. Generate the
replay bundle, render all three branches, and compose the comparison with:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" -m scripts.demo_collision_intervention --output output/demo_collision_intervention
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" -m scripts.render_demo_branches_blender --states-dir output/demo_collision_intervention --branches normal trajectory_changed target_removed --resolution 640 540 --fps 24 --samples 1
$env:PATH = "C:\Users\uya7hc\.conda\envs\kubric-demo\Library\bin;$env:PATH"
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" -m scripts.compose_intervention_demo --states-dir output/demo_collision_intervention
```

The branch meanings are:

| Branch | Meaning |
| --- | --- |
| `normal` | The factual rollout with the small side-chain interaction. |
| `trajectory_changed` | The counterfactual rollout with the altered target trajectory and large rack-chain propagation. |
| `target_removed` | A demo-only visualization in which the target is removed at step 40; it is not an attested dataset recipe. |

For the full physics contract, replay metadata, contact graph, and trust
boundary, see [Trajectory interventions](trajectory_interventions.md).

## Materials and textures

The native Kubric material gallery is [examples/materials_textures.py](../examples/materials_textures.py).
It renders seven Principled BSDF examples: metal, rubber, plastic, ceramic,
glass, wood, and stone. The gallery varies base color, metallic response,
roughness, transmission, and index of refraction under a shared light rig.

Run it with:

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" -m examples.materials_textures
```

Outputs are written to `output/materials_textures/rgba_00000.png` and
`output/materials_textures/segmentation_00000.png`.

For every combination in one image, run
[examples/object_material_matrix.py](../examples/object_material_matrix.py):

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
& "C:\Users\uya7hc\.conda\envs\thesis\python.exe" -m examples.object_material_matrix
```

This writes the contact sheet [rgba_00000.png](../output/object_material_matrix/rgba_00000.png)
and 28 named images under `output/object_material_matrix/`, using the naming
pattern `<shape>_<material>.png`, for example `capsule_glass.png` and
`cylinder_wood.png`. The matrix rows are ordered `cube`, `sphere`, `cylinder`,
`capsule`; columns are ordered `metal`, `rubber`, `plastic`, `ceramic`,
`glass`, `wood`, `stone`.

The intervention appearance sampler additionally defines the renderer-independent
texture kinds `solid`, `noise`, `checker`, `wood`, `marble`, `speckle`, and
`image` in [interventions/appearance.py](../interventions/appearance.py). Those
texture values are sampled deterministically and stored in visual scene specs;
the native gallery above demonstrates the currently available Kubric renderer
material controls.
