# Shared visual-scene demo

This is the rendered evidence for the feature described in
[Trajectory interventions § Shared visual scenes](trajectory_interventions.md#shared-visual-scenes):
a pair samples **one** `VisualSceneSpec`, both branches read it, and the only
thing that differs between them is the intervened physics.

Two artifacts are published under `output/demo_visual_scene_sampling/`:

| Artifact | Contents |
| --- | --- |
| `shared_visual_scene_demo.mp4` | 1920×540, 72 frames — three branches side by side, one fixed camera, 640×540 and 64 spp per panel |
| `appearance_sampling_gallery.mp4` | 1440×808, 72 frames — six instances under one fixed viewpoint, so every visible difference comes from the sampler |

Both were rendered with the pip `bpy` Blender in the `thesis` environment,
following [`render_no_docker.md`](../render_no_docker.md).

## What the three panels show

`instance_328084c006cc046734ee` is a `break_contact` instance with seven objects,
two of which `objects.static_fraction` promoted to static `environment` obstacles.

| Panel | Branch | What it is |
| --- | --- | --- |
| left | `factual` | the published factual rollout |
| middle | `counterfactual` | the published `break_contact` rollout |
| right | `target removed (step 28)` | demo-only: the intervened object is deleted mid-clip |

All three panels are rendered from the same appearance record. The renderer
prints the SHA-256 of exactly what it handed to Blender, once per branch, and the
three digests agree:

```text
render-inputs digest per branch: ['c76b617e75046ba75d542ceb59984def12f60230b32988dc9dd9cc0f30720cf4']
all 3 branches rendered from ONE shared VisualSceneSpec: True
max |factual - counterfactual| position divergence: 0.2148 m
max |factual - target_removed| position divergence: 1.9667 m
```

That digest is of the *overridden* spec, because `--fixed-camera` replaces the
sampled camera path. A static camera is a repeated entry in
`CameraRenderSpec.positions` rather than a separate mode, so the override is still
a valid `VisualSceneSpec`, built with `dataclasses.replace` and hashed as such —
the demo reports the appearance it actually rendered rather than quoting the
stored hash it did not use. Rendered from the sampled camera instead, the same
instance prints `2b974a69f866d34a87d41c2b3eb54814228fe8c7c6c18d9246523c92d2baf095`,
which *is* the `visual_scene_hash` stored in its `pair.json`. Nothing else is
overridden: shapes, sizes, colours, material families, procedural textures, the
three-point light rig, and the background are what the sampler produced.

## Why a third panel

The first two panels are two similar tumbles, and nothing in them says which
object the intervention acted on. For this instance it is the object whose id is
literally `target`, pushed over `time_window` steps 3 → 16 of 72. The branches are
identical until step 4, and only three of the seven objects ever move differently:

| Object | max &#124;factual − counterfactual&#124; | Note |
| --- | --- | --- |
| `target` | 0.211 m | the intervened object |
| `object_1` | 0.215 m | downstream of the target |
| `object_2` | 0.187 m | downstream of the target |
| `object_0`, `object_4` | 0.000 m | static `environment` obstacles, they cannot move |
| `object_3`, `floor` | 0.000 m | never touched by the cascade |

The third panel makes that legible: the target is deleted part-way through, so
whatever was going to hit it sails through the space it used to occupy.
`object_1` and `object_2` end 0.59 m and 0.41 m from where the factual branch put
them, against 0.21 m for the counterfactual. Deletion is a step function in
Blender — `hide_render` and `hide_viewport` keyframes forced to `CONSTANT`
interpolation — so the object vanishes on one frame rather than fading.

Two properties of this branch are worth stating plainly:

- **It is presentation-only.** It carries `trust_model = "demo_only_removal_v1"`,
  it is never written to `pair.json`, and it is not a dataset recipe. The pair
  contract is still exactly two branches. This matches the `target_removed`
  branch of the [three-branch collision demo](trajectory_interventions.md#demo-only-removal-trust-boundary).
- **Its prefix is copied, not re-simulated.** The sampled `factual_path` is not
  stored in `pair.json`, and re-deriving it from logged target poses reproduces
  the rollout only to about 1.4 cm once the contact solver amplifies the
  difference. The branch therefore copies the published factual states up to the
  cut and restores every body's logged pose and velocity into Bullet there. The
  prefix is then bit-identical to the factual branch (0.000e+00 m), at the cost of
  Bullet solving the first post-cut step without its contact warm-start.

The cut defaults to the step the intervention window opens on, which is what keeps
all three branches sharing one prefix. The sampled windows here open on step 3 or
4 of 72, which reads as "the object was never there" rather than as a deletion, so
the published clip cuts at step 28 instead.

## The pixel claim

The digest above is a claim about the inputs. Comparing decoded frames is the
claim about the output. Allowing 16 code values for H.264 loss, the factual and
counterfactual clips agree everywhere except a small region:

- frame 0 differs in **0** pixels — both branches start from the same state;
- the peak frame differs in 1.73% of the image, the median frame in 0.29%;
- on the peak frame the entire difference sits inside `x[128:232] y[259:539]` —
  the intervened object and the object it stops touching.

A second instance (`instance_66906312e95c3f15ce07`, `maintain_contact`, one
environment obstacle) reproduces the result independently: one digest across its
branches, 0 differing pixels on frame 0, peak 2.67%, median 0.57%. Its sampled
appearance is a speckled floor with checker and glass objects, so the two also
show that the renderer follows whatever the sampler produced rather than a fixed
look.

Against the removal branch the same measurement localises the intervention in
time: 0.00% of pixels differ through frame 27, then 0.91% at frame 28, 1.70% at
frame 29, rising to 2.27% by frame 60.

## The gallery

`appearance_sampling_gallery.mp4` tiles six instances under one viewpoint:

| Tile | Recipe | Shapes | Textures | Material families |
| --- | --- | --- | --- | --- |
| `328084c0` | `break_contact`, +2 static env | cube sphere | marble solid speckle wood | ceramic glass metal rubber wood |
| `66906312` | `maintain_contact`, +1 static env | cube | checker solid speckle | ceramic glass plastic |
| `16ee376f` | `create_collision` | cube | checker marble noise speckle | ceramic metal plastic stone |
| `8ae581d0` | `maintain_contact`, +1 static env | cube sphere | noise solid speckle | glass metal plastic rubber stone wood |
| `35fca5e6` | `remove_collision` | cube sphere | checker noise solid speckle | metal plastic rubber wood |
| `03769be6` | `retime` | cube sphere | marble noise solid speckle | ceramic metal plastic rubber stone wood |

Same camera in all six, so every visible difference — olive metal floor versus
pink speckle versus purple marble, glass cubes versus lacquered spheres, warm
versus cool light rigs — comes from `sample_visual_scene`.

## Reproducing it

The batch that backs these clips is generated from a copy of
`configs/scene_ranges_visual.yaml` with `frame_range: [0, 72]` and
`frame_rate: 240`, one output frame per physics step, so the rollout is the same
240 Hz simulation sampled densely enough to encode:

```bash
python -m scripts.generate_dataset \
  --config output/demo_visual_scene_sampling/scene_ranges_visual_video.yaml \
  --output output/demo_visual_scene_sampling/dataset_video \
  --seed 20260901 --num-instances 6 --max-attempts 40
```

The render harness that turns a published pair into these clips is deliberately
**not** part of the tracked tree. Wiring Blender to `VisualSceneSpec` is the next
change; until it lands in `scripts/`, the harness lives beside its outputs under
`output/demo_visual_scene_sampling/` (untracked, like the rest of `output/`) with
its own `README.md` and per-step `logs/`. It reads a published pair with
`read_paired_artifact()`, rebuilds the appearance with
`visual_scene_from_payload()`, and renders every branch from that one object:
`MaterialSpec` → a `kb.PrincipledBSDFMaterial` plus a node graph for the sampled
`TextureSpec`, `LightSpec` → `kb.RectAreaLight`, `BackgroundSpec` →
`scene.background`, and `CameraRenderSpec` keyframed per frame.
