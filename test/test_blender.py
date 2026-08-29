# Copyright 2026 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kubric.safeimport.bpy import bpy

import numpy as np

from kubric import core
from kubric.renderer import blender
from kubric.renderer import blender_utils


def test_prepare_blender_object():
    @blender_utils.prepare_blender_object
    def add_asset(self, asset):
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        return cube

    cube_asset = core.Cube()
    cube_obj = add_asset(None, cube_asset)

    assert cube_obj.name.split(".")[0] == cube_asset.uid
    assert cube_obj.rotation_mode == "QUATERNION"
    assert cube_obj in bpy.context.scene.collection.objects.values()


def test_blender_scene_properties(tmp_path):
    scene = core.Scene(
        frame_start=2,
        frame_end=3,
        frame_rate=5,
        resolution=(7, 11),
    )
    renderer = blender.Blender(scene, tmp_path)
    assert renderer in scene.views
    assert renderer.scene == scene

    assert renderer.blender_scene.frame_start == 2
    assert renderer.blender_scene.frame_end == 3
    assert renderer.blender_scene.render.fps == 5
    assert renderer.blender_scene.render.resolution_x == 7
    assert renderer.blender_scene.render.resolution_y == 11


def test_blender_camera_on_init(tmp_path):
    cam = core.PerspectiveCamera(
        position=(1, 2, 3), quaternion=(0, 1, 0, 0), focal_length=3, sensor_width=4
    )
    renderer = blender.Blender(core.Scene(camera=cam), tmp_path)

    assert renderer in cam.linked_objects
    blender_cam = cam.linked_objects[renderer]
    assert renderer.blender_scene.camera == blender_cam
    assert blender_cam in renderer.blender_scene.collection.objects.values()
    assert tuple(blender_cam.location) == (1, 2, 3)
    assert tuple(blender_cam.rotation_quaternion) == (0, 1, 0, 0)
    assert blender_cam.data.lens == 3
    assert blender_cam.data.sensor_width == 4


def test_blender_camera_assign_after_init(tmp_path):
    scene = core.Scene()
    renderer = blender.Blender(scene, tmp_path)

    cam = core.PerspectiveCamera(
        position=(1, 2, 3), quaternion=(0, 1, 0, 0), focal_length=3, sensor_width=4
    )

    scene.camera = cam

    assert renderer in cam.linked_objects
    blender_cam = cam.linked_objects[renderer]
    assert renderer.blender_scene.camera == blender_cam
    assert blender_cam in renderer.blender_scene.collection.objects.values()
    assert tuple(blender_cam.location) == (1, 2, 3)
    assert tuple(blender_cam.rotation_quaternion) == (0, 1, 0, 0)
    assert blender_cam.data.lens == 3
    assert blender_cam.data.sensor_width == 4


def test_blender_adaptive_sampling_default(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path)
    assert renderer.adaptive_sampling is False
    assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_set_adaptive_sampling(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path)
    renderer.adaptive_sampling = False
    assert renderer.adaptive_sampling is False
    assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_init_adaptive_sampling(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path, adaptive_sampling=False)
    assert renderer.adaptive_sampling is False
    assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_use_denoising_default(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path)
    assert renderer.use_denoising is True
    assert renderer.blender_scene.cycles.use_denoising is True


def test_blender_set_use_denoising(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path)
    renderer.use_denoising = False
    assert renderer.use_denoising is False
    assert renderer.blender_scene.cycles.use_denoising is False


def test_blender_use_denoising_init(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path, use_denoising=False)
    assert renderer.use_denoising is False
    assert renderer.blender_scene.cycles.use_denoising is False


def test_blender_samples_per_pixel_default(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path)
    assert renderer.samples_per_pixel == 128
    assert renderer.blender_scene.cycles.samples == 128


def test_blender_set_samples_per_pixel(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path)
    renderer.samples_per_pixel = 64
    assert renderer.samples_per_pixel == 64
    assert renderer.blender_scene.cycles.samples == 64


def test_blender_samples_per_pixel_init(tmp_path):
    renderer = blender.Blender(core.Scene(), tmp_path, samples_per_pixel=256)
    assert renderer.samples_per_pixel == 256
    assert renderer.blender_scene.cycles.samples == 256


def test_blender_cylinder_has_expected_dimensions(tmp_path):
    scene = core.Scene(resolution=(16, 16))
    renderer = blender.Blender(scene, tmp_path)
    cylinder = core.Cylinder(scale=(0.25, 0.25, 0.75), position=(0, 0, 0))
    scene.add(cylinder)

    blender_obj = cylinder.linked_objects[renderer]
    bpy.context.view_layer.update()  # dimensions are cached until the depsgraph runs
    np.testing.assert_allclose(blender_obj.dimensions, (0.5, 0.5, 1.5), atol=1e-5)


def test_blender_capsule_dimensions_include_caps(tmp_path):
    scene = core.Scene(resolution=(16, 16))
    renderer = blender.Blender(scene, tmp_path)
    # 2 * 0.5 cylindrical section plus a 0.25 radius cap at each end.
    capsule = core.Capsule(scale=(0.25, 0.25, 0.5), position=(0, 0, 0))
    scene.add(capsule)

    blender_obj = capsule.linked_objects[renderer]
    np.testing.assert_allclose(blender_obj.dimensions, (0.5, 0.5, 1.5), atol=1e-5)


def test_blender_capsule_mesh_is_closed_and_deterministic(tmp_path):
    scene = core.Scene(resolution=(16, 16))
    renderer = blender.Blender(scene, tmp_path)
    first = core.Capsule(scale=(0.25, 0.25, 0.5))
    second = core.Capsule(scale=(0.25, 0.25, 0.5))
    scene.add(first)
    scene.add(second)

    first_vertices, first_faces = blender_utils.get_vertices_and_faces(
        first.linked_objects[renderer])
    second_vertices, second_faces = blender_utils.get_vertices_and_faces(
        second.linked_objects[renderer])
    np.testing.assert_array_equal(first_vertices, second_vertices)
    np.testing.assert_array_equal(first_faces, second_faces)

    # A closed surface uses every edge exactly twice, so a hole or a duplicated
    # interior face from a botched join would show up here.
    edge_counts = {}
    for face in first_faces:
        for index, vertex in enumerate(face):
            edge = frozenset((vertex, face[(index + 1) % len(face)]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert set(edge_counts.values()) == {2}


def test_blender_cylinder_tracks_material(tmp_path):
    scene = core.Scene(resolution=(16, 16))
    renderer = blender.Blender(scene, tmp_path)
    material = core.PrincipledBSDFMaterial(color=core.Color(1.0, 0.0, 0.0, 1.0))
    cylinder = core.Cylinder(scale=(0.2, 0.2, 0.2), material=material)
    scene.add(cylinder)

    blender_obj = cylinder.linked_objects[renderer]
    assert blender_obj.active_material is not None
    assert blender_obj.active_material is material.linked_objects[renderer]


def test_blender_capsule_tracks_material(tmp_path):
    scene = core.Scene(resolution=(16, 16))
    renderer = blender.Blender(scene, tmp_path)
    material = core.PrincipledBSDFMaterial(color=core.Color(0.0, 1.0, 0.0, 1.0))
    capsule = core.Capsule(scale=(0.2, 0.2, 0.2), material=material)
    scene.add(capsule)

    blender_obj = capsule.linked_objects[renderer]
    assert blender_obj.active_material is not None
    assert blender_obj.active_material is material.linked_objects[renderer]


def test_get_render_layers_from_exr_uses_uppercase_cryptomatte_channels(monkeypatch):
    captured_channels = []

    class FakeExr:
        def header(self):
            return {
                "channels": {
                    "CryptoObject00.R": object(),
                    "CryptoObject00.G": object(),
                    "CryptoObject00.B": object(),
                    "CryptoObject00.A": object(),
                },
                "dataWindow": type(
                    "DataWindow",
                    (),
                    {
                        "min": type("Point", (), {"x": 0, "y": 0})(),
                        "max": type("Point", (), {"x": 0, "y": 0})(),
                    },
                ),
            }

    def fake_input_file(_filename):
        return FakeExr()

    def fake_read_channels_from_exr(_exr, channel_names):
        captured_channels.append(list(channel_names))
        return np.zeros((1, 1, len(channel_names)), dtype=np.float32)

    monkeypatch.setattr(blender_utils.OpenEXR, "InputFile", fake_input_file)
    monkeypatch.setattr(
        blender_utils, "read_channels_from_exr", fake_read_channels_from_exr
    )

    blender_utils.get_render_layers_from_exr("dummy.exr")

    assert captured_channels == [
        ["CryptoObject00.R", "CryptoObject00.B"],
        ["CryptoObject00.G", "CryptoObject00.A"],
    ]
