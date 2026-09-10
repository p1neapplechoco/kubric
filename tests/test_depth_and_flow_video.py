"""Unit tests for depth map and optical flow colorization and video generation."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from scripts.render_velocity_intervention import (
    _colorize_depth,
    _colorize_flow,
    _encode_mp4,
    _hsv_to_rgb,
    _viridis_lut,
)


def test_viridis_lut_properties():
  lut = _viridis_lut()
  assert lut.shape == (256, 3)
  assert lut.dtype == np.uint8
  # Check standard Viridis anchor bounds: starts dark purple, ends bright yellow
  assert lut[0, 0] > 50 and lut[0, 2] > 70  # purple
  assert lut[-1, 0] > 200 and lut[-1, 1] > 200  # yellow


def test_colorize_depth_shapes_and_values():
  # T=5 frames, H=32, W=48
  depth = np.full((5, 32, 48), 50.0, dtype=np.float32)  # far clip
  depth[:, 10:20, 15:30] = 5.0  # object at 5m
  depth[:, 20:30, 15:30] = 15.0  # object at 15m
  far = 50.0

  rgb = _colorize_depth(depth, far)
  assert rgb.shape == (5, 32, 48, 3)
  assert rgb.dtype == np.uint8

  # Void/far pixels (depth >= far * 0.999) should be black
  assert np.all(rgb[:, 0, 0] == 0)

  # Foreground object at 5m is closer than 15m, so it should be brighter
  assert rgb[:, 15, 20].sum() > rgb[:, 25, 20].sum()

  # Test 4D input (T, H, W, 1)
  rgb_4d = _colorize_depth(depth[..., np.newaxis], far)
  assert rgb_4d.shape == (5, 32, 48, 3)
  np.testing.assert_array_equal(rgb, rgb_4d)


def test_hsv_to_rgb():
  # Test red (H=0, S=1, V=1)
  hsv_red = np.array([[[0.0, 1.0, 1.0]]], dtype=np.float32)
  rgb_red = _hsv_to_rgb(hsv_red)
  np.testing.assert_allclose(rgb_red[0, 0], [1.0, 0.0, 0.0], atol=1e-5)

  # Test green (H=1/3, S=1, V=1)
  hsv_green = np.array([[[1.0 / 3.0, 1.0, 1.0]]], dtype=np.float32)
  rgb_green = _hsv_to_rgb(hsv_green)
  np.testing.assert_allclose(rgb_green[0, 0], [0.0, 1.0, 0.0], atol=1e-5)

  # Test blue (H=2/3, S=1, V=1)
  hsv_blue = np.array([[[2.0 / 3.0, 1.0, 1.0]]], dtype=np.float32)
  rgb_blue = _hsv_to_rgb(hsv_blue)
  np.testing.assert_allclose(rgb_blue[0, 0], [0.0, 0.0, 1.0], atol=1e-5)

  # Test black (V=0)
  hsv_black = np.array([[[0.5, 1.0, 0.0]]], dtype=np.float32)
  rgb_black = _hsv_to_rgb(hsv_black)
  np.testing.assert_allclose(rgb_black[0, 0], [0.0, 0.0, 0.0], atol=1e-5)


def test_colorize_flow_shapes_and_values():
  # Zero flow (static)
  flow_zero = np.zeros((4, 32, 32, 2), dtype=np.float32)
  rgb_zero = _colorize_flow(flow_zero)
  assert rgb_zero.shape == (4, 32, 32, 3)
  assert rgb_zero.dtype == np.uint8
  # Zero motion has value 0 -> completely black
  assert np.all(rgb_zero == 0)

  # Moving flow
  flow_moving = np.zeros((4, 32, 32, 2), dtype=np.float32)
  flow_moving[:, 10:20, 10:20, 0] = 5.0  # moving horizontally
  flow_moving[:, 20:30, 20:30, 1] = 5.0  # moving vertically
  rgb_moving = _colorize_flow(flow_moving)
  assert rgb_moving.shape == (4, 32, 32, 3)
  assert rgb_moving.dtype == np.uint8

  # Moving regions have non-zero RGB
  assert rgb_moving[:, 15, 15].sum() > 0
  assert rgb_moving[:, 25, 25].sum() > 0

  # Different movement directions should have distinct hues
  assert not np.array_equal(rgb_moving[:, 15, 15], rgb_moving[:, 25, 25])


def test_encode_mp4_creates_valid_videos():
  depth = np.full((12, 64, 64), 30.0, dtype=np.float32)
  depth[:, 20:40, 20:40] = 6.0
  depth_rgb = _colorize_depth(depth, far_clip=30.0)

  flow = np.zeros((12, 64, 64, 2), dtype=np.float32)
  flow[:, 20:40, 20:40, 0] = 8.0
  flow_rgb = _colorize_flow(flow)

  with tempfile.TemporaryDirectory() as tmpdir:
    depth_video = Path(tmpdir) / "depth.mp4"
    flow_video = Path(tmpdir) / "flow.mp4"

    _encode_mp4(depth_video, depth_rgb, frame_rate=24)
    _encode_mp4(flow_video, flow_rgb, frame_rate=24)

    assert depth_video.exists()
    assert depth_video.stat().st_size > 500
    assert flow_video.exists()
    assert flow_video.stat().st_size > 500
