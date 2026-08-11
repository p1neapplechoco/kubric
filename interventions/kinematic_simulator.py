"""PyBullet simulation with mass-carrying kinematic path interventions."""

from __future__ import annotations

import contextlib
import math
import numbers
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pybullet as pb

from kubric import core
from kubric.simulator.pybullet import PyBullet

from interventions.logging import ContactLogger, SimulationLog


_ZERO3 = (0.0, 0.0, 0.0)
_QUATERNION_TOLERANCE = 1e-6


def _logical_id(asset: core.PhysicalObject) -> str:
  """Returns the explicit logical id, falling back to Kubric's unique uid."""
  value = asset.metadata.get("logical_id", asset.uid)
  if not isinstance(value, str) or not value:
    raise ValueError("logical IDs must be non-empty strings")
  return value


def _path_array(path: Any) -> np.ndarray:
  """Copies and validates an exact ``[T, 7]`` XYZ+WXYZ path."""
  try:
    untyped = np.asarray(path)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("path must be a numeric [T, 7] array") from error
  object_complex = untyped.dtype.kind == "O" and any(
      isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real)
      for value in untyped.flat
  )
  if np.iscomplexobj(untyped) or object_complex:
    raise ValueError("path must contain real values")
  try:
    result = np.array(untyped, dtype=np.float64, copy=True)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("path must be a numeric [T, 7] array") from error
  if result.ndim != 2 or result.shape[1:] != (7,):
    raise ValueError("path must have shape [T, 7]")
  if result.shape[0] < 2:
    raise ValueError("path must contain at least two samples")
  if not np.isfinite(result).all():
    raise ValueError("path must contain only finite values")
  norms = np.asarray(
      [math.hypot(*(float(value) for value in row)) for row in result[:, 3:7]]
  )
  if not np.all(np.abs(norms - 1.0) <= _QUATERNION_TOLERANCE):
    raise ValueError("path quaternions must be unit-normalized WXYZ values")
  return result


def _positive_mass(value: Any) -> float:
  if isinstance(value, bool) or not isinstance(value, numbers.Real):
    raise ValueError("push_mass must be a positive finite real number")
  try:
    result = float(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("push_mass must be a positive finite real number") from error
  if not math.isfinite(result) or result <= 0.0:
    raise ValueError("push_mass must be a positive finite real number")
  return result


def _change_observers(asset: core.Asset) -> Dict[str, Tuple[Any, ...]]:
  """Snapshots traitlets callbacks so the private wrapper can unlink cleanly."""
  notifiers = getattr(asset, "_trait_notifiers", {})
  return {
      name: tuple(events.get("change", ()))
      for name, events in notifiers.items()
      if events.get("change")
  }


class KinematicSimulator(PyBullet):
  """A private PyBullet view for deterministic trajectory interventions.

  The Kubric asset remains logically static throughout.  Immediately before each
  simulated path sample, its Bullet body temporarily receives ``push_mass`` and
  the backward-difference velocity.  Its pose is corrected at the next sample,
  so any within-step drift remains both physical and observable in the log.
  """

  def __init__(self, scene: core.Scene, scratch_dir: Any = None):
    self._closed = False
    self._asset_observers: Dict[core.Asset, List[Tuple[str, Any]]] = {}
    self._checkpoint_ids: set[int] = set()
    if scratch_dir is None:
      super().__init__(scene)
    else:
      super().__init__(scene, scratch_dir=scratch_dir)
    try:
      self._physics_client.setTimeStep(1.0 / float(scene.step_rate))
      self._validate_logical_ids()
    except BaseException:
      self.close()
      raise

  def add(self, asset: core.Asset) -> None:
    """Links an asset while retaining the observers needed for clean close()."""
    before = _change_observers(asset)
    super().add(asset)
    after = _change_observers(asset)
    added = self._asset_observers.setdefault(asset, [])
    for name, callbacks in after.items():
      previous = before.get(name, ())
      for callback in callbacks:
        if not any(callback is item for item in previous):
          added.append((name, callback))

  def remove_asset(self, asset: core.Asset) -> None:
    """Allows close() to unlink cleanly even after an external disconnect."""
    if self.is_connected:
      super().remove_asset(asset)

  @property
  def bullet_client(self):
    """The private client wrapper, with this connection id pre-bound."""
    return self._physics_client

  @property
  def is_connected(self) -> bool:
    if self._closed:
      return False
    try:
      return bool(self._physics_client.isConnected())
    except pb.error:
      return False

  def _require_connected(self) -> None:
    if not self.is_connected:
      raise RuntimeError("KinematicSimulator is closed or disconnected")

  def __enter__(self) -> "KinematicSimulator":
    self._require_connected()
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> bool:
    del exc_type, exc_value, traceback
    self.close()
    return False

  def close(self) -> None:
    """Unlinks and disconnects this simulator; repeated calls are harmless."""
    if self._closed:
      return
    scene = self.scene
    try:
      if scene is not None and self in scene.views:
        scene.unlink_view(self)
    finally:
      for asset, callbacks in tuple(self._asset_observers.items()):
        for name, callback in callbacks:
          try:
            asset.unobserve(callback, names=name, type="change")
          except (KeyError, ValueError):
            pass
      self._asset_observers.clear()
      if scene is not None:
        for name, callbacks in self.scene_observers.items():
          for callback in callbacks:
            try:
              scene.unobserve(callback, names=name, type="change")
            except (KeyError, ValueError):
              pass
      try:
        if self.is_connected:
          self._physics_client.disconnect()
      finally:
        # Kubric's wrapper does not invalidate this id itself.  Without doing so,
        # its later __del__ may disconnect an unrelated connection that reused it.
        self._physics_client._client = -1  # pylint: disable=protected-access
        self._checkpoint_ids.clear()
        self._closed = True

  def _physical_assets(self) -> Tuple[core.PhysicalObject, ...]:
    assets = (
        asset
        for asset in self.scene.assets
        if isinstance(asset, core.PhysicalObject) and self in asset.linked_objects
    )
    return tuple(sorted(assets, key=lambda asset: _logical_id(asset)))

  def _validate_logical_ids(self) -> None:
    ids = [_logical_id(asset) for asset in self._physical_assets()]
    if len(ids) != len(set(ids)):
      raise ValueError("logical IDs must be unique; duplicate logical_id found")

  def _body(self, asset: core.PhysicalObject) -> int:
    try:
      body = asset.linked_objects[self]
    except KeyError as error:
      raise ValueError("target must be linked to this simulator and scene") from error
    if not isinstance(body, numbers.Integral):
      raise ValueError("target must be linked to a PyBullet body")
    return int(body)

  def save_checkpoint(self) -> int:
    self._require_connected()
    checkpoint = int(self.bullet_client.saveState())
    self._checkpoint_ids.add(checkpoint)
    return checkpoint

  def restore_checkpoint(self, checkpoint: int) -> None:
    self._require_connected()
    if checkpoint not in self._checkpoint_ids:
      raise ValueError("unknown or removed checkpoint")
    self.bullet_client.restoreState(stateId=checkpoint)

  def remove_checkpoint(self, checkpoint: int) -> None:
    self._require_connected()
    if checkpoint not in self._checkpoint_ids:
      raise ValueError("unknown or removed checkpoint")
    self.bullet_client.removeState(checkpoint)
    self._checkpoint_ids.remove(checkpoint)

  @contextlib.contextmanager
  def checkpoint(self) -> Iterator[int]:
    """Restores the saved state on exit and then removes the checkpoint."""
    checkpoint = self.save_checkpoint()
    try:
      yield checkpoint
    finally:
      if self.is_connected and checkpoint in self._checkpoint_ids:
        try:
          self.restore_checkpoint(checkpoint)
        finally:
          self.remove_checkpoint(checkpoint)

  def step_passive(self) -> None:
    """Advances exactly one physics step without writing Kubric keyframes."""
    self._require_connected()
    self.bullet_client.stepSimulation()

  def _snapshot(
      self, assets: Iterable[core.PhysicalObject]
  ) -> np.ndarray:
    rows = []
    for asset in assets:
      body = self._body(asset)
      position, quaternion_xyzw = self.bullet_client.getBasePositionAndOrientation(body)
      velocity, angular_velocity = self.bullet_client.getBaseVelocity(body)
      quaternion_wxyz = (
          quaternion_xyzw[3],
          quaternion_xyzw[0],
          quaternion_xyzw[1],
          quaternion_xyzw[2],
      )
      rows.append(
          tuple(position)
          + quaternion_wxyz
          + tuple(velocity)
          + tuple(angular_velocity)
      )
    return np.asarray(rows, dtype=np.float64)

  def _write_keyframe(
      self,
      assets: Iterable[core.PhysicalObject],
      state: np.ndarray,
      frame: int,
  ) -> None:
    for asset, values in zip(assets, state):
      asset.position = values[0:3]
      asset.quaternion = values[3:7]
      asset.velocity = values[7:10]
      asset.angular_velocity = values[10:13]
      for member in ("position", "quaternion", "velocity", "angular_velocity"):
        asset.keyframe_insert(member, frame)

  def run_with_intervention(
      self,
      target: core.PhysicalObject,
      path: Any,
      contact_logger: Optional[ContactLogger] = None,
      *,
      push_mass: float,
      branch: str = "factual",
      start_step: int = 0,
      write_keyframes: bool = False,
  ) -> SimulationLog:
    """Runs one physics step per path row with a temporarily massive target."""
    self._require_connected()
    push_mass_value = _positive_mass(push_mass)
    path_value = _path_array(path)
    if not isinstance(target, core.PhysicalObject):
      raise ValueError("target must be a PhysicalObject")
    if target not in self.scene.assets or self not in target.linked_objects:
      raise ValueError("target must be linked to this simulator and scene")
    if not target.static:
      raise ValueError("target must remain static")
    if isinstance(start_step, bool) or not isinstance(start_step, numbers.Integral):
      raise ValueError("start_step must be a nonnegative integer")
    start_step = int(start_step)
    if start_step < 0:
      raise ValueError("start_step must be a nonnegative integer")
    if not isinstance(branch, str) or not branch.strip():
      raise ValueError("branch must be a non-empty string")
    self._validate_logical_ids()
    target_id = _logical_id(target)
    body = self._body(target)
    assets = self._physical_assets()
    object_ids = tuple(_logical_id(asset) for asset in assets)
    body_to_object_id = {
        self._body(asset): object_id
        for asset, object_id in zip(assets, object_ids)
    }

    if contact_logger is None:
      logger = ContactLogger(body_to_object_id, self.scene.step_rate)
    else:
      logger = contact_logger
      if not callable(getattr(logger, "clear", None)) or not callable(
          getattr(logger, "log", None)
      ) or not hasattr(logger, "records"):
        raise TypeError("contact_logger must provide clear(), log(), and records")
    logger.clear()

    target.metadata["kinematic_emulation"] = True
    states = np.empty((len(path_value), len(assets), 13), dtype=np.float64)
    steps = tuple(start_step + offset for offset in range(len(path_value)))
    cadence = self.scene.step_rate // self.scene.frame_rate
    previous_position: Optional[np.ndarray] = None

    try:
      for offset, (absolute_step, command) in enumerate(zip(steps, path_value)):
        if previous_position is None:
          commanded_velocity = np.zeros(3, dtype=np.float64)
        else:
          commanded_velocity = (
              command[:3] - previous_position
          ) * float(self.scene.step_rate)
        previous_position = command[:3].copy()

        self.bullet_client.changeDynamics(body, -1, mass=push_mass_value)
        self.bullet_client.resetBaseVelocity(
            body,
            linearVelocity=commanded_velocity,
            angularVelocity=_ZERO3,
        )
        # Kubric's setters preserve and reapply the just-commanded velocity.
        target.position = command[:3]
        target.quaternion = command[3:7]
        measured_velocity, measured_angular = self.bullet_client.getBaseVelocity(body)
        if not np.allclose(
            measured_velocity, commanded_velocity, rtol=1e-12, atol=1e-12
        ) or not np.allclose(measured_angular, _ZERO3, rtol=0.0, atol=1e-12):
          raise RuntimeError("target velocity changed before the physics step")

        self.bullet_client.stepSimulation()
        raw_contacts = self.bullet_client.getContactPoints()
        logger.log(absolute_step, raw_contacts)
        state = self._snapshot(assets)
        states[offset] = state
        if write_keyframes and absolute_step % cadence == 0:
          frame = self.scene.frame_start + absolute_step // cadence
          self._write_keyframe(assets, state, frame)
    finally:
      # Bypass Kubric's static setter: its logical static flag never changes.
      if self.is_connected:
        self.bullet_client.changeDynamics(body, -1, mass=0.0)

    return SimulationLog(
        branch=branch,
        object_ids=object_ids,
        steps=steps,
        states=states,
        contacts=tuple(logger.records),
        step_rate=float(self.scene.step_rate),
        commanded_path=path_value,
        metadata={
            "target_id": target_id,
            "kinematic_emulation": True,
            "push_mass": push_mass_value,
            "dt": 1.0 / float(self.scene.step_rate),
            "velocity_estimator": "backward_difference",
        },
    )


__all__ = ["KinematicSimulator"]
