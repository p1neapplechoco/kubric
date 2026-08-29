"""PyBullet simulation with mass-carrying kinematic path interventions.

Purpose: execute prescribed target paths while collecting synchronized physics
states and contacts from Kubric's PyBullet backend.
Public API: KinematicSimulator and its compatibility alias
KinematicDragSimulator.
Dependencies: NumPy, Kubric, and PyBullet; the interventions package exposes this
backend lazily so backend-neutral imports do not load those runtime dependencies.
Trust boundary: private Bullet save-state snapshots are bound to the creating
simulator, physics client, and backend lifetime; they are not portable artifacts.
"""

from __future__ import annotations

import contextlib
import math
import numbers
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pybullet as pb

from kubric import core
from kubric.simulator.pybullet import PyBullet

from interventions.logging import ContactLogger, SimulationLog


_ZERO3 = (0.0, 0.0, 0.0)
_QUATERNION_TOLERANCE = 1e-6
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _logical_id(asset: core.PhysicalObject) -> str:
  """Returns the required process-stable logical identifier."""
  if "logical_id" not in asset.metadata:
    raise ValueError("asset metadata['logical_id'] is required")
  value = asset.metadata["logical_id"]
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


def _commanded_velocities(path: np.ndarray, step_rate: float) -> np.ndarray:
  """Prevalidates Kubric pose casts and all backward differences."""
  if np.any(np.abs(path) > _FLOAT32_MAX):
    raise ValueError("path values must be representable as float32 Kubric traits")
  velocities = np.zeros((len(path), 3), dtype=np.float64)
  with np.errstate(over="ignore", invalid="ignore"):
    velocities[1:] = (path[1:, :3] - path[:-1, :3]) * step_rate
  if not np.isfinite(velocities).all():
    raise ValueError("path produces a non-finite commanded velocity")
  if np.any(np.abs(velocities) > _FLOAT32_MAX):
    raise ValueError("path velocity must be representable as float32")
  return velocities


def _target_manifold_penetrations(
    step: int,
    raw_contacts: Iterable[Any],
    target_body: int,
    body_to_object_id: Dict[int, str],
    target_id: str,
    target_scale: float,
) -> Tuple[Dict[str, Any], ...]:
  """Returns canonical post-step target penetrations, including force zero."""
  depths: Dict[str, float] = {}
  for raw_contact in raw_contacts:
    try:
      fields = tuple(raw_contact)
    except TypeError:
      continue
    if len(fields) != 14:
      continue
    body_a, body_b = fields[1], fields[2]
    if (
        isinstance(body_a, bool)
        or not isinstance(body_a, numbers.Integral)
        or isinstance(body_b, bool)
        or not isinstance(body_b, numbers.Integral)
    ):
      continue
    if int(body_a) == target_body:
      peer_body = int(body_b)
    elif int(body_b) == target_body:
      peer_body = int(body_a)
    else:
      continue
    peer_id = body_to_object_id.get(peer_body)
    if peer_id is None or peer_id == target_id:
      continue
    distance = fields[8]
    if isinstance(distance, bool) or not isinstance(distance, numbers.Real):
      continue
    try:
      depth = max(0.0, -float(distance))
    except (TypeError, ValueError, OverflowError):
      continue
    if not math.isfinite(depth) or depth <= 0.0:
      continue
    depth = min(depth, target_scale)
    depths[peer_id] = max(depths.get(peer_id, 0.0), depth)
  return tuple(
      {"step": int(step), "object_id": peer_id, "depth": depths[peer_id]}
      for peer_id in sorted(depths)
  )


def _raw_contact_provenance(
    step: int,
    raw_contacts: Iterable[Any],
    body_to_object_id: Dict[int, str],
) -> Tuple[Dict[str, Any], ...]:
  """Preserves canonical Bullet fields for every retained positive contact."""
  entries = []
  for raw_contact in raw_contacts:
    try:
      fields = tuple(raw_contact)
    except TypeError:
      continue
    if len(fields) != 14:
      continue
    body_a, body_b = fields[1], fields[2]
    if (
        isinstance(body_a, bool)
        or not isinstance(body_a, numbers.Integral)
        or isinstance(body_b, bool)
        or not isinstance(body_b, numbers.Integral)
    ):
      continue
    object_a = body_to_object_id.get(int(body_a))
    object_b = body_to_object_id.get(int(body_b))
    if object_a is None or object_b is None or object_a == object_b:
      continue

    def vector(value: Any) -> Optional[Tuple[float, float, float]]:
      try:
        items = tuple(value)
      except TypeError:
        return None
      if len(items) != 3:
        return None
      result = []
      for item in items:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
          return None
        number = float(item)
        if not math.isfinite(number):
          return None
        result.append(0.0 if number == 0.0 else number)
      return tuple(result)  # type: ignore[return-value]

    position_on_a = vector(fields[5])
    position_on_b = vector(fields[6])
    normal_on_b = vector(fields[7])
    distance, force = fields[8], fields[9]
    if (
        position_on_a is None
        or position_on_b is None
        or normal_on_b is None
        or isinstance(distance, bool)
        or not isinstance(distance, numbers.Real)
        or isinstance(force, bool)
        or not isinstance(force, numbers.Real)
    ):
      continue
    distance_value = float(distance)
    force_value = float(force)
    if (
        not math.isfinite(distance_value)
        or not math.isfinite(force_value)
        or force_value <= 0.0
    ):
      continue
    entries.append({
        "step": int(step),
        "bullet_object_a": object_a,
        "bullet_object_b": object_b,
        "position_on_a": position_on_a,
        "position_on_b": position_on_b,
        "normal_on_b": normal_on_b,
        "contact_distance": (
            0.0 if distance_value == 0.0 else distance_value
        ),
        "normal_force": force_value,
    })
  return tuple(sorted(entries, key=lambda entry: (
      entry["step"], entry["bullet_object_a"], entry["bullet_object_b"],
      entry["position_on_a"], entry["position_on_b"], entry["normal_on_b"],
      entry["contact_distance"], entry["normal_force"],
  )))


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
  the backward-difference velocity. Gravity therefore acts on the temporarily
  massive target for that step; the next command corrects its pose. Any resulting
  within-step drift remains both physical and observable in the log.
  """

  def __init__(self, scene: core.Scene, scratch_dir: Any = None):
    self._closed = False
    self._closing = False
    self._asset_observers: Dict[core.Asset, List[Tuple[str, Any]]] = {}
    self._checkpoint_ids: set[int] = set()
    self._ownership_body: Optional[int] = None
    self._ownership_key: Optional[str] = None
    self._ownership_value: Optional[bytes] = None
    self._client_id: Optional[int] = None
    try:
      if scratch_dir is None:
        super().__init__(scene)
      else:
        super().__init__(scene, scratch_dir=scratch_dir)
    except BaseException:
      self._cleanup_partial_initialization(scene)
      raise
    try:
      self._client_id = int(self._physics_client._client)  # pylint: disable=protected-access
      self._install_ownership_canary()
      self._physics_client.resetSimulation = self._reject_reset_simulation
      self._physics_client.setTimeStep(1.0 / float(scene.step_rate))
      self._step_rate_observer = self._handle_step_rate_change
      self.scene_observers.setdefault("step_rate", []).append(
          self._step_rate_observer
      )
      scene.observe(self._step_rate_observer, names="step_rate", type="change")
      self._validate_logical_ids()
    except BaseException:
      if self._ownership_value is None:
        self._cleanup_partial_initialization(scene)
      else:
        self.close()
      raise

  def _cleanup_partial_initialization(self, scene: core.Scene) -> None:
    """Disconnects a client when Kubric scene linking fails inside super()."""
    client = getattr(self, "_physics_client", None)
    client_id = getattr(client, "_client", -1)
    self._closing = True
    try:
      if self in scene.views:
        try:
          scene.unlink_view(self)
        except BaseException:
          pass
      for asset in tuple(self._asset_observers):
        self._unobserve_asset(asset)
      for name, callbacks in getattr(self, "scene_observers", {}).items():
        for callback in callbacks:
          try:
            scene.unobserve(callback, names=name, type="change")
          except (KeyError, ValueError):
            pass
      if isinstance(client_id, numbers.Integral) and client_id >= 0:
        try:
          if pb.isConnected(int(client_id)):
            pb.disconnect(int(client_id))
        except pb.error:
          pass
    finally:
      if client is not None:
        client._client = -1  # pylint: disable=protected-access
      self._closed = True
      self._closing = False

  @staticmethod
  def _reject_reset_simulation(*args, **kwargs) -> None:
    del args, kwargs
    raise RuntimeError(
        "resetSimulation is unsupported; close and create a new simulator"
    )

  def _install_ownership_canary(self) -> None:
    """Marks this Bullet world so recycled integer client IDs are detectable."""
    token = uuid.uuid4().hex
    body = int(self._physics_client.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
    ))
    key = "kubric_kinematic_owner_{}".format(uuid.uuid4().hex)
    self._physics_client.addUserData(body, key, token)
    self._ownership_body = body
    self._ownership_key = key
    self._ownership_value = token.encode("utf-8")

  def _owns_client(self) -> bool:
    if (
        self._client_id is None
        or self._ownership_body is None
        or self._ownership_key is None
        or self._ownership_value is None
    ):
      return False
    try:
      if not pb.isConnected(self._client_id):
        return False
      user_data_id = pb.getUserDataId(
          self._ownership_body,
          self._ownership_key,
          physicsClientId=self._client_id,
      )
      if user_data_id < 0:
        return False
      return (
          pb.getUserData(user_data_id, physicsClientId=self._client_id)
          == self._ownership_value
      )
    except pb.error:
      return False

  def _handle_step_rate_change(self, change: Any) -> None:
    if self._owns_client():
      self._physics_client.setTimeStep(1.0 / float(change.new))

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

  def _unobserve_asset(self, asset: core.Asset) -> None:
    for name, callback in self._asset_observers.pop(asset, ()):
      try:
        asset.unobserve(callback, names=name, type="change")
      except (KeyError, ValueError):
        pass

  def remove_asset(self, asset: core.Asset) -> None:
    """Removes a body and its exact callbacks without leaving stale setters."""
    try:
      if not self._closing and self._owns_client():
        super().remove_asset(asset)
    finally:
      self._unobserve_asset(asset)

  @property
  def bullet_client(self):
    """The private client wrapper, with this connection id pre-bound."""
    self._require_connected()
    return self._physics_client

  @property
  def physics_client(self) -> int:
    """Returns the owned integer client id, rejecting recycled connections."""
    self._require_connected()
    assert self._client_id is not None
    return self._client_id

  @property
  def is_connected(self) -> bool:
    return not self._closed and self._owns_client()

  def _require_connected(self) -> None:
    if not self.is_connected:
      raise RuntimeError(
          "KinematicSimulator connection ownership is stale, closed, or disconnected"
      )

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
    owned = self._owns_client()
    self._closing = True
    scene = self.scene
    try:
      if scene is not None and self in scene.views:
        scene.unlink_view(self)
    finally:
      for asset in tuple(self._asset_observers):
        self._unobserve_asset(asset)
      if scene is not None:
        for name, callbacks in self.scene_observers.items():
          for callback in callbacks:
            try:
              scene.unobserve(callback, names=name, type="change")
            except (KeyError, ValueError):
              pass
      try:
        if owned and self._client_id is not None:
          pb.disconnect(self._client_id)
      except pb.error:
        pass
      finally:
        # Kubric's wrapper does not invalidate this id itself. Without doing so,
        # its later __del__ may disconnect an unrelated connection that reused it.
        self._physics_client._client = -1  # pylint: disable=protected-access
        self._checkpoint_ids.clear()
        self._closed = True
        self._closing = False

  def __del__(self):
    try:
      self.close()
    except BaseException:
      try:
        self._physics_client._client = -1  # pylint: disable=protected-access
      except BaseException:
        pass

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

  @contextlib.contextmanager
  def _suspend_asset_observers(
      self, assets: Iterable[core.PhysicalObject]
  ) -> Iterator[None]:
    detached = []
    for asset in assets:
      for name, callback in self._asset_observers.get(asset, ()):
        asset.unobserve(callback, names=name, type="change")
        detached.append((asset, name, callback))
    try:
      yield
    finally:
      for asset, name, callback in detached:
        tracked = self._asset_observers.get(asset, ())
        still_tracked = any(
            tracked_name == name and tracked_callback is callback
            for tracked_name, tracked_callback in tracked
        )
        if (
            self._owns_client()
            and asset in self.scene.assets
            and self in asset.linked_objects
            and still_tracked
        ):
          asset.observe(callback, names=name, type="change")

  def _publish_keyframes(
      self,
      assets: Tuple[core.PhysicalObject, ...],
      records: Iterable[Tuple[np.ndarray, int]],
      final_state: np.ndarray,
  ) -> None:
    """Publishes recorded states without feeding float32 traits into Bullet."""
    with self._suspend_asset_observers(assets):
      try:
        for state, frame in records:
          self._write_keyframe(assets, state, frame)
      finally:
        for asset, values in zip(assets, final_state):
          asset.position = values[0:3]
          asset.quaternion = values[3:7]
          asset.velocity = values[7:10]
          asset.angular_velocity = values[10:13]

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
    step_rate = float(self.scene.step_rate)
    commanded_velocities = _commanded_velocities(path_value, step_rate)
    self._validate_logical_ids()
    target_id = _logical_id(target)
    body = self._body(target)
    assets = self._physical_assets()
    object_ids = tuple(_logical_id(asset) for asset in assets)
    body_to_object_id = {
        self._body(asset): object_id
        for asset, object_id in zip(assets, object_ids)
    }
    realized_scale = np.abs(np.asarray(target.scale, dtype=np.float64))
    target_scale = (
        float(np.max(realized_scale))
        if isinstance(target, core.Sphere)
        else math.hypot(*(float(component) for component in realized_scale.flat))
    )

    if contact_logger is None:
      logger = ContactLogger(body_to_object_id, step_rate)
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
    cadence = int(step_rate) // self.scene.frame_rate
    keyframe_records: List[Tuple[np.ndarray, int]] = []
    manifold_penetrations: List[Dict[str, Any]] = []
    raw_contact_provenance: List[Dict[str, Any]] = []

    for offset, (absolute_step, command, commanded_velocity) in enumerate(
        zip(steps, path_value, commanded_velocities)
    ):
      if float(self.scene.step_rate) != step_rate:
        raise RuntimeError("scene.step_rate changed during intervention run")
      try:
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
        raw_contacts = tuple(self.bullet_client.getContactPoints())
        raw_contact_provenance.extend(_raw_contact_provenance(
            absolute_step, raw_contacts, body_to_object_id
        ))
        manifold_penetrations.extend(_target_manifold_penetrations(
            absolute_step,
            raw_contacts,
            body,
            body_to_object_id,
            target_id,
            target_scale,
        ))
        logger.log(absolute_step, raw_contacts)
        state = self._snapshot(assets)
        states[offset] = state
        if float(self.scene.step_rate) != step_rate:
          raise RuntimeError("scene.step_rate changed during intervention run")
        if write_keyframes and absolute_step % cadence == 0:
          frame = self.scene.frame_start + absolute_step // cadence
          keyframe_records.append((state.copy(), frame))
      finally:
        # Bypass Kubric's static setter: its logical static flag never changes.
        if self.is_connected:
          self.bullet_client.changeDynamics(body, -1, mass=0.0)

    if write_keyframes:
      self._publish_keyframes(assets, keyframe_records, states[-1])

    return SimulationLog(
        branch=branch,
        object_ids=object_ids,
        steps=steps,
        states=states,
        contacts=tuple(logger.records),
        step_rate=step_rate,
        commanded_path=path_value,
        metadata={
            "target_id": target_id,
            "kinematic_emulation": True,
            "push_mass": push_mass_value,
            "dt": 1.0 / step_rate,
            "velocity_estimator": "backward_difference",
            "raw_contact_provenance": tuple(raw_contact_provenance),
            "target_manifold_penetrations": tuple(manifold_penetrations),
        },
    )


KinematicDragSimulator = KinematicSimulator


__all__ = ["KinematicDragSimulator", "KinematicSimulator"]
