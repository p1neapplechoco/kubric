"""Cross-platform durability and advisory-locking primitives.

Purpose: provide the filesystem primitives the publication paths depend on --
flushing a directory entry, taking an exclusive advisory lock, and renaming a
staged payload into place -- with one implementation per platform, so that
``interventions.logging``, ``interventions.dataset``, and
``interventions.twin_runner`` cannot drift apart.

Public API: fsync_directory(), exclusive_lock(), publish_rename(),
publish_replace(), DIRECTORY_FSYNC_SUPPORTED, PUBLISH_RETRY_DELAYS.

Dependencies: standard library only. POSIX uses ``fcntl``; Windows uses ``msvcrt``.
Neither backend is imported at call time, so importing this module is always safe.

Trust boundary: these helpers provide durability and mutual exclusion between
cooperating publishers on a single machine. They are not a security boundary and
give no protection against a writer that ignores the lock, and no guarantee across
network filesystems where advisory locks are unreliable.
"""

from __future__ import annotations

import contextlib
import errno
import os
import time
from pathlib import Path

try:
  import fcntl
except ImportError:  # pragma: no cover - exercised only off POSIX platforms.
  fcntl = None

try:
  import msvcrt
except ImportError:  # pragma: no cover - exercised only on POSIX platforms.
  msvcrt = None


_WINDOWS = os.name == "nt"

# Windows has no blocking flock equivalent that is safe to use here: msvcrt's
# LK_LOCK gives up after ten seconds, so contention is handled by polling.
_LOCK_POLL_SECONDS = 0.01

# POSIX can fsync a directory to make a rename durable. Windows cannot: os.open()
# on a directory fails with PermissionError, and no CRT handle exposes the entry.
DIRECTORY_FSYNC_SUPPORTED = not _WINDOWS

# Windows denies a rename while any other process holds an open handle anywhere in
# the tree, which real-time virus scanners and search indexers routinely do for a
# few milliseconds after files are created. The failure is indistinguishable from
# a real permission error by errno alone, so the budget is deliberately short: it
# absorbs a scanner but still surfaces a genuinely unwritable destination quickly.
PUBLISH_RETRY_DELAYS = (0.01, 0.02, 0.05, 0.1, 0.25) if _WINDOWS else ()


def fsync_directory(directory: Path) -> None:
  """Flush a directory entry to stable storage where the platform allows it.

  On Windows this is a no-op. Publication there still relies on ``os.replace``
  for atomic visibility, and on the payload's own ``os.fsync`` for content
  durability, but the directory entry itself is not separately flushed.
  """
  if not DIRECTORY_FSYNC_SUPPORTED:
    return
  flags = os.O_RDONLY
  if hasattr(os, "O_DIRECTORY"):
    flags |= os.O_DIRECTORY
  descriptor = os.open(str(directory), flags)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _publish(operation: str, source, destination) -> None:
  for delay in PUBLISH_RETRY_DELAYS:
    try:
      getattr(os, operation)(source, destination)
      return
    except PermissionError:
      time.sleep(delay)
  getattr(os, operation)(source, destination)


def publish_rename(source, destination) -> None:
  """Renames ``source`` onto a destination that must not already exist.

  Behaves exactly like ``os.rename`` apart from retrying the transient Windows
  sharing violation described on PUBLISH_RETRY_DELAYS. The final attempt is
  unguarded, so a persistent failure propagates its original exception.
  """
  _publish("rename", source, destination)


def publish_replace(source, destination) -> None:
  """Atomically replaces ``destination`` with ``source``.

  Behaves exactly like ``os.replace`` apart from the retry described on
  publish_rename.
  """
  _publish("replace", source, destination)


def _acquire(descriptor: int, blocking: bool) -> None:
  if _WINDOWS:
    while True:
      os.lseek(descriptor, 0, os.SEEK_SET)
      try:
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return
      except OSError as error:
        if not blocking:
          raise BlockingIOError(
              errno.EAGAIN, "advisory lock is held by another writer"
          ) from error
        time.sleep(_LOCK_POLL_SECONDS)
  flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
  fcntl.flock(descriptor, flags)


def _release(descriptor: int) -> None:
  if _WINDOWS:
    os.lseek(descriptor, 0, os.SEEK_SET)
    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    return
  fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextlib.contextmanager
def exclusive_lock(lock_path: Path, *, blocking: bool = True):
  """Hold an exclusive advisory lock on ``lock_path`` for the duration of the block.

  Raises BlockingIOError when ``blocking`` is false and the lock is already held.
  """
  if fcntl is None and msvcrt is None:  # pragma: no cover - unsupported platform.
    raise RuntimeError("advisory file locking is unavailable on this platform")
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
  try:
    _acquire(descriptor, blocking)
    try:
      yield
    finally:
      _release(descriptor)
  finally:
    os.close(descriptor)
