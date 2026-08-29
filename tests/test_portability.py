"""Tests for the cross-platform filesystem primitives shared by publishers."""

import errno
import os

import pytest

from interventions import _portability


@pytest.mark.parametrize(
    "publish, underlying",
    (
        (_portability.publish_rename, "rename"),
        (_portability.publish_replace, "replace"),
    ),
)
def test_publish_calls_the_underlying_primitive_exactly_once_on_success(
    tmp_path, monkeypatch, publish, underlying
):
  calls = []
  real = getattr(os, underlying)

  def tracked(source, destination):
    calls.append((source, destination))
    return real(source, destination)

  monkeypatch.setattr(os, underlying, tracked)
  source = tmp_path / "source"
  source.mkdir()
  destination = tmp_path / "destination"

  publish(source, destination)

  assert calls == [(source, destination)]
  assert destination.is_dir()
  assert not source.exists()


@pytest.mark.skipif(
    len(_portability.PUBLISH_RETRY_DELAYS) < 2,
    reason="this platform has no transient sharing violation to retry",
)
@pytest.mark.parametrize(
    "publish, underlying",
    (
        (_portability.publish_rename, "rename"),
        (_portability.publish_replace, "replace"),
    ),
)
def test_publish_retries_a_transient_sharing_violation(
    tmp_path, monkeypatch, publish, underlying
):
  attempts = []
  real = getattr(os, underlying)

  def flaky(source, destination):
    attempts.append((source, destination))
    if len(attempts) < 3:
      raise PermissionError(errno.EACCES, "Access is denied")
    return real(source, destination)

  monkeypatch.setattr(os, underlying, flaky)
  monkeypatch.setattr(_portability.time, "sleep", lambda _seconds: None)
  source = tmp_path / "source"
  source.mkdir()
  destination = tmp_path / "destination"

  publish(source, destination)

  assert len(attempts) == 3
  assert destination.is_dir()


@pytest.mark.parametrize(
    "publish, underlying",
    (
        (_portability.publish_rename, "rename"),
        (_portability.publish_replace, "replace"),
    ),
)
def test_publish_reraises_after_exhausting_its_retry_budget(
    tmp_path, monkeypatch, publish, underlying
):
  attempts = []

  def always_denied(source, destination):
    attempts.append((source, destination))
    raise PermissionError(errno.EACCES, "Access is denied")

  monkeypatch.setattr(os, underlying, always_denied)
  monkeypatch.setattr(_portability.time, "sleep", lambda _seconds: None)

  with pytest.raises(PermissionError):
    publish(tmp_path / "source", tmp_path / "destination")

  assert len(attempts) == len(_portability.PUBLISH_RETRY_DELAYS) + 1


@pytest.mark.parametrize(
    "publish, underlying",
    (
        (_portability.publish_rename, "rename"),
        (_portability.publish_replace, "replace"),
    ),
)
def test_publish_does_not_retry_a_permanent_failure(
    tmp_path, monkeypatch, publish, underlying
):
  attempts = []

  def missing(source, destination):
    attempts.append((source, destination))
    raise FileNotFoundError(errno.ENOENT, "No such file or directory")

  monkeypatch.setattr(os, underlying, missing)

  with pytest.raises(FileNotFoundError):
    publish(tmp_path / "source", tmp_path / "destination")

  assert len(attempts) == 1
