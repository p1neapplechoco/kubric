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

"""Model for redirection of stdout/err to file."""

import ctypes
import os


class RedirectStream(object):
  """A simple class to redirect stdout/err to file.

  Usage:
    with RedirectStream(sys.stdout, filename="stdout.txt"):
      print("commands will have stdout directed to file")
  """

  # On POSIX the process-wide C runtime is reachable as CDLL(None). Windows has no
  # such handle, so the UCRT (and the legacy CRT, for older toolchains) is probed by
  # name instead. Flushing is best-effort: failing to flush must never prevent the
  # file descriptor from being restored in __exit__.
  _C_RUNTIMES = ("ucrtbase", "msvcrt") if os.name == "nt" else (None,)

  @staticmethod
  def _flush_c_stream():
    for name in RedirectStream._C_RUNTIMES:
      try:
        libc = ctypes.CDLL(name)
        libc.fflush(None)
      except (OSError, TypeError, AttributeError, ValueError):
        continue

  def __init__(self, stream, filename=os.devnull, disabled=False):
    self.stream = stream
    self.filename = filename
    self.disabled = disabled
    self.fd = None
    self.dup_stream = None

  def __enter__(self):
    if self.disabled: return
    try:
      self.stream.flush()  # ensures python stream unaffected
      self.fd = open(self.filename, "w+", encoding="utf-8")  # pylint: disable=consider-using-with
      self.dup_stream = os.dup(self.stream.fileno())
      os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream
    except Exception as e:  # pylint: disable=broad-except
      # TODO: redirect stream breaks in jupyter notebooks.
      #       This try except is a hacky workaround
      print(e)

  def __exit__(self, _, value, traceback):
    if self.disabled: return
    RedirectStream._flush_c_stream()  # best-effort; empties the C stream buffer
    try:
      if self.dup_stream is not None:
        # Restoring the descriptor must happen even if flushing above did nothing,
        # otherwise the caller is left with a stream pointing at a closed file.
        os.dup2(self.dup_stream, self.stream.fileno())
        os.close(self.dup_stream)
        self.dup_stream = None
    finally:
      if self.fd is not None:
        self.fd.close()
        self.fd = None
