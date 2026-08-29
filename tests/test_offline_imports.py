"""Package import boundaries for backend-independent intervention tools."""

import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_offline_submodules_import_when_optional_backends_are_blocked():
  script = r'''
import importlib.abc
import sys


class BackendBlocker(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname.split(".", 1)[0] in {"kubric", "pybullet"}:
      raise ModuleNotFoundError(
          "blocked optional backend dependency: " + fullname,
          name=fullname,
      )
    return None


sys.meta_path.insert(0, BackendBlocker())

import interventions
from interventions.appearance import MaterialSpec, TextureSpec, VisualObjectSpec
from interventions.graph_extraction import TemporalGraph, extract_ground_truth
from interventions.logging import ContactRecord, SimulationLog
from interventions.schema import GroundTruth, SceneConfig
from interventions.tagging import derive_tags
from interventions.trajectory import build_path, validate_path

assert TextureSpec(kind="solid", colors=((0.0, 0.0, 0.0, 1.0),)).kind == "solid"
assert MaterialSpec is not None and VisualObjectSpec is not None
assert interventions.TemporalGraph is TemporalGraph
assert interventions.extract_ground_truth is extract_ground_truth
assert interventions.ContactRecord is ContactRecord
assert interventions.SimulationLog is SimulationLog
assert interventions.GroundTruth is GroundTruth
assert interventions.SceneConfig is SceneConfig
assert interventions.derive_tags is derive_tags
assert interventions.build_path is build_path
assert interventions.validate_path is validate_path
assert "interventions.kinematic_simulator" not in sys.modules
assert "interventions.twin_runner" not in sys.modules
assert "KinematicSimulator" in interventions.__all__
assert "KinematicSimulator" in dir(interventions)
assert "read_paired_artifact" in interventions.__all__
assert "read_paired_artifact" in dir(interventions)
assert not any(
    name.split(".", 1)[0] in {"kubric", "pybullet"}
    for name in sys.modules
)
'''

  result = subprocess.run(
      [sys.executable, "-c", script],
      cwd=_PROJECT_ROOT,
      check=False,
      capture_output=True,
      text=True,
  )

  assert result.returncode == 0, result.stderr
