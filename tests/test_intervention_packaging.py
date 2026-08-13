"""Static packaging contract for the trajectory-intervention extension."""

import ast
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


_ROOT = Path(__file__).resolve().parents[1]


def _install_requirement_names():
  lines = (_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
  return {
      canonicalize_name(Requirement(line).name)
      for line in lines
      if line.strip() and not line.lstrip().startswith("#")
  }


def _setup_keyword(name):
  tree = ast.parse((_ROOT / "setup.py").read_text(encoding="utf-8"))
  setup_calls = [
      node for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and isinstance(node.func.value, ast.Name)
      and node.func.value.id == "setuptools"
      and node.func.attr == "setup"
  ]
  assert len(setup_calls) == 1
  keywords = {keyword.arg: keyword.value for keyword in setup_calls[0].keywords}
  assert name in keywords
  return ast.literal_eval(keywords[name])


def test_intervention_runtime_dependencies_are_declared():
  assert {"pyyaml", "scipy"} <= _install_requirement_names()


def test_scene_ranges_is_packaged_from_one_canonical_source():
  config = _ROOT / "configs" / "scene_ranges.yaml"
  assert config.is_file()
  assert list(_ROOT.glob("**/scene_ranges.yaml")) == [config]
  assert _setup_keyword("data_files") == [
      ("share/kubric/configs", ["configs/scene_ranges.yaml"]),
  ]
