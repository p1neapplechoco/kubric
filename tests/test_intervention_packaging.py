"""Static packaging contract for the trajectory-intervention extension."""

import ast
from email.parser import BytesParser
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


_ROOT = Path(__file__).resolve().parents[1]


def _copy_minimal_build_source(tmp_path):
  source = tmp_path / "source"
  source.mkdir()
  for filename in ("LICENSE", "README.md", "requirements.txt", "setup.py"):
    shutil.copy2(_ROOT / filename, source / filename)
  (source / "kubric").mkdir()
  shutil.copy2(
      _ROOT / "kubric" / "__init__.py", source / "kubric" / "__init__.py"
  )
  (source / "configs").mkdir()
  shutil.copy2(
      _ROOT / "configs" / "scene_ranges.yaml",
      source / "configs" / "scene_ranges.yaml",
  )
  return source


def _run_build_command(command, *, cwd):
  env = os.environ.copy()
  env.update({
      "PIP_DISABLE_PIP_VERSION_CHECK": "1",
      "PIP_NO_INDEX": "1",
      "PYTHONNOUSERSITE": "1",
  })
  return subprocess.run(
      command,
      cwd=cwd,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )


def _set_local_source_version(init_path):
  contents = init_path.read_text(encoding="utf-8")
  init_path.write_text(
      contents.replace('__version__ = "HEAD"', '__version__ = "LOCAL"'),
      encoding="utf-8",
  )


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


def test_default_pep517_wheel_has_stable_version_and_restores_source(tmp_path):
  source = _copy_minimal_build_source(tmp_path)
  wheel_dir = tmp_path / "wheelhouse"
  wheel_dir.mkdir()
  init_path = source / "kubric" / "__init__.py"
  _set_local_source_version(init_path)
  original_init = init_path.read_bytes()

  result = _run_build_command(
      [
          sys.executable,
          "-m",
          "pip",
          "wheel",
          ".",
          "--no-deps",
          "--no-build-isolation",
          "--no-index",
          "--wheel-dir",
          str(wheel_dir),
      ],
      cwd=source,
  )

  assert result.returncode == 0, result.stdout + result.stderr
  assert init_path.read_bytes() == original_init
  wheels = list(wheel_dir.glob("*.whl"))
  assert len(wheels) == 1
  assert wheels[0].name == "kubric-0.0.0-py3-none-any.whl"

  with zipfile.ZipFile(wheels[0]) as archive:
    metadata_members = [
        name for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")
    ]
    config_members = [
        name for name in archive.namelist()
        if name.endswith("/share/kubric/configs/scene_ranges.yaml")
    ]
    assert len(metadata_members) == 1
    assert len(config_members) == 1
    metadata = BytesParser().parsebytes(archive.read(metadata_members[0]))
    assert metadata["Name"] == "kubric"
    assert metadata["Version"] == "0.0.0"
    assert archive.read(config_members[0]) == (
        source / "configs" / "scene_ranges.yaml"
    ).read_bytes()
    wheel_init = archive.read("kubric/__init__.py").decode("utf-8")
    assert re.search(r'^__version__ = "0\.0\.0"$', wheel_init, re.MULTILINE)


def test_setup_failure_restores_source_version(tmp_path):
  source = _copy_minimal_build_source(tmp_path)
  init_path = source / "kubric" / "__init__.py"
  _set_local_source_version(init_path)
  original_init = init_path.read_bytes()

  result = _run_build_command(
      [sys.executable, "setup.py", "not-a-real-command"], cwd=source
  )

  assert result.returncode != 0
  assert init_path.read_bytes() == original_init


def test_explicit_version_modes_remain_available(tmp_path):
  source = _copy_minimal_build_source(tmp_path)
  init_path = source / "kubric" / "__init__.py"
  original_init = init_path.read_bytes()
  cases = (
      (("--tag", "v1.2.3"), "kubric", r"1\.2\.3"),
      (("--nightly",), "kubric-nightly", r"\d{4}\.\d{1,2}\.\d{1,2}"),
      (
          ("--secondly",),
          "kubric-secondly",
          r"\d{4}(?:\.\d{1,2}){5}",
      ),
  )

  for mode_args, expected_name, version_pattern in cases:
    result = _run_build_command(
        [sys.executable, "setup.py", *mode_args, "--name", "--version"],
        cwd=source,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    generated_name, generated_version = result.stdout.splitlines()
    assert generated_name == expected_name
    assert re.fullmatch(version_pattern, generated_version)
    assert init_path.read_bytes() == original_init
