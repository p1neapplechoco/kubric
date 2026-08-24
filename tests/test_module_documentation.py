"""AST-only checks for public module documentation contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATHS = (
    "interventions/__init__.py",
    "interventions/dataset.py",
    "interventions/graph_extraction.py",
    "interventions/kinematic_simulator.py",
    "interventions/logging.py",
    "interventions/schema.py",
    "interventions/tagging.py",
    "interventions/trajectory.py",
    "interventions/twin_runner.py",
    "scripts/__init__.py",
    "scripts/generate_dataset.py",
    "scripts/generate_instance.py",
    "scripts/trajectory_demo_spec.py",
    "scripts/demo_collision_intervention.py",
    "scripts/render_demo_branches_blender.py",
    "scripts/compose_intervention_demo.py",
)
_CONTRACT_HEADINGS = (
    "Purpose:",
    "Public API:",
    "Dependencies:",
    "Trust boundary:",
)
_REQUIRED_ENTRIES = {
    "scripts/trajectory_demo_spec.py": (
        "DemoObjectSpec",
        "DemoSceneSpec",
        "validate_demo_spec",
        "canonical_spec_payload",
        "spec_sha256",
        "demo_spec_summary",
    ),
    "scripts/demo_collision_intervention.py": (
        "build_demo_inputs",
        "generate_demo",
        "write_demo_bundle",
        "main",
    ),
    "scripts/render_demo_branches_blender.py": ("main",),
    "scripts/compose_intervention_demo.py": (
        "compose_intervention_demo",
        "main",
    ),
}
_EXACT_MAIN_DOCSTRINGS = {
    "scripts/generate_dataset.py": (
        "Runs resumable batch generation and returns its stable CLI exit status."
    ),
    "scripts/generate_instance.py": (
        "Runs one inspectable generation attempt and returns its CLI exit status."
    ),
    "scripts/demo_collision_intervention.py": (
        "Generates and atomically publishes the canonical three-branch replay."
    ),
    "scripts/render_demo_branches_blender.py": (
        "Preflights requested replays, renders them atomically, and returns zero."
    ),
    "scripts/compose_intervention_demo.py": (
        "Validates CLI inputs, composes the comparison atomically, and returns zero."
    ),
}


def _parse(module_path: str) -> ast.Module:
  source_path = _ROOT / module_path
  return ast.parse(source_path.read_text(encoding="utf-8"), filename=module_path)


def _public_definitions(tree: ast.Module) -> dict[str, ast.AST]:
  definition_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
  return {
      node.name: node
      for node in tree.body
      if isinstance(node, definition_types) and not node.name.startswith("_")
  }


@pytest.mark.parametrize("module_path", _MODULE_PATHS)
def test_module_docstring_declares_contract(module_path: str) -> None:
  module_docstring = ast.get_docstring(_parse(module_path), clean=True) or ""
  missing = [
      heading for heading in _CONTRACT_HEADINGS if heading not in module_docstring
  ]
  assert not missing, f"{module_path} module docstring is missing {missing!r}"


@pytest.mark.parametrize("module_path", _MODULE_PATHS)
def test_public_top_level_api_is_documented(module_path: str) -> None:
  definitions = _public_definitions(_parse(module_path))
  missing = [
      name
      for name, node in definitions.items()
      if not (ast.get_docstring(node, clean=True) or "").strip()
  ]
  assert not missing, f"{module_path} public definitions lack docstrings: {missing!r}"


@pytest.mark.parametrize(
    ("module_path", "required_names"), _REQUIRED_ENTRIES.items()
)
def test_required_demo_entries_are_documented(
    module_path: str, required_names: tuple[str, ...]
) -> None:
  definitions = _public_definitions(_parse(module_path))
  missing = [name for name in required_names if name not in definitions]
  undocumented = [
      name
      for name in required_names
      if name in definitions
      and not (ast.get_docstring(definitions[name], clean=True) or "").strip()
  ]
  assert not missing, f"{module_path} lacks required public entries: {missing!r}"
  assert not undocumented, (
      f"{module_path} required entries lack docstrings: {undocumented!r}"
  )


@pytest.mark.parametrize(
    ("module_path", "expected_docstring"), _EXACT_MAIN_DOCSTRINGS.items()
)
def test_cli_main_docstring_is_stable(
    module_path: str, expected_docstring: str
) -> None:
  main_node = _public_definitions(_parse(module_path))["main"]
  assert ast.get_docstring(main_node, clean=True) == expected_docstring
