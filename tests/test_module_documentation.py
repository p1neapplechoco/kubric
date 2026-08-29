"""AST-only checks for public module documentation contracts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATHS = (
    "interventions/__init__.py",
    "interventions/_portability.py",
    "interventions/appearance.py",
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


def _production_module_paths() -> tuple[str, ...]:
  package_roots = (_ROOT / "interventions", _ROOT / "scripts")
  return tuple(sorted(
      source_path.relative_to(_ROOT).as_posix()
      for package_root in package_roots
      for source_path in package_root.rglob("*.py")
      if source_path.is_file() and "__pycache__" not in source_path.parts
  ))


def _validate_contract_sections(docstring: str, module_path: str) -> None:
  lines = docstring.splitlines()
  positions = []
  for heading in _CONTRACT_HEADINGS:
    matches = [
        index for index, line in enumerate(lines) if line.startswith(heading)
    ]
    assert len(matches) == 1, (
        f"{module_path} module docstring must contain exactly one {heading!r} "
        "heading at the start of a line"
    )
    positions.append(matches[0])

  assert positions == sorted(positions), (
      f"{module_path} module docstring headings must appear in this order: "
      f"{_CONTRACT_HEADINGS!r}"
  )
  for index, (heading, position) in enumerate(zip(_CONTRACT_HEADINGS, positions)):
    next_position = (
        positions[index + 1] if index + 1 < len(positions) else len(lines)
    )
    content = "\n".join(
        (lines[position][len(heading):], *lines[position + 1:next_position])
    )
    assert content.strip(), (
        f"{module_path} module docstring {heading!r} section must contain "
        "non-whitespace content"
    )


def _public_definitions(tree: ast.Module) -> dict[str, ast.AST]:
  definition_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
  return {
      node.name: node
      for node in tree.body
      if isinstance(node, definition_types) and not node.name.startswith("_")
  }


def test_documented_module_inventory_matches_production_modules() -> None:
  assert tuple(sorted(_MODULE_PATHS)) == _production_module_paths()


@pytest.mark.parametrize("module_path", _MODULE_PATHS)
def test_module_docstring_declares_contract(module_path: str) -> None:
  module_docstring = ast.get_docstring(_parse(module_path), clean=True) or ""
  _validate_contract_sections(module_docstring, module_path)


@pytest.mark.parametrize(
    "module_docstring",
    (
        (
            "Public API: stable API\n"
            "Purpose: substantive purpose\n"
            "Dependencies: direct dependency\n"
            "Trust boundary: explicit boundary"
        ),
        (
            "Purpose:   \n"
            "Public API: stable API\n"
            "Dependencies: direct dependency\n"
            "Trust boundary: explicit boundary"
        ),
    ),
)
def test_module_contract_checker_rejects_malformed_sections(
    monkeypatch: pytest.MonkeyPatch, module_docstring: str
) -> None:
  tree = ast.parse(repr(module_docstring))
  monkeypatch.setattr(sys.modules[__name__], "_parse", lambda _: tree)

  with pytest.raises(AssertionError):
    test_module_docstring_declares_contract("synthetic.py")


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
