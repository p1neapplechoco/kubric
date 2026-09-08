"""Tests for the velocity-dataset batch driver and its HuggingFace publisher."""

import collections
import json

import pytest

from interventions import velocity_scenes
from scripts.generate_velocity_dataset import (
    _is_complete,
    _physics_complete,
    assign_split,
    generate_dataset,
)
from scripts.upload_velocity_dataset import (
    TOKEN_VARIABLES,
    build_dataset_card,
    build_split_index,
    resolve_token,
    upload_dataset,
)


_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _manifest(instances=(), **overrides):
  manifest = {
      "master_seed": 7,
      "requested": len(instances),
      "accepted": len(instances),
      "attempts": len(instances),
      "branches": ["factual", "counterfactual", "subject_removed"],
      "render_profile": {
          "resolution": [512, 384],
          "samples_per_pixel": 48,
          "device": "GPU",
      },
      "splits": {"seed": 1, "fractions": dict(_FRACTIONS), "counts": {}},
      "instances": list(instances),
  }
  manifest.update(overrides)
  return manifest


def _entry(instance_id, split="train", index=0, num_objects=6):
  return {
      "instance_id": instance_id,
      "index": index,
      "split": split,
      "subject_id": "subject",
      "num_objects": num_objects,
      "object_ids": ["floor", "subject"],
      "branches": ["factual", "counterfactual", "subject_removed"],
      "visual_scene_hash": "0" * 64,
      "motion": {},
      "qc_metrics": {},
      "rendered": True,
  }


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def test_assign_split_is_a_pure_function_of_the_instance_id():
  first = assign_split("instance_abc", _FRACTIONS, 11)
  assert all(
      assign_split("instance_abc", _FRACTIONS, 11) == first for _ in range(10)
  )


def test_assign_split_depends_on_the_split_seed():
  ids = ["instance_{:04x}".format(value) for value in range(400)]
  one = [assign_split(name, _FRACTIONS, 1) for name in ids]
  two = [assign_split(name, _FRACTIONS, 2) for name in ids]
  assert one != two


def test_assign_split_does_not_depend_on_batch_size_or_order():
  """Extending a batch must not move an instance between splits.

  This is the property that keeps a held-out scene held out when a dataset is
  regenerated with more instances or a different rejection pattern.
  """
  ids = ["instance_{:04x}".format(value) for value in range(50)]
  reference = {name: assign_split(name, _FRACTIONS, 3) for name in ids}
  for name in reversed(ids):
    assert assign_split(name, _FRACTIONS, 3) == reference[name]


def test_assign_split_approximates_the_configured_fractions():
  counts = collections.Counter(
      assign_split("instance_{:08x}".format(value), _FRACTIONS, 5)
      for value in range(20000)
  )
  for name, fraction in _FRACTIONS.items():
    assert counts[name] / 20000 == pytest.approx(fraction, abs=0.015)


def test_assign_split_handles_a_single_bucket():
  assert assign_split("instance_abc", {"train": 1.0}, 5) == "train"


def test_assign_split_ignores_the_scale_of_the_fractions():
  scaled = {name: value * 37.0 for name, value in _FRACTIONS.items()}
  ids = ["instance_{:04x}".format(value) for value in range(200)]
  assert [assign_split(name, _FRACTIONS, 9) for name in ids] == [
      assign_split(name, scaled, 9) for name in ids
  ]


def test_assign_split_never_returns_a_zero_weighted_bucket():
  fractions = {"train": 1.0, "val": 0.0, "test": 0.0}
  assigned = {
      assign_split("instance_{:04x}".format(value), fractions, 4)
      for value in range(500)
  }
  assert assigned == {"train"}


@pytest.mark.parametrize(
    "fractions", ({}, {"train": 0.0}, {"train": -1.0, "val": 2.0})
)
def test_assign_split_rejects_unusable_fractions(fractions):
  with pytest.raises(ValueError):
    assign_split("instance_abc", fractions, 1)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_instance_id_is_computable_without_sampling():
  """Resume depends on this: the id must not require a rollout to learn."""
  assert velocity_scenes.instance_id_for(11, 3) == velocity_scenes.instance_id_for(
      11, 3
  )
  assert velocity_scenes.instance_id_for(11, 3) != velocity_scenes.instance_id_for(
      11, 4
  )
  assert velocity_scenes.instance_id_for(11, 3) != velocity_scenes.instance_id_for(
      12, 3
  )


def _write_instance(root, branches=("factual",), *, size=1):
  root.mkdir(parents=True, exist_ok=True)
  (root / "instance.json").write_text("{}", encoding="utf-8")
  (root / "render.json").write_text("{}", encoding="utf-8")
  for branch in branches:
    branch_dir = root / branch
    branch_dir.mkdir(exist_ok=True)
    for name in ("video.mp4", "segmentation.npz", "tracking.npz", "graph.json"):
      (branch_dir / name).write_bytes(b"x" * size)
  return root


def test_is_complete_accepts_a_fully_written_instance(tmp_path):
  root = _write_instance(tmp_path / "instance_a", ("factual", "counterfactual"))
  assert _is_complete(root, ("factual", "counterfactual"))


def test_is_complete_rejects_a_missing_branch(tmp_path):
  root = _write_instance(tmp_path / "instance_a", ("factual",))
  assert not _is_complete(root, ("factual", "counterfactual"))


def test_is_complete_rejects_a_truncated_artifact(tmp_path):
  """A run killed mid-encode leaves a zero-byte mp4; resume must not trust it."""
  root = _write_instance(tmp_path / "instance_a", ("factual",))
  (root / "factual" / "video.mp4").write_bytes(b"")
  assert not _is_complete(root, ("factual",))


def test_is_complete_rejects_a_missing_render_record(tmp_path):
  root = _write_instance(tmp_path / "instance_a", ("factual",))
  (root / "render.json").unlink()
  assert not _is_complete(root, ("factual",))


def test_physics_complete_does_not_require_rendered_artifacts(tmp_path):
  root = tmp_path / "instance_a"
  root.mkdir()
  assert not _physics_complete(root)
  (root / "instance.json").write_text("{}", encoding="utf-8")
  assert _physics_complete(root)


# ---------------------------------------------------------------------------
# Batch generation, physics only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ranges():
  return velocity_scenes.load_ranges("configs/scene_ranges_velocity.yaml")


def test_generate_dataset_publishes_a_manifest_and_instances(tmp_path, _ranges):
  manifest = generate_dataset(
      _ranges, tmp_path, 20260908, 2, 8, render=False, progress=False
  )
  assert manifest["status"] == "complete"
  assert manifest["accepted"] == 2
  assert len(manifest["instances"]) == 2
  assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
  for entry in manifest["instances"]:
    summary_path = (
        tmp_path / "instances" / entry["instance_id"] / "instance.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["split"] == entry["split"]
    assert summary["trust_model"] == velocity_scenes.VELOCITY_TRUST_MODEL


def test_generate_dataset_is_reproducible_from_the_seed(tmp_path, _ranges):
  first = generate_dataset(
      _ranges, tmp_path / "a", 4242, 2, 8, render=False, progress=False
  )
  second = generate_dataset(
      _ranges, tmp_path / "b", 4242, 2, 8, render=False, progress=False
  )
  assert [entry["instance_id"] for entry in first["instances"]] == [
      entry["instance_id"] for entry in second["instances"]
  ]
  assert [entry["split"] for entry in first["instances"]] == [
      entry["split"] for entry in second["instances"]
  ]


def test_generate_dataset_resume_republishes_without_rewriting(tmp_path, _ranges):
  generate_dataset(
      _ranges, tmp_path, 20260908, 2, 8, render=False, progress=False
  )
  stamps = {
      path: path.stat().st_mtime_ns
      for path in tmp_path.rglob("instance.json")
  }
  assert stamps

  resumed = generate_dataset(
      _ranges, tmp_path, 20260908, 2, 8, render=False, resume=True, progress=False
  )
  assert resumed["accepted"] == 2
  assert {
      path: path.stat().st_mtime_ns for path in tmp_path.rglob("instance.json")
  } == stamps


def test_generate_dataset_reports_an_incomplete_batch(tmp_path, _ranges):
  """Running out of attempts is reported, not silently returned as success."""
  manifest = generate_dataset(
      _ranges, tmp_path, 20260908, 5, 1, render=False, progress=False
  )
  assert manifest["status"] == "incomplete"
  assert manifest["attempts"] == 1
  assert manifest["accepted"] < 5


def test_generate_dataset_records_rejected_attempts(tmp_path, _ranges):
  """A rejection sampler's failures belong in the manifest, not swallowed."""
  manifest = generate_dataset(
      _ranges, tmp_path, 20260908, 14, 16, render=False, progress=False
  )
  assert manifest["rejections"], "expected at least one placement rejection"
  for rejection in manifest["rejections"]:
    assert rejection["status"] in ("sample_failed", "qc_rejected")
    assert rejection["reason"]
    assert rejection["index"] not in {
        entry["index"] for entry in manifest["instances"]
    }


def test_generate_dataset_rejects_an_unknown_branch(tmp_path, _ranges):
  with pytest.raises(ValueError, match="unknown branches"):
    generate_dataset(
        _ranges,
        tmp_path,
        20260908,
        1,
        1,
        render=False,
        branches=["factual", "imaginary"],
        progress=False,
    )


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def test_resolve_token_prefers_an_explicit_value(monkeypatch):
  monkeypatch.setenv("HF_TOKEN", "from-env")
  assert resolve_token("  explicit  ") == "explicit"


def test_resolve_token_reads_the_environment(monkeypatch):
  for name in TOKEN_VARIABLES:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv(TOKEN_VARIABLES[-1], "from-env")
  assert resolve_token() == "from-env"


def test_resolve_token_fails_loudly_when_it_cannot_prompt(monkeypatch):
  for name in TOKEN_VARIABLES:
    monkeypatch.delenv(name, raising=False)
  with pytest.raises(RuntimeError, match="no access token"):
    resolve_token(prompt=False)


def test_build_split_index_sorts_and_records_paths():
  manifest = _manifest([
      _entry("instance_b", "val", 1),
      _entry("instance_a", "train", 0),
      _entry("instance_c", "train", 2),
  ])
  rows = build_split_index(manifest)
  assert [row["instance_id"] for row in rows] == [
      "instance_a",
      "instance_c",
      "instance_b",
  ]
  assert rows[0]["path"] == "instances/instance_a"


def test_dataset_card_frontmatter_is_parseable_yaml():
  yaml = pytest.importorskip("yaml")
  manifest = _manifest(
      [_entry("instance_a"), _entry("instance_b", "val", 1)],
      splits={"seed": 1, "fractions": _FRACTIONS, "counts": {"train": 1, "val": 1}},
  )
  card = build_dataset_card(manifest, "user/name")
  assert card.startswith("---\n")
  front = yaml.safe_load(card.split("---")[1])
  assert front["dataset_info"]["splits"] == [
      {"name": "train", "num_examples": 1},
      {"name": "val", "num_examples": 1},
  ]


def test_dataset_card_reports_the_observed_body_count():
  """The card is generated from the manifest, so it cannot overstate the data."""
  manifest = _manifest([
      _entry("instance_a", num_objects=5),
      _entry("instance_b", num_objects=7),
  ])
  assert "**4-6 bodies**" in build_dataset_card(manifest, "user/name")


def test_dataset_card_names_the_repo_and_the_seed():
  manifest = _manifest([_entry("instance_a")], master_seed=1234)
  card = build_dataset_card(manifest, "user/name")
  assert "# user/name" in card
  assert "1234" in card


def test_upload_dry_run_writes_the_card_and_index_but_sends_nothing(tmp_path):
  manifest = _manifest([_entry("instance_a"), _entry("instance_b", "val", 1)])
  (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

  result = upload_dataset(tmp_path, "user/name", dry_run=True)

  assert result["dry_run"] is True
  assert result["uploaded"] is False
  assert "url" not in result
  assert result["instances"] == 2
  assert (tmp_path / "README.md").is_file()
  assert len(json.loads((tmp_path / "split_index.json").read_text("utf-8"))) == 2


def test_upload_defaults_to_a_private_repo(tmp_path):
  """Publishing is hard to undo, so the default must not be public."""
  manifest = _manifest([_entry("instance_a")])
  (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
  assert upload_dataset(tmp_path, "user/name", dry_run=True)["private"] is True


def test_upload_requires_a_manifest(tmp_path):
  with pytest.raises(FileNotFoundError, match="manifest.json"):
    upload_dataset(tmp_path, "user/name", dry_run=True)


def test_upload_refuses_an_empty_batch(tmp_path):
  (tmp_path / "manifest.json").write_text(
      json.dumps(_manifest([])), encoding="utf-8"
  )
  with pytest.raises(ValueError, match="no instances"):
    upload_dataset(tmp_path, "user/name", dry_run=True)


def test_upload_never_puts_the_token_in_published_files(tmp_path):
  manifest = _manifest([_entry("instance_a")])
  (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

  upload_dataset(tmp_path, "user/name", token="hf_secret_value", dry_run=True)

  for name in ("README.md", "split_index.json", "manifest.json"):
    assert "hf_secret_value" not in (tmp_path / name).read_text(encoding="utf-8")
