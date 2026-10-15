"""Historical readers are recovered only from exact immutable witnesses."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from restore_approved_replay import restore
import restore_approved_replay as replay


def fixture(root, name="backend/app/services/reader.py"):
    evidence = root / "training/results"
    source = evidence / "original/source-witnesses" / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original implementation\n")
    current = root / name
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(b"current implementation\n")
    plan = evidence / "release-only-v38-v1/plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({"baseline_hashes": {"sources": {
        name: hashlib.sha256(source.read_bytes()).hexdigest()}}}))
    return source, current, plan


def test_exact_archived_code_restored_without_touching_current_code(tmp_path):
    source, current, _ = fixture(tmp_path)
    output = tmp_path / ".training/original"
    result = restore(tmp_path, output)
    assert (output / "backend/app/services/reader.py").read_bytes() == source.read_bytes()
    assert current.read_bytes() == b"current implementation\n"
    assert result["sources_restored"] == 1 and result["guards_changed"] is False


def test_missing_or_changed_witness_refuses_before_creating_output(tmp_path):
    source, _, _ = fixture(tmp_path)
    source.write_bytes(b"changed")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="No hash-matched"):
        restore(tmp_path, output)
    assert not output.exists()


def test_existing_tree_is_never_overwritten(tmp_path):
    fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="Preserve existing"):
        restore(tmp_path, output)


def test_traversal_in_archived_plan_cannot_escape_destination(tmp_path):
    _, _, plan = fixture(tmp_path)
    plan.write_text(json.dumps({"baseline_hashes": {"sources": {"../outside": "0" * 64}}}))
    with pytest.raises(ValueError, match="Invalid historical"):
        restore(tmp_path, tmp_path / "output")


@pytest.mark.parametrize("corrupted", [False, True])
def test_git_fallback_requires_exact_pinned_content(tmp_path, monkeypatch, corrupted):
    source, _, plan = fixture(tmp_path)
    original = source.read_bytes()
    source.unlink()
    record = json.loads(plan.read_text())
    record["baseline_hashes"]["preceding_source_commit"] = "d214a3e"
    plan.write_text(json.dumps(record))

    def git(args, **kwargs):
        assert args == ["git", "-C", str(tmp_path), "show", "d214a3e:backend/app/services/reader.py"]
        return SimpleNamespace(returncode=0, stdout=b"wrong" if corrupted else original)

    monkeypatch.setattr(replay.subprocess, "run", git)
    if corrupted:
        with pytest.raises(ValueError, match="No hash-matched"):
            restore(tmp_path, tmp_path / "output")
        assert not (tmp_path / "output").exists()
    else:
        result = restore(tmp_path, tmp_path / "output")
        assert next(iter(result["source_witnesses"].values())).startswith("git:d214a3e:")
