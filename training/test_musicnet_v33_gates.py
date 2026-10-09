import json

import pytest
import reserved_musicnet_v33 as reserved
from bass_v32_regression_data import load as regressions
from musicnet_v32_data import load as fitting


def test_reserved_declaration_reads_metadata_only_and_keeps_all_excerpts(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    piano = tmp_path / "piano"
    source.mkdir()
    piano.mkdir()
    (source / "manifest.json").write_text("{}")
    (piano / "plan.json").write_text(
        json.dumps(
            {
                "source_root": str(source),
                "source_manifest_sha256": reserved.digest(source / "manifest.json"),
            }
        )
    )
    rows = [
        {
            "id": "1234",
            "work": ["composer", "work"],
            "group": "reserved",
            "labels": str(tmp_path / "missing-labels.csv"),
            "labels_sha256": "labels",
            "clock": {"sample_rate": 44100},
            "excerpts": [
                {"audio": str(tmp_path / f"missing-{i}.wav"), "audio_sha256": str(i)}
                for i in range(3)
            ],
        },
        {"id": "5678", "group": "train"},
    ]
    monkeypatch.setattr(
        reserved, "sources", lambda root: rows if root == source else []
    )
    result = reserved.declaration(piano)
    assert len(result["jobs"]) == 3
    assert result["expected_recordings"] == 35
    assert [j["excerpt_index"] for j in result["jobs"]] == [0, 1, 2]
    assert result["procedural_seeds"] == list(range(278101, 278133))
    assert all(j["publisher_id"] == "1234" for j in result["jobs"])
    assert result["no_reserved_audio_or_labels_read"]
    (source / "manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Changed"):
        reserved.declaration(piano)


@pytest.mark.parametrize("groups", [("reserved",), ("test",), ()])
def test_fitting_rejects_test_and_reserved_groups_before_files(tmp_path, groups):
    with pytest.raises(ValueError, match="cannot read"):
        fitting(tmp_path / "missing-baseline", tmp_path / "missing-piano", groups)


def test_regression_reader_rejects_changed_manifest_before_cache_access(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        regressions(tmp_path, "changed")


def test_failed_regression_cannot_create_fresh_output(tmp_path, monkeypatch):
    import fresh_musicnet_anchor_v33 as fresh

    run = tmp_path / "run"
    run.mkdir()
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(fresh, "frozen", lambda _: ({}, {}, None, None))

    def reject(_):
        raise ValueError("regression failed")

    monkeypatch.setattr(fresh, "require_regression", reject)
    with pytest.raises(ValueError, match="regression failed"):
        fresh.verify(run, output)
    assert not output.exists()
