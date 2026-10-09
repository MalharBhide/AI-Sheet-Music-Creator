import json

import export_teacher_embedding_v32 as producer
import pytest


def test_failed_first_pass_refuses_export_before_asset_creation(tmp_path, monkeypatch):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "manifest.json").write_text(json.dumps({"items": []}))
    (fixtures / "plan.json").write_text("{}")
    (tmp_path / "original-first-pass.json").write_text(json.dumps({"passes": False}))
    monkeypatch.setattr(producer, "frozen", lambda _: ({}, {}, None, None))
    monkeypatch.setattr(producer, "require_regression", lambda _: None)
    output = tmp_path / "must-not-exist.npz"
    with pytest.raises(ValueError, match="first-pass"):
        producer.export(tmp_path, fixtures, output)
    assert not output.exists()
