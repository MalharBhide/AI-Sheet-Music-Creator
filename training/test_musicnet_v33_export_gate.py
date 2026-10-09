import pytest


def test_failed_regression_cannot_export_asset(tmp_path, monkeypatch):
    import export_musicnet_anchor_v33 as export

    output = tmp_path / "must-not-exist.npz"
    monkeypatch.setattr(export, "frozen", lambda _: ({}, {}, None, None))

    def reject(_):
        raise ValueError("failed regression")

    monkeypatch.setattr(export, "require_regression", reject)
    with pytest.raises(ValueError, match="failed regression"):
        export.export(tmp_path, tmp_path, output)
    assert not output.exists()
