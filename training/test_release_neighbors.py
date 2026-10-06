import numpy as np
import pytest
from release_neighbor_features import BASE_NAMES, features


def evidence(count):
    base = np.zeros((count, 52), np.float32)
    base[:, BASE_NAMES.index("note_mean")] = 0.8
    base[:, BASE_NAMES.index("onset_at_attack")] = 0.6
    base[:, BASE_NAMES.index("note_end_level")] = 0.4
    return base


def test_long_previous_hold_is_visible_at_its_release():
    events = np.array([[0, 5.01, 36, 80], [5, 6, 36, 80], [5, 6, 40, 80]])
    x = features(events, evidence(3))
    assert x[1, 0] > 0.7
    assert x[1, 1] == pytest.approx(0.04)
    assert x[1, 2] > 0.8
    assert x[1, 3] == pytest.approx(0.4)
    assert x[1, 5] == pytest.approx(0.01)
    assert not x[2].any()


def test_transposition_translation_and_order_do_not_change_geometry():
    events = np.array([[0, 5.01, 36, 80], [1, 5.01, 36, 70], [5, 6, 36, 80]])
    base = evidence(3)
    base[0, BASE_NAMES.index("note_end_level")] = 0.7
    expected = features(events, base)
    order = [2, 1, 0]
    np.testing.assert_allclose(features(events[order], base[order]), expected[order], atol=1e-6)
    shifted = events.copy()
    shifted[:, :2] += 10
    shifted[:, 2] += 12
    np.testing.assert_allclose(features(shifted, base), expected, atol=1e-6)


def test_silence_empty_and_invalid_clocks():
    assert features(np.empty((0, 4)), np.empty((0, 52))).shape == (0, 8)
    events = np.array([[3, 4, 36, 80]])
    assert not features(events, evidence(1)).any()
    events[0, 1] = 3
    with pytest.raises(ValueError, match="clock"):
        features(events, evidence(1))


def test_changed_source_is_rejected_before_decoder_or_output(tmp_path, monkeypatch):
    import prepare_groove_bass as producer

    rows = [
        {"group": g, "id": f"source-{index}"}
        for index, g in enumerate(["train"] * 9 + ["validation"] * 4 + ["reserved"] * 3)
    ]
    monkeypatch.setattr(producer, "drum_sources", lambda _: rows)

    def refuse(_):
        raise ValueError("Changed NSynth source")

    monkeypatch.setattr(producer, "sources", refuse)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="Changed NSynth"):
        producer.prepare(tmp_path, tmp_path, output)
    assert not output.exists()
