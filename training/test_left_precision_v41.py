import numpy as np
import pytest
from left_precision_v41 import CONTEXT_WIDTH, combine, model, probabilities
from periodicity_features import NAMES
from prepare_left_support_v41 import DURATION, KINDS, RATE, render, schedule
from test_chord_boundary_v36 import item


def notes(group="train"):
    return [{"id": f"p-{pitch}", "pitch": pitch, "group": group, "instrument_str": "fixture",
             "audio": f"/fixture/{pitch}.wav", "audio_sha256": "verified"} for pitch in range(24, 60)]


def test_both_independent_heads_must_permit_removal():
    reject = np.array([[.001, .999], [.999, .001], [.01, .99]])
    guard = np.array([[.99, .01], [.001, .999], [.02, .98]])
    ps = combine(reject, guard)
    np.testing.assert_allclose(ps[:, 1], [.01, .001, .98])
    np.testing.assert_allclose(ps.sum(axis=1), 1.)
    with pytest.raises(ValueError, match="dual support"):
        combine(reject, guard*2)


@pytest.mark.parametrize("width", [12, 20])
def test_physical_guardian_does_not_use_reference_labels_or_recording_identity(width):
    data = item([[3., 5., 48, 80], [3., 6., 52, 90]])
    data["frames"] = np.zeros((2, 40, 9), np.float32)
    data["periodicity"] = np.zeros((2, len(NAMES)), np.float32)
    network = model(width)
    normalizer = np.zeros(CONTEXT_WIDTH), np.ones(CONTEXT_WIDTH)
    before = probabilities(network, data, normalizer)
    data.update(reference=np.empty((0, 3)), pitch_reference=np.empty((0, 3)),
                id="different-song", corpus="other", group="test", y=np.zeros(2))
    np.testing.assert_array_equal(probabilities(network, data, normalizer), before)
    assert np.all(before[:, 0] > .98)
    with pytest.raises(ValueError, match="normalization"):
        probabilities(network, data, (normalizer[0], -normalizer[1]))


def test_authored_repeats_have_declared_spacing_and_no_source_partition_leakage():
    source = notes()
    first = schedule(source, "short-repeated")
    assert [x[0] for x in first[:3]] == [3., 3.75, 4.5]
    assert all(x[1] == .4 for x in first)
    for kind in KINDS:
        assert schedule(list(reversed(source)), kind) == schedule(source, kind)
    with pytest.raises(ValueError, match="partition"):
        schedule(notes("reserved"), "quiet-single")
    with pytest.raises(ValueError, match="partition"):
        schedule(source[:10]+notes("validation")[10:], "quiet-single")


@pytest.mark.parametrize("kind", ["short-repeated", "long-overlap", "quiet-chords"])
def test_authoring_protects_complete_decay_and_each_simultaneous_pitch(monkeypatch, kind):
    import prepare_left_support_v41 as author

    monkeypatch.setattr(author, "digest", lambda path: "verified")
    monkeypatch.setattr(author.sf, "read", lambda *args, **kwargs: (np.ones(64000, np.float32), 16000))
    wave, refs, support, used = render(notes(), kind)
    assert wave.shape == (round(DURATION*RATE),) and np.isfinite(wave).all()
    assert len(refs) == len(support) == len(schedule(notes(), kind))
    assert len(used) >= 1
    for ref, pitch in zip(refs, support, strict=True):
        assert ref[0] == pitch[0] and ref[2] == pitch[2]
        assert pitch[1] > ref[1]
        tail = 1. if kind != "short-repeated" else .08
        assert pitch[1]-ref[1] == pytest.approx(tail, abs=1/RATE)
    if kind == "quiet-chords":
        assert len({r[2] for r in refs if r[0] == 3.}) == 3
