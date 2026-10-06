"""Guard new acoustic training against reserved data and inconsistent clocks."""

import tarfile

import numpy as np
import pytest
from acquire_nsynth_subset import safe_member, verified
from bass_v25_data import load as parent_items
from periodicity_bass_data import load
from periodicity_model import INPUTS, model, probabilities
from prepare_nsynth_bass import make_sequence


@pytest.mark.parametrize("loader", [parent_items, load])
def test_reserved_fitting_request_is_rejected_before_file_access(tmp_path, loader):
    args = (tmp_path,) if loader is parent_items else (tmp_path, tmp_path)
    with pytest.raises(ValueError, match="cannot read"):
        loader(*args, groups=("test",))


@pytest.mark.parametrize("name", ["../../outside.wav", "/outside.wav"])
def test_nsynth_archive_path_escape_is_rejected(name):
    member = tarfile.TarInfo(name)
    with pytest.raises(ValueError, match="Unsafe"):
        safe_member(member)


def test_incomplete_nsynth_archive_is_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "incomplete.tar.gz"
    archive.write_bytes(b"partial")
    with pytest.raises(ValueError, match="Incomplete"):
        verified(archive)
    assert list(tmp_path.iterdir()) == [archive]


def test_authored_octave_and_repetition_sequence_labels_preserve_true_holds(monkeypatch):
    import prepare_nsynth_bass as producer

    records = [
        {
            "id": "instrument-" + str(pitch),
            "pitch": pitch,
            "audio": "not-user-audio-" + str(pitch),
            "audio_sha256": "fixture",
        }
        for pitch in (33, 45, 40, 48)
    ]
    monkeypatch.setattr(producer, "digest", lambda _: "fixture")
    monkeypatch.setattr(
        producer.sf,
        "read",
        lambda path, **kwargs: (
            np.sin(2 * np.pi * 55 * np.arange(64000) / 16000).astype(np.float32),
            16000,
        ),
    )
    for kind in ("repeat", "octave", "dyad", "layered"):
        wave, reference, pitch_reference, used = make_sequence(records, kind)
        assert wave.shape == (19 * 22050,) and np.isfinite(wave).all()
        assert {r[0] for r in reference} == {2.75, 7.25, 11.75}
        assert all(r[1] - r[0] == 3 for r in reference)
        assert all(r[1] - r[0] == 4 for r in pitch_reference)
        assert used and np.max(np.abs(wave)) <= 0.800001
        for onset in (2.75, 7.25, 11.75):
            pitches = [r[2] for r in reference if r[0] == onset]
            assert len(set(pitches)) == len(pitches)
            if kind == "octave":
                assert abs(pitches[0] - pitches[1]) == 12
        if kind == "repeat":
            assert len({r[2] for r in reference}) == 1


@pytest.mark.parametrize("change", ["shape", "nan", "scale"])
def test_recurrence_network_rejects_incompatible_inputs(change):
    network = model(64)
    values = np.zeros((2, INPUTS), np.float32)
    mean = np.zeros(INPUTS, np.float32)
    scale = np.ones(INPUTS, np.float32)
    if change == "shape":
        values = values[:, :-1]
    elif change == "nan":
        values[0, 0] = np.nan
    else:
        scale[0] = 0
    with pytest.raises(ValueError, match="inference contract"):
        probabilities(network, values, (mean, scale))


def test_empty_recurrence_network_preserves_binary_output_contract():
    result = probabilities(
        model(64),
        np.empty((0, INPUTS), np.float32),
        (np.zeros(INPUTS, np.float32), np.ones(INPUTS, np.float32)),
    )
    assert result.shape == (0, 2)
