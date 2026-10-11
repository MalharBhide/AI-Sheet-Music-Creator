import numpy as np
import pytest
from acquire_mutopia_boundaries import SPLIT
from chord_boundary_v36 import STAY, decode
from prepare_mutopia_boundaries import DURATION, RATE, render
from release_only_v38 import labels, model, probabilities
from test_chord_boundary_v36 import item


def test_unmatched_notes_teach_abstention_and_only_release_is_supervised():
    data = item([[3., 4.08, 48, 80], [7., 8., 52, 80]], [[3., 4., 48]])
    target = labels(data)
    assert np.all(target[:, :2] == -100)
    assert target[:, 2].tolist() == [0, STAY]
    data["eligible"][:] = False
    assert np.all(labels(data) == -100)


def test_real_supported_hold_remains_a_stay_target():
    data = item([[3., 4.08, 48, 80]], [[3., 4., 48]])
    data["pitch_reference"] = np.array([[3., 4.3, 48]])
    assert labels(data)[0, 2] == STAY


@pytest.mark.parametrize("width", [16, 24])
def test_fitted_release_head_cannot_remove_notes_or_move_attacks(width):
    import torch

    data = item([[3., 5., 48, 80], [3., 6., 52, 90]])
    network = model(width)
    with torch.no_grad():
        # Even an extreme learned release correction cannot alter inert heads.
        network.ending[-1].bias[0] = 100
    ps = probabilities(network, data, (np.zeros(100), np.ones(100)))
    after, indices = decode(data, ps, np.zeros(2), .8, 1.)
    np.testing.assert_array_equal(after[:, [0, 2, 3]], data["events"][:, [0, 2, 3]])
    np.testing.assert_allclose(after[:, 1], [4.92, 5.92])
    assert indices.tolist() == [0, 1]


def test_new_composition_split_is_fixed_and_reserved_work_is_separate():
    assert len([g for g in SPLIT.values() if g == "train"]) == 4
    assert SPLIT[263] == "validation"
    assert SPLIT[1899] == "reserved"


def test_renderer_keeps_simultaneous_notes_independent_and_protects_crop_holds():
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    part = pretty_midi.Instrument(0)
    part.notes = [pretty_midi.Note(80, 48, 14., 16.),
                  pretty_midi.Note(80, 52, 15.5, 18.),
                  pretty_midi.Note(80, 55, 15.5, 19.)]
    midi.instruments.append(part)
    wave, reference, support = render(midi, 15, 283099)
    assert wave.shape == (DURATION*RATE,)
    assert np.isfinite(wave).all()
    assert reference.tolist() == [[3.5, 6., 52.], [3.5, 7., 55.]]
    assert support[0].tolist() == [2.95, 4.1, 48.]
    assert np.max(np.abs(wave[:3*RATE])) == 0
    np.testing.assert_array_equal(render(midi, 15, 283099)[0], wave)
