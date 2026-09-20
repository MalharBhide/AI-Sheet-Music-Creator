"""Candidate decoding must match the original frequency-constrained detector."""

import numpy as np
import pytest


def test_original_candidate_bounds_and_untouched_evidence():
    pytest.importorskip('basic_pitch')
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_output_to_notes
    from pretty_midi import note_number_to_hz

    from app.services.accompaniment_candidates import decode_candidates

    acoustic = {'note': np.zeros((150, 88)), 'onset': np.zeros((150, 88)),
                'contour': np.zeros((150, 264))}
    # Boundary pitches and overlapping in-range notes exercise decoder bounds.
    for index, pitch in enumerate([35, 36, 60, 64, 95, 96]):
        start = 5 + index * 12
        acoustic['note'][start:start + 35, pitch - 21] = .8
        acoustic['onset'][start, pitch - 21] = .9
    before = {key: value.copy() for key, value in acoustic.items()}
    expected, _ = model_output_to_notes(
        {key: value.copy() for key, value in acoustic.items()},
        onset_thresh=.5, frame_thresh=.3,
        min_note_len=int(np.round(.09 * AUDIO_SAMPLE_RATE / FFT_HOP)),
        min_freq=note_number_to_hz(36), max_freq=note_number_to_hz(96),
        multiple_pitch_bends=False, melodia_trick=False, midi_tempo=93)
    actual = decode_candidates(acoustic, 93)

    def events(midi):
        return [(n.pitch, n.start, n.end, n.velocity)
                for part in midi.instruments for n in part.notes]

    assert events(actual) == events(expected)
    assert sorted(n[0] for n in events(actual)) == [36, 60, 64, 95]
    assert actual.get_tempo_changes()[1][0] == 93
    for key in acoustic:
        np.testing.assert_array_equal(acoustic[key], before[key])
