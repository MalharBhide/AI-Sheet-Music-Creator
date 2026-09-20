"""Decode original accompaniment candidates without altering acoustic evidence."""

import numpy as np


def decode_candidates(acoustic, bpm=120):
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_output_to_notes
    from pretty_midi import note_number_to_hz

    # The upstream decoder mutates note/onset arrays for pitch constraints.
    # Its original bounds also affect onset inference; filtering MIDI afterwards
    # is not equivalent. Preserve untouched full-range evidence for verification.
    bounded = {key: value.copy() for key, value in acoustic.items()}
    midi, _ = model_output_to_notes(bounded, onset_thresh=.5, frame_thresh=.3,
        min_note_len=int(np.round(.09 * AUDIO_SAMPLE_RATE / FFT_HOP)),
        min_freq=note_number_to_hz(36), max_freq=note_number_to_hz(96),
        multiple_pitch_bends=False, melodia_trick=False, midi_tempo=bpm)
    return midi
