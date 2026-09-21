"""Regression checks for supervision and pitch-relative acoustic features."""

from types import SimpleNamespace

import numpy as np
import pytest
from cache_note_verifier import targets

from app.services.note_evidence import CONTEXT_NAMES, FEATURE_NAMES, note_features


def test_supervision_does_not_train_late_correct_pitch_as_a_false_note():
    reference = np.array([[.5, 1., 60], [1.2, 1.8, 64]])
    events = np.array([[.51, .9, 60, 80], [1.29, 1.7, 64, 70], [.6, .9, 72, 50]])
    labels, mask = targets(reference, events)
    np.testing.assert_array_equal(labels, [1, 0, 0])
    np.testing.assert_array_equal(mask, [True, False, True])


def test_duplicate_detections_cannot_both_be_training_positives():
    reference = np.array([[.5, 1., 60]])
    events = np.array([[.5, .9, 60, 80], [.51, .9, 60, 80]])
    labels, _ = targets(reference, events)
    assert labels.sum() == 1


def test_empty_notes_need_no_signal_processing():
    assert note_features(np.array([]), 22050, {}, []).shape == (0, len(FEATURE_NAMES))


def test_features_validate_audio_and_preserve_real_note_attributes():
    notes = [SimpleNamespace(start=.1, end=.3, pitch=21, velocity=50),
             SimpleNamespace(start=.4, end=.8, pitch=108, velocity=100)]
    samples = np.zeros(22050, dtype=np.float32)
    arrays = {'note': np.full((87, 88), .5), 'onset': np.full((87, 88), .6)}
    result = note_features(samples, 22050, arrays, notes)
    assert result.shape == (2, 26)
    assert np.isfinite(result).all()
    assert result[0, 1] == pytest.approx(50 / 127)
    assert result[1, 1] == pytest.approx(100 / 127)
    assert result[0, 10] == 0  # No phantom evidence below the piano range.
    assert result[1, 11] == 0
    assert notes[0].pitch == 21 and notes[1].end == .8
    with pytest.raises(ValueError, match='22050'):
        note_features(samples, 16000, arrays, notes)
    samples[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        note_features(samples, 22050, arrays, notes)


def test_context_keeps_legacy_features_exact_and_handles_pitch_boundaries():
    notes = [SimpleNamespace(start=0., end=.3, pitch=21, velocity=50),
             SimpleNamespace(start=.4, end=1., pitch=108, velocity=100)]
    time = np.arange(22050) / 22050
    samples = (.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    rng = np.random.default_rng(17)
    arrays = {'note': rng.random((87, 88)), 'onset': rng.random((87, 88))}
    original = note_features(samples, 22050, arrays, notes)
    context = note_features(samples, 22050, arrays, notes, include_context=True)
    assert context.shape == (2, len(FEATURE_NAMES) + len(CONTEXT_NAMES))
    np.testing.assert_array_equal(original, context[:, :len(FEATURE_NAMES)])
    assert np.isfinite(context).all()
    assert note_features(np.array([]), 22050, {}, [], include_context=True).shape == (0, 52)
