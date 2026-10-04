"""Source MIDI clocks, pedal support and source-group partitioning stay explicit."""

from types import SimpleNamespace

import numpy as np
import pytest
from prepare_slakh_training import verify_codec
from slakh_references import crop, intervals, partition


def midi(notes, pedal):
    return SimpleNamespace(instruments=[SimpleNamespace(is_drum=False,
        notes=[SimpleNamespace(start=s, end=e, pitch=p) for s, e, p in notes],
        control_changes=[SimpleNamespace(number=64, time=t, value=v) for t, v in pedal])])


def test_pedal_support_does_not_change_key_release_references_or_merge_real_repeats():
    keys, pitch = intervals(midi([(0., 1., 60), (2.5, 2.9, 60)], [(.5, 127), (3., 0)]), 4.)
    np.testing.assert_array_equal(keys, [[0., 1., 60.], [2.5, 2.9, 60.]])
    np.testing.assert_array_equal(pitch, [[0., 2.5, 60.], [2.5, 3., 60.]])


def test_pedal_up_at_note_release_and_unpedaled_notes_keep_original_ends():
    keys, support = intervals(midi([(0., 1., 60), (2., 3., 62)], [(.5, 127), (1., 0)]), 4.)
    np.testing.assert_array_equal(keys, support)


def test_crop_keeps_early_hold_pitch_support_without_inventing_a_new_attack():
    keys = np.array([[0., 4., 60.], [3., 5., 62.]])
    reference, pitch = crop(keys, keys, 2., 3.)
    np.testing.assert_array_equal(reference, [[1., 3., 62.]])
    np.testing.assert_array_equal(pitch, [[0., 2., 60.], [1., 3., 62.]])


def test_source_midi_cannot_silently_exceed_the_audio_clock():
    with pytest.raises(ValueError, match='audio clock'):
        intervals(midi([(0., 10., 60)], []), 4.)


def test_fixed_source_groups_remain_disjoint_and_duplicates_stop_preparation():
    identities = [f'Track{i:05}' for i in range(1, 21)]
    duplicate_map = {key: {'midi_duplicates': []} for key in identities}
    result = partition(identities, duplicate_map)
    assert result == partition(list(reversed(identities)), duplicate_map)
    assert {group: list(result.values()).count(group) for group in ('train', 'validation', 'test')} == {
        'train': 12, 'validation': 4, 'test': 4}
    duplicate_map[identities[1]]['midi_duplicates'] = [identities[0]]
    with pytest.raises(ValueError, match='group-level split'):
        partition(identities, duplicate_map)


def test_codec_checks_silent_intro_without_shifting_labels_from_predictions():
    rate = 2000
    original = np.zeros(rate * 8, dtype=np.float32)
    original[rate * 6:] = np.random.default_rng(182).normal(size=rate * 2)
    clock = verify_codec(original, original * .9, rate)
    assert clock['codec_lag_samples'] == 0 and clock['anchor_sample'] >= rate * 6
    with pytest.raises(ValueError, match='clock drift'):
        verify_codec(original, np.roll(original, 50), rate)
