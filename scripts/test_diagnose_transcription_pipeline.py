"""Check diagnostic accounting; these tests do not certify transcription quality."""

from diagnose_transcription_pipeline import compare, make_note


def test_deleted_notes_are_missing_not_perfect_timing():
    result = compare([make_note(48, 0, 1)], [])
    assert result['missing'] == 1
    assert result['matched'] == 0
    assert result['onset_mae_ms'] is None
    assert result['release_mae_ms'] is None


def test_wrong_pitch_counts_as_missing_and_extra():
    result = compare([make_note(48, 0, 1)], [make_note(49, 0, 1)])
    assert result['missing'] == result['extra'] == 1
    assert result['matched'] == 0


def test_duplicate_detection_cannot_match_reference_twice():
    result = compare([make_note(48, 0, 1)], [make_note(48, 0, 1), make_note(48, .01, 1)])
    assert result['matched'] == 1
    assert result['extra'] == 1
    assert result['onset_mae_ms'] == 0


def test_timing_and_distant_unmatched_notes_are_both_reported():
    result = compare([make_note(48, 0, 1), make_note(48, 2, 3)],
                     [make_note(48, .1, 1.2), make_note(48, 4, 5)])
    assert result['matched'] == result['missing'] == result['extra'] == 1
    assert abs(result['onset_mae_ms'] - 100) < 1e-8
    assert abs(result['release_mae_ms'] - 200) < 1e-8


def test_missing_middle_repeat_does_not_shift_later_pairing():
    result = compare([make_note(48, i, i+.5) for i in range(3)],
                     [make_note(48, 0, .5), make_note(48, 2, 2.5)])
    assert result['matched'] == 2
    assert result['missing'] == 1
    assert result['onset_mae_ms'] == 0
