import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.services.audio_analysis import (
    _tempo_from_onsets,
    arrange_melody_register,
    balance_piano_arrangement,
    clean_notes,
    estimate_grid_phase,
    estimate_key,
    estimate_tempo,
    preserve_melody_releases,
    reduce_accompaniment,
)


def n(pitch, start=0, end=1, velocity=90):
    return SimpleNamespace(pitch=pitch, start=start, end=end, velocity=velocity)


def cleaned(notes, role='piano', detail='balanced'):
    return clean_notes(notes, role=role, detail=detail, tempo_bpm=120, grid='sixteenth')


def test_cleaner_keeps_piano_chords_but_removes_glitches_and_duplicate_attacks():
    notes = [n(60), n(64), n(67), n(60, .015, .9), n(72, .1, .13), n(80, velocity=3),
             n(20), n(70, float('nan'), 1)]
    assert [(v.pitch, v.start, v.end) for v in cleaned(notes)] == [
        (60, 0, 1), (64, 0, 1), (67, 0, 1)]


def test_repeated_piano_key_trims_old_note_without_deleting_new_attack():
    notes = cleaned([n(60, 0, 2), n(60, 1, 3)])
    assert [(v.start, v.end) for v in notes] == [(0, 1), (1, 3)]


def test_common_grid_phase_prevents_jitter_from_turning_quarters_into_uneven_rhythm():
    notes = [n(60 + i % 5, .0625 + .5 * i + (-.005 if i % 2 else .005),
               .4 + .5 * i) for i in range(20)]
    offset = estimate_grid_phase(notes, tempo_bpm=120, grid='sixteenth')
    assert abs(offset) == pytest.approx(.0625)
    for item in notes:
        item.start = max(0, item.start - offset)
        item.end = max(item.start, item.end - offset)
    result = cleaned(notes)
    assert np.diff([item.start for item in result]) == pytest.approx(np.full(19, .5))


def test_phase_correction_is_zero_for_aligned_or_irregular_onsets():
    aligned = [n(60, .5 * i, .5 * i + .4) for i in range(20)]
    assert estimate_grid_phase(aligned, tempo_bpm=120, grid='sixteenth') == pytest.approx(0)
    irregular = [n(60, i * .5 + (i % 16) * .125 / 16, i * .5 + .4) for i in range(32)]
    assert estimate_grid_phase(irregular, tempo_bpm=120, grid='sixteenth') == 0


def test_melody_selects_salient_line_and_releases_old_pitch():
    notes = cleaned([n(60, 0, 2), n(72, 0, 1, 50), n(62, 1, 3), n(74, 1, 3, 85)],
                    role='vocals')
    assert [(v.pitch, v.start, v.end) for v in notes] == [(60, 0, 1), (62, 1, 3)]


def test_bass_does_not_keep_vocal_or_upper_harmonics():
    notes = cleaned([n(36), n(72, velocity=110)], role='bass')
    assert [v.pitch for v in notes] == [36]


def test_trained_vocal_cleanup_preserves_fast_notes_and_low_singers():
    result = cleaned([n(41, 0, .08), n(43, .13, .21), n(45, .26, .34)], role='vocals')
    assert [item.pitch for item in result] == [41, 43, 45]
    assert [item.start for item in result] == [0, .125, .25]


def test_low_sung_melody_moves_as_a_whole_line_without_filling_rests():
    lead = SimpleNamespace(notes=[n(48, 0, .2), n(52, 1, 1.3), n(55, 2, 2.2)])
    bass = SimpleNamespace(notes=[n(36)])
    assert arrange_melody_register({'vocals': lead, 'bass': bass}) == 12
    assert [(item.pitch, item.start, item.end) for item in lead.notes] == [
        (60, 0, .2), (64, 1, 1.3), (67, 2, 2.2)]
    assert bass.notes[0].pitch == 36
    assert arrange_melody_register({'vocals': lead}) == 0
    assert arrange_melody_register({'piano': bass}) == 0
    assert arrange_melody_register({'vocals': SimpleNamespace(notes=[])}) == 0


def test_melody_register_preserves_wide_high_excursions():
    lead = SimpleNamespace(notes=[n(48, 0, 10), n(90, 11, 12)])
    assert arrange_melody_register({'vocals': lead}) == 0
    assert [item.pitch for item in lead.notes] == [48, 90]


def test_support_cannot_hold_or_reattack_a_melody_key_through_its_release():
    lead = SimpleNamespace(notes=[n(64, 1, 2), n(64, 3, 4)])
    support = SimpleNamespace(notes=[n(64, 0, 8), n(64, 1, 1.5), n(64, 1.5, 6),
                                      n(64, 2, 2.5), n(67, 0, 8)])
    assert preserve_melody_releases({'vocals': lead, 'other': support}) == 3
    assert [(item.pitch, item.start, item.end) for item in support.notes] == [
        (64, 0, 1), (64, 2, 2.5), (67, 0, 8)]
    assert [(item.start, item.end) for item in lead.notes] == [(1, 2), (3, 4)]
    assert preserve_melody_releases({'piano': support}) == 0


def test_accompaniment_reduction_limits_actual_sounding_polyphony():
    notes = cleaned([n(60 + index, index * .25, 4, 60 + index) for index in range(8)],
                    role='other')
    result = reduce_accompaniment(notes, detail='balanced')
    for at in np.arange(0, 4, .05):
        assert sum(item.start <= at < item.end for item in result) <= 3
    assert all(item.end == 4 for item in result)


def test_accompaniment_keeps_complete_holds_instead_of_stealing_for_loud_fragments():
    notes = [n(60, 0, 4, 60), n(64, 0, 4, 70),
             n(67, 1, 1.125, 110), n(69, 2, 2.125, 110), n(71, 3, 3.125, 110)]
    before = [vars(item).copy() for item in notes]
    result = reduce_accompaniment(notes, detail='balanced', melody=[n(80, 0, 4)])
    assert [(item.pitch, item.start, item.end) for item in result] == [(60, 0, 4), (64, 0, 4)]
    assert [vars(item) for item in notes] == before


def test_arrangement_does_not_thin_a_passage_that_already_fits():
    notes = [n(60, 0, 2, 100), n(62, 1, 3, 40), n(64, 2, 4, 40), n(65, 3, 5, 100)]
    result = reduce_accompaniment(notes, detail='balanced', melody=[n(80, 0, 5)])
    assert [(item.pitch, item.start, item.end) for item in result] == [
        (item.pitch, item.start, item.end) for item in notes]


def test_support_stays_below_melody_for_the_entire_overlap_and_keeps_rest_fills():
    melody = [n(72, 0, 2), n(67, 2, 3)]
    notes = [n(64, 0, 3), n(69, 0, 3), n(72, 0, 2), n(70, 3, 4)]
    result = reduce_accompaniment(notes, detail='balanced', melody=melody)
    assert [(item.pitch, item.start, item.end) for item in result] == [(64, 0, 3), (70, 3, 4)]
    assert [(item.pitch, item.start, item.end) for item in melody] == [(72, 0, 2), (67, 2, 3)]


def test_detailed_accompaniment_can_retain_five_notes():
    notes = cleaned([n(60 + i) for i in range(8)], role='other', detail='detailed')
    assert len(notes) == 5


def test_arrangement_places_melody_above_accompaniment_without_changing_notes():
    parts = {role: SimpleNamespace(notes=[n(60, 0, 8, 40), n(64, 8, 9, 100)])
             for role in ('vocals', 'bass', 'other')}
    balance_piano_arrangement(parts)
    assert min(item.velocity for item in parts['vocals'].notes) > max(
        item.velocity for item in parts['other'].notes)
    assert [item.velocity for item in parts['bass'].notes] == [54, 82]
    for part in parts.values():
        assert [(item.pitch, item.start, item.end) for item in part.notes] == [(60, 0, 8), (64, 8, 9)]
        assert part.notes[0].velocity < part.notes[1].velocity
    # Missing stems and silent recordings are valid.
    balance_piano_arrangement({'vocals': SimpleNamespace(notes=[])})


def test_tempo_override_does_not_import_or_read_audio(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'librosa', None)
    assert estimate_tempo(tmp_path / 'does-not-exist.wav', 87) == (87.0, [])


def test_tempo_analysis_samples_bounded_beginning_middle_and_end(tmp_path, monkeypatch):
    audio = tmp_path / 'long.wav'
    sf.write(str(audio), np.full(22050 * 240, .01, dtype='float32'), 22050)
    lengths = []

    def envelope(*, y, **kwargs):
        lengths.append(len(y))
        return np.ones(100)

    librosa = SimpleNamespace(onset=SimpleNamespace(onset_strength=envelope),
                              beat=SimpleNamespace(beat_track=lambda **kwargs: (np.array([96.2]), np.arange(20))))
    monkeypatch.setitem(sys.modules, 'librosa', librosa)
    bpm, warnings = estimate_tempo(audio, None)
    assert bpm == 96.2
    assert lengths == [30 * 22050] * 3
    assert warnings


def test_tempo_does_not_average_half_and_double_time_into_unobserved_bpm(tmp_path, monkeypatch):
    audio = tmp_path / 'ambiguous.wav'
    samples = np.full(22050 * 100, .01, dtype='float32')
    samples[:22050 * 30] = 0
    sf.write(str(audio), samples, 22050)
    candidates = iter([90.0] * 4 + [180.0] * 4)
    librosa = SimpleNamespace(onset=SimpleNamespace(onset_strength=lambda **kwargs: np.ones(100)),
                              beat=SimpleNamespace(beat_track=lambda **kwargs: (next(candidates), np.arange(20))))
    monkeypatch.setitem(sys.modules, 'librosa', librosa)
    bpm, warnings = estimate_tempo(audio, None)
    assert bpm in (90, 180)
    assert any('Different sections' in message for message in warnings)


def test_silence_reports_tempo_fallback(tmp_path):
    audio = tmp_path / 'silent.wav'
    sf.write(str(audio), np.zeros(22050), 22050)
    bpm, warnings = estimate_tempo(audio, None)
    assert bpm == 120
    assert 'could not be detected' in warnings[0]


def test_c_major_profile_estimates_key_without_changing_pitches():
    notes = [n(pitch, end=weight) for pitch, weight in
             [(60, 4), (62, .8), (64, 2), (65, 1), (67, 3), (69, .8), (71, .8)]]
    assert estimate_key(notes) == 'C major'
    assert [item.pitch for item in notes] == [60, 62, 64, 65, 67, 69, 71]
    assert estimate_key([n(60)]) is None


@pytest.mark.parametrize('bpm', [80, 100, 120, 150])
def test_real_beat_tracker_finds_regular_click_tempo(tmp_path, bpm):
    pytest.importorskip('librosa')
    rate = 22050
    samples = np.zeros(20 * rate, dtype='float32')
    rng = np.random.default_rng(22)
    for at in np.arange(.5, 19.5, 60 / bpm):
        onset = int(at * rate)
        click = rng.normal(0, .2, 600) * np.exp(-np.arange(600) / 80)
        samples[onset:onset + 600] += click
    audio = tmp_path / 'clicks.wav'
    sf.write(str(audio), samples, rate)
    detected, _ = estimate_tempo(audio, None)
    assert detected == pytest.approx(bpm, abs=3)


def test_competing_syncopated_pulse_does_not_force_the_120_bpm_prior():
    librosa = pytest.importorskip('librosa')
    rate = 22050
    frames = np.arange(30 * rate // 256)
    envelope = np.zeros(len(frames))
    # Main beat at 96 BPM, with a competing 128 BPM rhythmic layer. The former
    # single-prior tracker selects ~129; compare acoustic support instead.
    for spacing, strength in [(.625, 2.25), (.46875, 2)]:
        for at in np.arange(.3, 29.5, spacing):
            envelope += strength * np.exp(-.5 * ((frames - at * rate / 256) / 1.2) ** 2)
    old, _ = librosa.beat.beat_track(onset_envelope=envelope, sr=rate,
                                   hop_length=256, trim=False)
    assert float(np.asarray(old).reshape(-1)[0]) == pytest.approx(129, abs=2)
    assert _tempo_from_onsets(envelope, rate) == pytest.approx(96, abs=.2)
