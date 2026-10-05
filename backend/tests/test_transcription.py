import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from app.models import PipelineError, ScoreOptions
from app.services.piano_transcription import _append_chunk_notes, _audio_chunks, transcribe


def note(pitch, start, end):
    return SimpleNamespace(pitch=pitch, start=start, end=end, velocity=90)


def midi(*notes):
    return SimpleNamespace(instruments=[SimpleNamespace(notes=list(notes))])


def test_chunks_bound_model_input_and_include_exact_final_samples(tmp_path):
    audio, chunk = tmp_path / 'input.wav', tmp_path / 'chunk.wav'
    frames = 65 * 22050 + 1
    sf.write(str(audio), np.zeros(frames, dtype='float32'), 22050, format='RF64')
    spans, lengths = [], []
    for span in _audio_chunks(audio, chunk):
        spans.append(span)
        lengths.append(sf.info(str(chunk)).frames)
    assert spans == [(0, 0, 30), (29, 30, 60), (59, 60, frames / 22050)]
    assert lengths == [31 * 22050, 32 * 22050, 6 * 22050 + 1]


def test_single_sample_recording_is_padded_for_model_only(tmp_path):
    audio, chunk = tmp_path / 'input.wav', tmp_path / 'chunk.wav'
    sf.write(str(audio), [0.1], 22050)
    assert list(_audio_chunks(audio, chunk)) == [(0, 0, 1 / 22050)]
    assert sf.info(str(chunk)).duration == 1


def test_stereo_chunking_retains_channels_and_handles_subsecond_padding(tmp_path):
    audio, chunk = tmp_path / 'stereo.wav', tmp_path / 'chunk.wav'
    sf.write(str(audio), np.column_stack((np.full(4410, .2), np.full(4410, -.1))), 44100)
    assert list(_audio_chunks(audio, chunk, allow_stereo=True)) == [(0, 0, .1)]
    samples, rate = sf.read(str(chunk))
    assert rate == 44100
    assert samples.shape == (44100, 2)
    assert samples[:4410, 0].mean() == pytest.approx(.2, abs=.0001)
    assert samples[:4410, 1].mean() == pytest.approx(-.1, abs=.0001)


def test_chunk_reader_rejects_missing_normalization(tmp_path):
    audio = tmp_path / 'input.wav'
    sf.write(str(audio), [0.0] * 441, 44100)
    with pytest.raises(PipelineError, match='normalized'):
        list(_audio_chunks(audio, tmp_path / 'chunk.wav'))


def test_sustained_notes_merge_across_multiple_seams_without_context_duplicates():
    piano = SimpleNamespace(notes=[])
    boundary = _append_chunk_notes(piano, midi(note(60, 28, 31), note(64, 30.2, 30.8)),
                                   0, 0, 30, {})
    boundary = _append_chunk_notes(piano, midi(note(60, 0, 32), note(64, 1.2, 1.8)),
                                   29, 30, 60, boundary)
    _append_chunk_notes(piano, midi(note(60, 0, 5)), 59, 60, 65, boundary)
    assert [(n.pitch, n.start, n.end) for n in piano.notes] == [(60, 28, 64), (64, 30.2, 30.8)]


def test_new_onset_at_boundary_remains_a_separate_note():
    piano = SimpleNamespace(notes=[])
    boundary = _append_chunk_notes(piano, midi(note(60, 29, 30)), 0, 0, 30, {})
    _append_chunk_notes(piano, midi(note(60, 1, 2)), 29, 30, 60, boundary)
    assert [(n.start, n.end) for n in piano.notes] == [(29, 30), (30, 31)]


def test_detected_context_tail_survives_a_missing_continuation():
    piano = SimpleNamespace(notes=[])
    boundary = _append_chunk_notes(piano, midi(note(60, 28, 31)), 0, 0, 30, {})
    _append_chunk_notes(piano, midi(), 29, 30, 60, boundary)
    assert [(n.start, n.end) for n in piano.notes] == [(28, 31)]


def test_context_tail_stops_at_a_new_attack_or_recording_end():
    for final, events, expected in [
        (60, (note(60, 1.25, 2),), [(28, 30.25), (30.25, 31)]),
        (30.125, (), [(28, 30.125)]),
    ]:
        piano = SimpleNamespace(notes=[])
        boundary = _append_chunk_notes(piano, midi(note(60, 28, 31)), 0, 0, 30, {})
        _append_chunk_notes(piano, midi(*events), 29, 30, final, boundary)
        assert [(n.start, n.end) for n in piano.notes] == expected


def test_padding_predictions_are_clipped_or_discarded():
    piano = SimpleNamespace(notes=[])
    _append_chunk_notes(piano, midi(note(60, -.01, .5), note(64, .2, .7)), 0, 0, .125, {})
    assert [(n.pitch, n.start, n.end) for n in piano.notes] == [(60, 0, .125)]


@pytest.fixture
def fake_inference(monkeypatch):
    # Keep the model stubbed, but exercise the real MIDI writer/reader when the
    # transcription extra is installed (including the native integration image).
    pretty_midi = pytest.importorskip('pretty_midi')
    calls = []
    models = []
    results = []
    basic_pitch = ModuleType('basic_pitch')
    basic_pitch.ICASSP_2022_MODEL_PATH = 'test-model'
    inference = ModuleType('basic_pitch.inference')

    def model(path):
        assert path == 'test-model'
        instance = object()
        models.append(instance)
        return instance

    def predict(path, **kwargs):
        calls.append((sf.info(path).duration, kwargs))
        output = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=7)
        for pitch, start, end in results.pop(0):
            piano.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=end))
        piano.pitch_bends.append(pretty_midi.PitchBend(pitch=100, time=0))
        output.instruments.append(piano)
        return {}, output, []

    inference.Model = model
    inference.predict = predict
    monkeypatch.setitem(sys.modules, 'basic_pitch', basic_pitch)
    monkeypatch.setitem(sys.modules, 'basic_pitch.inference', inference)
    return SimpleNamespace(calls=calls, models=models, results=results, pretty_midi=pretty_midi)


def test_transcription_writes_one_midi_with_global_times_and_reuses_model(tmp_path, fake_inference):
    audio, output = tmp_path / 'input.wav', tmp_path / 'output/transcription.mid'
    sf.write(str(audio), np.full(65 * 22050, .01, dtype='float32'), 22050)
    fake_inference.results.extend([[(60, 28, 31)], [(60, 0, 3)], [(67, 3, 4)]])
    progress = []
    report = transcribe(audio, output, ScoreOptions(tempo_bpm=90, transcription_mode='melody'),
                        progress_callback=progress.append)
    result = fake_inference.pretty_midi.PrettyMIDI(str(output))
    assert len(fake_inference.models) == 1
    assert [duration for duration, _ in fake_inference.calls] == [31, 32, 6]
    assert all(options['model_or_model_path'] is fake_inference.models[0]
               for _, options in fake_inference.calls)
    assert progress == [30 / 65, 60 / 65, 1]
    assert len(result.instruments) == 1
    assert result.instruments[0].program == 0
    assert result.instruments[0].pitch_bends == []
    assert [(n.pitch, round(n.start, 2), round(n.end, 2))
            for n in result.instruments[0].notes] == [(60, 28, 32), (67, 62, 63)]
    assert result.get_tempo_changes()[1][0] == pytest.approx(90, abs=.001)
    assert report['tempo_bpm'] == 90
    assert report['note_count'] == 2
    assert not list(tmp_path.glob('transcription-*'))


def test_silence_still_produces_valid_empty_midi(tmp_path, fake_inference):
    audio, output = tmp_path / 'input.wav', tmp_path / 'transcription.mid'
    sf.write(str(audio), [0.0] * 2205, 22050)
    report = transcribe(audio, output, ScoreOptions(tempo_bpm=120, transcription_mode='melody'))
    assert output.read_bytes().startswith(b'MThd')
    result = fake_inference.pretty_midi.PrettyMIDI(str(output))
    assert not any(instrument.notes for instrument in result.instruments)
    assert fake_inference.calls == []
    assert report['note_count'] == 0
    assert 'No pitched notes' in report['warnings'][0]


def test_missing_piano_model_does_not_silently_use_basic_pitch(tmp_path, fake_inference):
    audio = tmp_path / 'input.wav'
    sf.write(str(audio), np.full(22050, .01), 22050)
    with pytest.raises(PipelineError, match='high-resolution piano model is not installed'):
        transcribe(audio, tmp_path / 'out.mid', ScoreOptions(tempo_bpm=120),
                   piano_model_path=tmp_path / 'missing.pth')
    assert fake_inference.models == []


def test_full_mix_uses_separate_sources_discards_drums_and_keeps_roles(
        tmp_path, fake_inference, monkeypatch):
    from app.services import (
        accompaniment_verifier,
        bass_articulation,
        bass_residual,
        bass_verifier,
        source_separation,
        vocal_melody,
    )

    audio, output = tmp_path / 'input.wav', tmp_path / 'out.mid'
    sf.write(str(audio), np.full(22050 * 2, .01), 22050)
    separated = []

    class Separator:
        def separate(self, path, directory):
            separated.append(path)
            return {role: path for role in ['vocals', 'bass', 'other', 'drums']}

    monkeypatch.setattr(source_separation, 'StemSeparator', Separator)
    # This test isolates routing and cleanup; the trained decoder is exercised
    # separately with real acoustic arrays and in native integration tests.
    class VocalModel:
        def predict(self, path, acoustic, bpm):
            return midi(note(72, 0, .4), note(74, .5, .9))

    monkeypatch.setattr(vocal_melody, 'VocalMelody', VocalModel)
    class NoteVerifier:
        name = 'Test verifier'

        def filter(self, path, acoustic, output):
            assert [n.pitch for n in output.instruments[0].notes] == [36, 60, 64, 67, 71]
            output.instruments[0].notes.pop()
            return 1

    monkeypatch.setattr(accompaniment_verifier, 'AccompanimentVerifier', NoteVerifier)
    class BassVerifier:
        name = 'Test bass verifier'

        def filter(self, path, acoustic, output):
            assert [n.pitch for n in output.instruments[0].notes] == [36, 48]
            return 0

    monkeypatch.setattr(bass_verifier, 'BassVerifier', BassVerifier)
    class BassArticulation:
        name = 'Test bass articulation'

        def filter(self, path, acoustic, output, baseline):
            assert isinstance(baseline, BassVerifier)
            assert [n.pitch for n in output.instruments[0].notes] == [36, 48]
            return 0

    monkeypatch.setattr(bass_articulation, 'BassArticulation', BassArticulation)

    class BassResidual:
        name = 'Test bass residual'

        def filter(self, path, acoustic, output, baseline):
            assert isinstance(baseline, BassVerifier)
            assert [n.pitch for n in output.instruments[0].notes] == [36, 48]
            return 0

    monkeypatch.setattr(bass_residual, 'BassResidual', BassResidual)

    fake_inference.results.extend([
        [],  # Vocal decoding must still run when Basic Pitch emits no events.
        [(36, 0, 1), (48, 0, 1)],
        [(35, 0, 1), (36, .25, .75), (60, 0, 1), (64, 0, 1), (67, 0, 1), (71, 0, 1), (96, 0, 1)],
    ])
    report = transcribe(audio, output,
                        ScoreOptions(tempo_bpm=120, transcription_mode='full_mix'))
    assert len(separated) == 1
    assert len(fake_inference.calls) == 3
    result = fake_inference.pretty_midi.PrettyMIDI(str(output))
    assert [part.name for part in result.instruments] == ['Vocals', 'Bass', 'Other']
    assert [len(part.notes) for part in result.instruments] == [2, 1, 2]
    assert report['sources'] == ['vocals', 'bass', 'other']
    assert report['engine'].startswith('Demucs htdemucs')
    assert report['accompaniment_verification'] == {
        'model': 'Test verifier', 'window_candidates': 5, 'window_rejections': 1}
    assert report['bass_verification'] == {
        'model': 'Test bass verifier', 'articulation_model': 'Test bass articulation',
        'residual_model': 'Test bass residual', 'window_candidates': 2,
        'window_rejections': 0, 'window_merged_boundaries': 0, 'window_residual_rejections': 0}
    assert report['support_notes_changed_for_bass'] == 1
    assert [(n.pitch, n.start, n.end) for n in result.instruments[1].notes] == [(36, 0, 1)]
    assert fake_inference.calls[2][1]['minimum_frequency'] is None
    assert fake_inference.calls[2][1]['maximum_frequency'] is None
    assert 'arrangement' in report['warnings'][0]


def test_piano_adapter_resamples_and_uses_model_note_events(tmp_path, monkeypatch):
    pytest.importorskip('scipy')
    from app.services.piano_transcription import _PianoEngine

    audio = tmp_path / 'chunk.wav'
    sf.write(str(audio), np.full(22050, .01), 22050)
    engine = _PianoEngine.__new__(_PianoEngine)
    lengths = []

    def predict(samples, path):
        lengths.append(len(samples))
        assert path is None
        assert np.all(samples[:4000] == 0)
        return {'est_note_events': [
            {'midi_note': 60, 'onset_time': .375, 'offset_time': 1.125, 'velocity': 82},
            {'midi_note': 64, 'onset_time': .375, 'offset_time': .75, 'velocity': 69},
            {'midi_note': 80, 'onset_time': .01, 'offset_time': .2, 'velocity': 70},
        ]}

    engine.model = SimpleNamespace(transcribe=predict)
    result = engine.predict(audio, 'piano', 96)
    assert lengths == [20000]
    assert [(n.pitch, n.start, n.end, n.velocity) for n in result.instruments[0].notes] == [
        (60, .125, .875, 82), (64, .125, .5, 69)]


def test_full_mix_reports_coarse_rhythm_loss_and_whole_melody_transposition(
        tmp_path, fake_inference, monkeypatch):
    from app.services import source_separation, vocal_melody

    class Separator:
        def separate(self, path, directory):
            return {'vocals': path}

    class VocalModel:
        def predict(self, path, acoustic, bpm):
            return midi(*(note(41 + i, i * .125, i * .125 + .08) for i in range(12)))

    monkeypatch.setattr(source_separation, 'StemSeparator', Separator)
    monkeypatch.setattr(vocal_melody, 'VocalMelody', VocalModel)
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.full(22050 * 2, .01), 22050)
    reports = []
    for grid in ('eighth', 'sixteenth'):
        fake_inference.results.append([])
        output = tmp_path / f'{grid}.mid'
        reports.append(transcribe(audio, output, ScoreOptions(
            tempo_bpm=120, transcription_mode='full_mix', grid=grid)))
    assert reports[0]['notes_by_source']['vocals'] == {'detected': 12, 'score': 7}
    assert any('Precise rhythm' in warning for warning in reports[0]['warnings'])
    assert reports[1]['notes_by_source']['vocals'] == {'detected': 12, 'score': 12}
    assert not any('Precise rhythm' in warning for warning in reports[1]['warnings'])
    assert reports[1]['melody_octave_shift'] == 2
    output = fake_inference.pretty_midi.PrettyMIDI(str(tmp_path / 'sixteenth.mid'))
    assert [item.pitch for item in output.instruments[0].notes] == list(range(65, 77))


def test_detailed_accompaniment_keeps_its_validated_detector_path(tmp_path, fake_inference):
    from app.services.piano_transcription import _GeneralEngine

    path = tmp_path / 'audio.wav'
    sf.write(path, np.full(22050, .01), 22050)
    fake_inference.results.append([(60, .1, .5)])
    engine = _GeneralEngine('detailed')
    output = engine.predict(path, 'other', 120)
    assert output.instruments[0].notes[0].pitch == 60
    assert engine.note_verifier is None
    assert engine.verification is None
    parameters = fake_inference.calls[0][1]
    assert parameters['minimum_frequency'] is not None
    assert parameters['melodia_trick'] is True


def test_balanced_bass_uses_own_consensus_once_and_preserves_bounded_decoder(tmp_path, fake_inference, monkeypatch):
    from app.services import bass_articulation, bass_residual, bass_verifier
    from app.services.piano_transcription import _GeneralEngine

    path = tmp_path / 'bass.wav'
    sf.write(path, np.full(30 * 22050, .01, dtype='float32'), 22050)
    constructions = []

    class BassVerifier:
        name = 'Test bass consensus'

        def __init__(self):
            constructions.append(self)

        def filter(self, path, acoustic, output):
            output.instruments[0].notes.pop()
            return 1

    monkeypatch.setattr(bass_verifier, 'BassVerifier', BassVerifier)
    articulations = []

    class BassArticulation:
        name = 'Test bass articulation'

        def __init__(self):
            articulations.append(self)

        def filter(self, path, acoustic, output, baseline):
            assert baseline is constructions[0]
            assert [n.pitch for n in output.instruments[0].notes] == [40]
            return 0

    monkeypatch.setattr(bass_articulation, 'BassArticulation', BassArticulation)

    residuals = []

    class BassResidual:
        name = 'Test bass residual'

        def __init__(self):
            residuals.append(self)

        def filter(self, path, acoustic, output, baseline):
            assert baseline is constructions[0]
            assert [n.pitch for n in output.instruments[0].notes] == [40]
            assert len(articulations) == 1
            return 0

    monkeypatch.setattr(bass_residual, 'BassResidual', BassResidual)

    fake_inference.results.extend([[(40, 3., 4.), (52, 3., 4.)]] * 2)
    engine = _GeneralEngine('balanced')
    for _ in range(2):
        output = engine.predict(path, 'bass', 90)
        assert [(n.pitch, n.start, n.end) for n in output.instruments[0].notes] == [(40, 3., 4.)]
    assert len(constructions) == 1 and engine.note_verifier is None and engine.verification is None
    assert len(articulations) == 1 and engine.bass_articulation is articulations[0]
    assert len(residuals) == 1 and engine.bass_residual is residuals[0]
    assert engine.bass_verification['window_candidates'] == 4
    assert engine.bass_verification['window_rejections'] == 2
    assert engine.bass_verification['window_merged_boundaries'] == 0
    assert engine.bass_verification['window_residual_rejections'] == 0
    parameters = fake_inference.calls[0][1]
    assert parameters['minimum_frequency'] == pytest.approx(27.5)
    assert parameters['maximum_frequency'] == pytest.approx(261.6255653)
    assert parameters['onset_threshold'] == .5 and parameters['frame_threshold'] == .3
    assert parameters['minimum_note_length'] == 90 and not parameters['melodia_trick']


def test_detailed_bass_keeps_original_decoder_without_consensus(tmp_path, fake_inference):
    from app.services.piano_transcription import _GeneralEngine

    path = tmp_path / 'bass.wav'
    sf.write(path, np.full(22050, .01), 22050)
    fake_inference.results.append([(40, .1, .5)])
    engine = _GeneralEngine('detailed')
    assert engine.predict(path, 'bass', 120).instruments[0].notes[0].pitch == 40
    assert engine.bass_verifier is None and engine.bass_verification is None
    assert engine.bass_articulation is None and engine.bass_residual is None
    assert fake_inference.calls[0][1]['melodia_trick'] is True


@pytest.mark.parametrize('manual', [False, True])
def test_repeated_note_tempo_reaches_written_midi_and_honors_manual_override(
        tmp_path, fake_inference, monkeypatch, manual):
    from app.services import piano_transcription

    audio, output = tmp_path / 'fixture.wav', tmp_path / 'out.mid'
    sf.write(audio, np.full(25 * 22050, .01, dtype='float32'), 22050)
    fake_inference.results.append([(64, .06 + .5 * i, .4 + .5 * i) for i in range(48)])
    monkeypatch.setattr(piano_transcription, 'estimate_tempo', lambda *_: (121.2, []))
    options = ScoreOptions(transcription_mode='melody', tempo_bpm=121.2 if manual else None)
    report = transcribe(audio, output, options)
    result = fake_inference.pretty_midi.PrettyMIDI(str(output))
    assert len(result.instruments[0].notes) == 48
    assert report['tempo_bpm'] == pytest.approx(121.2 if manual else 120)
    assert result.get_tempo_changes()[1][0] == pytest.approx(report['tempo_bpm'], abs=.001)
    if not manual:
        assert np.diff([n.start for n in result.instruments[0].notes]) == pytest.approx(np.full(47, .5), abs=.001)
    else:
        assert report['tempo_refinement_bpm'] == 0


def test_full_mix_rhythm_uses_lead_phase_instead_of_denser_backing(tmp_path, monkeypatch):
    import pretty_midi
    from app.services import piano_transcription, source_separation

    audio, output = tmp_path / 'fixture.wav', tmp_path / 'out.mid'
    sf.write(audio, np.full(8 * 22050, .01, dtype='float32'), 22050)
    class Separator:
        def separate(self, path, directory):
            return {role: Path(str(path) + '-' + role) for role in ('vocals', 'other')}
    class Engine:
        name = 'Fixture'
        def __init__(self, detail):
            pass
        def predict(self, path, role, bpm):
            part = pretty_midi.Instrument(0)
            if role == 'vocals':
                part.notes = [pretty_midi.Note(90, 72 + i % 3, .06 + .5 * i, .4 + .5 * i) for i in range(12)]
            else:
                part.notes = [pretty_midi.Note(85, p, .5 * i, .35 + .5 * i)
                              for i in range(12) for p in (48, 52, 55)]
            result = pretty_midi.PrettyMIDI()
            result.instruments.append(part)
            return result
    monkeypatch.setattr(source_separation, 'StemSeparator', Separator)
    monkeypatch.setattr(piano_transcription, '_GeneralEngine', Engine)
    report = transcribe(audio, output, ScoreOptions(transcription_mode='full_mix', tempo_bpm=120))
    result = pretty_midi.PrettyMIDI(str(output))
    assert report['timing_offset_seconds'] == pytest.approx(.06)
    assert [n.start for n in result.instruments[0].notes] == pytest.approx([.5 * i for i in range(12)], abs=.001)
