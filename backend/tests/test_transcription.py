import sys
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
    from app.services import accompaniment_verifier, source_separation, vocal_melody

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
            assert [n.pitch for n in output.instruments[0].notes] == [60, 64, 67, 71]
            output.instruments[0].notes.pop()
            return 1

    monkeypatch.setattr(accompaniment_verifier, 'AccompanimentVerifier', NoteVerifier)
    fake_inference.results.extend([
        [],  # Vocal decoding must still run when Basic Pitch emits no events.
        [(36, 0, 1), (48, 0, 1)],
        [(35, 0, 1), (60, 0, 1), (64, 0, 1), (67, 0, 1), (71, 0, 1), (96, 0, 1)],
    ])
    report = transcribe(audio, output,
                        ScoreOptions(tempo_bpm=120, transcription_mode='full_mix'))
    assert len(separated) == 1
    assert len(fake_inference.calls) == 3
    result = fake_inference.pretty_midi.PrettyMIDI(str(output))
    assert [part.name for part in result.instruments] == ['Vocals', 'Bass', 'Other']
    assert [len(part.notes) for part in result.instruments] == [2, 1, 3]
    assert report['sources'] == ['vocals', 'bass', 'other']
    assert report['engine'].startswith('Demucs htdemucs')
    assert report['accompaniment_verification'] == {
        'model': 'Test verifier', 'window_candidates': 4, 'window_rejections': 1}
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
