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
    sf.write(str(audio), np.zeros(65 * 22050, dtype='float32'), 22050)
    fake_inference.results.extend([[(60, 28, 31)], [(60, 0, 3)], [(67, 3, 4)]])
    progress = []
    transcribe(audio, output, ScoreOptions(tempo_bpm=90), progress_callback=progress.append)
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
    assert not list(tmp_path.glob('transcription-*'))


def test_silence_still_produces_valid_empty_midi(tmp_path, fake_inference):
    audio, output = tmp_path / 'input.wav', tmp_path / 'transcription.mid'
    sf.write(str(audio), [0.0] * 2205, 22050)
    fake_inference.results.append([])
    transcribe(audio, output, ScoreOptions())
    assert output.read_bytes().startswith(b'MThd')
    result = fake_inference.pretty_midi.PrettyMIDI(str(output))
    assert not any(instrument.notes for instrument in result.instruments)
