"""Runtime guards for the shipped checkpoint, separate from accuracy benchmarks."""

import numpy as np
import pytest
import soundfile as sf

from app.models import PipelineError
from app.services import vocal_melody


def test_missing_or_corrupt_checkpoint_has_a_clear_error(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    path = tmp_path / 'weights.pt'
    monkeypatch.setattr(vocal_melody, 'CHECKPOINT', path)
    for present in [False, True]:
        if present:
            path.write_bytes(b'not a model')
        with pytest.raises(PipelineError, match='missing or damaged'):
            vocal_melody.VocalMelody()


def test_shipped_model_preserves_silence_without_calling_feature_extraction(tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'silence.wav'
    sf.write(path, np.zeros(22050), 22050)
    model = vocal_melody.VocalMelody()
    result = model.predict(path, {}, 97)
    assert not result.instruments[0].notes


def test_runtime_rejects_unbounded_or_unnormalized_inputs(tmp_path):
    pytest.importorskip('torch')
    path = tmp_path / 'input.wav'
    model = vocal_melody.VocalMelody()
    for samples, rate in [(np.zeros(22050 * 34), 22050),
                          (np.zeros((22050, 2)), 22050), (np.zeros(44100), 44100)]:
        sf.write(path, samples, rate)
        with pytest.raises(PipelineError, match='normalized vocal window'):
            model.predict(path, {}, 97)
