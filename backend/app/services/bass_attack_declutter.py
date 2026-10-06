"""Acoustic rejection of unsupported post-V16 bass notes; retained events are unchanged."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.attack_local_evidence import observe
from app.services.attack_local_evidence import sequences as attack_sequences
from app.services.attack_local_network import model, probabilities
from app.services.bass_harmonic import BassHarmonic, features
from app.services.note_evidence import note_features
from app.services.release_local_evidence import release_sequences
from app.services.temporal_note_model import sequences as coarse_sequences

VERSION = 'attack-release-bass-support-v25-v1'
ASSET = Path(__file__).resolve().parents[1] / 'assets/bass-attack-declutter-v1.npz'
EXPECTED_SHA = 'f4c5d1fe9c411eca32b28b3eb2c6c7e38e292e1a0d4191d2d6fa81542b6e424c'


class AttackSupportModel:
    def __init__(self, saved):
        import torch

        metadata = {'version', 'format_version', 'width', 'remove_threshold', 'guardian_threshold', 'mean', 'scale'}
        if (not metadata.issubset(saved.files)
                or any(saved[k].shape != () for k in metadata - {'mean', 'scale'})
                or str(saved['version']) != VERSION or float(saved['format_version']) != 2.
                or float(saved['width']) not in (24., 32.)):
            raise ValueError('Incompatible bass acoustic support checkpoint contract')
        self.remove_threshold = float(saved['remove_threshold'])
        self.guardian_threshold = float(saved['guardian_threshold'])
        if (not np.isfinite([self.remove_threshold, self.guardian_threshold]).all()
                or self.remove_threshold not in (.6, .7, .8, .9, .95, .99)
                or self.guardian_threshold not in (.15, .3, .5)):
            raise ValueError('Invalid bass acoustic support confidence contract')
        self.network = model(int(saved['width']))
        template = self.network.state_dict()
        if set(saved.files) != metadata | {'weight_' + k for k in template}:
            raise ValueError('Incompatible bass acoustic support checkpoint arrays')
        state = {}
        for key, tensor in template.items():
            array = saved['weight_' + key]
            if array.dtype != np.float32 or array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                raise ValueError('Invalid bass acoustic support checkpoint weights')
            state[key] = torch.from_numpy(array.copy())
        self.normalizer = tuple(saved[k].copy() for k in ('mean', 'scale'))
        mean, scale = self.normalizer
        if (mean.dtype != np.float32 or scale.dtype != np.float32
                or mean.shape != (84,) or scale.shape != (84,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError('Invalid bass acoustic support normalization')
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def probability(self, frames, attack_frames, release_frames, context):
        return probabilities(self.network, frames, attack_frames, release_frames, context, self.normalizer)

    def keep(self, events, probability, guardian, duration, stricter=False):
        events, p, g = np.asarray(events, float), np.asarray(probability), np.asarray(guardian)
        n = len(events)
        if (events.shape != (n, 4) or p.shape != (n, 2) or g.shape != (n,)
                or not np.isfinite(duration) or duration <= 0
                or any(not np.isfinite(a).all() for a in (events, p, g))
                or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
                or np.any(events[:, 1] > duration) or np.any((p < 0) | (p > 1))
                or np.any((g < 0) | (g > 1)) or not np.allclose(p.sum(axis=1), 1., atol=1e-5)):
            raise ValueError('Invalid bass acoustic support probabilities or clock')
        eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5)
                    & (events[:, 2] >= 21) & (events[:, 2] < 60))
        threshold = self.remove_threshold + (1. - self.remove_threshold)/2 if stricter else self.remove_threshold
        return ~(eligible & (p.argmax(axis=1) == 1) & (p[:, 1] >= threshold)
                 & (g < self.guardian_threshold))


class BassAttackDeclutter:
    name = 'Bass attack/release verifier v1 (trained held-note evidence)'

    def __init__(self):
        if not ASSET.is_file() or hashlib.sha256(ASSET.read_bytes()).hexdigest() != EXPECTED_SHA:
            raise PipelineError('The bass acoustic support model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(ASSET, allow_pickle=False) as saved:
                self.model = AttackSupportModel(saved)
            self.guardian = BassHarmonic().models[1]
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            raise PipelineError('The bass acoustic support model contains incompatible data. Rebuild the backend and retry.') from exc

    def filter(self, path, acoustic, midi):
        notes = [note for part in midi.instruments for note in part.notes]
        if not notes:
            return 0
        waveform, rate = sf.read(str(path), dtype='float32')
        if rate != 22050 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise PipelineError('Bass acoustic verification needs finite mono audio at 22050 Hz.')
        duration = len(waveform)/rate
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float)
        # A legacy decoder can extend its final boundary slightly past the
        # waveform. Preserve its existing clipping behavior instead of making
        # this optional refinement fail an otherwise processable upload.
        if np.any(events[:, 1] > duration) or not np.isfinite(events).all():
            return 0
        eligible = ((events[:, 0] >= 2.5) & (events[:, 1] <= duration - 2.5)
                    & (events[:, 2] >= 21) & (events[:, 2] < 60))
        if not np.any(eligible):
            return 0
        base_x = note_features(waveform, rate, acoustic, notes, include_context=True)
        observed = observe(waveform, rate)
        frames = coarse_sequences(observed['cqt'], events)
        fine = attack_sequences(observed, events)
        endings = release_sequences(observed, events)
        p = self.model.probability(frames, fine, endings, features(events, base_x))
        keep = self.model.keep(events, p, self.guardian.probability(base_x), duration)
        accepted = {id(n) for n, k in zip(notes, keep, strict=True) if k}
        for part in midi.instruments:
            part.notes = [n for n in part.notes if id(n) in accepted]
        return int((~keep).sum())
