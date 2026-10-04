"""Verify accompaniment candidates with a checked, locally trained network."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError
from app.services.accompaniment_candidates import decode_candidates
from app.services.candidate_relations import (
    ALL_NAMES,
    RELATION_VERSION,
    relation_features,
    window_eligible,
)
from app.services.left_baseline_evidence import (
    ACOUSTIC_EVIDENCE_NAMES,
    EVIDENCE_VERSION,
    RELATION_EVIDENCE_NAMES,
    acoustic_evidence,
    append_evidence,
)
from app.services.note_context_model import (
    ContextNoteModel,
    consensus_keep,
    correction_keep,
    residual_keep,
    shared_events,
)
from app.services.note_evidence import FEATURE_NAMES, FEATURE_VERSION, note_features

CHECKPOINT = Path(__file__).resolve().parents[1] / 'assets' / 'accompaniment-verifier-v2.pt'
CHECKPOINT_SHA256 = '2b75ac2d3c80a2644c7df5c0ec5fab9c10086ff2fa072177868e7e6c3091fe60'
CONTEXT_CHECKPOINTS = (
    (CHECKPOINT.parent / 'accompaniment-context-expanded.npz',
     'd06bbe84ee9ad251960969e0097b217e340d52562ecf178295d7c4ae57c6ee8d'),
    (CHECKPOINT.parent / 'accompaniment-context-original.npz',
     '381c150b19bdb223fc9e85f88b78dd69892b0de5c7591189967f30b59a26a8e7'),
)
RESIDUAL_CHECKPOINT = CHECKPOINT.parent / 'accompaniment-residual-v4.npz'
RESIDUAL_SHA256 = 'ad958557c44312c8d082b9339801ae794a7673956f098106f7fb6381fa3cd24d'
LEFT_HAND_CHECKPOINT = CHECKPOINT.parent / 'accompaniment-left-hand-v1.npz'
LEFT_HAND_SHA256 = '7b3343637fa0d19ba1cd468845fcc309f30ac55b7fddc00f68bbc607abc2555b'
LEFT_REFINEMENT_CHECKPOINT = CHECKPOINT.parent / 'accompaniment-left-refinement-v3.npz'
LEFT_REFINEMENT_SHA256 = '6b0f9bb5260f9645831dd409abd6fcb17cd8acbdd09e7a13df9c356cd76414e2'

LEFT_RELATIONS_CHECKPOINT = CHECKPOINT.parent / 'left-relations-v7.npz'
LEFT_RELATIONS_SHA256 = '50c4a615d36a835459ca9f1d40879013e987dc3424c0f9496cff64b67353a681'
LEFT_GUARDIAN_CHECKPOINT = CHECKPOINT.parent / 'left-guardian-v7.npz'
LEFT_GUARDIAN_SHA256 = '34d70e8e621f6809d7066cd5a80011299464280406f0a0ebafbf4ed0b76f0cda'
BROAD_CHECKPOINT = CHECKPOINT.parent / 'accompaniment-broad-pitch-v1.npz'
BROAD_SHA256 = '1faaa1e98e1af9e95e8684ce58e5731fe21f219f9434673e64d63b8101d4b5dd'
BROAD_GUARDIAN_CHECKPOINT = CHECKPOINT.parent / 'accompaniment-broad-guardian-v1.npz'
BROAD_GUARDIAN_SHA256 = 'fbe2f75edf6dec98df6a3568c2e1dc3b900d55284f78b7718df54345b9ed7395'


class AccompanimentVerifier:
    name = 'Accompaniment note verifier v8 (trained pitch-preserving consensus)'

    def __init__(self):
        import torch

        from app.services.note_verifier import NoteVerifier

        if not CHECKPOINT.is_file() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise PipelineError('The accompaniment model is missing or damaged. Rebuild the backend and retry.')
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        if saved['feature_version'] != FEATURE_VERSION or saved['feature_names'] != list(FEATURE_NAMES):
            raise PipelineError('The accompaniment model and audio features have incompatible versions.')
        self.model = NoteVerifier().eval()
        self.model.load_state_dict(saved['state_dict'])
        self.mean, self.scale = saved['mean'].numpy(), saved['scale'].numpy()
        self.threshold = float(saved['threshold'])
        if (self.mean.shape != (len(FEATURE_NAMES),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0) or not 0 <= self.threshold <= 1):
            raise PipelineError('The accompaniment model has invalid normalization or threshold data.')
        self.context_models = []
        for path, digest in CONTEXT_CHECKPOINTS:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise PipelineError('The accompaniment context model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    context = ContextNoteModel(arrays)
                if context.threshold != .01:
                    raise ValueError('Unexpected context threshold')
                self.context_models.append(context)
            except (ValueError, KeyError, OSError) as exc:
                raise PipelineError('The accompaniment context model is invalid. Rebuild the backend and retry.') from exc
        if (not RESIDUAL_CHECKPOINT.is_file()
                or hashlib.sha256(RESIDUAL_CHECKPOINT.read_bytes()).hexdigest() != RESIDUAL_SHA256):
            raise PipelineError('The residual note model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(RESIDUAL_CHECKPOINT, allow_pickle=False) as arrays:
                self.residual_model = ContextNoteModel(arrays)
            if self.residual_model.threshold != .0375:
                raise ValueError('Unexpected residual threshold')
        except (ValueError, KeyError, OSError) as exc:
            raise PipelineError('The residual note model is invalid. Rebuild the backend and retry.') from exc
        if (not LEFT_HAND_CHECKPOINT.is_file()
                or hashlib.sha256(LEFT_HAND_CHECKPOINT.read_bytes()).hexdigest() != LEFT_HAND_SHA256):
            raise PipelineError('The left-hand note model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(LEFT_HAND_CHECKPOINT, allow_pickle=False) as arrays:
                self.left_hand_model = ContextNoteModel(arrays)
            if self.left_hand_model.threshold != .05:
                raise ValueError('Unexpected left-hand threshold')
        except (ValueError, KeyError, OSError) as exc:
            raise PipelineError('The left-hand note model is invalid. Rebuild the backend and retry.') from exc
        if (not LEFT_REFINEMENT_CHECKPOINT.is_file()
                or hashlib.sha256(LEFT_REFINEMENT_CHECKPOINT.read_bytes()).hexdigest() != LEFT_REFINEMENT_SHA256):
            raise PipelineError('The left-hand refinement model is missing or damaged. Rebuild the backend and retry.')
        try:
            with np.load(LEFT_REFINEMENT_CHECKPOINT, allow_pickle=False) as arrays:
                self.left_refinement_model = ContextNoteModel(arrays)
            if self.left_refinement_model.threshold != .0375:
                raise ValueError('Unexpected left-hand refinement threshold')
        except (ValueError, KeyError, OSError) as exc:
            raise PipelineError('The left-hand refinement model is invalid. Rebuild the backend and retry.') from exc

        for name, path, digest, threshold, contract in (
            ('left_relations_model', LEFT_RELATIONS_CHECKPOINT, LEFT_RELATIONS_SHA256, .0375,
             {'feature_names': ALL_NAMES, 'feature_version': RELATION_VERSION}),
            ('left_guardian_model', LEFT_GUARDIAN_CHECKPOINT, LEFT_GUARDIAN_SHA256, .3, {}),
        ):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise PipelineError('The left-hand consensus model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    model = ContextNoteModel(arrays, **contract)
                if model.threshold != threshold:
                    raise ValueError('Unexpected left-hand consensus threshold')
                setattr(self, name, model)
            except (ValueError, KeyError, OSError) as exc:
                raise PipelineError('The left-hand consensus model is invalid. Rebuild the backend and retry.') from exc

        for name, path, digest, threshold, names in (
            ('broad_model', BROAD_CHECKPOINT, BROAD_SHA256, .005, RELATION_EVIDENCE_NAMES),
            ('broad_guardian_model', BROAD_GUARDIAN_CHECKPOINT, BROAD_GUARDIAN_SHA256, .05,
             ACOUSTIC_EVIDENCE_NAMES),
        ):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise PipelineError('The pitch-preserving model is missing or damaged. Rebuild the backend and retry.')
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    model = ContextNoteModel(arrays, feature_names=names, feature_version=EVIDENCE_VERSION)
                if model.threshold != threshold:
                    raise ValueError('Unexpected pitch-preserving threshold')
                setattr(self, name, model)
            except (ValueError, KeyError, OSError) as exc:
                raise PipelineError('The pitch-preserving model is invalid. Rebuild the backend and retry.') from exc

    def filter(self, path, acoustic, midi):
        """Keep original pitch, timing and velocity; reject unsupported events only."""
        import torch

        notes = [n for part in midi.instruments for n in part.notes]
        if not notes:
            return 0
        samples, rate = sf.read(path, dtype='float32')
        if samples.ndim != 1 or rate != 22050 or not 0 < len(samples) <= 33 * rate:
            raise PipelineError('Accompaniment verification needs a normalized window of at most 33 seconds.')
        x = note_features(samples, rate, acoustic, notes, include_context=True)
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(torch.from_numpy((x[:, :len(FEATURE_NAMES)] - self.mean) / self.scale))).numpy()
        if not np.isfinite(probability).all():
            raise PipelineError('The accompaniment model returned invalid note confidence.')
        bounded = decode_candidates(acoustic)
        events = [[n.start, n.end, n.pitch, n.velocity] for n in notes]
        bounded_events = [[n.start, n.end, n.pitch, n.velocity]
                          for part in bounded.instruments for n in part.notes]
        context = np.asarray([model.probability(x) for model in self.context_models])
        shared = shared_events(events, bounded_events)
        keep = correction_keep(probability, context, shared,
                               threshold=self.threshold, prune=self.context_models[0].threshold)
        eligible = keep & shared & (probability <= .5)
        residual = np.ones(len(notes))
        if np.any(eligible):
            residual[eligible] = self.residual_model.probability(x[eligible])
        keep = residual_keep(keep, probability, shared, residual, threshold=self.residual_model.threshold)
        # This specialist was fitted only on low accompaniment candidates.
        # Independent bass-stem and treble/melody inference remain unchanged.
        low_shared = shared & np.asarray([36 <= n.pitch < 60 for n in notes])
        eligible = keep & low_shared & (probability <= .5)
        left_hand = np.ones(len(notes))
        if np.any(eligible):
            left_hand[eligible] = self.left_hand_model.probability(x[eligible])
        keep = residual_keep(keep, probability, low_shared, left_hand, threshold=self.left_hand_model.threshold)
        # Learn only remaining low-register errors. Earlier rejections stay
        # rejected; stronger, nonshared and treble candidates remain protected.
        eligible = keep & low_shared & (probability <= .5)
        refinement = np.ones(len(notes))
        if np.any(eligible):
            refinement[eligible] = self.left_refinement_model.probability(x[eligible])
        keep = residual_keep(keep, probability, low_shared, refinement,
                             threshold=self.left_refinement_model.threshold)
        # Both separately fitted views must reject a surviving low note. Keep
        # physical window edges unchanged because their neighboring context is incomplete.
        eligible = keep & low_shared & (probability <= .5) & window_eligible(np.asarray(events), len(samples) / rate)
        relations, guardian = np.ones(len(notes)), np.ones(len(notes))
        if np.any(eligible):
            features = relation_features(np.asarray(events), x, probability)
            relations[eligible] = self.left_relations_model.probability(features[eligible])
            guardian[eligible] = self.left_guardian_model.probability(x[eligible])
        keep = consensus_keep(keep, probability, low_shared, relations, guardian,
                              threshold=self.left_relations_model.threshold,
                              guardian_threshold=self.left_guardian_model.threshold)
        # Preserve V7 rejections. This pair was validated on accompaniment
        # across the piano register; both independent views must reject.
        broad_shared = shared & np.asarray([36 <= n.pitch < 96 for n in notes])
        broad_shared &= window_eligible(np.asarray(events), len(samples) / rate)
        eligible = keep & broad_shared
        if np.any(eligible):
            features = relation_features(np.asarray(events), x, probability)
            evidence = append_evidence(features, self.left_relations_model.probability(features),
                                       self.left_guardian_model.probability(x))
            relation = self.broad_model.probability(evidence[eligible])
            guardian = self.broad_guardian_model.probability(acoustic_evidence(evidence[eligible]))
            rejected = np.ones(len(notes))
            rejected[eligible] = np.where((relation < self.broad_model.threshold)
                                         & (guardian < self.broad_guardian_model.threshold), 0., 1.)
            keep = residual_keep(keep, probability, broad_shared, rejected,
                                 threshold=self.broad_model.threshold, ceiling=1.)
        index = 0
        for part in midi.instruments:
            size = len(part.notes)
            part.notes = [note for note, accepted in zip(part.notes, keep[index:index + size], strict=True) if accepted]
            index += size
        return int(np.sum(~keep))
