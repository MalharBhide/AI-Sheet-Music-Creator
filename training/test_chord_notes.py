"""Regression checks for contextual chord training and evaluation."""

import numpy as np
import pytest
import torch
from app.services.note_verifier import NoteVerifier
from chord_note_model import ChordNoteVerifier, peer_matrix
from train_chord_notes import chord_groups, digest, gate_failures, measures, train, write


def checkpoint():
    torch.manual_seed(2)
    return {'state_dict': NoteVerifier().state_dict(), 'mean': torch.zeros(26), 'scale': torch.ones(26)}


def test_peers_exclude_duplicates_and_do_not_chain_arpeggios():
    events = np.array([[0, 1, 60, 80], [.03, 1, 64, 80], [.06, 1, 67, 80], [.01, 1, 60, 70]])
    peers = peer_matrix(events)
    assert peers[0, 1] and peers[1, 2]
    assert not peers[0, 2] and not peers[0, 3]
    assert not np.diag(peers).any()


def test_initial_residual_preserves_pretrained_logits():
    saved = checkpoint()
    model = ChordNoteVerifier(saved)
    features = torch.randn(4, 26)
    events = np.array([[0, 1, 48, 80], [0, 1, 55, 80], [1, 2, 62, 80], [3, 4, 60, 80]])
    torch.testing.assert_close(model(features, events), model.base(features))
    assert all(not p.requires_grad for p in model.base.parameters())


def test_context_is_permutation_invariant_and_finite_without_peers():
    model = ChordNoteVerifier(checkpoint())
    torch.nn.init.normal_(model.residual[-1].weight)
    features = torch.randn(4, 26)
    events = np.array([[0, 1, 48, 80], [0, 1, 55, 80], [.02, 1, 60, 80], [3, 4, 60, 80]])
    order = np.array([3, 1, 0, 2])
    actual = model(features, events)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(model(features[order], events[order]), actual[order])
    assert model(torch.zeros((0, 26)), np.empty((0, 4))).shape == (0,)


def test_training_updates_context_but_not_frozen_encoder():
    model = ChordNoteVerifier(checkpoint())
    before = {k: v.clone() for k, v in model.base.state_dict().items()}
    optimizer = torch.optim.AdamW(model.residual.parameters(), lr=.01)
    features = torch.randn(3, 26)
    events = np.array([[0, 1, 48, 80], [0, 1, 55, 80], [0, 1, 56, 80]])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features, events), torch.tensor([1., 1., 0.]))
    loss.backward()
    optimizer.step()
    assert model.residual[-1].weight.abs().sum() > 0
    for k, v in model.base.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_chord_metrics_require_all_pitches_and_correct_attacks():
    reference = np.array([[0., 1., 60], [.02, 1., 64], [.06, 1., 67], [2., 3., 60], [2., 3., 67]])
    assert chord_groups(reference) == [[0, 1], [3, 4]]
    predicted = np.column_stack((reference, np.full(5, 80)))
    assert measures(reference, predicted)['complete_chords'] == 2
    assert measures(reference, predicted[[0, 2, 3, 4]])['complete_chords'] == 1
    predicted[1, :2] += .2
    assert measures(reference, predicted)['complete_chords'] == 1


def test_extra_pitch_is_counted_even_when_chord_complete():
    reference = np.array([[0., 1., 60], [0., 1., 64], [0., 1., 67]])
    prediction = np.array([[0., 1., p, 80] for p in (60, 64, 66, 67)])
    result = measures(reference, prediction)
    assert result['complete_chords'] == 1
    assert result['predicted'] - result['tp'] == 1


def test_better_aggregate_score_cannot_hide_lost_chord_tones():
    baseline = {'aggregate': {'false_positives': 100}, 'per_work': {
        'quiet-chords': {'recall': .8, 'f1': .7, 'complete_chord_recall': .6}}}
    candidate = {'aggregate': {'false_positives': 80}, 'per_work': {
        'quiet-chords': {'recall': .8, 'f1': .75, 'complete_chord_recall': .59}}}
    assert gate_failures(baseline, candidate) == ['quiet-chords: complete chord recall']
    candidate['per_work']['quiet-chords']['complete_chord_recall'] = .6
    assert gate_failures(baseline, candidate) == []


def test_gate_rejects_incomplete_validation_and_cosmetic_change():
    baseline = {'aggregate': {'false_positives': 100}, 'per_work': {
        'piano-work': {'recall': .8, 'f1': .7, 'complete_chord_recall': .6}}}
    assert gate_failures(baseline, baseline) == ['False positives did not decrease by 5%']
    assert gate_failures(baseline, {'per_work': {}}) == ['Different evaluation works']


def test_source_drift_stops_training_before_loading_examples(tmp_path):
    from app.services.accompaniment_verifier import CHECKPOINT

    write(tmp_path / 'plan.json', {'encoder_sha256': digest(CHECKPOINT),
        'code_sha256': {'train_chord_notes.py': '0' * 64}})
    write(tmp_path / 'manifest.json', {'plan_sha256': digest(tmp_path / 'plan.json')})
    with pytest.raises(ValueError, match='sources changed'):
        train(tmp_path)
    assert not (tmp_path / 'candidate.pt').exists()
