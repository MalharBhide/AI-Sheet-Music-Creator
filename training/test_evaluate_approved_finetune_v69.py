from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate_approved_finetune_v69 as evaluate
from approved_model_finetune import load_approved

ROOT = Path(__file__).resolve().parents[1]


def checkpoint():
    approved = load_approved(ROOT)
    winner = {'profile': {'width': 12}, 'epoch': 25}
    saved = {'version': evaluate.NETWORK_VERSION, 'fine_tune_version': evaluate.VERSION,
             'initialized_from_sha256': evaluate.APPROVED_SHA256, 'width': 12, 'epoch': 25,
             'state_dict': approved.network.state_dict(),
             'mean': torch.from_numpy(approved.normalizer[0].copy()),
             'scale': torch.from_numpy(approved.normalizer[1].copy())}
    return approved, winner, saved


def test_checkpoint_restores_only_the_approved_architecture_and_normalization():
    approved, winner, saved = checkpoint()
    model, normalizer = evaluate.restore_network(saved, winner, approved)
    for key, value in model.state_dict().items():
        assert torch.equal(value, approved.network.state_dict()[key])
    for value, expected in zip(normalizer, approved.normalizer, strict=True):
        np.testing.assert_array_equal(value, expected)


def test_support_feature_updates_can_restore_without_changing_rejection():
    approved, winner, saved = checkpoint()
    winner['profile']['freeze_rejection'] = True
    key = 'condition.bias'
    saved['state_dict'][key] = saved['state_dict'][key].clone()+.01
    network, _ = evaluate.restore_network(saved, winner, approved)
    assert torch.equal(network.state_dict()[key], saved['state_dict'][key])


@pytest.mark.parametrize('field', ['version', 'fine_tune_version', 'initialized_from_sha256', 'width', 'epoch'])
def test_checkpoint_from_another_run_or_architecture_is_rejected(field):
    approved, winner, saved = checkpoint()
    saved[field] = None
    with pytest.raises(ValueError, match='checkpoint'):
        evaluate.restore_network(saved, winner, approved)


@pytest.mark.parametrize('fault', ['mean', 'scale', 'nonfinite_weight', 'wrong_weight_dtype'])
def test_refitted_normalization_and_nonfinite_weights_are_rejected(fault):
    approved, winner, saved = checkpoint()
    if fault in ('mean', 'scale'):
        saved[fault][0] += .01
    else:
        key = next(iter(saved['state_dict']))
        saved['state_dict'][key] = saved['state_dict'][key].clone()
        if fault == 'wrong_weight_dtype':
            saved['state_dict'][key] = saved['state_dict'][key].double()
        else:
            saved['state_dict'][key].reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError):
        evaluate.restore_network(saved, winner, approved)


def test_failed_validation_cannot_open_regression_data(tmp_path, monkeypatch):
    accessed = []
    monkeypatch.setattr(evaluate, 'frozen', lambda *args: (_ for _ in ()).throw(ValueError('No validation winner')))
    monkeypatch.setattr(evaluate, 'consumed', lambda *args: accessed.append(True))
    with pytest.raises(ValueError, match='validation'):
        evaluate.evaluate_run(tmp_path, ROOT)
    assert accessed == []


def test_completed_regression_cannot_be_replaced(tmp_path, monkeypatch):
    target = tmp_path/'consumed-regression.json'
    target.write_text('{"passes": false}')
    monkeypatch.setattr(evaluate, 'frozen', lambda *args: pytest.fail('Must reject before reading another winner'))
    with pytest.raises(ValueError, match='Preserve'):
        evaluate.evaluate_run(tmp_path, ROOT)
    assert target.read_text() == '{"passes": false}'


@pytest.mark.parametrize('prefix', ['reject.'])
def test_frozen_rejection_checkpoint_cannot_change_original_note_classifier(prefix):
    approved, winner, saved = checkpoint()
    winner['profile']['freeze_rejection'] = True
    evaluate.restore_network(saved, winner, approved)
    key = next(k for k in saved['state_dict'] if k.startswith(prefix))
    saved['state_dict'][key] = saved['state_dict'][key].clone()
    saved['state_dict'][key].reshape(-1)[0] += .01
    with pytest.raises(ValueError, match='classifier changed'):
        evaluate.restore_network(saved, winner, approved)
