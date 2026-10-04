import accompaniment_baseline_v9 as baseline
import numpy as np
import train_polyphonic_consensus as training
from test_accompaniment_baseline_v9 import heads
from test_repeat_boundaries import fixture


def item_with_extra(monkeypatch):
    monkeypatch.setattr(baseline, 'heads', heads)
    item = fixture()
    item['events'] = np.vstack((item['events'], [3., 3.2, 72., 80.]))
    item['x'] = np.vstack((item['x'], np.zeros(74)))
    item['keep'] = item['baseline_keep'] = item['shared'] = np.ones(4, dtype=bool)
    item['p'] = np.full(4, .9)
    item['baseline_events'] = baseline.project(item)
    return item


def test_new_validation_compares_against_v9_holds_and_requires_both_heads(monkeypatch):
    item = item_with_extra(monkeypatch)
    confidence = np.array([1., 1., 1., 0.])
    best, _ = training.select([item], [confidence], [confidence])
    result, _ = best
    assert result['passes'] and result['false_notes_removed'] == 1
    assert result['per_recording'][0]['held_references_preserved']
    assert 'deployed_v9' in result['per_recording'][0]
    best, _ = training.select([item], [confidence], [np.ones(4)])
    assert best is None


def test_full_support_gate_stops_erasing_pitch_missing_from_attack_crop(monkeypatch):
    item = item_with_extra(monkeypatch)
    item['pitch_reference'] = np.array([[0., 4., 48.], [0., 4., 72.]])
    confidence = np.array([1., 1., 1., 0.])
    best, _ = training.select([item], [confidence], [confidence])
    assert best is None
