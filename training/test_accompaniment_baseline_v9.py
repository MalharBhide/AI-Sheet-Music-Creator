"""Future models must improve the actual V9 pipeline, including hold articulation."""

from types import SimpleNamespace

import accompaniment_baseline_v9 as baseline
import numpy as np
from test_repeat_boundaries import fixture


def heads():
    return [SimpleNamespace(threshold=.05, probability=lambda x: np.zeros(len(x))),
            SimpleNamespace(threshold=.025, probability=lambda x: np.zeros(len(x)))]


def test_v9_baseline_includes_hold_merges_and_rechecks_changed_pair_context(monkeypatch):
    monkeypatch.setattr(baseline, 'heads', heads)
    item = fixture()
    joined = baseline.project(item)
    np.testing.assert_array_equal(joined, [[0., 4., 48., 80.]])
    assert len(item['_boundary_cache']) == 2
    again = baseline.project(item, np.array([True, False, True]))
    np.testing.assert_array_equal(again, [[0., 1., 48., 80.], [2., 4., 48., 60.]])
    assert len(item['_boundary_cache']) == 2
    # In-place feature changes invalidate cached evidence rather than quietly
    # retaining decisions computed from a different acoustic neighborhood.
    item['x'] *= 0
    baseline.project(item)
    assert len(item['_boundary_cache']) == 2


def test_deleting_a_v9_hold_fragment_fails_even_if_onset_counts_stay_equal(monkeypatch):
    monkeypatch.setattr(baseline, 'heads', heads)
    item = fixture()
    item['baseline_events'] = baseline.project(item)
    result = baseline.score([item], [np.array([1., 0., 1.])], .1)
    assert not result['passes']
    assert not result['per_recording'][0]['held_references_preserved']
    assert result['per_recording'][0]['lost_reference_pitch_seconds'] == 1.


def test_full_pitch_support_protects_crop_crossing_holds_missing_from_onset_reference(monkeypatch):
    monkeypatch.setattr(baseline, 'heads', heads)
    item = fixture()
    item['reference'] = np.array([[3., 4., 60.]])
    item['pitch_reference'] = np.array([[0., 4., 48.]])
    item['baseline_events'] = baseline.project(item)
    result = baseline.score([item], [np.array([1., 0., 1.])], .1)
    row = result['per_recording'][0]
    assert row['matched_references_preserved'] and row['held_references_preserved']
    assert not row['reference_pitch_coverage_preserved'] and not row['passes']
