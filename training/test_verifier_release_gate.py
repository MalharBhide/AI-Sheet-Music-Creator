"""A better average or an older baseline must not hide a release regression."""

from copy import deepcopy

from verifier_release_gate import decision


def report():
    baseline = {'precision': .8, 'recall': .8, 'f1': .8, 'false_positives': 20}
    trained = {'precision': .9, 'recall': .8, 'f1': .847, 'false_positives': 10}
    return {'per_recording': [{'id': 'quiet', 'baseline': baseline, 'trained': trained}],
            'aggregate': {'piano': {'baseline': baseline, 'previous_v2': deepcopy(baseline),
                                    'trained': trained}}}


def test_average_improvement_cannot_hide_individual_recording_failure():
    result = report()
    result['per_recording'].append({'id': 'lost melody', 'baseline': {'f1': .9, 'recall': .9},
                                    'trained': {'f1': .89, 'recall': .9}})
    assert not decision(result)['promoted']
    assert any('lost melody' in reason for reason in decision(result)['reasons'])


def test_candidate_must_also_beat_the_current_release():
    result = report()
    result['aggregate']['piano']['previous_v2']['precision'] = .95
    assert not decision(result)['promoted']


def test_clear_measured_improvement_passes():
    assert decision(report())['promoted']


def test_no_op_is_not_a_model_improvement():
    result = report()
    result['aggregate']['piano']['trained'] = deepcopy(result['aggregate']['piano']['previous_v2'])
    assert not decision(result)['promoted']
