"""Finish a bounded dataset batch and evaluate its frozen winner, never deploy.

The separately running Demucs preparation must finish its complete predeclared
manifest. A failed producer cannot silently become partial fitting data. This
runner creates no user jobs and cannot export weights or change the website.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from bass_residual_coverage_release import evaluate
from current_bass_v11_baseline import hashes
from prepare_bass_original_expanded import prepare as prepare_original
from prepare_bass_v11_baseline import prepare as prepare_baseline
from prepare_robust_training_stems import digest, preserve
from train_bass_residual_coverage import contracts, train


def producer_running():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'/src/training/prepare_slakh_bass_demucs_expanded.py' in command:
            return True
    return False


def ready(directory):
    path = directory / 'manifest.json'
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if (manifest['plan_sha256'] != digest(directory / 'plan.json')
            or not manifest['no_test_inference'] or not manifest['no_user_audio_or_scores']
            or plan['demucs_offsets'] != [30, 60, 90] or plan['oracle_offsets'] != []
            or dict(Counter(item['group'] for item in manifest['items'])) != {'train': 36, 'validation': 12}):
        raise ValueError('Changed or incomplete expanded Demucs preparation')
    for name, expected in plan['code_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Changed expanded Demucs preparation source')
    return True


def run(directory):
    destination = directory / 'bass-expanded-training-v3'
    destination.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    plan = {'baseline_hashes': hashes(), 'training_contracts': contracts(),
            'code_sha256': {name: digest(root / 'training' / name) for name in (
                'continue_bass_v11_training.py', 'prepare_bass_original_expanded.py',
                'prepare_slakh_bass_demucs_expanded.py')},
            'fitting_roots': ['slakh-bass-v1', 'bass-positive-data-v1',
                              'slakh-bass-demucs-expanded-v1', 'bass-original-expanded-v1'],
            'consumed_regressions': ['bass-held-regression-v1', 'slakh-bass-positive-reserved-v2',
                                     'bass-boundary-stress-v1', 'bass-articulation-stress-v2'],
            'new_training_clips': 100, 'new_validation_clips': 28,
            'no_user_audio_or_scores': True, 'no_automatic_export_deployment_or_git': True,
            'release_requirement': 'A frozen validation winner must pass all 92 consumed regressions. Fresh stress, portable/native parity and full integration still required before any website release.'}
    preserve(destination / 'plan.json', plan)

    def status(state, **details):
        payload = {'state': state, 'plan_sha256': digest(destination / 'plan.json'),
                   'production_weights_changed': False, 'user_uploads_processed': 0, **details}
        (destination / 'status.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload), flush=True)

    try:
        status('preparing_80_original_training_validation_clips')
        prepare_original(directory)
        status('waiting_for_48_licensed_demucs_clips')
        expanded = directory / 'slakh-bass-demucs-expanded-v1'
        deadline = time.monotonic() + 7200
        while not ready(expanded):
            if not producer_running():
                raise RuntimeError('Demucs producer stopped without its complete manifest')
            if time.monotonic() >= deadline:
                raise TimeoutError('Expanded Demucs preparation exceeded two hours')
            time.sleep(20)
        # Fitting code/weights are frozen before data acquisition; stop if edited.
        if hashes() != plan['baseline_hashes'] or contracts() != plan['training_contracts']:
            raise ValueError('Changed frozen expanded training contract')
        for name, expected in plan['code_sha256'].items():
            if digest(root / 'training' / name) != expected:
                raise ValueError('Changed expanded training/preparation script')
        status('preparing_duration_correct_v11_baseline')
        cache = directory / 'bass-v11-expanded-baseline-v2'
        prepare_baseline([directory / name for name in plan['fitting_roots']],
                         [directory / name for name in plan['consumed_regressions']], cache)
        status('fitting_four_residual_heads')
        candidate = directory / 'bass-residual-expanded-v3'
        train(cache, candidate)
        batch = json.loads((candidate / 'batch-selection.json').read_text())
        if not batch['selected']:
            status('withheld_no_validation_winner', candidate=str(candidate))
            return
        status('evaluating_frozen_winner_on_consumed_regression', candidate=str(candidate))
        evaluate(candidate)
        report = json.loads((candidate / 'consumed-regression.json').read_text())
        status('passed_regression_needs_fresh_and_runtime_release_checks' if report['passes'] else 'withheld_failed_regression',
               candidate=str(candidate), passes=report['passes'],
               failed_recordings=report['failed_recordings'], false_notes_removed=report['false_notes_removed'])
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
