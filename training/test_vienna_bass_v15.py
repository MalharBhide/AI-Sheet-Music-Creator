"""Real piano data must keep performer partitions and pedal/clock truth."""

import json
from types import SimpleNamespace

import numpy as np
import prepare_vienna_bass_v15 as preparation
import pretty_midi
import pytest


def test_shifted_crop_retains_pedal_support_and_distinct_same_key_attack(monkeypatch):
    midi = pretty_midi.PrettyMIDI()
    part = pretty_midi.Instrument(0)
    part.notes = [pretty_midi.Note(80, 36, 29., 31.), pretty_midi.Note(90, 36, 32.25, 32.8),
                  pretty_midi.Note(90, 72, 32., 33.)]
    part.control_changes = [pretty_midi.ControlChange(64, 127, 30.), pretty_midi.ControlChange(64, 0, 33.5)]
    midi.instruments = [part]
    monkeypatch.setattr(preparation.pretty_midi, 'PrettyMIDI', lambda path: midi)
    monkeypatch.setattr(preparation.sf, 'info', lambda path: SimpleNamespace(duration=120.))
    keys, pitch = preparation.reference({'midi': 'instrument.mid', 'audio': 'piano.wav', 'shift_seconds': .5})
    np.testing.assert_allclose(keys, [[2.75, 3.3, 36]])
    np.testing.assert_allclose(pitch, [[0., 2.75, 36], [2.75, 4., 36]])


def fixtures(tmp_path, monkeypatch):
    (tmp_path / 'note-verifier-v3-data').mkdir()
    rights = {'author': 'Werner Goebl', 'license': 'CC-BY-4.0',
              'archives': [{'url': url, 'sha256': sha} for url, sha in preparation.CHECKSUMS.items()]}
    (tmp_path / 'vienna-source.json').write_text(json.dumps(rights))
    # The historical wrong clock file must never be used.
    (tmp_path / 'vienna-clock-audit.json').write_text('not valid JSON')
    clocks = [{'id': f'vienna-Piece{piece}_p{player:02}', 'group': 'train' if player<=14 else 'validation' if player<=18 else 'test',
               'initial_shift': -.5, 'spectral_correction': .01}
              for piece in range(4) for player in range(1, 23)]
    path = tmp_path / 'note-verifier-v3-data/clock-audit.json'
    path.write_text(json.dumps(clocks))

    def source(folder, pattern):
        assert not any(f'_p{player:02}' in pattern for player in range(19, 23)), 'Reserved audio was read'
        return [tmp_path / pattern]

    monkeypatch.setattr(type(tmp_path), 'rglob', source)
    monkeypatch.setattr(preparation.sf, 'info', lambda path: SimpleNamespace(duration=40. if 'Piece3' in path.name else 120.))
    monkeypatch.setattr(preparation, 'digest', lambda path: 'fixture-sha')
    return rights, clocks, path


def test_corrected_clocks_splits_duration_selection_and_no_test_audio(tmp_path, monkeypatch):
    fixtures(tmp_path, monkeypatch)
    records = preparation.declared_sources(tmp_path)
    assert len(records) == 54
    assert {g: sum(r['group']==g for r in records) for g in ('train', 'validation')} == {'train': 42, 'validation': 12}
    assert sum('demucs-bass' in r['variants'] for r in records) == 12
    assert all(r['shift_seconds'] == -.49 for r in records)
    assert not {r['source_group'] for r in records if r['group']=='train'} & {r['source_group'] for r in records if r['group']=='validation'}


def test_noncommercial_or_changed_performer_assignment_is_rejected(tmp_path, monkeypatch):
    rights, clocks, path = fixtures(tmp_path, monkeypatch)
    rights['license'] = 'CC-BY-NC-4.0'
    (tmp_path / 'vienna-source.json').write_text(json.dumps(rights))
    with pytest.raises(ValueError, match='rights'):
        preparation.declared_sources(tmp_path)
    rights['license'] = 'CC-BY-4.0'
    (tmp_path / 'vienna-source.json').write_text(json.dumps(rights))
    clocks[18]['group'] = 'train'
    path.write_text(json.dumps(clocks))
    with pytest.raises(ValueError, match='performer split'):
        preparation.declared_sources(tmp_path)
