"""Acoustic support runtime contracts and retained-note preservation."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from app.services.attack_local_network import model
from app.services.bass_attack_declutter import VERSION, AttackSupportModel, BassAttackDeclutter


def fields():
    torch.set_num_threads(1)
    net = model(24)
    data = {'version':VERSION, 'format_version':2, 'width':24,
            'remove_threshold':.9, 'guardian_threshold':.5,
            'mean':np.zeros(84,np.float32), 'scale':np.ones(84,np.float32)}
    data.update({'weight_'+k:v.numpy() for k,v in net.state_dict().items()})
    return data


def loaded(tmp_path, changes=None):
    values=fields()
    if changes:
        values.update(changes)
    path=tmp_path/'random-initializer-fixture.npz'
    np.savez_compressed(path, **values)
    with np.load(path,allow_pickle=False) as saved:
        return AttackSupportModel(saved)


@pytest.mark.parametrize('changes',[{'width':24.5}, {'remove_threshold':np.nan},
    {'guardian_threshold':.7}, {'mean':np.zeros(84,np.float64)},
    {'scale':np.zeros(84,np.float32)}, {'unexpected':np.array(1)}])
def test_incompatible_checkpoint_rejected(tmp_path,changes):
    with pytest.raises(ValueError):
        loaded(tmp_path,changes)


def test_runtime_empty_probability_shape(tmp_path):
    m=loaded(tmp_path)
    assert m.probability(np.zeros((0,40,9)),np.zeros((0,61,18)),np.zeros((0,61,18)),np.zeros((0,84))).shape==(0,2)


def test_guard_and_margin_abstain_without_mutating_any_event(tmp_path):
    m=loaded(tmp_path)
    events=np.array([[4,6,48,90],[7,8,37,60],[9,10,48,90]],float)
    p=np.array([[.01,.99],[.08,.92],[.01,.99]])
    g=np.array([.5,.1,.9])
    original=events.copy()
    np.testing.assert_array_equal(m.keep(events,p,g,30),[True,False,True])
    np.testing.assert_array_equal(m.keep(events,p,g,30,stricter=True),[True,True,True])
    np.testing.assert_array_equal(events,original)


@pytest.mark.parametrize('duration,end',[(3.,2.),(8.,8.001)])
def test_short_and_decoder_overshoot_abstain_before_feature_extraction(tmp_path,monkeypatch,duration,end):
    import app.services.bass_attack_declutter as module
    path=tmp_path/'authored.wav'
    sf.write(path,np.zeros(int(duration*22050),np.float32),22050)
    note=SimpleNamespace(start=1.,end=end,pitch=48,velocity=90)
    part=SimpleNamespace(notes=[note])
    midi=SimpleNamespace(instruments=[part])
    wrapper=BassAttackDeclutter.__new__(BassAttackDeclutter)
    monkeypatch.setattr(module,'note_features',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('Unexpected acoustic analysis')))
    assert wrapper.filter(path,None,midi)==0 and part.notes[0] is note
    assert note.start==1. and note.end==end


def test_filter_preserves_source_parts_note_identity_and_all_retained_fields(tmp_path,monkeypatch):
    import app.services.bass_attack_declutter as module
    path=tmp_path/'authored.wav'
    sf.write(path,np.zeros(30*22050,np.float32),22050)
    notes=[SimpleNamespace(start=4.,end=6.,pitch=48,velocity=90,tag='held'),
           SimpleNamespace(start=5.,end=5.4,pitch=37,velocity=50,tag='noise'),
           SimpleNamespace(start=7.,end=8.,pitch=48,velocity=90,tag='repeat')]
    parts=[SimpleNamespace(notes=notes[:2]),SimpleNamespace(notes=notes[2:])]
    midi=SimpleNamespace(instruments=parts)
    wrapper=BassAttackDeclutter.__new__(BassAttackDeclutter)
    wrapper.model=loaded(tmp_path)
    monkeypatch.setattr(wrapper.model,'probability',lambda *_:np.array([[.99,.01],[.01,.99],[.99,.01]]))
    wrapper.guardian=SimpleNamespace(probability=lambda _:np.array([.9,.1,.9]))
    monkeypatch.setattr(module,'note_features',lambda *_a,**_k:np.zeros((3,52)))
    monkeypatch.setattr(module,'observe',lambda *_:{'cqt':None})
    monkeypatch.setattr(module,'coarse_sequences',lambda *_:np.zeros((3,40,9)))
    monkeypatch.setattr(module,'attack_sequences',lambda *_:np.zeros((3,61,18)))
    monkeypatch.setattr(module,'release_sequences',lambda *_:np.zeros((3,61,18)))
    before=[vars(n).copy() for n in notes]
    assert wrapper.filter(path,None,midi)==1
    assert midi.instruments is parts and parts[0].notes[0] is notes[0] and parts[1].notes[0] is notes[2]
    assert [vars(n) for n in (notes[0],notes[2])]==[before[0],before[2]]


def test_sealed_release_loads_and_corruption_is_rejected(tmp_path, monkeypatch):
    import app.services.bass_attack_declutter as module
    from app.models import PipelineError

    verifier = BassAttackDeclutter()
    assert verifier.model.remove_threshold == .8
    assert verifier.model.guardian_threshold == .5
    broken = tmp_path / 'broken.npz'
    broken.write_bytes(b'not a model')
    monkeypatch.setattr(module, 'ASSET', broken)
    with pytest.raises(PipelineError, match='missing or damaged'):
        BassAttackDeclutter()
