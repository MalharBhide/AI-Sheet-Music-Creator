"""Fine timing branch must preserve the warm-start coarse representation."""

import numpy as np
import pytest
import torch
from app.services.temporal_note_model import create_model
from attack_local_model import model, probabilities, warm_start


@pytest.mark.parametrize('width',[24,32])
def test_warm_start_preserves_hidden_state(width):
    old=create_model(48).eval()
    new=warm_start(model(width),{'width':48,'state_dict':old.state_dict()}).eval()
    frames=torch.rand(3,40,9)
    context=torch.rand(3,84)
    attack=torch.rand(3,61,18)
    with torch.inference_mode():
        old_hidden=old.classifier[:5](torch.cat((old.sequence(frames.transpose(1,2)),context),dim=1))
        new_hidden=new.classifier[:5](torch.cat((new.coarse(frames.transpose(1,2)),context,
                                               new.local(attack.transpose(1,2))),dim=1))
    torch.testing.assert_close(old_hidden,new_hidden,rtol=1e-5,atol=1e-6)


def test_four_actions_empty_inputs_and_batching_do_not_mutate_threads():
    network=model(24)
    threads=torch.get_num_threads()
    args=(np.zeros((7,40,9)),np.zeros((7,61,18)),np.zeros((7,84)),(np.zeros(84),np.ones(84)))
    p=probabilities(network,*args,batch_size=2)
    assert p.shape==(7,4) and np.all(p.argmax(axis=1)==1)
    np.testing.assert_allclose(p.sum(axis=1),1.,atol=2e-7)
    empty=probabilities(network,*[a[:0] for a in args[:3]],args[3])
    assert empty.shape==(0,4) and torch.get_num_threads()==threads


def test_feature_shape_and_invalid_fine_energy_are_rejected():
    network=model(24)
    with pytest.raises(ValueError):
        probabilities(network,np.zeros((1,40,9)),np.zeros((1,40,18)),np.zeros((1,84)),(np.zeros(84),np.ones(84)))
    with pytest.raises(ValueError):
        probabilities(network,np.zeros((1,40,9)),np.full((1,61,18),2.),np.zeros((1,84)),(np.zeros(84),np.ones(84)))
