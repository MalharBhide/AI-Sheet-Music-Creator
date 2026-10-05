"""Dual-scale observed evidence for early/unchanged/late/remove predictions."""

import numpy as np
from attack_local_features import CHANNELS, STEPS

VERSION = 'attack-local-bass-joint-v19-v1'


def model(width=32):
    import torch
    from app.services.temporal_note_model import create_model
    from torch import nn

    if width not in (24,32):
        raise ValueError('Unsupported attack-local width')

    class AttackLocalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.coarse=create_model(48).sequence
            self.local=nn.Sequential(nn.Conv1d(CHANNELS,width,5,padding=2),nn.ReLU(),
                                    nn.Conv1d(width,width,5,padding=2),nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(8),nn.Flatten())
            self.classifier=nn.Sequential(nn.Linear(48*8+84+width*8,96),nn.ReLU(),nn.Dropout(.1),
                                          nn.Linear(96,32),nn.ReLU(),nn.Linear(32,4))
            nn.init.zeros_(self.classifier[-1].weight)
            with torch.no_grad():
                self.classifier[-1].bias.copy_(torch.tensor([-3.,3.,-3.,-3.]))

        def forward(self,frames,attack_frames,context):
            coarse=self.coarse(frames.transpose(1,2))
            local=self.local(attack_frames.transpose(1,2))
            return self.classifier(torch.cat((coarse,context,local),dim=1))

    return AttackLocalNet()


def warm_start(network, saved):
    import torch

    if saved['width']!=48:
        raise ValueError('Incompatible coarse encoder width')
    state=saved['state_dict']
    coarse={k.removeprefix('sequence.'):v for k,v in state.items() if k.startswith('sequence.')}
    network.coarse.load_state_dict(coarse,strict=True)
    with torch.no_grad():
        first=network.classifier[0]
        first.weight.zero_()
        first.weight[:,:48*8+84].copy_(state['classifier.0.weight'])
        first.bias.copy_(state['classifier.0.bias'])
        network.classifier[3].weight.copy_(state['classifier.3.weight'])
        network.classifier[3].bias.copy_(state['classifier.3.bias'])
    return network


def probabilities(network, frames, attack_frames, context, normalizer, batch_size=256):
    import torch

    frames,attack_frames,context=[np.asarray(a,np.float32) for a in (frames,attack_frames,context)]
    mean,scale=[np.asarray(a,np.float32) for a in normalizer]
    n=len(context)
    if (context.shape!=(n,84) or frames.shape!=(n,40,9) or attack_frames.shape!=(n,STEPS,CHANNELS)
            or mean.shape!=(84,) or scale.shape!=(84,) or np.any(scale<=0) or batch_size<1
            or any(not np.isfinite(a).all() for a in (frames,attack_frames,context,mean,scale))
            or any(np.any((a<0)|(a>1)) for a in (frames,attack_frames))):
        raise ValueError('Invalid attack-local model inputs')
    network.eval()
    result=[]
    with torch.inference_mode():
        for start in range(0,n,batch_size):
            logits=network(torch.from_numpy(frames[start:start+batch_size]),
                           torch.from_numpy(attack_frames[start:start+batch_size]),
                           torch.from_numpy(((context[start:start+batch_size]-mean)/scale).astype(np.float32)))
            result.append(torch.softmax(logits,dim=-1).numpy())
    return np.concatenate(result) if result else np.empty((0,4),np.float32)
