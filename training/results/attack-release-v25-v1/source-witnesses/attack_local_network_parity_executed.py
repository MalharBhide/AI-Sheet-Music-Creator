"""Three acoustic time scales for supported notes and brief held continuations."""

import numpy as np
from app.services.attack_local_evidence import CHANNELS, STEPS

VERSION = 'attack-release-bass-support-v25-v1'


def model(width=24):
    import torch
    from app.services.temporal_note_model import create_model
    from torch import nn

    if width not in (24, 32):
        raise ValueError('Unsupported attack/release width')

    class AttackReleaseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.coarse = create_model(48).sequence
            self.local = nn.Sequential(nn.Conv1d(CHANNELS,width,5,padding=2),nn.ReLU(),
                nn.Conv1d(width,width,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(8),nn.Flatten())
            self.ending = nn.Sequential(nn.Conv1d(CHANNELS,width,5,padding=2),nn.ReLU(),
                nn.Conv1d(width,width,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(8),nn.Flatten())
            self.classifier = nn.Sequential(nn.Linear(48*8+84+2*width*8,96),nn.ReLU(),nn.Dropout(.1),
                nn.Linear(96,32),nn.ReLU(),nn.Linear(32,2))
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

        def forward(self,frames,attacks,endings,context):
            return self.classifier(torch.cat((self.coarse(frames.transpose(1,2)),context,
                self.local(attacks.transpose(1,2)),self.ending(endings.transpose(1,2))),dim=1))

    return AttackReleaseNet()


def probabilities(network,frames,attacks,endings,context,normalizer,batch_size=256):
    import torch

    frames,attacks,endings,context=[np.asarray(a,np.float32) for a in (frames,attacks,endings,context)]
    mean,scale=[np.asarray(a,np.float32) for a in normalizer]
    n=len(context)
    if (context.shape!=(n,84) or frames.shape!=(n,40,9)
            or attacks.shape!=(n,STEPS,CHANNELS) or endings.shape!=(n,STEPS,CHANNELS)
            or mean.shape!=(84,) or scale.shape!=(84,) or np.any(scale<=0) or batch_size<1
            or any(not np.isfinite(a).all() for a in (frames,attacks,endings,context,mean,scale))
            or any(np.any((a<0)|(a>1)) for a in (frames,attacks,endings))):
        raise ValueError('Invalid attack/release acoustic model inputs')
    result=[]
    network.eval()
    with torch.inference_mode():
        for start in range(0,n,batch_size):
            logits=network(torch.from_numpy(frames[start:start+batch_size]),
                torch.from_numpy(attacks[start:start+batch_size]),
                torch.from_numpy(endings[start:start+batch_size]),
                torch.from_numpy(((context[start:start+batch_size]-mean)/scale).astype(np.float32)))
            result.append(torch.softmax(logits,dim=-1).numpy())
    return np.concatenate(result) if result else np.empty((0,2),np.float32)
