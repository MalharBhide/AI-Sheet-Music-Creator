"""Note-centred spectral sequences for a supervised candidate KEEP classifier.

This module does not change production routing. Input channels are observed
pitch-relative acoustics, never reference labels or fitted confidences.
"""

import numpy as np

VERSION = 'bass-v14-temporal-cqt-v1'
RATE = 22050
HOP = 256
OFFSETS = (0, -1, 1, -12, -19, -24, 12, 19, 24)
STEPS = 40
FEATURE_COUNT = 84


def spectrum(samples, rate):
    import librosa

    samples = np.asarray(samples, np.float32)
    if rate != RATE or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Temporal evidence needs finite mono 22050 Hz audio')
    if len(samples) < rate:
        samples = np.pad(samples, (0, rate - len(samples)))
    magnitude = np.abs(librosa.cqt(samples, sr=rate, hop_length=HOP,
                                  fmin=float(librosa.midi_to_hz(21)), n_bins=88)).T
    peak = float(np.max(magnitude, initial=0.))
    if peak <= 1e-12:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip((20 * np.log10(np.maximum(magnitude, 1e-12) / peak) + 80) / 80,
                   0., 1.).astype(np.float32)


def sequences(observed, events):
    observed, events = np.asarray(observed, np.float32), np.asarray(events, float)
    if (observed.ndim != 2 or observed.shape[1] != 88 or not len(observed)
            or not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1))
            or events.shape != (len(events), 4) or not np.isfinite(events).all()
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or np.any((events[:, 2] < 21) | (events[:, 2] > 108))
            or np.any(events[:, 2] != np.floor(events[:, 2]))):
        raise ValueError('Invalid temporal sequence evidence')
    result = np.zeros((len(events), STEPS, len(OFFSETS)), np.float32)
    clock = np.arange(len(observed)) * HOP / RATE
    for index, (start, end, pitch, _) in enumerate(events):
        times = np.concatenate((start - np.linspace(.24, .03, 8),
                                np.linspace(start, end, 24), end + np.linspace(.03, .24, 8)))
        for channel, offset in enumerate(OFFSETS):
            bin_index = int(pitch) - 21 + offset
            if 0 <= bin_index < 88:
                result[index, :, channel] = np.interp(times, clock, observed[:, bin_index])
    return result


def create_model(width=32):
    import torch
    from torch import nn

    class TemporalNoteNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence = nn.Sequential(nn.Conv1d(len(OFFSETS), width, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
                                          nn.AdaptiveAvgPool1d(8), nn.Flatten())
            self.classifier = nn.Sequential(nn.Linear(width * 8 + FEATURE_COUNT, 96), nn.ReLU(),
                                            nn.Dropout(.1), nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, frames, context):
            return self.classifier(torch.cat((self.sequence(frames.transpose(1, 2)), context), dim=1)).squeeze(1)

    return TemporalNoteNet()


def probability(model, frames, context, normalizer, batch_size=256):
    import torch

    frames, context = np.asarray(frames, np.float32), np.asarray(context, np.float32)
    mean, scale = [np.asarray(a, np.float32) for a in normalizer]
    if (frames.shape != (len(context), STEPS, len(OFFSETS)) or context.shape != (len(context), FEATURE_COUNT)
            or mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,)
            or not np.isfinite(frames).all() or not np.isfinite(context).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
            or batch_size < 1):
        raise ValueError('Invalid temporal model inputs')
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            stop = start + batch_size
            x = (context[start:stop] - mean) / scale
            result.append(torch.sigmoid(model(torch.from_numpy(frames[start:stop]), torch.from_numpy(x))).numpy())
    return np.concatenate(result) if result else np.empty(0, np.float32)
