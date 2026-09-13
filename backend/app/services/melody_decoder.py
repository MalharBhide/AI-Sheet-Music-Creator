"""Supervised, pitch-equivariant melody decoding from continuous vocal evidence."""

import numpy as np
import torch
from torch import nn

RATE = 50
FEATURE_VERSION = 'vocadito-melody-v1'
CHANNELS = 10


def features(samples, rate, acoustic):
    """Return continuous acoustic evidence on a common 20 ms clock."""
    import librosa
    from basic_pitch.note_creation import model_frames_to_time

    if samples.ndim != 1 or rate != 22050 or not np.isfinite(samples).all():
        raise ValueError('Melody features require finite mono samples at 22050 Hz')
    times = np.arange(int(np.ceil(len(samples) / rate * RATE))) / RATE
    if not len(samples) or np.max(np.abs(samples)) < 1e-6:
        frames = 1 + len(samples) // 256
        return np.zeros((len(times), 88, CHANNELS), dtype=np.float32), (
            np.full(frames, np.nan), np.zeros(frames), np.zeros(frames))
    clock = model_frames_to_time(len(acoustic['note']))
    def interpolate(values, source):
        return np.stack([
            np.interp(times, source, column, left=0, right=0) for column in values.T
        ], axis=1)
    note = interpolate(acoustic['note'], clock)
    onset = interpolate(acoustic['onset'], clock)
    f0, voiced, probability = librosa.pyin(
        samples, sr=rate, fmin=float(librosa.midi_to_hz(45)),
        fmax=float(librosa.midi_to_hz(96)), frame_length=2048,
        hop_length=256, resolution=.2)
    pitches = librosa.hz_to_midi(f0)
    confidence = np.where(voiced, probability, 0)
    pyin_clock = np.arange(len(pitches)) * 256 / rate
    distance = (pitches[:, None] - np.arange(21, 109)[None, :]) / .65
    pitch_evidence = np.nan_to_num(np.exp(-.5 * distance ** 2)) * confidence[:, None]
    pyin = interpolate(pitch_evidence, pyin_clock)
    rms = librosa.feature.rms(y=samples, frame_length=512, hop_length=256)[0]
    energy = np.clip((librosa.amplitude_to_db(rms, ref=np.max) + 60) / 60, 0, 1)
    cqt = np.abs(librosa.cqt(samples, sr=rate, hop_length=256,
                            fmin=float(librosa.midi_to_hz(21)), n_bins=88)).T
    cqt = interpolate(np.clip((librosa.amplitude_to_db(cqt, ref=np.max) + 60) / 60, 0, 1),
                      np.arange(len(cqt)) * 256 / rate)
    below = np.pad(note[:, :-12], ((0, 0), (12, 0)))
    above = np.pad(note[:, 12:], ((0, 0), (0, 12)))
    harmonic = np.pad(cqt[:, 12:], ((0, 0), (0, 12)))
    def broadcast(values):
        return np.broadcast_to(values[:, None], note.shape)
    x = np.stack([note, onset, below, above, pyin, cqt, harmonic,
                  broadcast(np.interp(times, np.arange(len(rms)) * 256 / rate, energy)),
                  broadcast(note.max(axis=1)),
                  broadcast(np.interp(times, pyin_clock, confidence))], axis=-1)
    return x.astype(np.float32), (pitches, confidence, rms)


class MelodyDecoder(nn.Module):
    """Share the same temporal classifier across all 88 piano pitches."""

    def __init__(self):
        super().__init__()
        self.pitch = nn.Sequential(
            nn.Conv1d(CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv1d(16, 2, 1))
        self.silence = nn.Sequential(nn.Conv1d(3, 8, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(8, 1, 1))

    def forward(self, x):
        batch, time, pitch, channels = x.shape
        shared = x.permute(0, 2, 3, 1).reshape(batch * pitch, channels, time)
        result = self.pitch(shared).reshape(batch, pitch, 2, time)
        silence = self.silence(x[:, :, 0, 7:10].transpose(1, 2))
        logits = torch.cat([result[:, :, 0], silence], dim=1).transpose(1, 2)
        return logits, result[:, :, 1].transpose(1, 2)


def labels(notes, length):
    """Annotations contain onset seconds, frequency Hz, duration seconds."""
    import librosa

    y = np.full(length, 88, dtype=np.int64)
    attacks = np.zeros((length, 88), dtype=np.float32)
    times = np.arange(length) / RATE
    for onset, hz, duration in notes:
        pitch = int(np.rint(librosa.hz_to_midi(hz))) - 21
        if not 0 <= pitch < 88:
            continue
        y[(times >= onset) & (times < onset + duration)] = pitch
        frame = round(onset * RATE)
        attacks[max(0, frame - 1):min(length, frame + 2), pitch] = 1
    return y, attacks


def decode(logits, attacks, *, minimum=.08, onset_threshold=.5, smoothing=3):
    """Decode one monophonic line; never fill a gap without pitch evidence."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks

    if not len(logits):
        return np.empty((0, 3), dtype=float)
    classes = np.argmax(logits, axis=1)
    if smoothing > 1:
        # Median of categorical pitch IDs would manufacture intermediate notes.
        # Smooth each class score before choosing a class instead.
        classes = median_filter(logits, size=(smoothing, 1), mode='nearest').argmax(axis=1)
    edges = np.r_[0, np.flatnonzero(np.diff(classes)) + 1, len(classes)]
    notes = []
    for first, end in zip(edges[:-1], edges[1:], strict=True):
        pitch = classes[first]
        if pitch == 88:
            continue
        peaks, _ = find_peaks(attacks[first:end, pitch], height=onset_threshold,
                              distance=max(1, round(minimum * RATE)))
        boundaries = [first]
        for peak in peaks + first:
            if peak - boundaries[-1] >= minimum * RATE and end - peak >= minimum * RATE:
                boundaries.append(int(peak))
        boundaries.append(end)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            if (stop - start) / RATE >= minimum:
                notes.append([start / RATE, stop / RATE, int(pitch + 21)])
    return np.asarray(notes, dtype=float).reshape(-1, 3)
