"""Pitch-relative waveform recurrence independent of annotations and detector scores."""

import numpy as np
from scipy.signal import resample_poly

VERSION = "waveform-periodicity-v1"
RATE, FRAME, HOP, MAX_LAG = 8000, 2048, 64, 1024
LAG_RATIOS = (0.5, 1.0, 2.0, 3.0, 2.0 ** (-1.0 / 12.0), 2.0 ** (1.0 / 12.0))
CHANNELS = len(LAG_RATIOS) + 1
POSITIONS = 9
NAMES = tuple(
    f"position_{i}_{name}"
    for i in range(POSITIONS)
    for name in (
        "half_period",
        "period",
        "double_period",
        "triple_period",
        "semitone_shorter",
        "semitone_longer",
        "relative_energy",
    )
)


def observe(samples, rate):
    samples = np.asarray(samples, np.float32)
    if rate != 22050 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Periodicity needs finite mono 22050 Hz audio")
    duration = len(samples) / rate
    wave = resample_poly(samples, 160, 441).astype(np.float64)
    wave = np.pad(wave, (FRAME // 2, FRAME // 2))
    frames = np.lib.stride_tricks.sliding_window_view(wave, FRAME)[::HOP]
    correlations, levels = [], []
    for first in range(0, len(frames), 128):
        block = frames[first : first + 128].copy()
        block -= block.mean(axis=1, keepdims=True)
        # Linear autocorrelation, not circular FFT correlation. Normalization
        # uses both overlapping segment energies at each lag, not total energy.
        transform = np.fft.rfft(block, n=2 * FRAME, axis=1)
        ac = np.fft.irfft(transform * transform.conj(), n=2 * FRAME, axis=1)[:, : MAX_LAG + 1]
        cumulative = np.pad(np.cumsum(block * block, axis=1), ((0, 0), (1, 0)))
        lags = np.arange(MAX_LAG + 1)
        first_energy = cumulative[:, FRAME - lags]
        second_energy = cumulative[:, -1:] - cumulative[:, lags]
        denominator = np.sqrt(first_energy * second_energy)
        ac = np.divide(ac, denominator, out=np.zeros_like(ac), where=denominator > 1e-12)
        correlations.append(np.clip(ac, -1.0, 1.0).astype(np.float32))
        levels.append(np.sqrt(cumulative[:, -1] / FRAME))
    levels = np.concatenate(levels)
    return {
        "duration": duration,
        "correlations": np.concatenate(correlations),
        "energy": (levels / max(float(levels.max(initial=0)), 1e-12)).astype(np.float32),
    }


def features(observed, events):
    events = np.asarray(events, float)
    ac, energy = observed["correlations"], observed["energy"]
    if (
        events.shape != (len(events), 4)
        or not np.isfinite(events).all()
        or np.any(events[:, 0] < 0)
        or np.any(events[:, 1] <= events[:, 0])
        or np.any(events[:, 1] > observed["duration"] + 1e-6)
        or np.any((events[:, 2] < 21) | (events[:, 2] > 108))
        or np.any(events[:, 2] != np.floor(events[:, 2]))
        or ac.ndim != 2
        or ac.shape[1] != MAX_LAG + 1
        or not len(ac)
        or energy.shape != (len(ac),)
        or not np.isfinite(ac).all()
        or not np.isfinite(energy).all()
        or np.any(np.abs(ac) > 1)
        or np.any((energy < 0) | (energy > 1))
    ):
        raise ValueError("Invalid periodicity observation or event clock")
    clock = np.arange(len(ac)) * HOP / RATE
    result = np.zeros((len(events), len(NAMES)), np.float32)
    for index, (start, end, pitch, _) in enumerate(events):
        span = end - start
        times = np.asarray(
            [
                start - 0.08,
                start + 0.08,
                start + 0.2,
                start + 0.25 * span,
                start + 0.5 * span,
                start + 0.75 * span,
                end - 0.08,
                end + 0.08,
                end + 0.2,
            ]
        )
        matrix = np.zeros((POSITIONS, CHANNELS), np.float32)
        period = RATE / (440.0 * 2.0 ** ((pitch - 69.0) / 12.0))
        for column, ratio in enumerate(LAG_RATIOS):
            position = period * ratio
            low = int(np.floor(position))
            if low >= MAX_LAG:
                continue
            fraction = position - low
            envelope = ac[:, low] * (1 - fraction) + ac[:, low + 1] * fraction
            matrix[:, column] = np.interp(times, clock, envelope, left=0.0, right=0.0)
        matrix[:, -1] = np.interp(times, clock, energy, left=0.0, right=0.0)
        result[index] = matrix.ravel()
    return result
