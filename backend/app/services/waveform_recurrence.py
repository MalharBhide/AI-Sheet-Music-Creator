"""Compute identical recurrence features without a duration-sized correlation table."""

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


def features(samples, rate, events, batch_size=64):
    samples, events = np.asarray(samples, np.float32), np.asarray(events, float)
    if (
        rate != 22050
        or samples.ndim != 1
        or not len(samples)
        or not np.isfinite(samples).all()
        or events.shape != (len(events), 4)
        or not np.isfinite(events).all()
        or np.any(events[:, 0] < 0)
        or np.any(events[:, 1] <= events[:, 0])
        or np.any(events[:, 1] > len(samples) / rate + 1e-6)
        or np.any((events[:, 2] < 21) | (events[:, 2] > 108))
        or np.any(events[:, 2] != np.floor(events[:, 2]))
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 256
    ):
        raise ValueError("Invalid bounded recurrence waveform or event clock")
    result = np.zeros((len(events), len(NAMES)), np.float32)
    if not len(events):
        return result
    wave = resample_poly(samples, 160, 441).astype(np.float64)
    wave = np.pad(wave, (FRAME // 2, FRAME // 2))
    frames = np.lib.stride_tricks.sliding_window_view(wave, FRAME)[::HOP]
    peak = 0.0
    for first in range(0, len(frames), 128):
        block = frames[first : first + 128].copy()
        block -= block.mean(axis=1, keepdims=True)
        # Match the fitting observer's accumulation and normalization exactly.
        energy = np.sqrt(np.cumsum(block * block, axis=1)[:, -1] / FRAME)
        peak = max(peak, float(energy.max(initial=0)))
    peak = max(peak, 1e-12)
    last_clock = (len(frames) - 1) * HOP / RATE
    for first in range(0, len(events), batch_size):
        selected = events[first : first + batch_size]
        starts, ends, pitches, _ = selected.T
        spans = ends - starts
        times = np.column_stack(
            (
                starts - 0.08,
                starts + 0.08,
                starts + 0.2,
                starts + 0.25 * spans,
                starts + 0.5 * spans,
                starts + 0.75 * spans,
                ends - 0.08,
                ends + 0.08,
                ends + 0.2,
            )
        )
        valid = (times >= 0) & (times <= last_clock)
        positions = np.clip(times * RATE / HOP, 0, len(frames) - 1)
        low = np.floor(positions).astype(int)
        high = np.minimum(low + 1, len(frames) - 1)
        indices = np.unique(np.concatenate((low.ravel(), high.ravel())))
        lookup = {int(index): offset for offset, index in enumerate(indices)}
        correlations, levels = [], []
        for offset in range(0, len(indices), 128):
            block = frames[indices[offset : offset + 128]].copy()
            block -= block.mean(axis=1, keepdims=True)
            transform = np.fft.rfft(block, n=2 * FRAME, axis=1)
            ac = np.fft.irfft(transform * transform.conj(), n=2 * FRAME, axis=1)[:, : MAX_LAG + 1]
            cumulative = np.pad(np.cumsum(block * block, axis=1), ((0, 0), (1, 0)))
            lags = np.arange(MAX_LAG + 1)
            denominator = np.sqrt(
                cumulative[:, FRAME - lags] * (cumulative[:, -1:] - cumulative[:, lags])
            )
            ac = np.divide(ac, denominator, out=np.zeros_like(ac), where=denominator > 1e-12)
            correlations.append(np.clip(ac, -1, 1).astype(np.float32))
            levels.append((np.sqrt(cumulative[:, -1] / FRAME) / peak).astype(np.float32))
        ac, energy = np.concatenate(correlations), np.concatenate(levels)
        for index, pitch in enumerate(pitches):
            rows_low = np.array([lookup[int(v)] for v in low[index]])
            rows_high = np.array([lookup[int(v)] for v in high[index]])
            # Use the actual fitting clock coordinates rather than assuming
            # floating-point division gives exactly identical interpolation.
            left_clock, right_clock = low[index] * HOP / RATE, high[index] * HOP / RATE
            fractions = np.divide(
                times[index] - left_clock,
                right_clock - left_clock,
                out=np.zeros(9),
                where=right_clock > left_clock,
            )
            matrix = np.zeros((9, 7), np.float32)
            period = RATE / (440.0 * 2.0 ** ((pitch - 69.0) / 12.0))
            for column, ratio in enumerate(LAG_RATIOS):
                lag = period * ratio
                bottom = int(np.floor(lag))
                if bottom >= MAX_LAG:
                    continue
                fraction = lag - bottom
                # The original observer forms float32 lag envelopes before
                # float64 clock interpolation. Preserve that rounding order.
                envelope_low = (
                    ac[rows_low, bottom] * (1 - fraction) + ac[rows_low, bottom + 1] * fraction
                )
                envelope_high = (
                    ac[rows_high, bottom] * (1 - fraction) + ac[rows_high, bottom + 1] * fraction
                )
                matrix[:, column] = np.where(
                    valid[index], envelope_low * (1 - fractions) + envelope_high * fractions, 0
                )
            matrix[:, -1] = np.where(
                valid[index], energy[rows_low] * (1 - fractions) + energy[rows_high] * fractions, 0
            )
            result[first + index] = matrix.ravel()
    return result
