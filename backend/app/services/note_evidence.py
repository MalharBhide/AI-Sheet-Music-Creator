"""Pitch-relative acoustic evidence for verifying candidate accompaniment notes."""

import numpy as np

FEATURE_VERSION = 'note-evidence-v1'
FEATURE_NAMES = (
    'duration', 'velocity', 'note_mean', 'note_max', 'note_q20', 'note_q80',
    'note_occupancy', 'onset_at_attack', 'onset_max', 'onset_mean',
    'octave_below', 'octave_above', 'fifth_below', 'fifth_above',
    'nineteen_below', 'nineteen_above', 'relative_note', 'polyphony',
    'cqt_mean', 'cqt_max', 'cqt_attack_rise', 'cqt_semitone_below',
    'cqt_semitone_above', 'cqt_octave_below', 'cqt_octave_above', 'cqt_nineteen_below',
)


def note_features(samples, rate, acoustic, notes):
    """No filename, key signature or absolute pitch is a classifier feature."""
    if not notes:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    import librosa
    from basic_pitch.note_creation import model_frames_to_time

    if rate != 22050 or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('Note verification requires finite mono 22050 Hz audio')
    evidence, onset = np.asarray(acoustic['note']), np.asarray(acoustic['onset'])
    if evidence.ndim != 2 or evidence.shape[1] != 88 or onset.shape != evidence.shape or not len(evidence):
        raise ValueError('Invalid acoustic evidence shape')
    clock = model_frames_to_time(len(evidence))
    spectrum = np.abs(librosa.cqt(samples, sr=rate, hop_length=256,
                                 fmin=float(librosa.midi_to_hz(21)), n_bins=88)).T
    spectrum = np.clip((librosa.amplitude_to_db(spectrum, ref=np.max) + 80) / 80, 0, 1)
    cqt_clock = np.arange(len(spectrum)) * 256 / rate

    def span(clock, start, end):
        first = min(len(clock) - 1, max(0, int(np.searchsorted(clock, start))))
        stop = min(len(clock), max(first + 1, int(np.searchsorted(clock, end))))
        return slice(first, stop)

    def column(matrix, pitch):
        return matrix[:, pitch] if 0 <= pitch < 88 else np.zeros(len(matrix))

    rows = []
    for item in notes:
        pitch = int(item.pitch) - 21
        if not 0 <= pitch < 88 or not 0 <= item.start < item.end:
            raise ValueError('Invalid candidate note')
        region = span(clock, item.start, item.end)
        frames, attacks = evidence[region], onset[region, pitch]
        values = frames[:, pitch]
        attack = onset[span(clock, max(0, item.start - .05), item.start + .08), pitch]
        cq = spectrum[span(cqt_clock, item.start, item.end)]
        cq_start = column(spectrum[span(cqt_clock, item.start, item.start + .08)], pitch).mean()
        cq_before = column(spectrum[span(cqt_clock, max(0, item.start - .08), item.start)], pitch).mean()
        row = [np.log1p(min(item.end - item.start, 8)) / np.log(9), item.velocity / 127,
               values.mean(), values.max(), np.quantile(values, .2), np.quantile(values, .8),
               np.mean(values >= .3), attack.max(), attacks.max(), attacks.mean()]
        row.extend(column(frames, pitch + interval).mean() for interval in (-12, 12, -7, 7, -19, 19))
        row.extend([np.mean(values / np.maximum(.01, frames.max(axis=1))),
                    np.mean(np.sum(frames >= .3, axis=1)) / 12,
                    cq[:, pitch].mean(), cq[:, pitch].max(), cq_start - cq_before])
        row.extend(column(cq, pitch + interval).mean() for interval in (-1, 1, -12, 12, -19))
        rows.append(row)
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite note evidence')
    return result
