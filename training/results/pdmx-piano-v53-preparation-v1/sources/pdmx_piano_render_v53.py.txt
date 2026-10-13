"""Whole-work synthesis before cropping; source holds are never retriggered."""


import numpy as np
from offline_piano_v47 import BLOCK, RATE, TAIL, Piano, support_envelope


class WholeWorkPiano(Piano):
    def render(self, events):
        events = np.asarray(events, float)
        if events.shape != (len(events), 4) or not len(events) or not np.isfinite(events).all():
            raise ValueError("Invalid rendered key events")
        clocks = np.rint(events[:, :2]*RATE).astype(np.int64)
        if (np.any(clocks % BLOCK) or np.any(clocks[:, 0] < 0) or np.any(clocks[:, 1] <= clocks[:, 0])
                or clocks[:, 1].max() > 1200*RATE or len(events) > 10000
                or not np.allclose(clocks/RATE, events[:, :2], rtol=0, atol=1e-10)
                or np.any(events[:, 2:] != np.floor(events[:, 2:]))
                or np.any((events[:, 2] < 21) | (events[:, 2] > 108))
                or np.any((events[:, 3] < 1) | (events[:, 3] > 127))):
            raise ValueError("Invalid synthesis-block key clock")
        frames = int(np.ceil((clocks[:, 1].max()+round((TAIL+3)*RATE))/BLOCK))*BLOCK
        wave = np.zeros(frames, np.float32)
        for pitch in np.unique(events[:, 2]):
            selected = events[events[:, 2] == pitch]
            selected = selected[np.argsort(selected[:, 0], kind="stable")]
            if np.any(selected[:-1, 1] > selected[1:, 0]):
                raise ValueError("Ambiguous rendered same-key overlap")
            self.check("fluid_synth_all_sounds_off", self.synth, 0)
            if np.any(self.samples(BLOCK)):
                raise ValueError("Piano voices leaked between independently damped keys")
            commands = sorted([(round(a*RATE), 1, int(v)) for a, b, p, v in selected]
                              +[(round(b*RATE), 0, 0) for a, b, p, v in selected])
            cursor, rendered = 0, np.empty(frames, np.float32)
            for boundary, on, velocity in commands:
                rendered[cursor:boundary] = self.samples(boundary-cursor)
                if on:
                    self.check("fluid_synth_noteon", self.synth, 0, int(pitch), velocity)
                else:
                    self.check("fluid_synth_noteoff", self.synth, 0, int(pitch))
                cursor = boundary
            rendered[cursor:] = self.samples(frames-cursor)
            wave += rendered*support_envelope(selected, frames)
        peak = float(np.max(np.abs(wave)))
        if not np.isfinite(peak) or peak <= 1e-7:
            raise ValueError("Empty sampled piano rendering")
        wave *= min(1., .8/peak)
        reference = events[:, :3].copy()
        support = reference.copy()
        support[:, 0] = np.maximum(0, support[:, 0]-.05)
        support[:, 1] += TAIL+.1
        return wave, reference, support

