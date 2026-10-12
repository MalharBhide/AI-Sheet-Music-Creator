"""Offline sampled piano with explicit synthesis-block key clocks and damping."""

import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np

RATE, BLOCK, TAIL = 22050, 64, .3
FONT = Path("/usr/share/sounds/sf3/MuseScore_General.sf3")
LIBRARY = Path("/usr/lib/aarch64-linux-gnu/libfluidsynth.so.3.1.4")
LICENSE = Path("/usr/share/doc/musescore-general-soundfont-small/copyright")
ASSETS = {FONT: "916aaca6b0eb9f9083eb42399614acb2ff4ec72ce28dffff1814592a9696b479",
    LIBRARY: "f23f44f36b1502f4f74e0b5eb11141c97b8e4af1f10f8293a7b95f83df819f4b",
    LICENSE: "9089e19387d8b508a94f9ea6475ab725376fd15323e51672799b9a5cacfe1671"}


def contracts():
    for path, expected in ASSETS.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Changed offline piano asset or license")
    return {str(path): sha for path, sha in ASSETS.items()}


def key_clock(events, bpm, transpose, velocity_scale):
    events = np.asarray(events, float)
    if (events.ndim != 2 or events.shape[1] != 4 or not len(events)
            or not np.isfinite(events).all() or np.any(events != np.floor(events))
            or np.any(events[:, 0] < 0) or np.any(events[:, 1] <= events[:, 0])
            or not np.isfinite(bpm) or not 60 <= bpm <= 180
            or transpose != int(transpose) or not np.isfinite(velocity_scale) or not .1 <= velocity_scale <= 1
            or np.any((events[:, 2]+transpose < 21) | (events[:, 2]+transpose > 108))
            or np.any((events[:, 3] < 1) | (events[:, 3] > 127))):
        raise ValueError("Invalid authored piano clock")
    intended = 3.+events[:, :2]/96*60/bpm
    frames = (np.ceil(intended*RATE/BLOCK)*BLOCK).astype(np.int64)
    if np.any(frames[:, 1] <= frames[:, 0]) or np.max(frames[:, 1]) > 35*RATE:
        raise ValueError("Unrenderable key duration")
    result = np.column_stack((frames/RATE, events[:, 2]+transpose,
        np.clip(np.rint(events[:, 3]*velocity_scale), 1, 127)))
    for pitch in np.unique(result[:, 2]):
        values = result[result[:, 2] == pitch]
        values = values[np.argsort(values[:, 0], kind="stable")]
        if np.any(values[:-1, 1] > values[1:, 0]):
            raise ValueError("Ambiguous rendered same-key overlap")
    return result, {"synthesis_block_frames": BLOCK, "sample_rate": RATE,
        "maximum_clock_delay_seconds": float(np.max(frames/RATE-intended)),
        "source_tempo_bpm": 120, "render_tempo_bpm": bpm, "transpose_semitones": transpose,
        "velocity_multiplier": velocity_scale, "human_timing_labels": False}


def support_envelope(events, frames):
    """An explicit per-key decay bounds all rendered pitch support."""
    result = np.zeros(frames, np.float32)
    for start, end, _, _ in events:
        first, last = round(start*RATE), round(end*RATE)
        if not 0 <= first < last <= frames:
            raise ValueError("Key outside rendered waveform")
        result[first:last] = 1
        tail_end = min(frames, last+round(TAIL*RATE))
        result[last:tail_end] = np.maximum(result[last:tail_end],
            1-np.arange(tail_end-last, dtype=np.float32)/round(TAIL*RATE))
    return result


class Piano:
    def __init__(self):
        self.contract = contracts()
        self.lib, self.settings, self.synth = ct.CDLL(str(LIBRARY)), None, None
        self.functions = {}
        signatures = {"new_fluid_settings": (ct.c_void_p, []),
            "delete_fluid_settings": (None, [ct.c_void_p]),
            "fluid_settings_setnum": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_double]),
            "fluid_settings_setint": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
            "new_fluid_synth": (ct.c_void_p, [ct.c_void_p]),
            "delete_fluid_synth": (None, [ct.c_void_p]),
            "fluid_synth_sfload": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
            "fluid_synth_program_select": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]),
            "fluid_synth_noteon": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]),
            "fluid_synth_noteoff": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int]),
            "fluid_synth_all_sounds_off": (ct.c_int, [ct.c_void_p, ct.c_int]),
            "fluid_synth_get_internal_bufsize": (ct.c_int, [ct.c_void_p]),
            "fluid_synth_write_float": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_void_p,
                ct.c_int, ct.c_int, ct.c_void_p, ct.c_int, ct.c_int])}
        for name, (restype, args) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = restype, args
            self.functions[name] = function
        try:
            self.settings = self.functions["new_fluid_settings"]()
            if not self.settings:
                raise ValueError("Could not create offline synthesis settings")
            for name, value in ((b"synth.sample-rate", RATE), (b"synth.gain", .6)):
                self.check("fluid_settings_setnum", self.settings, name, value)
            for name, value in ((b"synth.reverb.active", 0), (b"synth.chorus.active", 0),
                                (b"synth.polyphony", 1024), (b"synth.cpu-cores", 1)):
                self.check("fluid_settings_setint", self.settings, name, value)
            self.synth = self.functions["new_fluid_synth"](self.settings)
            if not self.synth or self.functions["fluid_synth_get_internal_bufsize"](self.synth) != BLOCK:
                raise ValueError("Changed synthesis block clock")
            identifier = self.functions["fluid_synth_sfload"](self.synth, str(FONT).encode(), 1)
            if identifier < 0:
                raise ValueError("Could not load licensed piano samples")
            self.check("fluid_synth_program_select", self.synth, 0, identifier, 0, 0)
        except BaseException:
            self.close()
            raise

    def check(self, name, *args):
        if self.functions[name](*args) != 0:
            raise ValueError("Offline synthesis failed: "+name)

    def samples(self, frames):
        if frames < 0 or frames % BLOCK:
            raise ValueError("Synthesis writes must use complete blocks")
        if frames == 0:
            return np.empty(0, np.float32)
        buffer = np.empty((frames, 2), np.float32)
        pointer = buffer.ctypes.data_as(ct.c_void_p)
        self.check("fluid_synth_write_float", self.synth, frames, pointer, 0, 2, pointer, 1, 2)
        wave = buffer.mean(axis=1)
        if not np.isfinite(wave).all():
            raise ValueError("Nonfinite piano samples")
        return wave

    def render(self, events):
        events = np.asarray(events, float)
        if events.shape != (len(events), 4) or not len(events) or not np.isfinite(events).all():
            raise ValueError("Invalid rendered key events")
        clocks = np.rint(events[:, :2]*RATE).astype(np.int64)
        if (np.any(clocks % BLOCK) or np.any(clocks[:, 0] < 0) or np.any(clocks[:, 1] <= clocks[:, 0])
                or clocks[:, 1].max() > 35*RATE or len(events) > 1000
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

    def close(self):
        if self.synth:
            self.functions["delete_fluid_synth"](self.synth)
            self.synth = None
        if self.settings:
            self.functions["delete_fluid_settings"](self.settings)
            self.settings = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
