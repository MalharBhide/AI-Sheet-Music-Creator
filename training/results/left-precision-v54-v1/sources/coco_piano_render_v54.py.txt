"""Whole-key rendering with explicit, audited natural sample expiration.

FluidSynth documents that noteoff can fail when no voices match. The inherited
renderer serially renders one pitch, without an audio driver or concurrent DSP.
Accept that result only when the synchronous active voice count is exactly zero.
Authored key duration is not evidence that a decaying sample remains audible.
"""

import ctypes as ct

from pdmx_piano_render_v53 import WholeWorkPiano


class HeldWorkPiano(WholeWorkPiano):
    def __init__(self):
        self.natural_expirations = []
        super().__init__()
        try:
            function = self.lib.fluid_synth_get_active_voice_count
            function.restype, function.argtypes = ct.c_int, [ct.c_void_p]
            self.functions["fluid_synth_get_active_voice_count"] = function
        except BaseException:
            self.close()
            raise

    def check(self, name, *args):
        result = self.functions[name](*args)
        if result == 0:
            return
        if (name == "fluid_synth_noteoff" and result == -1 and len(args) == 3
                and args[0] == self.synth and args[1] == 0
                and 21 <= args[2] <= 108
                and self.functions["fluid_synth_get_active_voice_count"](self.synth) == 0):
            self.natural_expirations.append({"channel": 0, "pitch": int(args[2]),
                                             "active_voices": 0})
            return
        raise ValueError("Offline synthesis failed: " + name)
