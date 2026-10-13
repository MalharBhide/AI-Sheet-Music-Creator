"""Accept only neutral tick-zero piano resets, preserving every musical event."""

import io

import mido
from pdmx_piano_clock_v53 import notes as strict_notes


def notes(data):
    midi = mido.MidiFile(file=io.BytesIO(data))
    for track in midi.tracks:
        tick, kept = 0, []
        for message in track:
            if not isinstance(message.time, int) or message.time < 0:
                raise ValueError("Invalid symbolic delta clock")
            tick += message.time
            if message.type == "pitchwheel":
                if tick != 0 or message.pitch != 0 or message.channel == 9:
                    raise ValueError("Nonneutral or late source pitchwheel")
                # A neutral zero-tick reset cannot change a fresh synth's pitch.
                # Removing it does not alter any subsequent delta or absolute time.
                continue
            if message.type == "program_change" and tick == 0:
                if message.program != 0 or message.channel == 9:
                    raise ValueError("Unsupported initial piano program")
                # MIDI channels start with program 0; repeated initial program-0
                # messages are redundant even when tracks interleave at tick 0.
                continue
            kept.append(message)
        track[:] = kept
    output = io.BytesIO()
    midi.save(file=output)
    return strict_notes(output.getvalue())
