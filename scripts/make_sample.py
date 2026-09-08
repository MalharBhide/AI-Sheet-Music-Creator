"""Generate a short original test melody, using only the Python standard library."""
import argparse
import math
import struct
import wave
from pathlib import Path


def make_sample(path: Path) -> None:
    rate = 22050
    samples = []
    # C major arpeggio: a regression fixture, not a recording of a published work.
    for midi in [60, 64, 67, 72, 67, 64, 60, 60]:
        frequency = 440 * 2 ** ((midi - 69) / 12)
        for index in range(int(rate * 0.5)):
            t = index / rate
            envelope = min(1, t / 0.01) * min(1, max(0, (0.46 - t) / 0.04)) * math.exp(-2 * t)
            value = sum(math.sin(2 * math.pi * frequency * harmonic * t) / harmonic ** 2
                        for harmonic in range(1, 5))
            samples.append(struct.pack('<h', int(13000 * envelope * value)))
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b''.join(samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', nargs='?', default='sample.wav')
    args = parser.parse_args()
    make_sample(Path(args.output))
