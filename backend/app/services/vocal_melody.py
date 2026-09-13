"""Load the checked, locally trained melody model for one bounded vocal window."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError

CHECKPOINT = Path(__file__).resolve().parents[1] / 'assets' / 'vocal-melody-v1.pt'
CHECKPOINT_SHA256 = '60209b56b6618335f22d1e4e9fc4604d7a2e501df3242d4a18e3df01d81034f9'


class VocalMelody:
    name = 'Vocadito melody decoder v1'

    def __init__(self):
        import torch

        from app.services.melody_decoder import FEATURE_VERSION, MelodyDecoder

        if not CHECKPOINT.is_file() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise PipelineError('The trained melody checkpoint is missing or damaged. Rebuild the backend and retry.')
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        if checkpoint['feature_version'] != FEATURE_VERSION:
            raise PipelineError('The melody model and audio features have incompatible versions.')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = MelodyDecoder().eval()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.decoder = checkpoint['decoder']

    def predict(self, path: Path, acoustic: dict, bpm: float):
        import pretty_midi
        import torch

        from app.services.melody_decoder import RATE, decode, features

        samples, rate = sf.read(path, dtype='float32')
        if samples.ndim != 1 or rate != 22050 or len(samples) > 33 * rate:
            raise PipelineError('The melody decoder requires a normalized vocal window of at most 33 seconds.')
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        part = pretty_midi.Instrument(program=0, name='Vocals')
        midi.instruments.append(part)
        if not len(samples) or np.max(np.abs(samples)) < 1e-6:
            return midi
        x, _ = features(samples, rate, acoustic)
        with torch.inference_mode():
            logits, attacks = self.model(torch.from_numpy(x)[None])
        events = decode(logits[0].numpy(), torch.sigmoid(attacks[0]).numpy(), **self.decoder)
        duration = len(samples) / rate
        for start, end, pitch in events:
            end = min(float(end), duration)
            if end <= start:
                continue
            # Dynamics are an acoustic proxy, not model certainty. Arrangement
            # balancing subsequently brings the lead above the supporting parts.
            energy = float(np.mean(x[int(start * RATE):max(int(start * RATE) + 1, int(end * RATE)), 0, 7]))
            velocity = int(np.clip(60 + 35 * energy, 50, 100))
            part.notes.append(pretty_midi.Note(velocity, int(pitch), float(start), end))
        return midi
