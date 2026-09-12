"""Demucs four-source separation for one bounded window at a time."""

from pathlib import Path

import numpy as np
import soundfile as sf

from app.models import PipelineError


class StemSeparator:
    def __init__(self):
        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise PipelineError("Full-song transcription needs Demucs and PyTorch. Install the full transcription dependencies, or choose a single-instrument mode.") from exc
        try:
            self.model = get_model(name="htdemucs")
            self.model.eval()
            torch.set_num_threads(min(4, torch.get_num_threads()))
        except Exception as exc:
            raise PipelineError("The full-song separation model could not be loaded. Check the server's model cache and network connection, then retry.") from exc

    def separate(self, audio_path: Path, directory: Path) -> dict[str, Path]:
        import torch
        import torchaudio.functional as audio_functions
        from demucs.apply import apply_model

        samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
        if waveform.shape[0] == 1:
            waveform = waveform.expand(self.model.audio_channels, -1).contiguous()
        waveform = audio_functions.resample(waveform, rate, self.model.samplerate)
        deviation = float(waveform.std())
        if not np.isfinite(deviation) or deviation < 1e-7:
            return {}
        mean = waveform.mean()
        normalized = (waveform - mean) / deviation
        try:
            with torch.inference_mode():
                separated = apply_model(self.model, normalized[None], device="cpu",
                                        shifts=0, split=True, overlap=0.25,
                                        segment=7.8, progress=False, num_workers=0)[0]
        except Exception as exc:
            raise PipelineError("Source separation failed. Check the server worker log and available memory, then retry.") from exc
        paths = {}
        mixture_rms = float(np.sqrt(np.mean(samples ** 2)))
        for name, waveform in zip(self.model.sources, separated, strict=True):
            if name == "drums":
                continue
            waveform = waveform * deviation + mean
            mono = audio_functions.resample(waveform.mean(dim=0), self.model.samplerate,
                                             22050).cpu().numpy()
            if float(np.sqrt(np.mean(mono ** 2))) < max(1e-5, mixture_rms * 0.012):
                continue
            path = directory / f"{name}.wav"
            sf.write(str(path), mono, 22050, subtype="FLOAT")
            paths[name] = path
        return paths
