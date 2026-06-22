from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

from app.config import get_settings


class TranscriptionError(RuntimeError):
    pass


def transcribe_audio_to_piano_midi(audio_path: Path, midi_path: Path) -> Path:
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="basic_pitch_") as temp_dir:
        output_dir = Path(temp_dir)
        try:
            _run_basic_pitch_python_api(audio_path, output_dir)
        except Exception as api_error:
            try:
                _run_basic_pitch_cli(audio_path, output_dir)
            except Exception as cli_error:
                raise TranscriptionError(
                    "Basic Pitch could not transcribe the audio. "
                    f"Python API error: {api_error}. CLI error: {cli_error}."
                ) from cli_error

        generated_midi = _latest_midi_file(output_dir)
        if generated_midi is None:
            raise TranscriptionError("Basic Pitch finished but did not produce a MIDI file.")

        shutil.copyfile(generated_midi, midi_path)

    _force_piano_program(midi_path)
    return midi_path


def _run_basic_pitch_python_api(audio_path: Path, output_dir: Path) -> None:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    predict_and_save(
        [str(audio_path)],
        str(output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )


def _run_basic_pitch_cli(audio_path: Path, output_dir: Path) -> None:
    settings = get_settings()
    binary = shutil.which(settings.basic_pitch_bin)
    if not binary:
        raise TranscriptionError(
            f"Could not find '{settings.basic_pitch_bin}'. Install basic-pitch or set BASIC_PITCH_BIN."
        )

    result = subprocess.run(
        [binary, str(output_dir), str(audio_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise TranscriptionError(detail)


def _latest_midi_file(output_dir: Path) -> Path | None:
    midi_files = list(output_dir.rglob("*.mid")) + list(output_dir.rglob("*.midi"))
    if not midi_files:
        return None
    return max(midi_files, key=lambda path: path.stat().st_mtime)


def _force_piano_program(midi_path: Path) -> None:
    try:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in midi.instruments:
            instrument.program = 0
            instrument.is_drum = False
            instrument.notes = [
                note for note in instrument.notes if 21 <= note.pitch <= 108
            ]
        midi.write(str(midi_path))
    except Exception:
        return

