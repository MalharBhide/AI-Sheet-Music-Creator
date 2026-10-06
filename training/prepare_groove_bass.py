"""Prepare licensed drum/NSynth mixtures through the actual shipped V25 bass route."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from acquire_groove_subset import SHA as GROOVE_SHA
from app.services.bass_harmonic import features as context_features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe as acoustic_observe
from attack_local_features import sequences as attack_sequences
from bass_training_data import eligible
from current_bass_v25_baseline import hashes
from periodicity_features import features, observe
from prepare_nsynth_bass import KINDS as PITCHED_KINDS
from prepare_nsynth_bass import make_sequence, sources
from prepare_robust_training_stems import digest, preserve
from release_local_features import release_sequences
from release_neighbor_features import features as neighbor_features
from scipy.signal import resample_poly

KINDS = ("drums_only",) + PITCHED_KINDS


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/prepare_groove_bass.py",
        "training/acquire_groove_subset.py",
        "training/prepare_nsynth_bass.py",
        "training/periodicity_features.py",
        "training/attack_local_features.py",
        "training/release_local_features.py",
        "training/release_neighbor_features.py",
    )
    return {name: digest(root / name) for name in names}


def drum_sources(source):
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    witness = json.loads((source / "source.json").read_text())
    rows = manifest["items"]
    if (
        manifest["plan_sha256"] != digest(source / "plan.json")
        or plan["producer_sha256"] != digest(Path(__file__).with_name("acquire_groove_subset.py"))
        or plan["source_sha256"] != digest(source / "source.json")
        or witness["sha256"] != GROOVE_SHA
        or not witness["complete_original_object_verified"]
        or len(rows) != 16
        or len({r["id"] for r in rows}) != 16
        or any(r["split"] == "test" for r in rows)
        or {g: sum(r["group"] == g for r in rows) for g in ("train", "validation", "reserved")}
        != {"train": 9, "validation": 4, "reserved": 3}
    ):
        raise ValueError("Changed original drum source or performer split")
    groups = {
        g: {r["drummer"] for r in rows if r["group"] == g}
        for g in ("train", "validation", "reserved")
    }
    if any(
        groups[a] & groups[b]
        for a, b in (("train", "validation"), ("train", "reserved"), ("validation", "reserved"))
    ):
        raise ValueError("Groove performer leakage")
    return rows


def mixture(drum, notes, kind):
    path = Path(drum["audio"])
    if digest(path) != drum["audio_sha256"]:
        raise ValueError("Changed original drum recording")
    audio, rate = sf.read(path, dtype="float32")
    if (
        rate not in (22050, 44100, 48000)
        or audio.ndim not in (1, 2)
        or not np.isfinite(audio).all()
    ):
        raise ValueError("Invalid original drum waveform")
    if audio.ndim == 2:
        if audio.shape[1] != 2:
            raise ValueError("Unsupported original drum channels")
        audio = audio.mean(axis=1)
    divisor = math.gcd(rate, 22050)
    audio = resample_poly(audio, 22050 // divisor, rate // divisor).astype(np.float32)
    wave = np.zeros(19 * 22050, np.float32)
    length = min(len(wave), len(audio))
    peak = float(np.max(np.abs(audio), initial=0))
    if peak <= 0:
        raise ValueError("Silent original drum recording")
    wave[:length] = 0.35 * audio[:length] / peak
    reference, support, used = [], [], []
    if kind != "drums_only":
        pitched, reference, support, used = make_sequence(notes, kind)
        wave += (0.12 if kind == "quiet" else 0.5) * pitched
    peak = float(np.max(np.abs(wave), initial=0))
    if peak > 0.9:
        wave *= 0.9 / peak
    return wave, reference, support, used


def prepare(source, nsynth, output):
    drums = [r for r in drum_sources(source) if r["group"] in ("train", "validation")]
    items = sources(nsynth)
    # Reserve all four instrument identities and their 24 sequences untouched.
    usable = [i for i in items if i["group"] in ("train", "validation")]
    instruments = sorted({i["instrument_str"] for i in usable})
    if len(instruments) != 14 or len(drums) != 13:
        raise ValueError("Incomplete NSynth fitting instruments")
    plan = {
        "source_root": str(source),
        "nsynth_root": str(nsynth),
        "nsynth_manifest_sha256": digest(nsynth / "manifest.json"),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "kinds": list(KINDS),
        "onsets": [2.75, 7.25, 11.75],
        "key_hold_seconds": 3.0,
        "protected_decay_seconds": 1.0,
        "mp3_bitrates": [48, 128, 192],
        "expected_counts": {"train": 63, "validation": 28},
        "reserved_drum_performers_untouched": True,
        "drum_midi_is_not_piano_pitch_truth": True,
        "rights": "Google LLC NSynth and GMD recordings, CC BY 4.0",
        "reserved_instruments_untouched": True,
        "no_user_audio_or_scores": True,
    }
    if output.exists():
        if (
            json.loads((output / "plan.json").read_text()) != plan
            or (output / "manifest.json").exists()
        ):
            raise ValueError("Preserve changed or completed NSynth candidates")
    else:
        output.mkdir()
        preserve(output / "plan.json", plan)
    engine = _GeneralEngine("balanced")
    records = []
    captured = {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    for source_index, drum in enumerate(drums):
        group = drum["group"]
        choices = sorted({i["instrument_str"] for i in usable if i["group"] == group})
        for index, kind in enumerate(KINDS):
            instrument = choices[(source_index * len(KINDS) + index) % len(choices)]
            notes = [i for i in usable if i["instrument_str"] == instrument]
            identifier = "groove-v25-" + drum["id"] + "-" + kind
            folder = output / identifier
            folder.mkdir(exist_ok=True)
            completion = folder / "record.json"
            if completion.exists():
                record = json.loads(completion.read_text())
                if (
                    record["plan_sha256"] != digest(output / "plan.json")
                    or record["cache_sha256"] != digest(folder / "features.npz")
                    or record["audio_sha256"] != digest(folder / "decoded.wav")
                ):
                    raise ValueError("Changed completed NSynth candidate")
            else:
                wave, reference, pitch_reference, used = mixture(drum, notes, kind)
                sf.write(folder / "source.wav", wave, 22050, subtype="FLOAT")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(folder / "source.wav"),
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        str(plan["mp3_bitrates"][index % 3]) + "k",
                        str(folder / "encoded.mp3"),
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(folder / "encoded.mp3"),
                        "-ar",
                        "22050",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_f32le",
                        str(folder / "decoded.wav"),
                    ],
                    check=True,
                )
                audio = folder / "decoded.wav"
                wave, rate = sf.read(audio, dtype="float32")
                if rate != 22050 or len(wave) != 19 * 22050:
                    raise ValueError("Changed MP3 physical clock")
                midi = engine.predict(audio, "bass", 120.0)
                retained = [n for p in midi.instruments for n in p.notes]
                events = np.asarray(
                    [[n.start, n.end, n.pitch, n.velocity] for n in retained], float
                ).reshape(-1, 4)
                base_x = note_features(
                    wave, rate, captured["acoustic"], retained, include_context=True
                )
                observed = acoustic_observe(wave, rate)
                np.savez_compressed(
                    folder / "features.npz",
                    frames=coarse_sequences(observed["cqt"], events),
                    attack_frames=attack_sequences(observed, events),
                    release_frames=release_sequences(observed, events),
                    release_neighbors=neighbor_features(events, base_x),
                    events=events,
                    base_x=base_x,
                    x=context_features(events, base_x),
                    periodicity=features(observe(wave, rate), events),
                    eligible=eligible(events, 19.0),
                    plan_sha256=digest(output / "plan.json"),
                )
                record = {
                    "id": identifier,
                    "source_group": "groove-" + drum["drummer"] + "/nsynth-" + instrument,
                    "drum_source": drum["id"],
                    "drummer": drum["drummer"],
                    "instrument_str": instrument,
                    "drum_audio_sha256": drum["audio_sha256"],
                    "group": group,
                    "corpus": "groove-"
                    + ("drums-only" if kind == "drums_only" else "pitched-mixture"),
                    "audio": str(audio),
                    "audio_sha256": digest(audio),
                    "cache_sha256": digest(folder / "features.npz"),
                    "duration": 19.0,
                    "seconds": 19.0,
                    "reference": reference,
                    "pitch_reference": pitch_reference,
                    "source_notes": used,
                    "plan_sha256": digest(output / "plan.json"),
                    "candidate_notes": len(events),
                    "kind": kind,
                    "bitrate": plan["mp3_bitrates"][index % 3],
                }
                preserve(completion, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "cached": len(records),
                        "total": 91,
                        "id": identifier,
                        "notes": record["candidate_notes"],
                    }
                ),
                flush=True,
            )
    if plan["baseline_hashes"] != hashes() or plan["code_sha256"] != contracts():
        raise ValueError("Groove mixture pipeline changed during inference")
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "reserved_instruments_untouched": True,
            "reserved_drum_performers_untouched": True,
            "no_user_audio_or_scores": True,
        },
    )
    print(
        json.dumps(
            {
                "passes": True,
                "recordings": len(records),
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("nsynth", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.nsynth, args.output)
