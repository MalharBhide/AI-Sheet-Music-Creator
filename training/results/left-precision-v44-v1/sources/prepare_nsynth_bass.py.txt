"""Cache new labeled NSynth instrument sequences through the actual shipped V25 route."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_harmonic import features as context_features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from bass_training_data import eligible
from current_bass_v25_baseline import hashes
from periodicity_features import features, observe
from prepare_robust_training_stems import digest, preserve
from scipy.signal import resample_poly

KINDS = ("single", "repeat", "octave", "dyad", "quiet", "layered")


def contracts():
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/prepare_nsynth_bass.py",
        "training/acquire_nsynth_subset.py",
        "training/periodicity_features.py",
    )
    return {name: digest(root / name) for name in names}


def sources(folder):
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["plan_sha256"] != digest(folder / "plan.json")
        or plan["producer_sha256"] != digest(Path(__file__).with_name("acquire_nsynth_subset.py"))
        or manifest["instrument_counts"] != {"train": 11, "validation": 3, "reserved": 4}
        or len(manifest["items"]) != 497
        or not manifest["no_decoder_inference"]
        or not plan["no_user_audio_or_scores"]
    ):
        raise ValueError("Changed NSynth instrument split or acquisition source")
    partitions = {
        g: {i["instrument_str"] for i in manifest["items"] if i["group"] == g}
        for g in ("train", "validation", "reserved")
    }
    if any(
        partitions[a] & partitions[b]
        for a, b in (("train", "validation"), ("train", "reserved"), ("validation", "reserved"))
    ):
        raise ValueError("NSynth instrument leakage")
    return manifest["items"]


def make_sequence(items, kind):
    ordered = sorted(items, key=lambda i: hashlib.sha256(i["id"].encode()).hexdigest())
    by_pitch = {i["pitch"]: i for i in ordered}
    first = ordered[0]
    other = next(i for i in ordered if abs(i["pitch"] - first["pitch"]) >= 3)
    octave = by_pitch.get(first["pitch"] + 12) or by_pitch.get(first["pitch"] - 12)
    if octave is None:
        pair = next(
            ((i, by_pitch[i["pitch"] + 12]) for i in ordered if i["pitch"] + 12 in by_pitch), None
        )
        if pair is not None:
            first, octave = pair
    if kind == "octave" and octave is None:
        raise ValueError("Selected NSynth instrument lacks a played-octave pair")
    other = next(i for i in ordered if abs(i["pitch"] - first["pitch"]) >= 3)
    choices = {
        "single": [[first], [other], [ordered[2]]],
        "repeat": [[first]] * 3,
        "octave": [[first, octave]] * 3,
        "dyad": [[first, other]] * 3,
        "quiet": [[first], [other], [first]],
        "layered": [[first, other], [other], [first, other]],
    }[kind]
    waveform = np.zeros(19 * 22050, np.float32)
    reference = []
    pitch_reference = []
    used = []
    for start, notes in zip((2.75, 7.25, 11.75), choices, strict=True):
        for note in notes:
            path = Path(note["audio"])
            if digest(path) != note["audio_sha256"]:
                raise ValueError("Changed NSynth note bytes")
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000 or audio.shape != (64000,) or not np.isfinite(audio).all():
                raise ValueError("Invalid original NSynth note clock")
            audio = resample_poly(audio, 441, 320).astype(np.float32)
            position = round(start * 22050)
            gain = 0.12 if kind == "quiet" and note is first else 1.0
            waveform[position : position + len(audio)] += gain * audio
            reference.append([start, start + 3.0, note["pitch"]])
            # Dataset key release is at 3 s; its full 1 s decay tail remains
            # protected pitch support. Do not teach the filter to remove decay.
            pitch_reference.append([start, start + 4.0, note["pitch"]])
            used.append(note["id"])
    peak = float(np.max(np.abs(waveform)))
    if peak <= 0:
        raise ValueError("Silent selected NSynth source")
    waveform *= 0.8 / peak
    return waveform, reference, pitch_reference, sorted(set(used))


def prepare(source, output):
    items = sources(source)
    # Reserve all four instrument identities and their 24 sequences untouched.
    usable = [i for i in items if i["group"] in ("train", "validation")]
    instruments = sorted({i["instrument_str"] for i in usable})
    if len(instruments) != 14:
        raise ValueError("Incomplete NSynth fitting instruments")
    plan = {
        "source_root": str(source),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "baseline_hashes": hashes(),
        "code_sha256": contracts(),
        "kinds": list(KINDS),
        "onsets": [2.75, 7.25, 11.75],
        "key_hold_seconds": 3.0,
        "protected_decay_seconds": 1.0,
        "mp3_bitrates": [48, 128, 192],
        "expected_counts": {"train": 66, "validation": 18},
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
    for instrument in instruments:
        notes = [i for i in usable if i["instrument_str"] == instrument]
        group = notes[0]["group"]
        for index, kind in enumerate(KINDS):
            identifier = "nsynth-v25-" + instrument + "-" + kind
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
                wave, reference, pitch_reference, used = make_sequence(notes, kind)
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
                np.savez_compressed(
                    folder / "features.npz",
                    events=events,
                    base_x=base_x,
                    x=context_features(events, base_x),
                    periodicity=features(observe(wave, rate), events),
                    eligible=eligible(events, 19.0),
                    plan_sha256=digest(output / "plan.json"),
                )
                record = {
                    "id": identifier,
                    "source_group": "nsynth-" + instrument,
                    "group": group,
                    "corpus": "nsynth-" + notes[0]["instrument_family_str"],
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
                        "total": 84,
                        "id": identifier,
                        "notes": record["candidate_notes"],
                    }
                ),
                flush=True,
            )
    if plan["baseline_hashes"] != hashes() or plan["code_sha256"] != contracts():
        raise ValueError("NSynth pipeline changed during inference")
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "reserved_instruments_untouched": True,
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
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
