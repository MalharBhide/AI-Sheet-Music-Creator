"""Author fitting-only NSynth sequences with quiet notes, repeats and overlap.

Short articulations are explicitly authored sample gates, not claimed human key
releases. Their complete 80 ms damping tails remain protected pitch support.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_harmonic import features as context_features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe, sequences
from bass_temporal_data import annotate
from bass_training_data import eligible
from current_bass_v32_baseline import hashes
from periodicity_bass_data import validate
from periodicity_features import features as recurrence
from periodicity_features import observe as recurrence_observe
from prepare_nsynth_bass import sources
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from release_local_features import release_sequences
from release_neighbor_features import features as neighbor_features
from scipy.signal import resample_poly

VERSION = "authored-left-support-v41-v1"
RATE, DURATION = 22050, 24.
KINDS = ("quiet-single", "quiet-overlap", "quiet-chords", "long-overlap",
         "short-repeated", "variable-repeated", "short-chords", "broken-chords")
KEYS = ("events", "base_x", "x", "periodicity", "eligible", "frames",
        "attack_frames", "release_frames", "release_neighbors")


def contracts():
    names = ("prepare_left_support_v41.py", "prepare_nsynth_bass.py", "acquire_nsynth_subset.py",
             "prepare_slakh_bass.py", "prepare_slakh_training.py", "attack_local_features.py",
             "release_local_features.py", "release_neighbor_features.py", "periodicity_features.py")
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def schedule(notes, kind):
    """Deterministic pitch choices use more source samples; never reserved audio."""
    if kind not in KINDS or not notes or any(n["group"] not in ("train", "validation") for n in notes):
        raise ValueError("Illegal fitting articulation or source partition")
    if len({n["instrument_str"] for n in notes}) != 1 or len({n["group"] for n in notes}) != 1:
        raise ValueError("Mixed instrument or partition")
    ordered = sorted(notes, key=lambda n: hashlib.sha256((VERSION+kind+n["id"]).encode()).hexdigest())
    a = ordered[0]
    b = next(n for n in ordered if abs(n["pitch"]-a["pitch"]) >= 3)
    c = next(n for n in ordered if min(abs(n["pitch"]-p["pitch"]) for p in (a, b)) >= 3)
    d = next(n for n in ordered if n["id"] not in {a["id"], b["id"], c["id"]})
    result = []

    def add(start, hold, selected, gains):
        result.extend((start, hold, n, gain) for n, gain in zip(selected, gains, strict=True))

    for j, start in enumerate((3., 7.5, 12., 16.5)):
        if kind == "quiet-single":
            add(start, 3., ([a, b, c, d][j],), (1. if j == 0 else .08,))
        elif kind == "quiet-overlap":
            add(start, 3., (a, b), (1., .08))
            add(start+.75, 3., (c,), (.16,))
        elif kind == "quiet-chords":
            add(start, 3., (a, b, c), (1., .12, .06))
        elif kind == "long-overlap":
            add(start, 3., (a, b), (.4, 1.))
            add(start+1., 3., (c,), (.4,))
        elif kind in ("short-repeated", "variable-repeated"):
            step = .75 if kind == "short-repeated" else (.45, .9, .6, 1.2)[j]
            hold = .4 if kind == "short-repeated" else step*.65
            for k in range(3):
                add(start+k*step, hold, (a,), (1., .12, .4)[k:k+1])
        elif kind == "short-chords":
            add(start, (.4, .8, 1.6, .6)[j], (a, b, c), (1., .3, .12))
        else:
            for k, n in enumerate((a, b, c)):
                add(start+k*.3, .8, (n,), (1.,))
    return result


def render(notes, kind):
    wave = np.zeros(round(DURATION*RATE), np.float32)
    reference, support, used = [], [], []
    for start, hold, note, gain in schedule(notes, kind):
        path = Path(note["audio"])
        if digest(path) != note["audio_sha256"]:
            raise ValueError("Changed NSynth original note")
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16000 or audio.shape != (64000,) or not np.isfinite(audio).all():
            raise ValueError("Invalid NSynth sample clock")
        audio = resample_poly(audio, 441, 320).astype(np.float32)
        if hold < 3.:
            end, tail = round(hold*RATE), round(.08*RATE)
            audio = audio[:end+tail].copy()
            audio[end:] *= np.linspace(1., 0., tail, dtype=np.float32)
            protected_duration = (end+tail)/RATE
        else:
            protected_duration = 4.
        position = round(start*RATE)
        if position+len(audio) > len(wave):
            raise ValueError("Authored support exceeds recording")
        wave[position:position+len(audio)] += gain*audio
        reference.append([start, start+hold, note["pitch"]])
        support.append([start, start+protected_duration, note["pitch"]])
        used.append(note["id"])
    wave *= .8/max(1e-9, float(np.max(np.abs(wave))))
    return wave, reference, support, sorted(set(used))


def prepare(source, output):
    usable = [n for n in sources(source) if n["group"] in ("train", "validation")]
    instruments = sorted({n["instrument_str"] for n in usable})
    if len(instruments) != 14:
        raise ValueError("Incomplete original fitting instruments")
    plan = {"version": VERSION, "source_root": str(source), "source_manifest_sha256": digest(source/"manifest.json"),
        "baseline_hashes": hashes(), "code_sha256": contracts(), "kinds": KINDS,
        "duration": DURATION, "bitrate_cycle": [48, 128, 192], "expected_counts": {"train": 88, "validation": 24},
        "labels": "Original NSynth long key hold 3 s, full 1 s decay protected. Short sample gates use declared durations and complete authored 80 ms damping tails. All mixed pitches protected; no guessed harmony or drum labels.",
        "reserved_instruments_untouched": True, "no_user_audio_or_scores": True}
    if output.exists():
        if json.loads((output/"plan.json").read_text()) != json.loads(json.dumps(plan)) or (output/"manifest.json").exists():
            raise ValueError("Preserve completed or changed preparation")
    else:
        output.mkdir()
        preserve(output/"plan.json", plan)
    engine, captured = _GeneralEngine("balanced"), {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records = []
    for instrument in instruments:
        notes = [n for n in usable if n["instrument_str"] == instrument]
        for index, kind in enumerate(KINDS):
            identifier = "support-v41-"+instrument+"-"+kind
            work = output/identifier
            work.mkdir(exist_ok=True)
            if (work/"record.json").exists():
                record = json.loads((work/"record.json").read_text())
                if (record["plan_sha256"] != digest(output/"plan.json")
                        or digest(work/"features.npz") != record["cache_sha256"]
                        or digest(work/"decoded.wav") != record["audio_sha256"]):
                    raise ValueError("Changed completed fitting observation")
            else:
                wave, reference, support, used = render(notes, kind)
                bitrate = plan["bitrate_cycle"][index % 3]
                wave, clock = encode(wave, RATE, work, bitrate)
                if wave.shape != (round(DURATION*RATE),):
                    raise ValueError("Changed MP3 fitting clock")
                audio = work/"decoded.wav"
                midi = engine.predict(audio, "bass", 120.)
                retained = [n for p in midi.instruments for n in p.notes]
                events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in retained], float).reshape(-1, 4)
                base = note_features(wave, RATE, captured["acoustic"], retained, include_context=True)
                observed = observe(wave, RATE)
                np.savez_compressed(work/"features.npz", events=events, base_x=base,
                    x=context_features(events, base), eligible=eligible(events, DURATION),
                    frames=coarse_sequences(observed["cqt"], events), attack_frames=sequences(observed, events),
                    release_frames=release_sequences(observed, events), release_neighbors=neighbor_features(events, base),
                    periodicity=recurrence(recurrence_observe(wave, RATE), events), plan_sha256=digest(output/"plan.json"))
                record = {"id": identifier, "source_group": "nsynth-"+instrument, "group": notes[0]["group"],
                    "corpus": "nsynth-authored-"+notes[0]["instrument_family_str"], "kind": kind,
                    "audio": str(audio), "audio_sha256": digest(audio), "cache_sha256": digest(work/"features.npz"),
                    "duration": DURATION, "seconds": DURATION, "reference": reference, "pitch_reference": support,
                    "source_notes": used, "candidate_notes": len(events), "codec_clock": clock, "bitrate": bitrate,
                    "plan_sha256": digest(output/"plan.json")}
                preserve(work/"record.json", record)
            records.append(record)
            print(json.dumps({"cached": len(records), "total": 112, "id": identifier, "notes": record["candidate_notes"]}), flush=True)
    if hashes() != plan["baseline_hashes"] or contracts() != plan["code_sha256"]:
        raise ValueError("Changed preparation sources or historical runtime")
    preserve(output/"manifest.json", {"version": VERSION, "items": records, "plan_sha256": digest(output/"plan.json"),
        "reserved_instruments_untouched": True, "no_user_audio_or_scores": True})


def load(folder):
    plan = json.loads((folder/"plan.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    source = Path(plan["source_root"])
    original = {n["id"]: n for n in sources(source) if n["group"] in ("train", "validation")}
    if (plan["version"] != VERSION or manifest["version"] != VERSION or plan["baseline_hashes"] != hashes()
            or plan["code_sha256"] != contracts() or manifest["plan_sha256"] != digest(folder/"plan.json")
            or plan["source_manifest_sha256"] != digest(source/"manifest.json")
            or not manifest["reserved_instruments_untouched"] or not manifest["no_user_audio_or_scores"]
            or len(manifest["items"]) != 112 or len({r["id"] for r in manifest["items"]}) != 112
            or {g: sum(r["group"] == g for r in manifest["items"]) for g in ("train", "validation")} != plan["expected_counts"]):
        raise ValueError("Changed or leaking authored fitting set")
    items = []
    for record in manifest["items"]:
        ns = [original[n] for n in record["source_notes"]]
        instrument = record["source_group"].removeprefix("nsynth-")
        if (not ns or any(n["group"] != record["group"] or n["instrument_str"] != instrument for n in ns)
                or record["kind"] not in KINDS or record["plan_sha256"] != manifest["plan_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]
                or digest(folder/record["id"]/"features.npz") != record["cache_sha256"]):
            raise ValueError("Changed authored fitting source, waveform or cache")
        for n in ns:
            if digest(Path(n["audio"])) != n["audio_sha256"]:
                raise ValueError("Changed original source note")
        with np.load(folder/record["id"]/"features.npz", allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed authored cache identity")
            item = {**record, **{k: saved[k] for k in KEYS}}
        np.testing.assert_array_equal(item["eligible"], eligible(item["events"], DURATION))
        np.testing.assert_array_equal(item["release_neighbors"], neighbor_features(item["events"], item["base_x"]))
        item.update(reference=np.asarray(record["reference"], float).reshape(-1, 3),
                    pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3))
        items.append(annotate(validate(item)))
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    prepare(**vars(parser.parse_args()))
