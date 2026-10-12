"""Measure original new-velocity MP3 phrases with declared holds and repeats."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import prepare_left_support_v41 as support
from acquire_nsynth_dynamics_v44 import VELOCITIES, sources
from prepare_robust_training_stems import digest, preserve

VERSION = "authored-nsynth-dynamics-v44-v1"
COUNTS = {"train": 352, "validation": 96}
KINDS, RATE, DURATION = support.KINDS, support.RATE, support.DURATION


def contracts():
    root = Path(__file__).resolve().parent
    return {**support.contracts(), **{name: digest(root/name) for name in (
        "acquire_nsynth_dynamics_v44.py", "prepare_nsynth_dynamics_v44.py")}}


def tasks(notes):
    instruments = sorted({n["instrument_str"] for n in notes})
    return [(instrument, velocity, kind) for instrument in instruments
            for velocity in VELOCITIES for kind in KINDS]


def identity(instrument, velocity, kind):
    return f"dynamics-v44-{instrument}-{velocity:03d}-{kind}"


def labels(notes, kind):
    schedule = support.schedule(notes, kind)
    reference = [[start, start+hold, note["pitch"]] for start, hold, note, _ in schedule]
    pitch = [[start, start+(4. if hold == 3. else (round(hold*RATE)+round(.08*RATE))/RATE), note["pitch"]]
             for start, hold, note, _ in schedule]
    return reference, pitch, sorted({n["id"] for _, _, n, _ in schedule})


def prepare(source, output):
    notes = sources(source)
    jobs = tasks(notes)
    if len(jobs) != 448:
        raise ValueError("Incomplete dynamics tasks")
    plan = {"version": VERSION, "source_root": str(source), "source_manifest_sha256": digest(source/"manifest.json"),
        "baseline_hashes": support.hashes(), "code_sha256": contracts(), "expected_counts": COUNTS,
        "tasks": jobs, "duration": DURATION, "bitrate_cycle": [48, 128, 192],
        "supervision": "Original velocity-specific bass/keyboard samples, not amplitude substitutes for velocity 75. All eight authored articulations retained. Original long keys held three seconds with the entire one-second tail protected. Short gates and 80ms damping are explicitly authored, not claimed human key releases.",
        "reserved_instruments_untouched": True, "no_user_audio_or_scores": True}
    if output.exists():
        if json.loads((output/"plan.json").read_text()) != json.loads(json.dumps(plan)) or (output/"manifest.json").exists():
            raise ValueError("Preserve completed or changed dynamics preparation")
    else:
        output.mkdir()
        preserve(output/"plan.json", plan)
    engine, captured = support._GeneralEngine("balanced"), {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records = []
    for index, (instrument, velocity, kind) in enumerate(jobs):
        selected = [n for n in notes if n["instrument_str"] == instrument and n["velocity"] == velocity]
        identifier = identity(instrument, velocity, kind)
        work = output/identifier
        work.mkdir(exist_ok=True)
        if (work/"record.json").exists():
            record = json.loads((work/"record.json").read_text())
            if (record["plan_sha256"] != digest(output/"plan.json")
                    or digest(work/"features.npz") != record["cache_sha256"]
                    or digest(work/"decoded.wav") != record["audio_sha256"]):
                raise ValueError("Changed completed dynamics observation")
        else:
            wave, reference, pitch, used = support.render(selected, kind)
            if (reference, pitch, used) != labels(selected, kind):
                raise ValueError("Changed authored dynamics labels")
            bitrate = plan["bitrate_cycle"][index % 3]
            wave, clock = support.encode(wave, RATE, work, bitrate)
            if wave.shape != (round(DURATION*RATE),):
                raise ValueError("Changed MP3 dynamics clock")
            audio = work/"decoded.wav"
            midi = engine.predict(audio, "bass", 120.)
            retained = [n for p in midi.instruments for n in p.notes]
            events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in retained], float).reshape(-1, 4)
            base = support.note_features(wave, RATE, captured["acoustic"], retained, include_context=True)
            observed = support.observe(wave, RATE)
            np.savez_compressed(work/"features.npz", events=events, base_x=base,
                x=support.context_features(events, base), eligible=support.eligible(events, DURATION),
                frames=support.coarse_sequences(observed["cqt"], events), attack_frames=support.sequences(observed, events),
                release_frames=support.release_sequences(observed, events), release_neighbors=support.neighbor_features(events, base),
                periodicity=support.recurrence(support.recurrence_observe(wave, RATE), events),
                plan_sha256=digest(output/"plan.json"))
            record = {"id": identifier, "source_group": "nsynth-"+instrument, "group": selected[0]["group"],
                "corpus": "nsynth-dynamics-"+selected[0]["instrument_family_str"], "kind": kind, "velocity": velocity,
                "audio": str(audio), "audio_sha256": digest(audio), "cache_sha256": digest(work/"features.npz"),
                "duration": DURATION, "seconds": DURATION, "reference": reference, "pitch_reference": pitch,
                "source_notes": used, "candidate_notes": len(events), "codec_clock": clock, "bitrate": bitrate,
                "plan_sha256": digest(output/"plan.json")}
            preserve(work/"record.json", record)
        records.append(record)
        print(json.dumps({"cached": len(records), "total": len(jobs), "id": identifier,
                          "notes": record["candidate_notes"]}), flush=True)
    if support.hashes() != plan["baseline_hashes"] or contracts() != plan["code_sha256"]:
        raise ValueError("Changed dynamics preparation sources")
    preserve(output/"manifest.json", {"version": VERSION, "items": records, "plan_sha256": digest(output/"plan.json"),
        "reserved_instruments_untouched": True, "no_user_audio_or_scores": True})


def load(folder):
    plan = json.loads((folder/"plan.json").read_text())
    manifest = json.loads((folder/"manifest.json").read_text())
    source = Path(plan["source_root"])
    notes = sources(source)
    original = {n["id"]: n for n in notes}
    records = manifest["items"]
    if (plan["version"] != VERSION or manifest["version"] != VERSION
            or plan["baseline_hashes"] != support.hashes() or plan["code_sha256"] != contracts()
            or manifest["plan_sha256"] != digest(folder/"plan.json")
            or plan["source_manifest_sha256"] != digest(source/"manifest.json")
            or plan["expected_counts"] != COUNTS or dict(Counter(r["group"] for r in records)) != COUNTS
            or plan["tasks"] != [list(t) for t in tasks(notes)] or len(records) != 448
            or {r["id"] for r in records} != {identity(*t) for t in tasks(notes)}
            or not all(doc[k] for doc in (plan, manifest) for k in ("reserved_instruments_untouched", "no_user_audio_or_scores"))):
        raise ValueError("Changed or leaking dynamics observations")
    result = []
    for record in records:
        instrument = record["source_group"].removeprefix("nsynth-")
        selected = [n for n in notes if n["instrument_str"] == instrument and n["velocity"] == record["velocity"]]
        reference, pitch, used = labels(selected, record["kind"])
        if (record["reference"] != reference or record["pitch_reference"] != pitch or record["source_notes"] != used
                or any(original[n]["group"] != record["group"] for n in used)
                or record["plan_sha256"] != manifest["plan_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]
                or digest(folder/record["id"]/"features.npz") != record["cache_sha256"]):
            raise ValueError("Changed dynamics labels, waveform or cache")
        with np.load(folder/record["id"]/"features.npz", allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed dynamics cache identity")
            item = {**record, **{key: saved[key] for key in support.KEYS}}
        np.testing.assert_array_equal(item["eligible"], support.eligible(item["events"], DURATION))
        np.testing.assert_array_equal(item["x"], support.context_features(item["events"], item["base_x"]))
        np.testing.assert_array_equal(item["release_neighbors"], support.neighbor_features(item["events"], item["base_x"]))
        item.update(reference=np.asarray(reference, float).reshape(-1, 3), pitch_reference=np.asarray(pitch, float).reshape(-1, 3))
        result.append(support.annotate(support.validate(item)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    prepare(**vars(parser.parse_args()))
