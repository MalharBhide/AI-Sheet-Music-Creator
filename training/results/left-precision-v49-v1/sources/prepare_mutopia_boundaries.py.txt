"""Render new public-domain score clocks and observe actual approved bass survivors."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
from acquire_mutopia_boundaries import SPLIT
from app.services.bass_harmonic import features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe, sequences
from bass_temporal_data import annotate
from bass_training_data import eligible
from current_bass_v32_baseline import hashes
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from release_local_features import release_sequences
from release_neighbor_features import features as neighbors

RATE = 22050
DURATION = 26
STARTS = (15, 45)


def source_rows(source):
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    if (manifest["plan_sha256"] != digest(source / "plan.json")
            or plan["producer_sha256"] != digest(Path(__file__).with_name("acquire_mutopia_boundaries.py"))
            or {int(k): v for k, v in plan["split"].items()} != SPLIT
            or {r["id"]: r["group"] for r in manifest["items"]} != SPLIT):
        raise ValueError("Changed new-dataset source split")
    for row in manifest["items"]:
        if (row["license"] != "Public Domain" or row["composer"] != "Scott Joplin"
                or row["midi_sha256"] != digest(source / f"{row['id']}.mid")
                or row["page_sha256"] != digest(source / f"{row['id']}.html")):
            raise ValueError("Changed publisher MIDI or license witness")
    return manifest["items"]


def render(midi, first, seed):
    """Own harmonic renderer; exact key releases and declared 60 ms decay tails."""
    rng = np.random.default_rng(seed)
    notes = sorted((n.start, n.end, n.pitch, n.velocity)
                   for part in midi.instruments if not part.is_drum for n in part.notes)
    samples = np.zeros(DURATION*RATE, np.float32)
    reference, support = [], []
    slope, decay = rng.uniform(1., 2.3), rng.uniform(.8, 3.)
    for start, end, pitch, velocity in notes:
        if end <= first or start >= first+20:
            continue
        left, right = max(3., start-first+3), min(23., end-first+3)
        if right <= left:
            continue
        if start >= first:
            reference.append((left, right, pitch))
        support.append((max(0, left-.05), min(DURATION, right+.1), pitch))
        lo, hi = round(left*RATE), round((right+.06)*RATE)
        t = np.arange(hi-lo)/RATE
        hz = 440*2**((pitch-69)/12)
        tone = sum(np.sin(2*np.pi*hz*k*t+rng.uniform(-.2, .2))/k**slope
                   for k in range(1, 9) if hz*k < RATE/2)
        envelope = np.minimum(1, t/.008)*(.25+.75*np.exp(-t/decay))
        envelope *= np.clip((right-left+.06-t)/.06, 0, 1)
        samples[lo:hi] += (tone*envelope*.12*velocity/100).astype(np.float32)
    return samples, np.asarray(reference).reshape(-1, 3), np.asarray(support).reshape(-1, 3)


def prepare(source, output):
    rows = source_rows(source)
    baseline = hashes()
    plan = {"source_root": str(source), "source_manifest_sha256": digest(source / "manifest.json"),
            "baseline_hashes": baseline, "producer_sha256": digest(Path(__file__)),
            "starts": STARTS, "variants": 2, "duration": DURATION, "rate": RATE,
            "counts": {"train": 16, "validation": 4}, "reserved_works_untouched": True,
            "label_policy": "Exact rendered MIDI keys; overlapping pitches protected with 50 ms attack/100 ms release padding. 60 ms synthesized acoustic tails. No pedal labels invented.",
            "no_user_audio_or_scores": True}
    if not output.exists():
        output.mkdir()
        preserve(output / "plan.json", plan)
    elif json.loads((output / "plan.json").read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("Changed preparation contract")
    if (output / "manifest.json").exists():
        raise ValueError("Preserve completed dataset")
    engine = _GeneralEngine("balanced")
    native, captured = engine.predict_function, {}

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records = []
    for row in rows:
        if row["group"] == "reserved":
            continue
        midi = pretty_midi.PrettyMIDI(str(source / f"{row['id']}.mid"))
        for crop, start in enumerate(STARTS):
            for variant in range(2):
                identifier = f"mutopia-v38-{row['id']}-{crop}-{variant}"
                folder = output / identifier
                folder.mkdir(exist_ok=True)
                completion = folder / "record.json"
                if completion.exists():
                    record = json.loads(completion.read_text())
                    if (record["cache_sha256"] != digest(folder / "features.npz")
                            or record["audio_sha256"] != digest(Path(record["audio"]))
                            or record["plan_sha256"] != digest(output / "plan.json")):
                        raise ValueError("Changed completed new observation")
                else:
                    wave, reference, support = render(midi, start, row["id"]*100+crop*2+variant)
                    wave, clock = encode(wave, RATE, folder, (48, 128)[variant])
                    path = folder / "decoded.wav"
                    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
                        predicted = engine.predict(path, "bass", 120.)
                    notes = [n for p in predicted.instruments for n in p.notes]
                    events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes]).reshape(-1, 4)
                    base = note_features(wave, RATE, captured["acoustic"], notes, include_context=True)
                    observed = observe(wave, RATE)
                    np.savez_compressed(folder / "features.npz", events=events, base_x=base,
                        x=features(events, base), eligible=eligible(events, DURATION),
                        frames=coarse_sequences(observed["cqt"], events),
                        attack_frames=sequences(observed, events), release_frames=release_sequences(observed, events),
                        release_neighbors=neighbors(events, base), plan_sha256=digest(output / "plan.json"))
                    record = {"id": identifier, "group": row["group"], "source_group": f"mutopia/Joplin/{row['id']}",
                        "corpus": "mutopia-joplin-synthesized", "publisher_id": row["id"],
                        "audio": str(path), "audio_sha256": digest(path), "duration": DURATION,
                        "reference": reference.tolist(), "pitch_reference": support.tolist(),
                        "codec_clock": clock, "cache_sha256": digest(folder / "features.npz"),
                        "plan_sha256": digest(output / "plan.json")}
                    preserve(completion, record)
                records.append(record)
                print(json.dumps({"completed": len(records), "total": 20, "id": identifier}), flush=True)
    if hashes() != baseline:
        raise ValueError("Production baseline changed during preparation")
    preserve(output / "manifest.json", {"plan_sha256": digest(output / "plan.json"),
        "items": records, "reserved_works_untouched": True, "no_user_audio_or_scores": True})


def load(folder):
    plan = json.loads((folder / "plan.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    source = Path(plan["source_root"])
    source_rows(source)
    if (plan["source_manifest_sha256"] != digest(source / "manifest.json")
            or plan["baseline_hashes"] != hashes()
            or plan["producer_sha256"] != digest(Path(__file__))
            or manifest["plan_sha256"] != digest(folder / "plan.json")
            or not manifest["reserved_works_untouched"]
            or {g: sum(r["group"] == g for r in manifest["items"]) for g in ("train", "validation")}
                != {"train": 16, "validation": 4}
            or len({r["id"] for r in manifest["items"]}) != 20):
        raise ValueError("Incomplete or changed new fitting dataset")
    result = []
    for record in manifest["items"]:
        path = folder / record["id"] / "features.npz"
        if (record["group"] != SPLIT[record["publisher_id"]]
                or record["source_group"] != f"mutopia/Joplin/{record['publisher_id']}"
                or digest(path) != record["cache_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]):
            raise ValueError("Changed new fitting observation or source group")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed observation plan")
            item = {**record, **{k: saved[k] for k in saved.files if k != "plan_sha256"}}
        item.update(reference=np.asarray(record["reference"]).reshape(-1, 3),
                    pitch_reference=np.asarray(record["pitch_reference"]).reshape(-1, 3))
        result.append(annotate(item))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    prepare(**vars(parser.parse_args()))
