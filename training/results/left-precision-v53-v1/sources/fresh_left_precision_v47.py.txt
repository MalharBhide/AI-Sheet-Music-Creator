"""One untouched MP3 first pass, only after complete consumed preservation."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from app.services.bass_harmonic import features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe, sequences
from bass_temporal_data import annotate
from bass_training_data import eligible
from evaluate_left_precision_v47 import frozen
from left_precision_v43 import gate, probabilities
from left_precision_v44 import baseline, released_models
from left_precision_v46 import evaluate
from periodicity_features import features as recurrence
from periodicity_features import observe as recurrence_observe
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import texture
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from release_local_features import release_sequences
from release_neighbor_features import features as neighbors


def verify(run, production, output):
    plan, winner, network, normalizer = frozen(run, production)
    regression = json.loads((run/"consumed-regression.json").read_text())
    if (not regression["passes"] or regression["recordings"] != 603
            or regression["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or regression["batch_selection_sha256"] != digest(run/"batch-selection.json")
            or any(not regression[k]["passes"] or len(regression[k]["per_recording"]) != 603
                   or regression[k]["false_notes_removed"] <= 0 for k in ("raw", "strict"))):
        raise ValueError("Complete preserving regression with benefit required before fresh inference")
    if output.exists() or (run/"original-first-pass.json").exists():
        raise ValueError("Preserve original first pass; no replacements or retuning")
    if "292101-292140" not in plan["subsequent_gate"]:
        raise ValueError("Changed predeclared independent seeds")
    torch.set_num_threads(2)
    names = ("fresh_left_precision_v47.py", "prepare_bass_positive_data.py", "prepare_bass_texture_v12.py",
             "prepare_slakh_bass.py", "periodicity_features.py", "attack_local_features.py", "release_local_features.py")
    root = Path(__file__).resolve().parent
    producer = {"checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run/"batch-selection.json"),
        "consumed_regression_sha256": digest(run/"consumed-regression.json"),
        "source_sha256": {n: digest(root/n) for n in names}, "seeds": list(range(292101, 292141)),
        "recordings": 40, "bitrate_cycle": [48, 128, 192], "no_user_audio_or_scores": True,
        "no_replacements_or_retuning": True}
    output.mkdir()
    preserve(output/"plan.json", producer)
    engine, captured = _GeneralEngine("balanced"), {}
    native = engine.predict_function

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records, items = [], []
    for index, seed in enumerate(producer["seeds"]):
        family = "original" if index % 2 == 0 else "texture"
        wave, reference = (original if family == "original" else texture)(seed)
        identifier = f"left-v47-fresh-{seed}"
        work = output/identifier
        work.mkdir()
        wave, clock = encode(wave, RATE, work, producer["bitrate_cycle"][index % 3])
        if wave.shape != (30*RATE,):
            raise ValueError("Changed first-pass MP3 physical clock")
        audio = work/"decoded.wav"
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            predicted = engine.predict(audio, "bass", 120.)
        notes = [n for p in predicted.instruments for n in p.notes]
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes], float).reshape(-1, 4)
        base = note_features(wave, RATE, captured["acoustic"], notes, include_context=True)
        observed = observe(wave, RATE)
        arrays = {"events": events, "base_x": base, "x": features(events, base),
            "eligible": eligible(events, 30.), "frames": coarse_sequences(observed["cqt"], events),
            "attack_frames": sequences(observed, events), "release_frames": release_sequences(observed, events),
            "release_neighbors": neighbors(events, base), "periodicity": recurrence(recurrence_observe(wave, RATE), events)}
        np.savez_compressed(work/"features.npz", **arrays, plan_sha256=digest(output/"plan.json"))
        record = {"id": identifier, "group": "test", "source_group": "original-bass-seed-"+str(seed),
            "corpus": "new-first-pass-"+family, "audio": str(audio), "audio_sha256": digest(audio),
            "duration": 30., "reference": reference.tolist(), "pitch_reference": reference.tolist(),
            "codec_clock": clock, "cache_sha256": digest(work/"features.npz")}
        records.append(record)
        items.append(annotate({**record, **arrays, "reference": reference, "pitch_reference": reference.copy()}))
        print(json.dumps({"completed": len(records), "total": 40, "id": identifier}), flush=True)
    preserve(output/"manifest.json", {"items": records, "plan_sha256": digest(output/"plan.json"),
        "no_user_audio_or_scores": True, "now_consumed_regression": True})
    released = released_models(production)
    deployed = baseline(items, released)
    stricter_deployed = baseline(items, released, stricter=True)
    heads = [probabilities(network, i, normalizer) for i in items]
    ps = [gate(h, winner["veto_threshold"]) for h in heads]
    stricter_ps = [gate(h, winner["veto_threshold"]*.8) for h in heads]
    memo = {}
    raw = evaluate(items, deployed, ps, winner["threshold"], memo)
    strict = evaluate(items, stricter_deployed, stricter_ps, min(.9999, winner["threshold"]+.005), memo)
    gains = {k: {family: sum(r["false_notes_removed"] for r in report["per_recording"]
                if r["corpus"] == "new-first-pass-"+family) for family in ("original", "texture")}
             for k, report in (("raw", raw), ("strict", strict))}
    result = {"passes": raw["passes"] and strict["passes"] and all(g > 0 for by_family in gains.values() for g in by_family.values()),
        "raw": raw, "strict": strict, "family_false_note_gains": gains, "recordings": 40,
        "checkpoint_sha256": winner["checkpoint_sha256"], "first_pass_complete": True,
        "test_manifest_sha256": digest(output/"manifest.json"), "batch_selection_sha256": digest(run/"batch-selection.json"),
        "consumed_regression_sha256": digest(run/"consumed-regression.json"), "producer_sha256": digest(Path(__file__)),
        "no_test_used_for_selection": True, "no_user_audio_or_scores": True, "new_weights_deployed": False}
    frozen(run, production)
    if producer["source_sha256"] != {n: digest(root/n) for n in names}:
        raise ValueError("Changed first-pass producer")
    preserve(run/"original-first-pass.json", result)
    print(json.dumps({"passes": result["passes"], "recordings": 40, "family_false_note_gains": gains}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("production", type=Path)
    parser.add_argument("output", type=Path)
    verify(**vars(parser.parse_args()))
