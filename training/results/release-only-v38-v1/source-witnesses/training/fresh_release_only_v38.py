"""One fixed 44-clip first pass after a frozen winner passes all 519 regressions."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from app.services.bass_harmonic import features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe, sequences
from bass_temporal_data import annotate
from bass_training_data import eligible
from evaluate_release_only_v38 import frozen, require_regression, score
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import texture
from prepare_mutopia_boundaries import STARTS, render, source_rows
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from release_local_features import release_sequences
from release_neighbor_features import features as neighbors


def verify(run, output):
    require_regression(run)
    plan, winner, network, normalizer = frozen(run)
    if output.exists() or (run / "original-first-pass.json").exists():
        raise ValueError("Preserve original first pass; no replacement or retuning")
    if "283101-283140" not in plan["fresh_policy"]:
        raise ValueError("Changed predeclared first-pass seeds")
    source = Path(json.loads((Path(plan["new_root"]) / "plan.json").read_text())["source_root"])
    reserved = [r for r in source_rows(source) if r["group"] == "reserved"]
    if len(reserved) != 1 or reserved[0]["id"] != 1899:
        raise ValueError("Changed reserved new-dataset work")
    jobs = [("procedural", seed, index) for index, seed in enumerate(range(283101, 283141))]
    jobs.extend(("mutopia", (crop, variant), crop*2+variant)
                for crop in range(2) for variant in range(2))
    root = Path(__file__).resolve().parents[1]
    names = ("fresh_release_only_v38.py", "evaluate_release_only_v38.py",
             "prepare_mutopia_boundaries.py", "prepare_bass_positive_data.py", "prepare_bass_texture_v12.py",
             "prepare_slakh_bass.py", "release_local_features.py", "attack_local_features.py")
    producer = {"checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "consumed_regression_sha256": digest(run / "consumed-regression.json"),
        "source_sha256": {name: digest(root / "training" / name) for name in names},
        "procedural_seeds": list(range(283101, 283141)), "reserved_publisher_id": 1899,
        "recordings": 44, "bitrate_cycle": [48, 128, 192],
        "no_user_audio_or_scores": True, "no_retuning_or_replacement": True}
    output.mkdir()
    preserve(output / "plan.json", producer)
    engine = _GeneralEngine("balanced")
    native, captured = engine.predict_function, {}

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        captured["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    items, records = [], []
    for family, choice, index in jobs:
        if family == "procedural":
            wave, reference = (original if index % 2 == 0 else texture)(choice)
            support, duration = reference.copy(), 30.
            identifier = f"release-v38-fresh-{choice}"
            group = "original-bass-seed-"+str(choice)
        else:
            crop, variant = choice
            midi = pretty_midi.PrettyMIDI(str(source / "1899.mid"))
            wave, reference, support = render(midi, STARTS[crop], 189900+index)
            duration = 26.
            identifier = f"release-v38-reserved-1899-{crop}-{variant}"
            group = "mutopia/Joplin/1899"
        folder = output / identifier
        folder.mkdir()
        wave, clock = encode(wave, RATE, folder, producer["bitrate_cycle"][index % 3])
        audio = folder / "decoded.wav"
        wave, rate = sf.read(audio, dtype="float32")
        if rate != RATE or len(wave) != round(duration*RATE) or wave.ndim != 1:
            raise ValueError("Changed first-pass MP3 sample clock")
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            predicted = engine.predict(audio, "bass", 120.)
        notes = [n for p in predicted.instruments for n in p.notes]
        events = np.asarray([[n.start, n.end, n.pitch, n.velocity] for n in notes]).reshape(-1, 4)
        base = note_features(wave, RATE, captured["acoustic"], notes, include_context=True)
        observed = observe(wave, RATE)
        arrays = {"events": events, "base_x": base, "x": features(events, base),
            "eligible": eligible(events, duration), "frames": coarse_sequences(observed["cqt"], events),
            "attack_frames": sequences(observed, events), "release_frames": release_sequences(observed, events),
            "release_neighbors": neighbors(events, base)}
        np.savez_compressed(folder / "features.npz", **arrays, plan_sha256=digest(output / "plan.json"))
        record = {"id": identifier, "group": "test", "source_group": group,
            "corpus": "new-first-pass-"+family, "audio": str(audio), "audio_sha256": digest(audio),
            "duration": duration, "reference": reference.tolist(), "pitch_reference": support.tolist(),
            "codec_clock": clock, "cache_sha256": digest(folder / "features.npz")}
        records.append(record)
        items.append(annotate({**record, **arrays, "reference": reference, "pitch_reference": support}))
        print(json.dumps({"completed": len(records), "total": 44, "id": identifier}), flush=True)
    preserve(output / "manifest.json", {"items": records, "plan_sha256": digest(output / "plan.json"),
        "no_user_audio_or_scores": True, "now_consumed_regression": True})
    result = {**score(items, network, normalizer, winner), "recordings": 44,
        "checkpoint_sha256": winner["checkpoint_sha256"], "first_pass_complete": True,
        "test_manifest_sha256": digest(output / "manifest.json"),
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "consumed_regression_sha256": digest(run / "consumed-regression.json"),
        "producer_sha256": digest(Path(__file__)), "no_user_audio_or_scores": True,
        "no_test_used_for_selection": True, "new_weights_deployed": False}
    frozen(run)
    if producer["source_sha256"] != {name: digest(root / "training" / name) for name in names}:
        raise ValueError("Changed code during first pass")
    preserve(run / "original-first-pass.json", result)
    print(json.dumps({"passes": result["passes"], "recordings": 44}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    verify(**vars(parser.parse_args()))
