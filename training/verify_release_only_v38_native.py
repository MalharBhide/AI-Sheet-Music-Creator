"""Exercise actual staged Basic Pitch routing on all fresh clips with corrections."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from app.services import chord_release, piano_transcription
from evaluate_release_only_v38 import frozen, require_regression
from prepare_robust_training_stems import digest, preserve


def verify(run, fixtures, stage):
    target = run / "native-routing.json"
    if target.exists():
        raise ValueError("Preserve completed native routing proof")
    _, winner, _, _ = frozen(run)
    require_regression(run)
    stage_proof = json.loads((stage / "stage.json").read_text())
    proof = json.loads((run / "runtime-parity.json").read_text())
    for name, sha in stage_proof["source_sha256"].items():
        if digest(stage / "backend" / name) != sha:
            raise ValueError("Changed staged release")
    if (Path(piano_transcription.__file__).resolve() != (stage / "backend/app/services/piano_transcription.py").resolve()
            or Path(chord_release.__file__).resolve() != (stage / "backend/app/services/chord_release.py").resolve()
            or stage_proof["runtime_parity_sha256"] != digest(run / "runtime-parity.json")):
        raise ValueError("Native test did not import the staged runtime")
    first = json.loads((run / "original-first-pass.json").read_text())
    manifest = json.loads((fixtures / "manifest.json").read_text())
    if first["test_manifest_sha256"] != digest(fixtures / "manifest.json"):
        raise ValueError("Changed fresh corpus")
    changed = {r["id"] for r in first["raw"]["per_recording"] if r["retimed_notes"] > 0}
    if not changed:
        raise ValueError("Native routing must exercise an actual correction")
    engine = piano_transcription._GeneralEngine("balanced")
    portable = chord_release.ChordRelease(stage / "backend/app/assets/chord-release-v1.npz", proof["asset_sha256"])
    rows = []
    torch.set_num_threads(2)
    for record in manifest["items"]:
        if record["id"] not in changed:
            continue
        path = fixtures / record["id"] / "features.npz"
        if digest(path) != record["cache_sha256"] or digest(Path(record["audio"])) != record["audio_sha256"]:
            raise ValueError("Changed native fixture")
        with np.load(path, allow_pickle=False) as saved:
            item = {k: saved[k] for k in ("events", "x", "attack_frames", "release_frames", "release_neighbors")}
        expected = chord_release.correct(item["events"], portable.model.probability(item),
                                         record["duration"], portable.model.threshold)
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            midi = engine.predict(Path(record["audio"]), "bass", 120.)
        actual = np.asarray([[n.start, n.end, n.pitch, n.velocity]
                             for part in midi.instruments for n in part.notes]).reshape(-1, 4)
        np.testing.assert_array_equal(actual, expected)
        assert engine.chord_release.model.threshold == winner["timing_threshold"]
        rows.append({"id": record["id"], "notes": len(actual),
                     "changed": int(np.sum(actual[:, 1] != item["events"][:, 1])), "passes": True})
    if sum(r["changed"] for r in rows) != first["raw"]["retimed_notes"]:
        raise ValueError("Incomplete native correction coverage")
    result = {"passes": True, "cases": rows, "all_fresh_corrected_cases_tested": True,
        "staged_module_origins_asserted": True, "stage_sha256": digest(stage / "stage.json"),
        "runtime_parity_sha256": digest(run / "runtime-parity.json"),
        "verifier_sha256": digest(Path(__file__)), "no_user_audio_or_scores": True}
    preserve(target, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "fixtures", "stage"):
        parser.add_argument(name, type=Path)
    verify(**vars(parser.parse_args()))
