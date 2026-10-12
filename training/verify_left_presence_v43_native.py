"""Exercise actual staged Basic Pitch routing on every fresh deletion case."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from app.services import chord_release, left_presence, piano_transcription
from export_left_presence_v43 import require_tests
from prepare_robust_training_stems import digest, preserve


def verify(run, production, fixtures, stage):
    target = run/"native-routing.json"
    if target.exists():
        raise ValueError("Preserve completed native routing proof")
    winner = json.loads((run/"batch-selection.json").read_text())["winner"]
    manifest = require_tests(run, fixtures, winner)
    staged = json.loads((stage/"stage.json").read_text())
    parity = json.loads((run/"runtime-parity.json").read_text())
    if (not parity["passes"] or staged["runtime_parity_sha256"] != digest(run/"runtime-parity.json")
            or staged["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or staged["asset_sha256"] != parity["asset_sha256"]
            or any(digest(stage/"backend"/name) != sha for name, sha in staged["source_sha256"].items())):
        raise ValueError("Changed staged runtime or parity witness")
    for module in (chord_release, left_presence, piano_transcription):
        if Path(module.__file__).resolve() != (stage/"backend/app/services"/(module.__name__.split(".")[-1]+".py")).resolve():
            raise ValueError("Native test did not import staged runtime")
    first = json.loads((run/"original-first-pass.json").read_text())
    changed = {r["id"] for r in first["raw"]["per_recording"] if r["removed_notes"] > 0}
    if not changed:
        raise ValueError("Native routing must exercise actual deletions")
    # Also verify an unchanged original and an unchanged texture passage.
    controls = set()
    for family in ("original", "texture"):
        controls.add(next(r["id"] for r in manifest["items"]
                          if r["corpus"] == "new-first-pass-"+family and r["id"] not in changed))
    engine = piano_transcription._GeneralEngine("balanced")
    portable = left_presence.LeftHandPresence(stage/"backend/app/assets/left-presence-v1.npz", parity["asset_sha256"])
    with np.load(stage/"backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        release = chord_release.ReleaseModel(saved)
    torch.set_num_threads(2)
    rows = []
    for record in manifest["items"]:
        if record["id"] not in changed | controls:
            continue
        path = fixtures/record["id"]/"features.npz"
        if digest(path) != record["cache_sha256"] or digest(Path(record["audio"])) != record["audio_sha256"]:
            raise ValueError("Changed native fixture")
        with np.load(path, allow_pickle=False) as saved:
            item = {key: saved[key] for key in ("events", "x", "frames", "attack_frames", "release_frames", "release_neighbors", "periodicity")}
        mask = portable.model.keep(item, record["duration"])
        expected = chord_release.correct(item["events"], release.probability(item), record["duration"], release.threshold)[mask]
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
            midi = engine.predict(Path(record["audio"]), "bass", 120.)
        actual = np.asarray([[n.start, n.end, n.pitch, n.velocity]
                             for part in midi.instruments for n in part.notes]).reshape(-1, 4)
        np.testing.assert_array_equal(actual, expected)
        assert engine.left_presence.model.threshold == winner["threshold"]
        assert engine.left_presence.model.veto_threshold == winner["veto_threshold"]
        rows.append({"id": record["id"], "notes": len(actual), "removed": int((~mask).sum()),
                     "control": record["id"] in controls, "passes": True})
        print(json.dumps(rows[-1]), flush=True)
    if (len(rows) != len(changed)+2 or sum(r["removed"] for r in rows) != first["raw"]["removed_notes"]
            or engine.bass_verification["window_presence_rejections"] != first["raw"]["removed_notes"]):
        raise ValueError("Incomplete native deletion coverage")
    if any(digest(stage/"backend"/name) != sha for name, sha in staged["source_sha256"].items()):
        raise ValueError("Staged runtime changed during native inference")
    result = {"passes": True, "cases": rows, "all_fresh_deletion_cases_tested": True,
        "unchanged_controls": 2, "staged_module_origins_asserted": True,
        "stage_sha256": digest(stage/"stage.json"), "runtime_parity_sha256": digest(run/"runtime-parity.json"),
        "verifier_sha256": digest(Path(__file__)), "production_root": str(production),
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    preserve(target, result)
    print(json.dumps({"passes": True, "cases": len(rows), "removed": sum(r["removed"] for r in rows)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "production", "fixtures", "stage"):
        parser.add_argument(name, type=Path)
    verify(**vars(parser.parse_args()))
