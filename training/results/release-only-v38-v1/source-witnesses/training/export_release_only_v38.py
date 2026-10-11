"""Export a passing frozen specialist and verify real runtime decisions before routing."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from app.services.attack_local_evidence import observe, sequences
from app.services.chord_release import ReleaseModel, correct
from app.services.release_local_evidence import release_sequences
from app.services.release_neighbors import features as neighbors
from chord_boundary_v36 import decode
from evaluate_release_only_v38 import consumed, frozen, require_regression
from musicnet_v32_data import load
from prepare_mutopia_boundaries import load as new_items
from prepare_robust_training_stems import digest, preserve
from release_only_v38 import VERSION, probabilities


def export(run, fixtures, output):
    plan, winner, network, normalizer = frozen(run)
    require_regression(run)
    first = json.loads((run / "original-first-pass.json").read_text())
    manifest = json.loads((fixtures / "manifest.json").read_text())
    first_plan = json.loads((fixtures / "plan.json").read_text())
    if (not first["passes"] or not first["first_pass_complete"] or first["recordings"] != 44
            or first["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or first["test_manifest_sha256"] != digest(fixtures / "manifest.json")
            or first["consumed_regression_sha256"] != digest(run / "consumed-regression.json")
            or first["batch_selection_sha256"] != digest(run / "batch-selection.json")
            or manifest["plan_sha256"] != digest(fixtures / "plan.json")
            or first_plan["procedural_seeds"] != list(range(283101, 283141))
            or first_plan["reserved_publisher_id"] != 1899
            or len(manifest["items"]) != 44 or len({r["id"] for r in manifest["items"]}) != 44
            or any(not first[k]["passes"] or len(first[k]["per_recording"]) != 44
                   or any(not r["passes"] for r in first[k]["per_recording"])
                   for k in ("raw", "strict"))):
        raise ValueError("Failed or incomplete first-pass release proof")
    if output.exists() or (run / "runtime-parity.json").exists():
        raise ValueError("Preserve completed export/verification")
    values = {"version": np.asarray(VERSION), "format_version": np.asarray(4),
              "width": np.asarray(winner["profile"]["width"]),
              "threshold": np.asarray(winner["timing_threshold"]),
              "mean": normalizer[0].astype(np.float32), "scale": normalizer[1].astype(np.float32)}
    values.update({"weight_"+k: v.numpy().astype(np.float32) for k, v in network.state_dict().items()})
    np.savez_compressed(output, **values)
    with np.load(output, allow_pickle=False) as saved:
        portable = ReleaseModel(saved)
    torch.set_num_threads(2)
    items = load(Path(plan["baseline_root"]), Path(plan["piano_root"]))
    items.extend(new_items(Path(plan["new_root"])))
    items.extend(consumed(plan))
    max_feature_error, max_probability_error, checked = 0., 0., 0
    for record in manifest["items"]:
        path = fixtures / record["id"] / "features.npz"
        if digest(path) != record["cache_sha256"] or digest(Path(record["audio"])) != record["audio_sha256"]:
            raise ValueError("Changed first-pass physical observations")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed physical cache plan")
            item = {**record, **{k: saved[k] for k in (
                "events", "x", "base_x", "attack_frames", "release_frames", "release_neighbors")}}
        samples, rate = sf.read(record["audio"], dtype="float32")
        observed = observe(samples, rate)
        for key, actual in (("attack_frames", sequences(observed, item["events"])),
                            ("release_frames", release_sequences(observed, item["events"])),
                            ("release_neighbors", neighbors(item["events"], item["base_x"]))):
            error = float(np.max(np.abs(actual-item[key]), initial=0))
            max_feature_error = max(max_feature_error, error)
            np.testing.assert_allclose(actual, item[key], rtol=1e-6, atol=1e-6)
        items.append(item)
    if len(items) != 1569 or len({i["id"] for i in items}) != 1569:
        raise ValueError("Incomplete parity coverage")
    for item in items:
        expected = probabilities(network, item, normalizer)
        actual = portable.probability(item)
        max_probability_error = max(max_probability_error,
            float(np.max(np.abs(expected[2]-actual), initial=0)))
        np.testing.assert_allclose(actual, expected[2], rtol=1e-6, atol=1e-6)
        for threshold in (winner["timing_threshold"], min(.9999, winner["timing_threshold"]+.03)):
            reference, _ = decode(item, expected, np.ones(len(actual)), threshold, 1.)
            changed = correct(item["events"], actual, item["duration"], threshold)
            np.testing.assert_array_equal(changed, reference)
        checked += len(actual)
    root = Path(__file__).resolve().parents[1]
    names = ("backend/app/services/chord_release.py", "backend/app/services/chord_release_network.py",
             "training/export_release_only_v38.py", "training/evaluate_release_only_v38.py")
    frozen(run)
    result = {"passes": True, "recordings": 1569, "physical_recordings": 44, "notes_checked": checked,
        "maximum_feature_error": max_feature_error, "maximum_probability_error": max_probability_error,
        "identical_decisions_both_settings": True, "asset": str(output), "asset_sha256": digest(output),
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "first_pass_sha256": digest(run / "original-first-pass.json"),
        "source_sha256": {name: digest(root / name) for name in names},
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    preserve(run / "runtime-parity.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    export(**vars(parser.parse_args()))
