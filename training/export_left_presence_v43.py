"""Export the frozen passing V43 winner; verify all decisions before routing."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from evaluate_left_precision_v43 import frozen
from left_precision_v39 import keep
from left_precision_v43 import VERSION, gate, probabilities
from prepare_robust_training_stems import digest, preserve
from train_left_precision_v43 import FOLDERS


def require_tests(run, fixtures, winner):
    regression = json.loads((run/"consumed-regression.json").read_text())
    first = json.loads((run/"original-first-pass.json").read_text())
    manifest = json.loads((fixtures/"manifest.json").read_text())
    plan = json.loads((fixtures/"plan.json").read_text())
    for result, count in ((regression, 563), (first, 40)):
        if (not result["passes"] or result["recordings"] != count
                or result["checkpoint_sha256"] != winner["checkpoint_sha256"]
                or result["batch_selection_sha256"] != digest(run/"batch-selection.json")
                or not result["no_test_used_for_selection"] or not result["no_user_audio_or_scores"]):
            raise ValueError("Failed or mismatched V43 evaluation")
        for name, threshold in (("raw", winner["threshold"]), ("strict", min(.9999, winner["threshold"]+.005))):
            rows = result[name]["per_recording"]
            if (not result[name]["passes"] or result[name]["threshold"] != threshold
                    or result[name]["false_notes_removed"] <= 0 or len(rows) != count
                    or len({r["id"] for r in rows}) != count
                    or any(not r["passes"] or not r["retained_clocks_identical"] for r in rows)):
                raise ValueError("Incomplete per-recording V43 preservation")
    if (not first["first_pass_complete"] or first["test_manifest_sha256"] != digest(fixtures/"manifest.json")
            or first["consumed_regression_sha256"] != digest(run/"consumed-regression.json")
            or any(value <= 0 for families in first["family_false_note_gains"].values() for value in families.values())
            or manifest["plan_sha256"] != digest(fixtures/"plan.json")
            or not manifest["now_consumed_regression"] or not manifest["no_user_audio_or_scores"]
            or plan["seeds"] != list(range(288101, 288141))
            or len(manifest["items"]) != 40 or len({r["id"] for r in manifest["items"]}) != 40):
        raise ValueError("Incomplete independent V43 first-pass provenance")
    return manifest


def export(run, production, fixtures, output):
    import app.services

    # Old data readers retain their exact original V32 sources and guards.
    # Only the newly authored runtime modules are appended from current source.
    app.services.__path__.append(str(production/"backend/app/services"))
    from app.services.attack_local_evidence import observe, sequences
    from app.services.chord_release import ReleaseModel, correct
    from app.services.left_presence import PresenceModel
    from app.services.release_local_evidence import release_sequences
    from app.services.release_neighbors import features as neighbors
    from app.services.temporal_note_model import sequences as coarse
    from app.services.waveform_recurrence import features as recurrence
    from evaluate_left_precision_v40 import consumed
    from musicnet_v32_data import load
    from prepare_left_support_v41 import load as authored
    from prepare_mutopia_recurrence_v42 import load as piano

    _, winner, network, normalizer = frozen(run, production)
    manifest = require_tests(run, fixtures, winner)
    if output.exists() or (run/"runtime-parity.json").exists():
        raise ValueError("Preserve the original export and runtime verification")
    runtime_names = ("backend/app/services/left_presence.py", "backend/app/services/left_presence_network.py",
                     "training/export_left_presence_v43.py")
    sources = {name: digest(production/name) for name in runtime_names}
    values = {"version": np.asarray(VERSION), "format_version": np.asarray(5),
        "width": np.asarray(winner["profile"]["width"]), "threshold": np.asarray(winner["threshold"]),
        "veto_threshold": np.asarray(winner["veto_threshold"]),
        "mean": normalizer[0].astype(np.float32), "scale": normalizer[1].astype(np.float32)}
    values.update({"weight_"+key: value.numpy().astype(np.float32) for key, value in network.state_dict().items()})
    np.savez_compressed(output, **values)
    with np.load(output, allow_pickle=False) as saved:
        portable = PresenceModel(saved)
    with np.load(production/"backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        release = ReleaseModel(saved)
    torch.set_num_threads(2)
    items = load(FOLDERS["baseline"], FOLDERS["piano"])+authored(FOLDERS["authored"])+piano(FOLDERS["mutopia"])
    items.extend(consumed(production))
    max_feature_error, max_probability_error, count = 0., 0., 0
    keys = ("events", "base_x", "x", "eligible", "frames", "attack_frames", "release_frames", "release_neighbors", "periodicity")
    for record in manifest["items"]:
        path = fixtures/record["id"]/"features.npz"
        if (record["group"] != "test" or digest(path) != record["cache_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]):
            raise ValueError("Changed independent physical observations")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed independent physical clock identity")
            item = {**record, **{key: saved[key] for key in keys}}
        samples, rate = sf.read(record["audio"], dtype="float32")
        observed = observe(samples, rate)
        for key, actual in (("frames", coarse(observed["cqt"], item["events"])),
                ("attack_frames", sequences(observed, item["events"])),
                ("release_frames", release_sequences(observed, item["events"])),
                ("release_neighbors", neighbors(item["events"], item["base_x"])),
                ("periodicity", recurrence(samples, rate, item["events"]))):
            max_feature_error = max(max_feature_error, float(np.max(np.abs(actual-item[key]), initial=0)))
            np.testing.assert_allclose(actual, item[key], rtol=1e-6, atol=1e-6)
        items.append(item)
        print(json.dumps({"physical_recordings": len(items)-1681, "expected": 40}), flush=True)
    if len(items) != 1721 or len({item["id"] for item in items}) != 1721:
        raise ValueError("Incomplete V43 portable parity coverage")
    for item in items:
        expected = probabilities(network, item, normalizer)
        deployed = correct(item["events"], release.probability(item), item["duration"], release.threshold)
        for stricter in (False, True):
            threshold = min(.9999, winner["threshold"]+.005) if stricter else winner["threshold"]
            reference = gate(expected, winner["veto_threshold"]*(.8 if stricter else 1))
            actual = portable.probability(item, stricter)
            max_probability_error = max(max_probability_error, float(np.max(np.abs(actual-reference), initial=0)))
            np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
            mask = portable.keep(item, item["duration"], stricter)
            np.testing.assert_array_equal(mask, keep(item["events"], reference, item["duration"], threshold))
            np.testing.assert_array_equal(deployed[mask], deployed[np.flatnonzero(mask)])
        count += len(item["events"])
    frozen(run, production)
    require_tests(run, fixtures, winner)
    if sources != {name: digest(production/name) for name in runtime_names}:
        raise ValueError("Runtime source changed during verification")
    result = {"passes": True, "recordings": 1721, "physical_recordings": 40, "events_checked": count,
        "maximum_feature_error": max_feature_error, "maximum_probability_error": max_probability_error,
        "identical_keep_decisions_both_settings": True, "retained_v38_clocks_identical": True,
        "asset": str(output), "asset_sha256": digest(output), "checkpoint_sha256": winner["checkpoint_sha256"],
        "first_pass_sha256": digest(run/"original-first-pass.json"), "runtime_source_sha256": sources,
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    preserve(run/"runtime-parity.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "production", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    export(**vars(parser.parse_args()))
