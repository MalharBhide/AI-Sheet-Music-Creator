"""Stage the qualified refinement and prove full-context runtime parity.

This writes a separate candidate asset only to audited local training storage.
It never overwrites the approved website asset or routes production inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from approved_export_gate_v63 import require_fresh_proof
from approved_model_finetune import load_approved, storage_preflight
from current_bass_v32_baseline import hashes
from evaluate_approved_finetune_v63 import frozen, require_qualification
from evaluate_left_precision_v54 import consumed
from fitting_v63 import fitting_items
from fresh_approved_finetune_v63 import require_fresh_protocol
from left_precision_release_gate_v57 import require_consumed_proof
from left_precision_v39 import keep
from left_precision_v43 import VERSION, gate, probabilities
from prepare_left_support_v41 import KEYS, contracts
from prepare_robust_training_stems import digest, preserve
from train_approved_finetune_v63 import VERSION as FINE_TUNE_VERSION


EXPORT_SOURCES = (
    "training/export_approved_finetune_v63.py", "training/approved_export_gate_v63.py",
    "training/test_approved_export_gate_v63.py", "training/test_export_approved_finetune_v63.py",
    "backend/app/services/left_presence_refinement.py", "backend/tests/test_left_presence_refinement.py",
    "backend/app/services/left_presence.py", "backend/app/services/left_presence_network.py",
    "backend/app/services/chord_release.py", "backend/app/services/attack_local_evidence.py",
    "backend/app/services/release_local_evidence.py", "backend/app/services/release_neighbors.py",
    "backend/app/services/temporal_note_model.py", "backend/app/services/waveform_recurrence.py",
)


def require_export_protocol(production):
    path = production/"training/results/approved-finetune-v63-v1/export-preregistration.json"
    plan = json.loads(path.read_text())
    if (plan["version"] != FINE_TUNE_VERSION
            or plan["source_sha256"] != {name: digest(production/name) for name in EXPORT_SOURCES}
            or plan["fresh_protocol_sha256"] != require_fresh_protocol(production)
            or plan["qualification_protocol_sha256"] != require_qualification(production)
            or plan["parity_recordings"] != 3563 or plan["physical_recordings"] != 40
            or plan["no_user_audio_or_scores"] is not True
            or plan["never_overwrite_approved_asset"] is not True):
        raise ValueError("Changed refinement export preregistration")
    return digest(path)


def require_tests(run, fixtures, winner, production):
    regression = json.loads((run/"consumed-regression.json").read_text())
    require_consumed_proof(regression, winner, digest(run/"batch-selection.json"),
                          digest(production/"training/evaluate_approved_finetune_v63.py"))
    qualification_sha = require_qualification(production)
    if regression["qualification_protocol_sha256"] != qualification_sha:
        raise ValueError("Changed consumed qualification binding")
    first = json.loads((run/"original-first-pass.json").read_text())
    manifest = json.loads((fixtures/"manifest.json").read_text())
    plan = json.loads((fixtures/"plan.json").read_text())
    fresh_protocol_sha = require_fresh_protocol(production)
    require_fresh_proof(first, winner, digest(run/"batch-selection.json"),
        digest(run/"consumed-regression.json"), digest(production/"training/fresh_approved_finetune_v63.py"),
        [r["id"] for r in manifest["items"]], fresh_protocol_sha, qualification_sha)
    if (first["test_manifest_sha256"] != digest(fixtures/"manifest.json")
            or manifest["plan_sha256"] != digest(fixtures/"plan.json")
            or manifest["now_consumed_regression"] is not True
            or manifest["no_user_audio_or_scores"] is not True
            or plan["fresh_protocol_sha256"] != fresh_protocol_sha
            or plan["seeds"] != list(range(305101, 305141))
            or plan["bitrate_cycle"] != [48, 128, 192]
            or plan["native_v32"] != hashes() or plan["observation_contracts"] != contracts()
            or plan["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or plan["batch_selection_sha256"] != digest(run/"batch-selection.json")
            or plan["consumed_regression_sha256"] != digest(run/"consumed-regression.json")
            or len(manifest["items"]) != 40 or len({r["id"] for r in manifest["items"]}) != 40):
        raise ValueError("Incomplete independent first-pass physical provenance")
    root = Path(__file__).resolve().parent
    expected = json.loads((production/"training/results/approved-finetune-v63-v1/fresh-preregistration.json").read_text())
    if (plan["source_sha256"] != expected["source_sha256"]
            or any(digest(root/name) != sha for name, sha in plan["source_sha256"].items())):
        raise ValueError("Changed independent first-pass sources")
    return manifest


def portable_values(network, normalizer, winner):
    if (winner["profile"]["width"] != 12 or winner["threshold"] != .9 or winner["veto_threshold"] != .2
            or any(a.dtype != np.float32 or a.shape != (163,) or not np.isfinite(a).all()
                   for a in normalizer) or np.any(normalizer[1] <= 0)):
        raise ValueError("Changed approved-model portable metadata")
    values = {"version": np.asarray(VERSION), "format_version": np.asarray(5),
              "width": np.asarray(12), "threshold": np.asarray(.9), "veto_threshold": np.asarray(.2),
              "mean": normalizer[0].copy(), "scale": normalizer[1].copy()}
    for key, value in network.state_dict().items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Invalid portable refinement weights")
        values["weight_"+key] = value.detach().cpu().numpy().copy()
    return values


def export(run, production, fixtures, output):
    # Preserve original data readers; only the new runtime bridge is appended.
    import app.services
    app.services.__path__.append(str(production/"backend/app/services"))
    from app.services.attack_local_evidence import observe, sequences
    from app.services.chord_release import ReleaseModel, correct
    from app.services.left_presence import PresenceModel
    from app.services.left_presence_refinement import IntersectedPresence
    from app.services.release_local_evidence import release_sequences
    from app.services.release_neighbors import features as neighbors
    from app.services.temporal_note_model import sequences as coarse
    from app.services.waveform_recurrence import features as recurrence
    protocol_sha = require_export_protocol(production)
    _, winner, network, normalizer = frozen(run, production)
    manifest = require_tests(run, fixtures, winner, production)
    root = Path("/training").resolve(strict=True)
    if (output.is_symlink() or output.suffix != ".npz" or not output.resolve().is_relative_to(root)
            or output.exists() or (run/"runtime-parity.json").exists()):
        raise ValueError("Preserve existing export; candidate must stay in audited training storage")
    storage = storage_preflight(root, 500_000_000,
        audited_links={"left-precision-v43-replay/training/results": production/"training/results"})
    np.savez_compressed(output, **portable_values(network, normalizer, winner))
    with np.load(output, allow_pickle=False) as saved:
        portable = PresenceModel(saved)
    approved = load_approved(production)
    combined = IntersectedPresence(approved, portable)
    with np.load(production/"backend/app/assets/chord-release-v1.npz", allow_pickle=False) as saved:
        release = ReleaseModel(saved)
    torch.set_num_threads(2)
    items = fitting_items()+consumed(production)
    feature_error, probability_error, event_count = 0., 0., 0
    for record in manifest["items"]:
        path = fixtures/record["id"]/"features.npz"
        if (record["group"] != "test" or digest(path) != record["cache_sha256"]
                or digest(Path(record["audio"])) != record["audio_sha256"]):
            raise ValueError("Changed fresh physical observations")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed independent feature cache identity")
            item = {**record, **{key: saved[key] for key in KEYS}}
        samples, rate = sf.read(record["audio"], dtype="float32")
        if rate != 22050 or samples.shape != (30*rate,) or not np.isfinite(samples).all():
            raise ValueError("Changed independent physical waveform clock")
        observed = observe(samples, rate)
        for key, actual in (("frames", coarse(observed["cqt"], item["events"])),
                ("attack_frames", sequences(observed, item["events"])),
                ("release_frames", release_sequences(observed, item["events"])),
                ("release_neighbors", neighbors(item["events"], item["base_x"])),
                ("periodicity", recurrence(samples, rate, item["events"]))):
            feature_error = max(feature_error, float(np.max(np.abs(actual-item[key]), initial=0)))
            np.testing.assert_allclose(actual, item[key], rtol=1e-6, atol=1e-6)
        items.append(item)
    if len(items) != 3563 or len({item["id"] for item in items}) != 3563:
        raise ValueError("Incomplete full fitting/regression/fresh parity coverage")
    for index, item in enumerate(items):
        heads = probabilities(network, item, normalizer)
        full = correct(item["events"], release.probability(item), item["duration"], release.threshold)
        for stricter in (False, True):
            threshold = min(.9999, winner["threshold"]+.005) if stricter else winner["threshold"]
            expected = gate(heads, winner["veto_threshold"]*(.8 if stricter else 1))
            actual = portable.probability(item, stricter)
            probability_error = max(probability_error, float(np.max(np.abs(actual-expected), initial=0)))
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            parent = approved.keep(item, item["duration"], stricter)
            expected_mask = parent & keep(item["events"], expected, item["duration"], threshold)
            mask = combined.keep(item, item["duration"], stricter)
            np.testing.assert_array_equal(mask, expected_mask)
            if np.any(mask & ~parent):
                raise ValueError("Runtime refinement restored a parent rejection")
            np.testing.assert_array_equal(full[mask], full[np.flatnonzero(expected_mask)])
        event_count += len(item["events"])
        if (index+1) % 100 == 0:
            print(json.dumps({"parity_recordings": index+1, "total": 3563}), flush=True)
    frozen(run, production)
    require_tests(run, fixtures, winner, production)
    if require_export_protocol(production) != protocol_sha:
        raise ValueError("Changed runtime export protocol")
    result = {"passes": True, "recordings": 3563, "physical_recordings": 40,
        "events_checked": event_count, "maximum_feature_error": feature_error,
        "maximum_probability_error": probability_error, "identical_keep_decisions_both_settings": True,
        "parent_rejections_never_restored": True, "retained_v38_clocks_identical": True,
        "asset": str(output), "asset_sha256": digest(output), "checkpoint_sha256": winner["checkpoint_sha256"],
        "first_pass_sha256": digest(run/"original-first-pass.json"),
        "export_protocol_sha256": protocol_sha, "storage_preflight": storage,
        "no_user_audio_or_scores": True, "new_weights_deployed": False}
    preserve(run/"runtime-parity.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "production", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    export(**vars(parser.parse_args()))
