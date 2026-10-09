"""Verify portable spectral probabilities and physical feature parity before installation."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from app.services.bass_attack_declutter import BassAttackDeclutter
from evaluate_musicnet_full_v34 import frozen, require_regression
from musicnet_full_encoding_model import probabilities
from prepare_robust_training_stems import digest, preserve


def verify(run, fixtures):
    root = Path(__file__).resolve().parents[1]
    from app.services.attack_local_evidence import observe as acoustic_observe
    from app.services.attack_local_evidence import sequences as attack_sequences
    from app.services.bass_full_encoding_declutter import FullEncodingSupportModel
    from app.services.release_local_evidence import release_sequences
    from app.services.release_neighbors import features as runtime_neighbors
    from app.services.temporal_note_model import sequences as coarse_sequences
    from app.services.waveform_recurrence import features as runtime_recurrence
    from bass_v32_regression_data import load as regression_items
    from fresh_musicnet_full_v34 import regression_gate
    from musicnet_v32_data import load

    runtime_files = [
        "backend/app/services/" + name + ".py"
        for name in (
            "bass_full_encoding_declutter",
            "bass_full_encoding_network",
            "bass_embedding_declutter",
            "bass_embedding_network",
            "waveform_recurrence",
            "release_neighbors",
        )
    ]
    source_hashes = {name: digest(root / name) for name in runtime_files}
    verifier_sha = digest(Path(__file__))
    target = run / "runtime-parity.json"
    if target.exists():
        raise ValueError("Preserve completed runtime verification")
    plan, winner, network, normalizer = frozen(run)
    require_regression(run)
    regression_gate(run, winner)
    torch.set_num_threads(2)
    export = json.loads((run / "portable-export.json").read_text())
    asset = Path(export["asset"])
    if (
        export["asset_sha256"] != digest(asset)
        or export["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or export["first_pass_sha256"] != digest(run / "original-first-pass.json")
        or export["exporter_sha256"]
        != digest(Path(__file__).with_name("export_musicnet_full_v34.py"))
        or export["runtime_source_sha256"]
        != {name: source_hashes[name] for name in export["runtime_source_sha256"]}
        or set(export["runtime_source_sha256"])
        != {
            "backend/app/services/bass_full_encoding_network.py",
            "backend/app/services/bass_full_encoding_declutter.py",
        }
    ):
        raise ValueError("Changed frozen portable asset or first-pass witness")
    with np.load(asset, allow_pickle=False) as saved:
        portable = FullEncodingSupportModel(saved)
        if (
            portable.remove_threshold != winner["threshold"]
            or portable.guardian_threshold != winner["guardian_threshold"]
            or int(saved["width"]) != winner["width"]
            or float(saved["cap"]) != winner["cap"]
        ):
            raise ValueError("Changed portable confidence or architecture")
        for key, expected_stats in zip(("mean", "scale"), normalizer, strict=True):
            np.testing.assert_array_equal(saved[key], expected_stats)
    rows = load(Path(plan["baseline_root"]), Path(plan["piano_root"]))
    rows.extend(
        regression_items(
            Path(plan["regression_root"]), plan["regression_manifest_sha256"]
        )
    )
    expected_fresh = plan["first_pass_declaration"]["expected_recordings"]
    fresh = json.loads((fixtures / "manifest.json").read_text())
    report = json.loads((run / "original-first-pass.json").read_text())
    if (
        not report["passes"]
        or report["test_manifest_sha256"] != digest(fixtures / "manifest.json")
        or len(fresh["items"]) != expected_fresh
        or fresh["plan_sha256"] != digest(fixtures / "plan.json")
        or report["checkpoint_sha256"] != winner["checkpoint_sha256"]
    ):
        raise ValueError("Changed first-pass observations")
    physical_count, maximum_feature_error = 0, 0.0
    for record in fresh["items"]:
        path = fixtures / record["id"] / "features.npz"
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed first-pass physical cache")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != fresh["plan_sha256"]:
                raise ValueError("Changed first-pass cache identity")
            item = {
                **record,
                **{
                    k: saved[k]
                    for k in (
                        "events",
                        "base_x",
                        "x",
                        "periodicity",
                        "eligible",
                        "frames",
                        "attack_frames",
                        "release_frames",
                        "release_neighbors",
                    )
                },
            }
        samples, rate = sf.read(record["audio"], dtype="float32")
        phase = runtime_recurrence(samples, rate, item["events"])
        maximum_feature_error = max(
            maximum_feature_error,
            float(np.max(np.abs(phase - item["periodicity"]), initial=0)),
        )
        np.testing.assert_allclose(phase, item["periodicity"], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            runtime_neighbors(item["events"], item["base_x"]), item["release_neighbors"]
        )
        observed = acoustic_observe(samples, rate)
        for key, actual in (
            ("frames", coarse_sequences(observed["cqt"], item["events"])),
            ("attack_frames", attack_sequences(observed, item["events"])),
            ("release_frames", release_sequences(observed, item["events"])),
        ):
            np.testing.assert_allclose(actual, item[key], rtol=1e-6, atol=1e-6)
        item["periodicity"] = phase
        rows.append(item)
        physical_count += 1
        if physical_count % 10 == 0:
            print(
                json.dumps(
                    {
                        "physical_feature_checks": physical_count,
                        "expected": expected_fresh,
                    }
                ),
                flush=True,
            )
    expected_total = (
        sum(plan["expected_fitting_counts"].values()) + 469 + expected_fresh
    )
    if len(rows) != expected_total or len({i["id"] for i in rows}) != expected_total:
        raise ValueError("Incomplete portable verification scope")
    guardian = BassAttackDeclutter().guardian
    maximum_probability_error = 0.0
    events_checked = 0
    for item in rows:
        expected = probabilities(network, item, normalizer)
        actual = portable.probability(item)
        maximum_probability_error = max(
            maximum_probability_error,
            float(np.max(np.abs(expected - actual), initial=0)),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        guard = guardian.probability(item["base_x"])
        for stricter in (False, True):
            threshold = (
                winner["threshold"] + (1 - winner["threshold"]) / 2
                if stricter
                else winner["threshold"]
            )
            keep = ~(
                item["eligible"]
                & (expected.argmax(axis=1) == 1)
                & (expected[:, 1] >= threshold)
                & (guard < winner["guardian_threshold"])
            )
            np.testing.assert_array_equal(
                portable.keep(
                    item["events"], actual, guard, item["duration"], stricter
                ),
                keep,
            )
        events_checked += len(item["events"])
    frozen(run)
    if source_hashes != {
        name: digest(root / name) for name in runtime_files
    } or verifier_sha != digest(Path(__file__)):
        raise ValueError("Runtime source changed during verification")
    preserve(
        target,
        {
            "passes": True,
            "recordings": len(rows),
            "events_checked": events_checked,
            "physical_recordings": physical_count,
            "maximum_feature_error": maximum_feature_error,
            "maximum_probability_error": maximum_probability_error,
            "identical_keep_decisions_at_both_settings": True,
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "asset_sha256": export["asset_sha256"],
            "first_pass_sha256": export["first_pass_sha256"],
            "runtime_source_sha256": source_hashes,
            "verifier_sha256": verifier_sha,
            "no_user_audio_or_scores": True,
            "new_weights_deployed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    verify(args.run, args.fixtures)
