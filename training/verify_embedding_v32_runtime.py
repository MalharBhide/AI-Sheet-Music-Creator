"""Verify portable spectral probabilities and physical feature parity before installation."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from app.services.bass_attack_declutter import BassAttackDeclutter
from app.services.bass_harmonic import features
from bass_temporal_data import annotate
from bass_training_data import eligible
from evaluate_teacher_embedding_v32 import frozen, require_regression
from prepare_robust_training_stems import digest, preserve
from release_neighbor_features import features as neighbor_features
from teacher_embedding_model import probabilities


def consumed(plan):
    from evaluate_periodicity_v26 import consumed_inputs as previous_inputs

    parent = previous_inputs()
    observations = Path(plan["periodicity_observation_root"])
    if (
        digest(observations / "manifest.json") != plan["periodicity_observation_manifest_sha256"]
        or digest(observations / "plan.json") != plan["periodicity_observation_plan_sha256"]
    ):
        raise ValueError("Changed consumed recurrence observations")
    observation_manifest = json.loads((observations / "manifest.json").read_text())
    records = {r["id"]: r for r in observation_manifest["items"]}
    if len(records) != 392:
        raise ValueError("Incomplete fusion regression observations")
    verifier = BassAttackDeclutter()
    items = []
    for item in parent:
        record = records.pop(item["id"])
        path = observations / (item["id"] + ".npz")
        if (
            digest(path) != record["cache_sha256"]
            or record["parent_cache_sha256"] != item["cache_sha256"]
        ):
            raise ValueError("Changed consumed recurrence cache/parent")
        p = verifier.model.probability(
            item["frames"], item["attack_frames"], item["release_frames"], item["x"]
        )
        keep = verifier.model.keep(
            item["events"], p, verifier.guardian.probability(item["base_x"]), item["duration"]
        )
        row = {
            **item,
            **{
                k: item[k][keep]
                for k in ("events", "base_x", "frames", "attack_frames", "release_frames")
            },
        }
        row.update(
            x=features(row["events"], row["base_x"]),
            eligible=eligible(row["events"], row["duration"]),
        )
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != observation_manifest["plan_sha256"]:
                raise ValueError("Changed regression observation clock identity")
            for key in ("events", "base_x", "x", "eligible"):
                np.testing.assert_array_equal(saved[key], row[key])
            row["periodicity"] = saved["periodicity"]
        row["release_neighbors"] = neighbor_features(row["events"], row["base_x"])
        items.append(annotate(row))
    if records:
        raise ValueError("Missing fusion regression record")
    return items


def verify(run, fixtures):
    import app.services

    # Legacy consumed-input readers retain their pinned original source tree.
    # Only new, independently hash-inventoried modules come from current source.
    root = Path(__file__).resolve().parents[1]
    app.services.__path__.append(str(root / "backend/app/services"))
    from app.services.attack_local_evidence import observe as acoustic_observe
    from app.services.attack_local_evidence import sequences as attack_sequences
    from app.services.bass_embedding_declutter import EmbeddingSupportModel
    from app.services.release_local_evidence import release_sequences
    from app.services.release_neighbors import features as runtime_neighbors
    from app.services.temporal_note_model import sequences as coarse_sequences
    from app.services.waveform_recurrence import features as runtime_recurrence
    from groove_bass_data import load

    target = run / "runtime-parity.json"
    if target.exists():
        raise ValueError("Preserve completed runtime verification")
    plan, winner, network, normalizer = frozen(run)
    require_regression(run)
    torch.set_num_threads(2)
    export = json.loads((run / "portable-export.json").read_text())
    asset = Path(export["asset"])
    if (
        export["asset_sha256"] != digest(asset)
        or export["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or export["first_pass_sha256"] != digest(run / "original-first-pass.json")
    ):
        raise ValueError("Changed frozen portable asset or first-pass witness")
    with np.load(asset, allow_pickle=False) as saved:
        portable = EmbeddingSupportModel(saved)
    rows = load(Path(plan["baseline_root"]), Path(plan["mixtures_root"]))
    rows.extend(consumed(plan))
    fresh = json.loads((fixtures / "manifest.json").read_text())
    report = json.loads((run / "original-first-pass.json").read_text())
    if (
        not report["passes"]
        or report["test_manifest_sha256"] != digest(fixtures / "manifest.json")
        or len(fresh["items"]) != 77
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
            maximum_feature_error, float(np.max(np.abs(phase - item["periodicity"]), initial=0))
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
                json.dumps({"physical_feature_checks": physical_count, "expected": 77}), flush=True
            )
    if len(rows) != 1377 or len({i["id"] for i in rows}) != 1377:
        raise ValueError("Incomplete portable verification scope")
    guardian = BassAttackDeclutter().guardian
    maximum_probability_error = 0.0
    events_checked = 0
    for item in rows:
        expected = probabilities(network, item, normalizer)
        actual = portable.probability(item)
        maximum_probability_error = max(
            maximum_probability_error, float(np.max(np.abs(expected - actual), initial=0))
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
                portable.keep(item["events"], actual, guard, item["duration"], stricter), keep
            )
        events_checked += len(item["events"])
    frozen(run)
    files = [
        "backend/app/services/" + name + ".py"
        for name in (
            "bass_embedding_declutter",
            "bass_embedding_network",
            "waveform_recurrence",
            "release_neighbors",
        )
    ]
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
            "runtime_source_sha256": {name: digest(root / name) for name in files},
            "verifier_sha256": digest(Path(__file__)),
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
