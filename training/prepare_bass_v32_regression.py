"""Cache the released V32 survivors of all 469 consumed regression fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
from app.services.bass_embedding_declutter import BassEmbeddingDeclutter
from app.services.bass_harmonic import features
from bass_training_data import eligible, identity
from current_bass_v32_baseline import ASSET_SHA, VERSION, hashes
from prepare_robust_training_stems import digest, preserve
from release_neighbor_features import features as neighbors


def prepare(run, fixtures, output):
    # Run the unchanged e56554c reader replay first on PYTHONPATH; none of the
    # old source contracts are weakened to accept the current website route.
    from verify_embedding_v32_runtime import consumed

    root = Path(__file__).resolve().parents[1]
    archived = root / "training/results/teacher-embedding-v32-v1"
    for name in (
        "plan.json",
        "batch-selection.json",
        "consumed-regression.json",
        "original-first-pass.json",
        "runtime-parity.json",
    ):
        if digest(run / name) != digest(archived / name):
            raise ValueError("Changed approved V32 study evidence")
    old_plan = json.loads((run / "plan.json").read_text())
    report = json.loads((run / "original-first-pass.json").read_text())
    manifest = json.loads((fixtures / "manifest.json").read_text())
    if (
        not report["passes"]
        or report["test_manifest_sha256"] != digest(fixtures / "manifest.json")
        or manifest["plan_sha256"] != digest(fixtures / "plan.json")
        or len(manifest["items"]) != 77
        or not manifest["now_consumed_regression"]
    ):
        raise ValueError("Changed complete consumed first-pass fixtures")
    items = consumed(old_plan)
    for record in manifest["items"]:
        path = fixtures / record["id"] / "features.npz"
        if (
            digest(path) != record["cache_sha256"]
            or digest(Path(record["audio"])) != record["audio_sha256"]
        ):
            raise ValueError("Changed consumed first-pass physical observation")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["plan_sha256"]) != manifest["plan_sha256"]:
                raise ValueError("Changed consumed first-pass identity")
            item = {
                **record,
                **{
                    k: saved[k]
                    for k in (
                        "events",
                        "base_x",
                        "x",
                        "periodicity",
                        "frames",
                        "attack_frames",
                        "release_frames",
                        "release_neighbors",
                        "eligible",
                    )
                },
            }
        item.update(
            reference=np.asarray(record["reference"], float).reshape(-1, 3),
            pitch_reference=np.asarray(record["pitch_reference"], float).reshape(-1, 3),
        )
        items.append(item)
    if (
        len(items) != 469
        or len({r["id"] for r in items}) != 469
        or any(i["group"] != "test" for i in items)
    ):
        raise ValueError("Incomplete or leaking V32 consumed regressions")
    current = hashes()
    output.mkdir(exist_ok=False)
    plan = {
        "version": VERSION,
        "baseline_hashes": current,
        "producer_sha256": digest(Path(__file__)),
        "prior_study_root": str(run),
        "prior_study_plan_sha256": digest(run / "plan.json"),
        "prior_first_pass_manifest_sha256": digest(fixtures / "manifest.json"),
        "required_regression_recordings": 469,
        "no_decoder_inference": True,
        "no_candidate_or_outcome_metrics": True,
        "no_fitting_selection_user_audio_or_scores": True,
        "all_rows_consumed_regression": True,
    }
    preserve(output / "plan.json", plan)
    folder = output / "features"
    folder.mkdir()
    verifier = BassEmbeddingDeclutter(root / "backend/app/assets/bass-embedding-v1.npz", ASSET_SHA)
    records = []
    for item in items:
        p = verifier.model.probability(item)
        keep = verifier.model.keep(
            item["events"], p, verifier.guardian.probability(item["base_x"]), item["duration"]
        )
        events, base = item["events"][keep], item["base_x"][keep]
        record = {
            k: item[k]
            for k in (
                "id",
                "group",
                "source_group",
                "corpus",
                "audio",
                "audio_sha256",
                "duration",
                "seconds",
            )
        }
        record.update(
            reference=item["reference"].tolist(),
            pitch_reference=item["pitch_reference"].tolist(),
            parent_cache_sha256=item["cache_sha256"],
            v32_removed=int((~keep).sum()),
            notes=len(events),
        )
        path = folder / (item["id"] + ".npz")
        arrays = {
            k: item[k][keep] for k in ("frames", "attack_frames", "release_frames", "periodicity")
        }
        np.savez_compressed(
            path,
            **arrays,
            events=events,
            base_x=base,
            x=features(events, base),
            release_neighbors=neighbors(events, base),
            eligible=eligible(events, item["duration"]),
            identity=identity(record),
            plan_sha256=digest(output / "plan.json"),
        )
        record["cache_sha256"] = digest(path)
        records.append(record)
    if current != hashes() or sum(i["v32_removed"] for i in records) != 17:
        raise ValueError("V32 changed or consumed decisions differ from released evidence")
    preserve(
        output / "manifest.json",
        {
            "version": VERSION,
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "all_rows_consumed_regression": True,
            "no_fitting_selection_user_audio_or_scores": True,
        },
    )
    print(
        json.dumps(
            {
                "passes": True,
                "recordings": len(records),
                "removed": 17,
                "manifest_sha256": digest(output / "manifest.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.run, args.fixtures, args.output)
