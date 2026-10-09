"""One frozen first pass on unused MusicNet piano works and new authored MP3s."""

import argparse
import contextlib
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from app.services.bass_harmonic import BassHarmonic, features
from app.services.note_evidence import note_features
from app.services.piano_transcription import _GeneralEngine
from app.services.temporal_note_model import sequences as coarse_sequences
from attack_local_features import observe as acoustic_observe
from attack_local_features import sequences as attack_sequences
from bass_joint_timing_v18 import margin
from bass_temporal_data import annotate
from bass_training_data import eligible
from evaluate_musicnet_anchor_v33 import frozen, require_regression
from musicnet_anchor_model import probabilities
from periodicity_features import features as recurrence_features
from periodicity_features import observe
from prepare_bass_positive_data import RATE, original
from prepare_bass_texture_v12 import texture
from prepare_musicnet_v32 import references
from prepare_robust_training_stems import digest, preserve
from prepare_slakh_bass import encode
from release_local_features import release_sequences
from release_neighbor_features import features as neighbor_features
from reserved_musicnet_v33 import declaration
from score_declutter_local import evaluate


def regression_gate(run, winner):
    require_regression(run)
    report = json.loads((run / "consumed-regression.json").read_text())
    plan, _, _, _ = frozen(run)
    observations = Path(plan["regression_root"]) / "manifest.json"
    manifest = json.loads(observations.read_text())
    ids = [i["id"] for i in manifest["items"]]
    if (
        len(ids) != 469
        or len(set(ids)) != 469
        or report["regression_manifest_sha256"] != digest(observations)
    ):
        raise ValueError("Changed complete V33 regression observation identities")
    for name, threshold in (
        ("raw", winner["threshold"]),
        ("margin", margin(winner["threshold"])),
    ):
        row = report[name]
        if (
            not row["passes"]
            or row["false_notes_removed"] < 0
            or [r["id"] for r in row["per_recording"]] != ids
            or row["remove_threshold"] != threshold
            or row["guardian_threshold"] != winner["guardian_threshold"]
            or row["retimed_notes"]
            or abs(row["onset_error_reduction_seconds"]) > 1e-9
        ):
            raise ValueError("Failed or changed frozen V33 confidence configuration")
    return digest(run / "consumed-regression.json")


def verify(run, output):
    report_path = run / "original-first-pass.json"
    if report_path.exists():
        raise ValueError(
            "Preserve original V33 first-pass result; no replacement tests"
        )
    plan, winner, network, normalizer = frozen(run)
    regression = regression_gate(run, winner)
    if plan["new_first_pass_seeds"] != list(range(278101, 278133)):
        raise ValueError("Changed V33 predeclared MP3 seeds")
    declared = declaration(Path(plan["piano_root"]))
    if declared != plan["first_pass_declaration"]:
        raise ValueError("Changed predeclared reserved MusicNet identities")
    jobs = [
        ("musicnet", job["id"], job, None, job["excerpt_index"])
        for job in declared["jobs"]
    ]
    jobs.extend(
        ("procedural", "musicnet-v33-fresh-" + str(seed), seed, None, index)
        for index, seed in enumerate(declared["procedural_seeds"])
    )
    expected = declared["expected_recordings"]
    if len(jobs) != expected or len({j[1] for j in jobs}) != expected:
        raise ValueError("Incomplete declared V33 first-pass identities")
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/fresh_musicnet_anchor_v33.py",
        "training/attack_local_features.py",
        "training/release_local_features.py",
        "training/prepare_musicnet_v32.py",
        "training/reserved_musicnet_v33.py",
        "training/periodicity_features.py",
        "training/musicnet_anchor_model.py",
        "training/acquire_musicnet_subset.py",
        "training/release_neighbor_features.py",
        "training/prepare_bass_positive_data.py",
        "training/prepare_bass_texture_v12.py",
        "training/prepare_slakh_bass.py",
        "backend/app/services/piano_transcription.py",
    )
    fixture_plan = {
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "consumed_regression_sha256": regression,
        "baseline_hashes": plan["baseline_hashes"],
        "first_pass_declaration": declared,
        "procedural_seeds": plan["new_first_pass_seeds"],
        "mp3_bitrates": [48, 128, 192],
        "code_sha256": {name: digest(root / name) for name in names},
        "expected_recordings": expected,
        "no_fitting_selection_user_audio_or_scores": True,
    }
    if output.exists():
        if (
            json.loads((output / "plan.json").read_text()) != fixture_plan
            or (output / "manifest.json").exists()
        ):
            raise ValueError("Preserve changed or complete first-pass fixtures")
    else:
        output.mkdir()
        preserve(output / "plan.json", fixture_plan)
    engine = _GeneralEngine("balanced")
    native = engine.predict_function
    evidence = {}

    def capture(*args, **kwargs):
        result = native(*args, **kwargs)
        evidence["acoustic"] = result[0]
        return result

    engine.predict_function = capture
    records = []
    items = []
    for family, identifier, choice, _kind, index in jobs:
        folder = output / identifier
        folder.mkdir(exist_ok=True)
        completion = folder / "record.json"
        if completion.exists():
            record = json.loads(completion.read_text())
            if (
                record["plan_sha256"] != digest(output / "plan.json")
                or record["cache_sha256"] != digest(folder / "features.npz")
                or record["audio_sha256"] != digest(folder / "decoded.wav")
            ):
                raise ValueError("Changed partial first-pass observation")
            with np.load(folder / "features.npz", allow_pickle=False) as saved:
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
            item.update(
                reference=np.asarray(record["reference"], float).reshape(-1, 3),
                pitch_reference=np.asarray(record["pitch_reference"], float).reshape(
                    -1, 3
                ),
            )
        else:
            if family == "musicnet":
                crop = choice["excerpt"]
                audio_path = Path(crop["audio"])
                labels_path = Path(choice["labels"])
                if (
                    digest(audio_path) != crop["audio_sha256"]
                    or digest(labels_path) != choice["labels_sha256"]
                ):
                    raise ValueError(
                        "Changed reserved original piano waveform or labels"
                    )
                samples, source_rate = sf.read(audio_path, dtype="float32")
                duration = 20.0
                if (
                    source_rate != 44100
                    or len(samples) != round(duration * source_rate)
                    or not np.isfinite(samples).all()
                    or np.max(np.abs(samples), initial=0) <= 0
                    or crop["end_sample"] - crop["start_sample"] != len(samples)
                ):
                    raise ValueError("Invalid reserved piano sample clock")
                if samples.ndim == 2:
                    samples = samples.mean(axis=1)
                if samples.ndim != 1:
                    raise ValueError("Invalid reserved piano channels")
                with labels_path.open() as stream:
                    reference, support = references(
                        list(csv.DictReader(stream)),
                        source_rate,
                        crop["start_sample"],
                        crop["end_sample"],
                        choice["clock"]["original_duration"],
                    )
                labels = np.asarray(reference, float).reshape(-1, 3)
                pitch_labels = np.asarray(support, float).reshape(-1, 3)
                corpus = "reserved-musicnet-real-piano"
                group = "musicnet/" + "/".join(choice["work"])
                used = [choice["publisher_id"]]
            else:
                samples, labels = (original if index < 16 else texture)(choice)
                pitch_labels = labels.copy()
                duration = 30.0
                corpus = "original-periodicity-first-pass"
                group = "original-bass-seed-" + str(choice)
                used = []
                source_rate = RATE
            _, clock = encode(
                samples, source_rate, folder, fixture_plan["mp3_bitrates"][index % 3]
            )
            audio = folder / "decoded.wav"
            samples, rate = sf.read(audio, dtype="float32")
            if (
                rate != RATE
                or samples.ndim != 1
                or len(samples) != round(duration * RATE)
            ):
                raise ValueError("Changed first-pass physical waveform clock")
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
                midi = engine.predict(audio, "bass", 90.0)
                notes = [n for part in midi.instruments for n in part.notes]
                base_x = note_features(
                    samples, rate, evidence["acoustic"], notes, include_context=True
                )
            events = np.asarray(
                [[n.start, n.end, n.pitch, n.velocity] for n in notes], float
            ).reshape(-1, 4)
            values = {
                "events": events,
                "base_x": base_x,
                "x": features(events, base_x),
                "periodicity": recurrence_features(observe(samples, rate), events),
                "eligible": eligible(events, duration),
                "release_neighbors": neighbor_features(events, base_x),
            }
            observed = acoustic_observe(samples, rate)
            values.update(
                frames=coarse_sequences(observed["cqt"], events),
                attack_frames=attack_sequences(observed, events),
                release_frames=release_sequences(observed, events),
            )
            np.savez_compressed(
                folder / "features.npz",
                **values,
                plan_sha256=digest(output / "plan.json"),
            )
            record = {
                "id": identifier,
                "group": "test",
                "source_group": group,
                "corpus": corpus,
                "audio": str(audio),
                "audio_sha256": digest(audio),
                "duration": duration,
                "seconds": duration,
                "reference": labels.tolist(),
                "pitch_reference": pitch_labels.tolist(),
                "source_notes": used,
                "codec_clock": clock,
                "cache_sha256": digest(folder / "features.npz"),
                "plan_sha256": digest(output / "plan.json"),
            }
            preserve(completion, record)
            item = {
                **record,
                **values,
                "reference": labels,
                "pitch_reference": pitch_labels,
            }
        records.append(record)
        items.append(annotate(item))
        print(
            json.dumps(
                {
                    "cached": len(records),
                    "total": expected,
                    "id": identifier,
                    "notes": len(item["events"]),
                }
            ),
            flush=True,
        )
    if fixture_plan["code_sha256"] != {name: digest(root / name) for name in names}:
        raise ValueError("First-pass producer changed during observation")
    frozen(run)
    regression_gate(run, winner)
    if declaration(Path(plan["piano_root"])) != declared:
        raise ValueError("Changed reserved source during first pass")
    preserve(
        output / "manifest.json",
        {
            "plan_sha256": digest(output / "plan.json"),
            "items": records,
            "now_consumed_regression": True,
            "no_fitting_selection_user_audio_or_scores": True,
        },
    )
    guardian = BassHarmonic().models[1]
    ps = [probabilities(network, i, normalizer) for i in items]
    guards = [guardian.probability(i["base_x"]) for i in items]
    memo = {}
    raw = evaluate(
        items, ps, guards, winner["threshold"], winner["guardian_threshold"], memo
    )
    stricter = evaluate(
        items,
        ps,
        guards,
        margin(winner["threshold"]),
        winner["guardian_threshold"],
        memo,
    )
    result = {
        "passes": bool(
            raw["passes"]
            and stricter["passes"]
            and raw["false_notes_removed"] > 0
            and stricter["false_notes_removed"] > 0
        ),
        "raw": raw,
        "margin": stricter,
        "recordings": expected,
        "first_pass_complete": True,
        "now_consumed_regression": True,
        "no_retuning": True,
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "batch_selection_sha256": digest(run / "batch-selection.json"),
        "test_manifest_sha256": digest(output / "manifest.json"),
        "no_user_audio_or_scores": True,
        "new_weights_deployed": False,
    }
    preserve(report_path, result)
    print(
        json.dumps({k: result[k] for k in ("passes", "recordings")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.run, args.output)
