"""Extract two audited CC0 piano arrangements into a composer-wide train-only group."""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

from audit_pdmx_archives import CATALOG, FILES, safe_path, verify_file
from pdmx_piano_clock_v53 import notes
from prepare_robust_training_stems import digest, preserve

VERSION = "pdmx-scarlatti-train-only-source-v53-v1"
SELECTED = {"2149496": "K.531", "5894834": "K.208"}
URL = "https://creativecommons.org/publicdomain/zero/1.0/"


def piano_metadata(row, metadata):
    data = metadata["data"]
    score = data["score"]
    return (row["composer_name"] == score.get("composer_name") == "Domenico Scarlatti"
        and row["license"] == score.get("license") == "cc-zero"
        and row["license_url"] == data.get("license_url") == URL
        and row["subset:no_license_conflict"] == row["subset:all_valid"] == "True"
        and row["tracks"] == "0" and score.get("parts") == 1
        and score.get("instruments") == [{"name": "Piano", "url_to_search": "/sheetmusic/piano", "count": 1}]
        and not any((score.get("is_blocked"), data.get("is_blocked"), data.get("opened_dispute")))
        and not score.get("publisher"))


def source_files(root, rows):
    values = {}
    for kind in ("metadata", "mid"):
        wanted = {safe_path(r[kind], kind): r for r in rows}
        with tarfile.open(root / f"PDMX-v9-{kind}.tar.gz", "r|gz") as archive:
            for member in archive:
                name = safe_path(member.name, kind)
                if name not in wanted:
                    continue
                row = wanted.pop(name)
                if not member.isfile() or not 0 < member.size < 1024 * 1024:
                    raise ValueError("Unsafe selected source member")
                raw = archive.extractfile(member).read()
                if len(raw) != member.size:
                    raise ValueError("Truncated selected source")
                values[Path(row["metadata"]).stem, kind] = raw
        if wanted:
            raise ValueError("Missing selected symbolic source")
    return values


def acquire(root, output):
    inputs = {"catalog": verify_file(root / "PDMX-v9.csv", *CATALOG)}
    for kind in ("metadata", "mid"):
        inputs[kind] = verify_file(root / f"PDMX-v9-{kind}.tar.gz", *FILES[kind][:2])
    rows = []
    with (root / "PDMX-v9.csv").open(newline="") as reader:
        for row in csv.DictReader(reader):
            if Path(row["metadata"]).stem in SELECTED:
                rows.append(row)
    if len(rows) != 2 or len({r["metadata"] for r in rows}) != 2:
        raise ValueError("Changed preselected source identities")
    output.mkdir(exist_ok=False)
    plan = {"version": VERSION, "publisher": "https://zenodo.org/records/15571083",
        "dataset_license": "CC BY 4.0", "score_license": "CC0 1.0", "inputs": inputs,
        "selected": SELECTED, "producer_sha256": digest(Path(__file__)),
        "clock_source_sha256": digest(Path(__file__).with_name("pdmx_piano_clock_v53.py")),
        "partition": "Entire Domenico Scarlatti composer group is training-only. No PDMX validation or test score. Existing held-out real-piano, NRG-CP and TinySOL validation gates remain unchanged. Catalog best_path groups are not work identities.",
        "source_review": "Exact public-domain keyboard works K.531 in E major and K.208 in A major. Matching conflict-free CC0 score and website metadata; explicit solo piano. K.208 uploader describes personal fingering and hand distribution, covered by the score grant. Neither selected description cites a modern edition or external performance. No custom audio is acquired. Other screened scores are excluded for ambiguous key clocks, pedal, harpsichord intent, conflicting work identifiers or modern-edition provenance.",
        "human_performance_labels": False, "no_user_audio_or_scores": True,
        "no_audio_rendered_or_inferred": True}
    preserve(output / "plan.json", plan)
    raw = source_files(root, rows)
    records = []
    for row in rows:
        identifier = Path(row["metadata"]).stem
        metadata = json.loads(raw[identifier, "metadata"])
        if not piano_metadata(row, metadata):
            raise ValueError("Changed selected piano rights or identity metadata")
        events = notes(raw[identifier, "mid"])
        for kind in ("metadata", "mid"):
            suffix = "json" if kind == "metadata" else "mid"
            (output / f"{identifier}.{suffix}").write_bytes(raw[identifier, kind])
        records.append({"id": "pdmx-v53-" + identifier, "score_id": identifier,
            "work": SELECTED[identifier], "group": "train", "source_group": "pdmx/domenico-scarlatti",
            "title": row["title"], "source_member": row["mid"], "metadata_member": row["metadata"],
            "source_url": metadata["data"]["score"]["url"],
            "midi": str(output / f"{identifier}.mid"),
            "midi_sha256": hashlib.sha256(raw[identifier, "mid"]).hexdigest(),
            "metadata_sha256": hashlib.sha256(raw[identifier, "metadata"]).hexdigest(),
            "source_notes": len(events), "source_seconds": float(events[:, 1].max())})
    preserve(output / "manifest.json", {"version": VERSION, "items": records,
        "plan_sha256": digest(output / "plan.json"), "no_user_audio_or_scores": True})
    (output / "ATTRIBUTION.txt").write_text("PDMX v9: Phillip Long, Zachary Novack, Julian McAuley, Taylor Berg-Kirkpatrick.\n"
        "https://zenodo.org/records/15571083 ; dataset CC BY 4.0.\n"
        "Selected uploaded scores: CC0 1.0. Unmodified source MIDI and metadata, local only.\n"
        + "".join(f"{r['title']}: {r['source_url']}\n" for r in records))
    print(json.dumps({"sources": len(records), "training_only": True}), flush=True)


def sources(folder):
    plan, manifest = [json.loads((folder / n).read_text()) for n in ("plan.json", "manifest.json")]
    rows = manifest["items"]
    if (plan["version"] != VERSION or manifest["version"] != VERSION or len(rows) != 2
            or plan["selected"] != SELECTED or manifest["plan_sha256"] != digest(folder / "plan.json")
            or plan["producer_sha256"] != digest(Path(__file__))
            or plan["clock_source_sha256"] != digest(Path(__file__).with_name("pdmx_piano_clock_v53.py"))
            or {r["score_id"] for r in rows} != set(SELECTED)
            or not plan["no_user_audio_or_scores"] or not manifest["no_user_audio_or_scores"]):
        raise ValueError("Changed train-only symbolic piano source")
    for row in rows:
        if (row["group"] != "train" or row["source_group"] != "pdmx/domenico-scarlatti"
                or Path(row["midi"]) != folder / f"{row['score_id']}.mid"
                or row["id"] != "pdmx-v53-" + row["score_id"]
                or row["work"] != SELECTED[row["score_id"]]
                or digest(Path(row["midi"])) != row["midi_sha256"]
                or digest(folder / f"{row['score_id']}.json") != row["metadata_sha256"]):
            raise ValueError("Changed source content or composer partition")
        events = notes(Path(row["midi"]).read_bytes())
        if len(events) != row["source_notes"] or float(events[:, 1].max()) != row["source_seconds"]:
            raise ValueError("Changed source key clock")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    acquire(**vars(parser.parse_args()))
