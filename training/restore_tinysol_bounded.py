"""Restore the original isolated-pitch corpus, retaining no full archive."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

BLOCK = 1024 * 1024


def download(url, path, expected):
    """Bound every write and promote only exact publisher bytes; never retry."""
    path = Path(path)
    partial = Path(str(path) + ".partial")
    if (not url.startswith("https://") or type(expected["bytes"]) is not int or expected["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
            or not re.fullmatch(r"[0-9a-f]{32}", expected["md5"])):
        raise ValueError("Invalid pinned source identity")
    if path.exists() or partial.exists() or path.is_symlink() or partial.is_symlink():
        raise ValueError("Preserve existing source or partial transfer")
    count, md5, sha = 0, hashlib.md5(), hashlib.sha256()
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        size = response.headers.get("Content-Length")
        if response.status != 200 or (size is not None and int(size) != expected["bytes"]):
            raise ValueError("Changed publisher transfer size or status")
        with partial.open("xb") as output:
            while chunk := response.read(BLOCK):
                if len(chunk) > BLOCK or count + len(chunk) > expected["bytes"]:
                    raise ValueError("Oversized bounded source transfer")
                count += len(chunk)
                output.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
                if count % (128 * BLOCK) < len(chunk):
                    print(json.dumps({"file": path.name, "downloaded_bytes": count,
                                      "total": expected["bytes"]}), flush=True)
    actual = {"bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}
    if actual != expected:
        raise ValueError("Incomplete or changed publisher bytes; partial preserved")
    partial.rename(path)
    return actual


def restore(production, root):
    from acquire_tinysol_v45 import acquire, checksums, sources
    from approved_model_finetune import storage_preflight

    production, root = Path(production), Path(root)
    evidence = production / "training/results/tinysol-v6-source-assessment-v1"
    expected = json.loads((evidence / "acquisition-plan.json").read_text())
    urls = json.loads((evidence / "metadata-assessment.json").read_text())
    archive = root / "TinySOL-v6-restoration.tar.gz"
    metadata = root / "TinySOL_metadata-v6.csv"
    output = root / "tinysol-v45-source"
    if output.exists():
        raise ValueError("Preserve existing original TinySOL source")
    # 2 GB piano + 3 GB dynamics + at most 2 GB archive/extracted source.
    audit = {"left-precision-v43-replay/training/results": production / "training/results"}
    print(json.dumps({"storage_preflight": storage_preflight(root, 7_000_000_000, audited_links=audit),
                      "existing_corpus_only": True, "no_inference": True}), flush=True)
    download(urls["metadata_url"], metadata, expected["metadata"])
    download(urls["audio_url"], archive, expected["source"])
    acquire(archive, metadata, output)
    rows = sources(output)
    original = json.loads((evidence / "preparation-v2-plan.json").read_text())
    if checksums(output / "manifest.json")["sha256"] != original["source_manifest_sha256"]:
        raise ValueError("Restored source differs from the original frozen corpus")
    archive.unlink()
    result = {"restoration_complete": True, "recordings": len(rows),
              "source_manifest_identical": True, "archive_removed_after_verification": True,
              "pitch_supervision_only": True, "no_timing_labels_invented": True,
              "no_inference": True, "no_user_audio_or_scores": True,
              "storage": storage_preflight(root, audited_links=audit)}
    (root / "tinysol-restoration-complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    restore(Path("/src"), Path("/training"))
