"""Stream complete publisher checksums; retain only metadata-selected training sources."""

import argparse
import bz2
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from assess_cocochorales import BASE, PUBLISHER, checksum_list, fetch
from prepare_robust_training_stems import digest, preserve

MAX_MEMBER = 128 * 1024 * 1024
ASSESSMENT_SHA = "559602e1bfedbdf833276fe80a7df45099902b7a25ac21024db8dcf750a77f7d"


class CheckedReader(io.RawIOBase):
    def __init__(self, source, expected):
        self.source, self.expected = source, expected
        self.count, self.progress = 0, 0
        self.md5, self.sha = hashlib.md5(), hashlib.sha256()

    def readable(self):
        return True

    def read(self, amount=-1):
        if amount < 0:
            raise ValueError("Unbounded source read")
        data = self.source.read(amount)
        self.count += len(data)
        self.md5.update(data)
        self.sha.update(data)
        if self.count > self.expected["bytes"]:
            raise ValueError("Oversized publisher stream")
        if self.count - self.progress >= 128 * 1024 * 1024:
            self.progress = self.count
            print(
                json.dumps(
                    {
                        "object": self.expected["name"],
                        "compressed_bytes": self.count,
                        "total": self.expected["bytes"],
                    }
                ),
                flush=True,
            )
        return data

    def verify(self):
        if self.count != self.expected["bytes"] or self.md5.hexdigest() != self.expected["md5"]:
            raise ValueError("Incomplete or corrupt publisher archive")


def safe_path(member):
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in member.name
        or member.issym()
        or member.islnk()
    ):
        raise ValueError("Unsafe publisher TAR member")
    if not member.isdir() and (not member.isfile() or not 0 <= member.size <= MAX_MEMBER):
        raise ValueError("Unexpected or oversized TAR member")
    return path


def select(assessment):
    result, used = [], set()
    for archive, group, count in zip(
        assessment["archives"], ("train", "validation"), (12, 6), strict=True
    ):
        rows = sorted(archive["records"], key=lambda r: int(r["original_midi"].split(".")[0]))
        if len(rows) != 2000 or len({r["original_midi"] for r in rows}) != len(rows):
            raise ValueError("Changed original source-group metadata")
        for index in range(count):
            row = rows[index * (len(rows) - 1) // (count - 1)]
            identifier = PurePosixPath(row["member"]).stem
            if (
                row["original_midi"] in used
                or row["ensemble"] != "string"
                or not re.fullmatch(r"string_track\d{6}", identifier)
            ):
                raise ValueError("Source-group leakage or changed ensemble")
            used.add(row["original_midi"])
            result.append({**row, "id": identifier, "group": group})
    return result


def initialize(assessment_root, output):
    if digest(assessment_root / "assessment.json") != ASSESSMENT_SHA:
        raise ValueError("Changed verified source assessment")
    assessment = json.loads((assessment_root / "assessment.json").read_text())
    sums = (assessment_root / "publisher-md5.txt").read_bytes()
    known = checksum_list(sums)
    publisher = fetch(PUBLISHER, 2 * 1024 * 1024)
    if b"CC-BY 4.0" not in publisher:
        raise ValueError("Dataset publisher no longer declares expected license")
    objects = []
    for group in ("train", "valid"):
        for component in ("note_expression", "main_dataset"):
            name = f"{component}/{group}/1.tar.bz2"
            with urllib.request.urlopen(
                urllib.request.Request(BASE + name, method="HEAD"), timeout=60
            ) as response:
                size = int(response.headers["Content-Length"])
                generation, etag = response.headers["x-goog-generation"], response.headers["ETag"]
                if (
                    response.status != 200
                    or not generation.isdigit()
                    or etag != '"' + known[name] + '"'
                    or not 0 < size < 6 * 1024**3
                ):
                    raise ValueError("Changed immutable source object identity")
                objects.append(
                    {
                        "name": name,
                        "bytes": size,
                        "generation": generation,
                        "etag": etag,
                        "md5": known[name],
                    }
                )
    plan = {
        "assessment_root": str(assessment_root),
        "assessment_sha256": ASSESSMENT_SHA,
        "publisher_url": PUBLISHER,
        "publisher_sha256": hashlib.sha256(publisher).hexdigest(),
        "publisher_md5_list_sha256": hashlib.sha256(sums).hexdigest(),
        "producer_sha256": digest(Path(__file__)),
        "objects": objects,
        "selected": select(assessment),
        "license": "CC BY 4.0",
        "attribution": assessment["attribution"],
        "selection": "Twelve train and six validation original MIDI identities, evenly spaced after numeric MIDI-ID sorting of the two verified metadata packages. No original test archives, label outcomes or predictions select examples. No other ensembles are used.",
        "no_fitting_inference_or_user_audio": True,
        "no_complete_archives_stored": True,
        "labels": "Retain original MIDI and synthesis-control CSVs. Use synthesis CSV frame timestamps (250 Hz) for expressive timing; verify pitch and label clock semantics before observation. No pickle loading.",
    }
    output.mkdir(exist_ok=False)
    (output / "publisher.html").write_bytes(publisher)
    preserve(output / "plan.json", plan)
    print(
        json.dumps(
            {
                "initialized": True,
                "selected_recordings": 18,
                "total_compressed_bytes": sum(o["bytes"] for o in objects),
            }
        ),
        flush=True,
    )


def consume(reader, selected, component, output):
    retained, seen, total = {}, set(), 0
    # Explicitly drain decompression after the TAR terminator so concatenated
    # pbzip2 streams and all compressed footers are checked, not merely hashed.
    with bz2.BZ2File(reader, "rb") as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r|") as archive:
            for member in archive:
                path = safe_path(member)
                if member.isdir():
                    continue
                if str(path) in seen:
                    raise ValueError("Duplicate TAR file identity")
                seen.add(str(path))
                total += member.size
                if total > 64 * 1024**3:
                    raise ValueError("Oversized decompressed publisher object")
                if not path.parts or path.parts[0] not in selected:
                    continue
                allowed = (".wav", ".mid", ".yaml") if component == "main_dataset" else (".csv",)
                if path.suffix not in allowed:
                    raise ValueError("Unexpected selected source member")
                source = archive.extractfile(member)
                data = source.read(MAX_MEMBER + 1)
                if len(data) != member.size:
                    raise ValueError("Truncated selected source member")
                target = output / component / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                retained[str(path)] = {"bytes": len(data), "sha256": digest(target)}
        while decompressed.read(1024 * 1024):
            pass
    # Tar decompression can read ahead, but it must still consume the complete
    # immutable original object. Unread trailing bytes cannot be ignored.
    while reader.read(1024 * 1024):
        pass
    reader.verify()
    return retained


def acquire_group(output, group):
    plan = json.loads((output / "plan.json").read_text())
    if digest(Path(__file__)) != plan["producer_sha256"] or plan["license"] != "CC BY 4.0":
        raise ValueError("Changed source acquisition plan")
    selected = {r["id"]: r for r in plan["selected"] if r["group"] == group}
    publisher_group = "valid" if group == "validation" else "train"
    objects = [o for o in plan["objects"] if o["name"].split("/")[1] == publisher_group]
    folder = output / "retained" / group
    folder.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        component = obj["name"].split("/")[0]
        witness = output / f"{group}-{component}-verified.json"
        if witness.exists():
            saved = json.loads(witness.read_text())
            if saved["object"] != obj or saved["plan_sha256"] != digest(output / "plan.json"):
                raise ValueError("Changed completed object witness")
            for name, row in saved["retained"].items():
                if digest(folder / component / name) != row["sha256"]:
                    raise ValueError("Changed retained source member")
            continue
        address = BASE + obj["name"] + "?generation=" + obj["generation"]
        request = urllib.request.Request(address, headers={"If-Match": obj["etag"]})
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                response.status != 200
                or response.headers["ETag"] != obj["etag"]
                or response.headers["x-goog-generation"] != obj["generation"]
                or int(response.headers["Content-Length"]) != obj["bytes"]
            ):
                raise ValueError("Changed immutable publisher stream")
            reader = CheckedReader(response, obj)
            retained = consume(reader, selected, component, folder)
        for identifier, row in selected.items():
            files = {name for name in retained if PurePosixPath(name).parts[0] == identifier}
            if component == "main_dataset":
                if (
                    len(files) != 11
                    or sum(name.endswith(".wav") for name in files) != 5
                    or sum(name.endswith(".mid") for name in files) != 5
                    or digest(folder / component / identifier / "metadata.yaml")
                    != row["member_sha256"]
                ):
                    raise ValueError("Incomplete selected source pair or changed metadata")
            elif len(files) != 4:
                raise ValueError("Incomplete four-part synthesis labels")
        preserve(
            witness,
            {
                "object": obj,
                "plan_sha256": digest(output / "plan.json"),
                "sha256": reader.sha.hexdigest(),
                "bytes": reader.count,
                "complete_md5_and_decompression_footer_verified": True,
                "retained": retained,
                "no_fitting_inference_or_user_audio": True,
            },
        )
        print(json.dumps({"verified": obj["name"], "retained_files": len(retained)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--group", choices=("train", "validation"))
    args = parser.parse_args()
    if args.assessment and not args.group:
        initialize(args.assessment, args.output)
    elif args.group and not args.assessment:
        acquire_group(args.output, args.group)
    else:
        parser.error("Choose initialization or one acquisition group")
