"""Acquire a predetermined public-domain, work-disjoint Joplin MIDI subset."""

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

SPLIT = {441: "train", 345: "train", 482: "train", 1540: "train",
         263: "validation", 1899: "reserved"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def acquire(output):
    output.mkdir(exist_ok=False)
    preserve(output / "plan.json", {
        "split": SPLIT, "producer_sha256": digest(Path(__file__)),
        "policy": "Only pages explicitly marking the edition Public Domain. All crops/variants of one work share its split. Reserved MIDI is downloaded but never rendered before candidate freezing.",
        "no_user_audio_or_scores": True})
    records = []
    for identifier, group in SPLIT.items():
        url = f"https://www.mutopiaproject.org/cgibin/piece-info.cgi?id={identifier}"
        with urllib.request.urlopen(url, timeout=45) as response:
            page = response.read(1_000_001)
        text = page.decode("utf-8")
        if len(page) > 1_000_000 or "Public Domain" not in text or "Joplin" not in text:
            raise ValueError("Changed publisher license or composer")
        links = re.findall(r'href=[\"\']([^\"\']+)[\"\']', text, re.I)
        midis = [urllib.parse.urljoin(url, link) for link in links
                 if link.lower().endswith((".mid", ".midi"))]
        if len(midis) != 1 or urllib.parse.urlparse(midis[0]).hostname not in (
                "www.mutopiaproject.org", "mutopiaproject.org", "www.ibiblio.org"):
            raise ValueError("Ambiguous publisher MIDI")
        with urllib.request.urlopen(midis[0].replace("http:", "https:"), timeout=45) as response:
            midi = response.read(1_000_001)
        if not midi.startswith(b"MThd") or not 14 < len(midi) < 1_000_000:
            raise ValueError("Invalid publisher MIDI object")
        (output / f"{identifier}.html").write_bytes(page)
        (output / f"{identifier}.mid").write_bytes(midi)
        title = re.search(r"<h2[^>]*>(.*?)</h2>", text, re.S | re.I)
        records.append({"id": identifier, "group": group, "composer": "Scott Joplin",
                        "title": re.sub("<[^>]+>", "", title[1]).strip() if title else str(identifier),
                        "publisher": url, "midi_url": midis[0], "license": "Public Domain",
                        "page_sha256": digest(output / f"{identifier}.html"),
                        "midi_sha256": digest(output / f"{identifier}.mid")})
        print(json.dumps(records[-1]), flush=True)
    preserve(output / "manifest.json", {"items": records,
        "plan_sha256": digest(output / "plan.json"), "no_user_audio_or_scores": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    acquire(parser.parse_args().output)
