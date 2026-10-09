"""Read verified Oxford pair identities only; reserve the complete pianist group."""

import json
from pathlib import Path

from acquire_oxford_midi_test import ARCHIVE, LICENSE
from prepare_robust_training_stems import digest

SHA256 = "0e440b140af7fb5d53d8ec34a17adff26479b2fada537688c36e5386d25a8a1b"
IDS = ("5", "10", "21", "23", "24", "25", "26", "27")


def declaration(source):
    plan = json.loads((source / "plan.json").read_text())
    witness = json.loads((source / "source.json").read_text())
    if (
        witness["archive_url"] != ARCHIVE
        or witness["sha256"] != SHA256
        or digest(source / "source.zip") != SHA256
        or witness["bytes"] != 69276729
        or (source / "source.zip").stat().st_size != witness["bytes"]
        or witness["plan_sha256"] != digest(source / "plan.json")
        or plan["license_url"] != LICENSE
        or plan["producer_sha256"]
        != digest(Path(__file__).with_name("acquire_oxford_midi_test.py"))
        or plan["license_sha256"] != digest(source / "LICENSE.publisher.txt")
        or plan["publisher_sha256"] != digest(source / "publisher.html")
        or witness["license"] != "CC BY 4.0"
        or not witness["all_members_crc_verified"]
        or not witness["no_fitting_or_inference"]
    ):
        raise ValueError("Changed or unverified original Oxford MIDI-test package")
    members = {row["member"]: row for row in witness["members"]}
    pairs = [
        {
            "id": identifier,
            "video_member": f"MIDItest/miditest_videos/{identifier}.mp4",
            "midi_member": f"MIDItest/miditest_MIDI/{identifier}.mid",
        }
        for identifier in IDS
    ]
    expected_media = {
        pair[key] for pair in pairs for key in ("video_member", "midi_member")
    }
    actual_media = {name for name in members if name.endswith((".mp4", ".mid"))}
    if len(members) != 21 or actual_media != expected_media:
        raise ValueError("Changed complete Oxford pair identities")
    return {
        "source_root": str(source),
        "source_sha256": digest(source / "source.json"),
        "archive_sha256": SHA256,
        "pairs": pairs,
        "source_group": "oxford-midi-test-one-recorded-pianist",
        "all_eight_reserved": True,
        "no_training_selection_or_decoder_inference": True,
        "not_part_of_frozen_v33_first_pass": True,
    }
