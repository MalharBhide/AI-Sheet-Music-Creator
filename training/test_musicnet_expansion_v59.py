import copy

import pytest
from audit_musicnet_expansion_v59 import RATE, label_summary, proposals


def row(group="train", identifier="1749", duration=500):
    return {"id": identifier, "group": group, "ensemble": "Solo Piano", "work": ["Composer", identifier, "Work"],
        "clock": {"original_duration": duration, "sample_rate": RATE},
        "excerpts": [{"start_sample": 15*RATE, "end_sample": 75*RATE}],
        "labels_sha256": "labels", "original_member": "musicnet/train_data/"+identifier+".wav"}


def test_only_fixed_later_passages_of_fitting_works_are_proposed():
    jobs = proposals([row(), row("validation", "b"), row("reserved", "c")])
    assert len(jobs) == 3
    assert [job["start_sample"]/RATE for job in jobs] == [95, 155, 215]
    assert all(job["group"] == "train" and job["publisher_id"] == "1749" for job in jobs)


def test_whole_duration_and_existing_sample_ranges_are_respected():
    assert len(proposals([row(duration=115)])) == 1
    assert proposals([row(duration=114.999)]) == []
    changed = row()
    changed["excerpts"].append({"start_sample": 100*RATE, "end_sample": 120*RATE})
    with pytest.raises(ValueError, match="overlaps"):
        proposals([changed])


@pytest.mark.parametrize("mutation", ["duplicate", "leakage", "unknown-group", "instrument", "clock", "nan"])
def test_held_out_works_and_invalid_source_clocks_cannot_enter_new_training(mutation):
    first, second = row(), row("validation", "b")
    if mutation == "duplicate":
        second["id"] = first["id"]
    elif mutation == "leakage":
        second["work"] = copy.deepcopy(first["work"])
    elif mutation == "unknown-group":
        first["group"] = "test"
    elif mutation == "instrument":
        first["ensemble"] = "String Quartet"
    elif mutation == "clock":
        first["clock"]["sample_rate"] = 22050
    else:
        first["clock"]["original_duration"] = float("nan")
    with pytest.raises(ValueError):
        proposals([first, second])


def label(start, end, pitch=48, instrument=1):
    return {"start_time": str(round(start*RATE)), "end_time": str(round(end*RATE)),
            "note": str(pitch), "instrument": str(instrument)}


def test_crossing_holds_are_counted_as_support_without_fabricated_boundaries():
    job = proposals([row()])[0]
    summary = label_summary([label(90, 96), label(96, 97, 52), label(114, 120), label(40, 41)], job)
    assert summary == {"complete_publisher_note_intervals": 1,
                       "crossing_intervals_for_pitch_support_only": 2, "distinct_pitches": 2}


@pytest.mark.parametrize("bad", [label(2, 1), label(1, 2, 20), label(1, 2, instrument=42), label(499, 501)])
def test_invalid_publisher_note_labels_are_rejected_even_outside_the_crop(bad):
    with pytest.raises(ValueError):
        label_summary([bad], proposals([row()])[0])
