import json

import numpy as np
import pytest
from acquire_coco_piano_v54 import commands, notes

HEADER = ",volume,vol_fluc,vibrato,brightness,attack,vol_peak_pos,pitch,onset,offset,note_length\n"


def voice(pitch, end=1500, volume=.5):
    return (HEADER + "0,0,0,0,0,0,0,0,0,0,0\n"
            + f"1,{volume},.2,.3,.4,.5,.6,{pitch},0,{end},{end}\n").encode()


def test_authored_mapping_keeps_full_chord_and_long_holds():
    raw = {i: voice(pitch, volume=.25 * (i + 1)) for i, pitch in enumerate((48, 55, 60, 67))}
    actual = commands(raw)
    assert [r[:3] for r in actual] == [(0, 1500, p) for p in (48, 55, 60, 67)]
    assert [r[3] for r in actual] == [32, 64, 96, 127]
    np.testing.assert_array_equal(notes(json.dumps(actual).encode()), actual)


def test_same_key_in_different_voices_is_rejected_without_merging():
    with pytest.raises(ValueError, match="Ambiguous"):
        commands({i: voice(p) for i, p in enumerate((48, 48, 60, 67))})


def test_missing_voice_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        commands({0: voice(48)})


@pytest.mark.parametrize("events", [
    [], [[0, 1500, 48]], [[0, 0, 48, 80]], [[-1, 1500, 48, 80]],
    [[0, 1500, 109, 80]], [[0, 1500, 48, 0]], [[0, 1500.5, 48, 80]],
    [[0, 1500, 48, 80], [1000, 2000, 48, 80]], [[0, 30001, 48, 80]],
])
def test_invalid_or_ambiguous_authored_commands_never_become_labels(events):
    with pytest.raises(ValueError):
        notes(json.dumps(events).encode())


def test_synthesis_mapping_preserves_long_holds_and_complete_work_over_35_seconds():
    from prepare_coco_piano_v54 import transform

    source = np.array([[0, 1500, 48, 80], [0, 10000, 55, 80]], float)
    actual, clock = transform(source, 0)
    assert actual[0, 0] == actual[1, 0]
    assert abs(actual[0, 1] - actual[0, 0] - 6) < 64 / 22050
    assert actual[1, 1] > 43
    assert clock["source_frame_seconds"] == .004
    assert clock["human_timing_labels"] is False


def test_preparation_has_three_variants_for_every_complete_source():
    from prepare_coco_piano_v54 import tasks

    rows = [{"id": f"source-{i}"} for i in range(36)]
    jobs = tasks(rows)
    assert len(jobs) == len({j["id"] for j in jobs}) == 108
    assert {j["source_id"] for j in jobs} == {r["id"] for r in rows}
