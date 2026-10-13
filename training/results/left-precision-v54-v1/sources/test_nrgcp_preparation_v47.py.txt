"""Source-independent checks for the synthetic piano labeling contract."""

import numpy as np
import pytest
from prepare_nrgcp_v47 import COUNTS, VARIANTS, tasks, transform


def test_variants_keep_whole_source_chords_and_repeated_note_rhythm():
    source = [[0, 96, 19, 95], [0, 288, 43, 95], [192, 288, 19, 90]]
    for variant, (bpm, velocity, _) in enumerate(VARIANTS):
        actual, clock = transform(source, variant)
        assert len(actual) == len(source)
        assert len(set(actual[:, 2]-np.array(source)[:, 2])) == 1
        assert actual[0, 0] == actual[1, 0]
        assert abs(actual[2, 0]-actual[0, 0]-120/bpm) < 64/22050
        assert abs(actual[1, 1]-actual[1, 0]-180/bpm) < 64/22050
        assert np.all(actual[:, 2] >= 21) and np.all(actual[:, 2] <= 108)
        np.testing.assert_array_equal(actual[:, 3], np.rint(np.array(source)[:, 3]*velocity))
        assert clock["human_timing_labels"] is False


def test_every_variant_stays_in_its_source_partition_and_bitrate():
    rows = [{"id": f"source-{n}", "group": "train" if n < 48 else "validation"} for n in range(60)]
    jobs = tasks(rows)
    assert len(jobs) == len({j["id"] for j in jobs}) == 180
    assert COUNTS == {"train": 144, "validation": 36}
    for row in rows:
        related = [j for j in jobs if j["source_id"] == row["id"]]
        assert [j["variant"] for j in related] == [0, 1, 2]
        assert [j["bitrate"] for j in related] == [48, 128, 192]


@pytest.mark.parametrize("variant", [-1, 3, 4])
def test_unknown_variants_cannot_silently_select_a_different_render(variant):
    with pytest.raises(ValueError):
        transform([[0, 96, 40, 90]], variant)


def test_impossible_whole_source_range_is_rejected_without_dropping_keys():
    with pytest.raises(ValueError):
        transform([[0, 96, 0, 90], [0, 96, 127, 90]], 1)
