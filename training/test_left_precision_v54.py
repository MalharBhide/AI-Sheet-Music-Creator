import numpy as np
import pytest
from left_precision_v54 import isolated_negative_weights


def item(corpus, labels, group="train"):
    return {"corpus": corpus, "group": group, "y": np.array(labels, float),
            "eligible": np.ones(len(labels), bool), "mask": np.ones(len(labels), bool)}


def test_emphasis_only_changes_isolated_wrong_pitch_relative_weight():
    items = [item("tinysol-isolated-pitch", [1, 0]), item("musicnet-real-piano", [0, 1])]
    selected = [i["eligible"] & i["mask"] for i in items]
    actual = isolated_negative_weights(items, selected, np.ones(4, np.float32), 8)
    np.testing.assert_allclose(actual / actual[0], [1, 8, 1, 1])
    assert actual.mean() == pytest.approx(1, abs=2e-7)


@pytest.mark.parametrize("group", ["validation", "test", "reserved"])
def test_held_out_labels_cannot_set_fitting_weights(group):
    original = item("tinysol-isolated-pitch", [0], group)
    with pytest.raises(ValueError, match="fitting"):
        isolated_negative_weights([original], [np.array([True])], np.ones(1), 8)


def test_unknown_or_ineligible_candidates_cannot_receive_negative_emphasis():
    original = item("tinysol-isolated-pitch", [1, 0])
    original["mask"][1] = False
    result = isolated_negative_weights([original], [np.array([True, False])], np.ones(1), 8)
    np.testing.assert_array_equal(result, [1])
    with pytest.raises(ValueError, match="supervised"):
        isolated_negative_weights([original], [np.array([True, True])], np.ones(2), 8)


@pytest.mark.parametrize("factor", [0, 65, float("nan"), float("inf")])
def test_invalid_fitting_emphasis_rejected(factor):
    original = item("tinysol-isolated-pitch", [0])
    with pytest.raises(ValueError):
        isolated_negative_weights([original], [np.array([True])], np.ones(1), factor)


def test_misaligned_fitting_weight_array_rejected():
    original = item("tinysol-isolated-pitch", [0])
    with pytest.raises(ValueError, match="Misaligned"):
        isolated_negative_weights([original], [np.array([True])], np.ones(2), 8)


def test_generated_command_presence_is_unknown_but_absence_remains_supervised(monkeypatch):
    import left_precision_v54 as study

    target = np.array([[0, 1, 1, 0]], np.float32)
    weight = np.ones_like(target)
    monkeypatch.setattr(study, "previous_frame_supervision", lambda item: (target, weight))
    actual_target, actual_weight = study.frame_supervision(
        {"corpus": "coco-authored-sampled-piano", "group": "train"})
    np.testing.assert_array_equal(actual_target, target)
    np.testing.assert_array_equal(actual_weight, [[1, 0, 0, 1]])
    np.testing.assert_array_equal(weight, np.ones_like(target))
    with pytest.raises(ValueError, match="fitting-only"):
        study.frame_supervision({"corpus": "coco-authored-sampled-piano", "group": "validation"})


def test_existing_frame_supervision_is_unchanged(monkeypatch):
    import left_precision_v54 as study

    target, weight = np.ones((1, 40)), np.ones((1, 40))
    monkeypatch.setattr(study, "previous_frame_supervision", lambda item: (target, weight))
    result = study.frame_supervision({"corpus": "musicnet-real-piano"})
    assert result[0] is target and result[1] is weight


@pytest.mark.parametrize("mutation", [None, "missing", "held-out", "duplicate", "overlap", "group-size"])
def test_new_parent_cohort_stays_complete_and_train_only(monkeypatch, mutation):
    import left_precision_v54 as study

    earlier = [{"id": f"earlier-{i}", "corpus": "earlier", "source_group": f"old/{i}"}
               for i in range(2740)]
    new = [{"id": f"new-{i}", "corpus": "coco-authored-sampled-piano", "group": "train",
            "source_group": f"coco-generated/{i // 3}.mid"} for i in range(108)]
    checked = []
    monkeypatch.setattr(study, "previous_partitions", lambda items: checked.append(items))
    if mutation == "missing":
        new.pop()
    elif mutation == "held-out":
        new[0]["group"] = "validation"
    elif mutation == "duplicate":
        new[0]["id"] = earlier[0]["id"]
    elif mutation == "overlap":
        earlier[0]["source_group"] = new[0]["source_group"]
    elif mutation == "group-size":
        new[0]["source_group"] = new[3]["source_group"]
    if mutation:
        with pytest.raises(ValueError, match="partition"):
            study.validate_partitions(earlier + new)
    else:
        study.validate_partitions(earlier + new)
    assert checked == [earlier]
