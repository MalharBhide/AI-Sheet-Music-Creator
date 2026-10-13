import copy

import pytest
from left_precision_v53 import validate_partitions


def cohort():
    result = []
    for group, corpus, count in (("train", "nrgcp-authored-sampled-piano", 144),
        ("train", "pdmx-authored-sampled-piano", 45), ("train", "earlier", 1933),
        ("validation", "musicnet-real-piano", 15), ("validation", "nrgcp-authored-sampled-piano", 36),
        ("validation", "tinysol-isolated-pitch", 237), ("validation", "earlier", 330)):
        for index in range(count):
            identifier = f"{group}/{corpus}/{index}"
            result.append({"id": identifier, "group": group, "corpus": corpus,
                "source_group": "pdmx/domenico-scarlatti" if corpus == "pdmx-authored-sampled-piano" else identifier})
    return result


def test_new_composer_is_train_only_and_existing_validation_complete():
    validate_partitions(cohort())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "new-validation", "leak", "unknown"])
def test_rejects_incomplete_or_changed_source_partitions(mutation):
    items = copy.deepcopy(cohort())
    if mutation == "missing":
        items.pop()
    elif mutation == "duplicate":
        items[1]["id"] = items[0]["id"]
    elif mutation == "new-validation":
        next(i for i in items if i["corpus"] == "pdmx-authored-sampled-piano")["group"] = "validation"
    elif mutation == "leak":
        items[-1]["source_group"] = "pdmx/domenico-scarlatti"
    elif mutation == "unknown":
        items[0]["group"] = "test"
    with pytest.raises(ValueError, match="partitions"):
        validate_partitions(items)
