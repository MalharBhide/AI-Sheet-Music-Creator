"""Combine regenerated native core observations with every original fitting group."""
from pathlib import Path
from collections import Counter
from regenerated_baseline_v63 import load as core_items
from musicnet_component_v63 import load as piano_items
from train_left_precision_v57 import (
    FOLDERS as ORIGINAL_FOLDERS, preparation_audit,
    authored_items, piano_chord_items, dynamics_items, isolated_items,
    rhythmic_items, new_piano_items, hold_items, generated_items,
)

FOLDERS = {**ORIGINAL_FOLDERS, "baseline": Path("/training/bass-v32-regenerated-v63-v1")}


def fitting_items():
    items = (core_items(FOLDERS["baseline"]) + piano_items(FOLDERS["piano"])
        + authored_items(FOLDERS["authored"]) + piano_chord_items(FOLDERS["mutopia"])
        + dynamics_items(FOLDERS["dynamics"]) + isolated_items(FOLDERS["tinysol"])
        + rhythmic_items(FOLDERS["rhythmic"]) + new_piano_items(FOLDERS["new_piano"])
        + hold_items(FOLDERS["new_holds"]) + generated_items(FOLDERS["new_generated"]))
    if (len(items) != 2920 or len({i["id"] for i in items}) != 2920
            or dict(Counter(i["group"] for i in items)) != {"train": 2302, "validation": 618}):
        raise ValueError("Complete original fitting scope is mandatory")
    return items
