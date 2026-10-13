"""Fresh learned permissions using new train-only piano work supervision."""

from left_precision_v48 import gate, model, probabilities
from left_precision_v50 import real_piano_weights

VERSION = "new-pdmx-piano-boundary-presence-v53-v1"
FRESH_SEEDS = tuple(range(298101, 298141))
__all__ = ["VERSION", "FRESH_SEEDS", "gate", "model", "probabilities",
           "real_piano_weights", "validate_partitions"]


def validate_partitions(items):
    training, validation = [[i for i in items if i["group"] == g] for g in ("train", "validation")]
    timed = [i for i in validation if i["corpus"] != "tinysol-isolated-pitch"]
    isolated = [i for i in validation if i["corpus"] == "tinysol-isolated-pitch"]
    new = [i for i in items if i["corpus"] == "pdmx-authored-sampled-piano"]
    if (len(items) != 2740 or len(training) != 2122 or len(timed) != 381 or len(isolated) != 237
            or len({i["id"] for i in items}) != len(items)
            or {i["source_group"] for i in training} & {i["source_group"] for i in validation}
            or sum(i["corpus"] == "musicnet-real-piano" for i in timed) != 15
            or sum(i["corpus"] == "nrgcp-authored-sampled-piano" for i in training) != 144
            or sum(i["corpus"] == "nrgcp-authored-sampled-piano" for i in timed) != 36
            or len(new) != 45 or any(i["group"] != "train"
                or i["source_group"] != "pdmx/domenico-scarlatti" for i in new)):
        raise ValueError("Incomplete or leaking new-piano fitting partitions")

