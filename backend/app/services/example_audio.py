"""Original piano study rendered from the reproducible quality fixture."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def piano_example() -> bytes:
    return (Path(__file__).resolve().parents[1] / 'assets/piano-study.wav').read_bytes()
