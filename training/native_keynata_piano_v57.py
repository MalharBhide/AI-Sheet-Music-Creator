"""Render every proposed Keynata fitting crop without running a transcriber."""

import argparse
from pathlib import Path

from prepare_keynata_piano_v57 import preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    preflight(**vars(parser.parse_args()))
