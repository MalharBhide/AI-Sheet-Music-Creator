from pathlib import Path

import pytest
from cocochorales_source import note_rows, references, sources


def test_synthesis_frame_clock_and_repeated_attacks_remain_separate(tmp_path):
    path = tmp_path / "voice.csv"
    path.write_text(
        "pitch,onset,offset,note_length\n0,0,0,0\n48,0,250,250\n0,250,260,10\n48,260,500,240\n"
    )
    assert note_rows(path) == [[0.0, 1.0, 48], [1.04, 2.0, 48]]
    attacks, support = references(note_rows(path), 5.0, 6.0)
    assert attacks == [[0.0, 1.0, 48], [1.04, 2.0, 48]]
    assert support == [[0, 2.0, 48], [0.99, 3.0, 48]]


@pytest.mark.parametrize(
    "row",
    [
        "48,nan,250,250",
        "48,0.5,250,249.5",
        "48,0,250,240",
        "48,250,0,-250",
        "48,0,0,0",
        "128,0,250,250",
    ],
)
def test_bad_note_clock_pitch_or_length_is_refused(tmp_path, row):
    path = tmp_path / "voice.csv"
    path.write_text("pitch,onset,offset,note_length\n" + row + "\n")
    with pytest.raises(ValueError):
        note_rows(path)


def test_crop_clips_hold_and_protects_decay_without_retiming():
    attacks, support = references([[4.0, 8.0, 48], [9.0, 10.0, 50]], 5.0, 11.0)
    assert attacks == [[4.0, 5.0, 48]]
    assert support == [[3.95, 5.0, 48]]
    with pytest.raises(ValueError, match="escape"):
        references([[4.0, 8.0, 48]], 5.0, 6.0)


def test_incomplete_acquisition_cannot_be_observed(tmp_path):
    # Missing source metadata is refused before any audio or decoder is accessed.
    with pytest.raises(FileNotFoundError):
        sources(Path(tmp_path))
