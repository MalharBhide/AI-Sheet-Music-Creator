import pytest
from musicnet_bass_data import load
from prepare_musicnet_bass import references, sources


def label(start, end, pitch=48, instrument=1):
    return dict(start_time=start, end_time=end, note=pitch, instrument=instrument)


def test_ongoing_holds_do_not_become_new_attacks():
    attacks, support = references(
        [label(500, 1500), label(1800, 4000), label(3500, 4000)], 100, 1000, 3000, 100
    )
    assert attacks == [[8.0, 20.0, 48]]
    assert support == [[0, 5.1, 48], [7.95, 20, 48]]


@pytest.mark.parametrize(
    "row",
    [
        label(-1, 100),
        label(100, 100),
        label(100, 10001),
        label(100, 200, instrument=42),
        label(100, 200, pitch=10),
    ],
)
def test_invalid_piano_label_clock_or_identity_refused(row):
    with pytest.raises(ValueError, match="note label"):
        references([row], 100, 1000, 3000, 100)


def test_reserved_fitting_refused_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        load(tmp_path, tmp_path, tmp_path, ("reserved",))


def test_unverified_acquisition_cannot_be_observed(tmp_path):
    (tmp_path / "manifest.json").write_text('{"plan_sha256":"invalid"}')
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="unverified"):
        sources(tmp_path)
