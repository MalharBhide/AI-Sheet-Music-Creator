import pytest
from coco_piano_render_v54 import HeldWorkPiano


def stub(result, voices):
    piano = object.__new__(HeldWorkPiano)
    piano.synth, piano.natural_expirations = 17, []
    piano.functions = {"fluid_synth_noteoff": lambda *args: result,
                       "fluid_synth_noteon": lambda *args: result,
                       "fluid_synth_get_active_voice_count": lambda *args: voices}
    return piano


def test_normal_key_release_remains_unchanged():
    piano = stub(0, 1)
    piano.check("fluid_synth_noteoff", 17, 0, 60)
    assert piano.natural_expirations == []


def test_only_no_remaining_voices_can_explain_unmatched_release():
    piano = stub(-1, 0)
    piano.check("fluid_synth_noteoff", 17, 0, 60)
    assert piano.natural_expirations == [{"channel": 0, "pitch": 60, "active_voices": 0}]


@pytest.mark.parametrize("result,voices,args", [
    (-1, 1, (17, 0, 60)), (-1, -1, (17, 0, 60)), (-2, 0, (17, 0, 60)),
    (-1, 0, (18, 0, 60)), (-1, 0, (17, 1, 60)), (-1, 0, (17, 0, 109)),
])
def test_real_or_unexplained_native_failure_stays_fatal(result, voices, args):
    with pytest.raises(ValueError, match="synthesis failed"):
        stub(result, voices).check("fluid_synth_noteoff", *args)


def test_failed_attack_is_never_ignored():
    with pytest.raises(ValueError, match="noteon"):
        stub(-1, 0).check("fluid_synth_noteon", 17, 0, 60, 80)
