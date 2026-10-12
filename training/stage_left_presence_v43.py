"""Stage a V43 website candidate without changing the released runtime."""

import argparse
import json
import shutil
from pathlib import Path

from evaluate_left_precision_v43 import frozen
from export_left_presence_v43 import require_tests
from prepare_robust_training_stems import digest, preserve
from stage_embedding_v32 import replace_once


def stage(run, production, fixtures, output):
    _, winner, _, _ = frozen(run, production)
    require_tests(run, fixtures, winner)
    parity = json.loads((run/"runtime-parity.json").read_text())
    asset = Path(parity["asset"])
    if (not parity["passes"] or parity["recordings"] != 1721 or parity["physical_recordings"] != 40
            or parity["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or parity["asset_sha256"] != digest(asset)
            or parity["maximum_feature_error"] > 1e-6 or parity["maximum_probability_error"] > 1e-6
            or not parity["identical_keep_decisions_both_settings"] or not parity["retained_v38_clocks_identical"]
            or parity["first_pass_sha256"] != digest(run/"original-first-pass.json")
            or any(digest(production/name) != sha for name, sha in parity["runtime_source_sha256"].items())):
        raise ValueError("Failed or changed V43 runtime parity")
    if output.exists():
        raise ValueError("Preserve staged runtime")
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "*.egg-info", "build")
    shutil.copytree(production/"backend", output/"backend", ignore=ignore)
    shutil.copytree(production/"scripts", output/"scripts", ignore=ignore)
    backend = output/"backend"
    shutil.copyfile(asset, backend/"app/assets/left-presence-v1.npz")
    (backend/"app/assets/left-presence-v1.LICENSE.txt").write_text("""Left-hand pitch-presence verifier v1 (V43)
Copyright 2026 Piano Scribe contributors. Model arrays: CC BY 4.0.
https://creativecommons.org/licenses/by/4.0/

Trained from scratch on licensed physical observations. No user audio or scores.
CC BY 4.0 sources: NSynth (Jesse Engel, Cinjon Resnick, Adam Roberts, Sander
Dieleman, Douglas Eck, Karen Simonyan, Mohammad Norouzi), MusicNet (John
Thickstun, Zaid Harchaoui, Sham Kakade), Vienna 4x22 (Werner Goebl), GuitarSet
(Qingyang Xi, Rachel Bittner, Johan Pauwels, Xuzhou Ye, Juan Bello), BabySlakh
(Ethan Manilow, Gordon Wichern, Prem Seetharaman, Jonathan Le Roux), and Groove
MIDI (Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, David Bamman).
Attribution and original source/license witnesses accompany inherited model
licenses and training manifests.

Public-domain Mutopia Scott Joplin editions: fitting IDs 441 (Kurt Holtrop),
345 (Tom Harke), 482 (Magnus Lewis-Smith), 1540 (Benjamin Bloomfield).
Validation 263 (Chris Sawer), reserved 1899 (Coyau), neither fitted. Audio for
this subset is synthesized by project-authored code, not downloaded performances.

The 88/24 new NSynth authored sequences retain the original 11/3 instrument
partition; four reserved instruments are not fitted. Short gates and acoustic
tails are authored synthesis, not measured human key releases.

Scope: balanced full-song bass/accompaniment, pitches 21–59, after V32 filtering.
Compute removal on unchanged V32 intervals; run V38 releases on their complete
input, then remove only approved candidates. Retained pitch, attack, release,
velocity and source parts are unchanged. First/last 2.5 seconds are protected.
Recipe, data provenance, evaluations and limits: docs/left-hand-precision-training.md
and training/results/left-precision-v43-v1. No accuracy guarantee for arbitrary songs.
""")
    path = backend/"app/services/piano_transcription.py"
    text = path.read_text()
    text = replace_once(text, "        self.chord_release = None", "        self.chord_release = None\n        self.left_presence = None")
    old = "            from app.services.chord_release import ChordRelease"
    text = replace_once(text, old, old+"\n            from app.services.left_presence import LeftHandPresence")
    old = "                self.bass_verification = {'model': self.bass_verifier.name,"
    text = replace_once(text, old, "                self.left_presence = LeftHandPresence(\n"
        "                    Path(__file__).resolve().parents[1] / 'assets/left-presence-v1.npz',\n"
        f"                    '{parity['asset_sha256']}')\n"+old)
    old = "                                          'chord_release_model': self.chord_release.name,"
    text = replace_once(text, old, old+"\n                                          'left_presence_model': self.left_presence.name,")
    old = "                                          'window_release_corrections': 0}"
    text = replace_once(text, old, "                                          'window_release_corrections': 0,\n                                          'window_presence_rejections': 0}")
    old = "            # Independently trained releases preserve pitches and all attacks.\n            self.bass_verification['window_release_corrections'] += self.chord_release.filter(path, arrays, midi)"
    text = replace_once(text, old,
        "            # Presence sees complete V32 intervals. Release correction must\n"
        "            # keep its complete approved input before applying this mask.\n"
        "            proposal = self.left_presence.prepare(path, arrays, midi)\n"+old+
        "\n            self.bass_verification['window_presence_rejections'] += proposal.apply(midi)")
    old = '            engine_name += f" + {bass_verification[\'chord_release_model\']}"'
    text = replace_once(text, old, old+'\n        if bass_verification.get(\'left_presence_model\'):\n            engine_name += f" + {bass_verification[\'left_presence_model\']}"')
    path.write_text(text)
    path = backend/"app/routes/dependencies.py"
    old = "            'chord_release_model': (BASS_EMBEDDING_CHECKPOINT.parent / 'chord-release-v1.npz').is_file(),"
    path.write_text(replace_once(path.read_text(), old, old+"\n            'left_hand_presence_model': (BASS_EMBEDDING_CHECKPOINT.parent / 'left-presence-v1.npz').is_file(),"))
    path = backend/"tests/test_transcription.py"
    text = path.read_text()
    old = "    monkeypatch.setattr(chord_release, 'ChordRelease', ChordRelease)"
    if text.count(old) != 2:
        raise ValueError("Changed existing routing test inventory")
    text = text.replace(old, old+"\n    _stub_left_presence(monkeypatch)")
    text += '''\n\ndef _stub_left_presence(monkeypatch):
    from types import SimpleNamespace

    from app.services import left_presence

    class LeftHandPresence:
        name = 'Test left-hand presence'

        def __init__(self, asset, expected_sha):
            assert asset.name == 'left-presence-v1.npz' and len(expected_sha) == 64

        def prepare(self, path, acoustic, output):
            assert getattr(output, '_embedding_checked', False)
            return SimpleNamespace(apply=lambda midi: 0)

    monkeypatch.setattr(left_presence, 'LeftHandPresence', LeftHandPresence)
'''
    text = replace_once(text, "'chord_release_model': 'Test independent chord releases', 'window_candidates': 2,",
        "'chord_release_model': 'Test independent chord releases', 'left_presence_model': 'Test left-hand presence', 'window_candidates': 2,")
    text = replace_once(text, "        'window_release_corrections': 0}", "        'window_release_corrections': 0, 'window_presence_rejections': 0}")
    path.write_text(text)
    names = ("app/services/piano_transcription.py", "app/routes/dependencies.py", "app/services/left_presence.py",
             "app/services/left_presence_network.py", "app/assets/left-presence-v1.npz", "app/assets/left-presence-v1.LICENSE.txt",
             "tests/test_transcription.py", "tests/test_left_presence.py")
    preserve(output/"stage.json", {"source_sha256": {name: digest(backend/name) for name in names},
        "checkpoint_sha256": winner["checkpoint_sha256"], "asset_sha256": parity["asset_sha256"],
        "runtime_parity_sha256": digest(run/"runtime-parity.json"), "stager_sha256": digest(Path(__file__)),
        "production_unchanged": True, "no_user_audio_or_scores": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "production", "fixtures", "output"):
        parser.add_argument(name, type=Path)
    stage(**vars(parser.parse_args()))
