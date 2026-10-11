"""Prepare a reviewable website release only after empirical and parity gates pass."""

import argparse
import json
import shutil
from pathlib import Path

from evaluate_release_only_v38 import frozen, require_regression
from prepare_robust_training_stems import digest, preserve
from stage_embedding_v32 import replace_once


def stage(run, output):
    _, winner, _, _ = frozen(run)
    require_regression(run)
    root = Path(__file__).resolve().parents[1]
    proof = json.loads((run / "runtime-parity.json").read_text())
    asset = Path(proof["asset"])
    if (not proof["passes"] or proof["recordings"] != 1569 or proof["physical_recordings"] != 44
            or proof["maximum_feature_error"] > 1e-6 or proof["maximum_probability_error"] > 1e-6
            or not proof["identical_decisions_both_settings"]
            or proof["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or proof["asset_sha256"] != digest(asset)
            or proof["first_pass_sha256"] != digest(run / "original-first-pass.json")
            or any(digest(root / name) != sha for name, sha in proof["source_sha256"].items())):
        raise ValueError("Failed or changed portable release proof")
    if output.exists():
        raise ValueError("Preserve existing staging")
    shutil.copytree(root / "backend", output / "backend",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "*.egg-info", "build"))
    shutil.copytree(root / "scripts", output / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    backend = output / "backend"
    shutil.copyfile(asset, backend / "app/assets/chord-release-v1.npz")
    (backend / "app/assets/chord-release-v1.LICENSE.txt").write_text("""Independent chord release specialist v1 (V38)
Copyright 2026 Piano Scribe contributors. Model arrays: CC BY 4.0.
https://creativecommons.org/licenses/by/4.0/

New score data: Mutopia Project public-domain Scott Joplin editions, IDs
441 (Kurt Holtrop), 345 (Tom Harke), 482 (Magnus Lewis-Smith), 1540 (Benjamin
Bloomfield). Validation 263 (Chris Sawer) and reserved 1899 (Coyau) never fitted.
https://www.mutopiaproject.org/
All audio for this subset synthesized by project-authored code; no downloaded
performances. Source pages and MIDI hashes accompany the training report.

Existing CC BY 4.0 corpora: MusicNet (John Thickstun, Zaid Harchaoui, Sham Kakade),
Vienna 4x22 (Werner Goebl), GuitarSet (Qingyang Xi, Rachel Bittner, Johan Pauwels,
Xuzhou Ye, Juan Bello), BabySlakh (Ethan Manilow, Gordon Wichern, Prem Seetharaman,
Jonathan Le Roux), NSynth (Jesse Engel, Cinjon Resnick, Adam Roberts, Sander
Dieleman, Douglas Eck, Karen Simonyan, Mohammad Norouzi) and Groove MIDI
(Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, David Bamman).
Inherited provenance and licenses remain beside previous model assets.

Scope: balanced full-song bass/accompaniment, pitches 21–59, independent release
corrections up to 80 ms after all prior filtering. Pitches, note starts, velocity,
source parts and first/last 2.5 seconds are preserved. No user audio or scores.
Recipe and measured limits: docs/chord-release-training-v38.md.
""")
    path = backend / "app/services/piano_transcription.py"
    text = path.read_text()
    text = replace_once(text, "        self.bass_embedding_declutter = None",
                        "        self.bass_embedding_declutter = None\n        self.chord_release = None")
    old = "            from app.services.bass_embedding_declutter import BassEmbeddingDeclutter"
    text = replace_once(text, old, old+"\n            from app.services.chord_release import ChordRelease")
    old = "                self.bass_verification = {'model': self.bass_verifier.name,"
    text = replace_once(text, old,
        "                self.chord_release = ChordRelease(\n"
        "                    Path(__file__).resolve().parents[1] / 'assets/chord-release-v1.npz',\n"
        f"                    '{proof['asset_sha256']}')\n"+old)
    old = "                                          'spectral_recurrence_model': self.bass_embedding_declutter.name,"
    text = replace_once(text, old, old+"\n                                          'chord_release_model': self.chord_release.name,")
    old = "                                          'window_spectral_recurrence_rejections': 0}"
    text = replace_once(text, old, "                                          'window_spectral_recurrence_rejections': 0,\n                                          'window_release_corrections': 0}")
    old = "            self.bass_verification['window_spectral_recurrence_rejections'] += self.bass_embedding_declutter.filter(path, arrays, midi)"
    text = replace_once(text, old, old+"\n            # Independently trained releases preserve pitches and all attacks.\n            self.bass_verification['window_release_corrections'] += self.chord_release.filter(path, arrays, midi)")
    old = '            engine_name += f" + {bass_verification[\'spectral_recurrence_model\']}"'
    text = replace_once(text, old, old+'\n        if bass_verification.get(\'chord_release_model\'):\n            engine_name += f" + {bass_verification[\'chord_release_model\']}"')
    path.write_text(text)
    path = backend / "app/routes/dependencies.py"
    text = path.read_text()
    old = "            'bass_spectral_recurrence_model': BASS_EMBEDDING_CHECKPOINT.is_file(),"
    text = replace_once(text, old, old+"\n            'chord_release_model': (BASS_EMBEDDING_CHECKPOINT.parent / 'chord-release-v1.npz').is_file(),")
    path.write_text(text)
    names = ("app/services/piano_transcription.py", "app/routes/dependencies.py",
             "app/services/chord_release.py", "app/services/chord_release_network.py",
             "app/assets/chord-release-v1.npz", "app/assets/chord-release-v1.LICENSE.txt")
    preserve(output / "stage.json", {"source_sha256": {name: digest(backend / name) for name in names},
        "checkpoint_sha256": winner["checkpoint_sha256"], "runtime_parity_sha256": digest(run / "runtime-parity.json"),
        "stager_sha256": digest(Path(__file__)), "production_unchanged": True, "no_user_audio_or_scores": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    stage(**vars(parser.parse_args()))
