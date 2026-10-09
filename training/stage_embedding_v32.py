"""Stage the release route only after frozen empirical and portable gates pass."""

import argparse
import json
import shutil
from pathlib import Path

from evaluate_teacher_embedding_v32 import frozen, require_regression
from prepare_robust_training_stems import digest, preserve


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Changed approved route insertion point")
    return text.replace(old, new)


def stage(run, output):
    _, winner, _, _ = frozen(run)
    require_regression(run)
    root = Path(__file__).resolve().parents[1]
    proof = json.loads((run / "runtime-parity.json").read_text())
    export = json.loads((run / "portable-export.json").read_text())
    asset = Path(export["asset"])
    if (
        not proof["passes"]
        or proof["recordings"] != 1377
        or proof["physical_recordings"] != 77
        or not proof["identical_keep_decisions_at_both_settings"]
        or proof["maximum_probability_error"] > 1e-6
        or proof["maximum_feature_error"] > 1e-6
        or proof["checkpoint_sha256"] != winner["checkpoint_sha256"]
        or proof["asset_sha256"] != digest(asset)
        or proof["first_pass_sha256"] != digest(run / "original-first-pass.json")
        or any(digest(root / name) != sha for name, sha in proof["runtime_source_sha256"].items())
    ):
        raise ValueError("Failed or changed runtime release gates")
    if output.exists():
        raise ValueError("Preserve existing release staging")
    shutil.copytree(
        root / "backend",
        output / "backend",
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "*.egg-info", "build"
        ),
    )
    backend = output / "backend"
    shutil.copyfile(asset, backend / "app/assets/bass-embedding-v1.npz")
    license_text = """Bass spectral/recurrence verifier v1 (candidate V32)

Copyright 2026 Piano Scribe contributors.
Newly trained portable model arrays: Creative Commons Attribution 4.0.
https://creativecommons.org/licenses/by/4.0/

Frozen teacher: bass-attack-declutter-v1.npz; its unchanged V25 license and
provenance remain beside this file. Vienna 4x22 (Werner Goebl), GuitarSet 1.1.0
(Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, Juan P. Bello),
BabySlakh v2 (Ethan Manilow, Gordon Wichern, Prem Seetharaman, Jonathan Le Roux)
and project-authored note sequences. All source recordings CC BY 4.0.

New correction fitting data, CC BY 4.0:
NSynth: Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck,
Karen Simonyan and Mohammad Norouzi, Google.
https://magenta.withgoogle.com/datasets/nsynth
Groove MIDI Dataset: Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck and
David Bamman, Google. Human-played electronic drum audio is nuisance evidence;
drum MIDI numbers are never piano note truth.
https://magenta.tensorflow.org/datasets/groove

MusicNet is not used in this candidate. No user recordings or scores used.
Recipe/provenance: docs/teacher-embedding-training.md and
training/results/teacher-embedding-v32-v1.
Scope: balanced full-song bass, pitches 21–59, after all eight V25 stages.
First/last 2.5 seconds are protected; strong acoustic guardian evidence remains.
Retained note objects preserve pitch, attack, endpoint, velocity and source part.
This limited-corpus evidence does not establish accuracy for arbitrary songs.
"""
    (backend / "app/assets/bass-embedding-v1.LICENSE.txt").write_text(license_text)
    path = backend / "app/services/piano_transcription.py"
    text = path.read_text()
    text = replace_once(
        text,
        "        self.bass_attack_declutter = None",
        "        self.bass_attack_declutter = None\n        self.bass_embedding_declutter = None",
    )
    text = replace_once(
        text,
        "            from app.services.bass_attack_declutter import BassAttackDeclutter",
        "            from app.services.bass_attack_declutter import BassAttackDeclutter\n            from app.services.bass_embedding_declutter import BassEmbeddingDeclutter",
    )
    text = replace_once(
        text,
        "                self.bass_attack_declutter = BassAttackDeclutter()",
        "                self.bass_attack_declutter = BassAttackDeclutter()\n                self.bass_embedding_declutter = BassEmbeddingDeclutter(\n                    Path(__file__).resolve().parents[1] / 'assets/bass-embedding-v1.npz',\n                    '"
        + export["asset_sha256"]
        + "')",
    )
    text = replace_once(
        text,
        "                                          'attack_release_model': self.bass_attack_declutter.name,",
        "                                          'attack_release_model': self.bass_attack_declutter.name,\n                                          'spectral_recurrence_model': self.bass_embedding_declutter.name,",
    )
    text = replace_once(
        text,
        "                                          'window_attack_release_rejections': 0}",
        "                                          'window_attack_release_rejections': 0,\n                                          'window_spectral_recurrence_rejections': 0}",
    )
    old = "            self.bass_verification['window_attack_release_rejections'] += self.bass_attack_declutter.filter(path, arrays, midi)"
    text = replace_once(
        text,
        old,
        old
        + "\n            # V32 evaluates actual V25 survivors without changing retained notes.\n            self.bass_verification['window_spectral_recurrence_rejections'] += self.bass_embedding_declutter.filter(path, arrays, midi)",
    )
    old = "            engine_name += f\" + {bass_verification['attack_release_model']}\""
    text = replace_once(
        text,
        old,
        old
        + "\n        if bass_verification.get('spectral_recurrence_model'):\n            engine_name += f\" + {bass_verification['spectral_recurrence_model']}\"",
    )
    path.write_text(text)
    path = backend / "app/routes/dependencies.py"
    text = path.read_text()
    text = replace_once(text, "import shutil", "import shutil\nfrom pathlib import Path")
    text = replace_once(
        text,
        "from app.services.vocal_melody import CHECKPOINT",
        "from app.services.vocal_melody import CHECKPOINT\n\nBASS_EMBEDDING_CHECKPOINT = Path(__file__).resolve().parents[1] / 'assets/bass-embedding-v1.npz'",
    )
    old = "            'bass_attack_release_model': BASS_ATTACK_RELEASE_CHECKPOINT.is_file(),"
    text = replace_once(
        text,
        old,
        old
        + "\n            'bass_spectral_recurrence_model': BASS_EMBEDDING_CHECKPOINT.is_file(),",
    )
    path.write_text(text)
    names = (
        "app/services/piano_transcription.py",
        "app/routes/dependencies.py",
        "app/assets/bass-embedding-v1.npz",
        "app/assets/bass-embedding-v1.LICENSE.txt",
        "app/services/bass_embedding_declutter.py",
        "app/services/bass_embedding_network.py",
        "app/services/waveform_recurrence.py",
        "app/services/release_neighbors.py",
    )
    preserve(
        output / "stage.json",
        {
            "source_sha256": {name: digest(backend / name) for name in names},
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "runtime_parity_sha256": digest(run / "runtime-parity.json"),
            "stager_sha256": digest(Path(__file__)),
            "production_unchanged": True,
            "no_user_audio_or_scores": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.run, args.output)
