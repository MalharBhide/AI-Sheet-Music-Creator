import json
from pathlib import Path

import pytest
from export_left_presence_v43 import require_tests


def test_export_gate_requires_complete_frozen_normal_and_stricter_reports(tmp_path):
    root = Path(__file__).resolve().parent/"results/left-precision-v43-v1"
    winner = json.loads((root/"batch-selection.json").read_text())["winner"]
    # The archived evidence is authoritative and passes before any edits.
    fixtures = root/"independent-observations"
    assert len(require_tests(root, fixtures, winner)["items"]) == 40
    for name in ("batch-selection.json", "consumed-regression.json", "original-first-pass.json"):
        (tmp_path/name).write_bytes((root/name).read_bytes())
    report = json.loads((tmp_path/"consumed-regression.json").read_text())
    report["strict"]["per_recording"].pop()
    (tmp_path/"consumed-regression.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="per-recording"):
        require_tests(tmp_path, fixtures, winner)
