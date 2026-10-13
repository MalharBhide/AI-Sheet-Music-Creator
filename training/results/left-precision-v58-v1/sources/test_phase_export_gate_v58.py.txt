import pytest
from phase_export_gate_v58 import require_fresh_proof


def proof():
    ids = {f"left-v58-fresh-{seed}" for seed in range(302101, 302141)}
    winner = {"checkpoint_sha256": "checkpoint", "threshold": .98}
    report = {"passes": True, "recordings": 40, "first_pass_complete": True,
        "checkpoint_sha256": "checkpoint", "batch_selection_sha256": "selection",
        "consumed_regression_sha256": "consumed", "producer_sha256": "producer",
        "no_test_used_for_selection": True, "no_user_audio_or_scores": True, "new_weights_deployed": False,
        "family_false_note_gains": {name: {"original": 20, "texture": 20} for name in ("raw", "strict")}}
    for name, threshold in (("raw", .98), ("strict", .985)):
        rows = [{"id": f"left-v58-fresh-{seed}", "corpus": "new-first-pass-"+("original" if i%2 == 0 else "texture"),
                 "passes": True, "fixed_reference_identities_preserved": True,
                 "retained_clocks_identical": True, "timing": {"passes": True}, "coverage": {"passes": True},
                 "recovered_notes": 0, "false_notes_removed": 1}
                for i, seed in enumerate(range(302101, 302141))]
        report[name] = {"passes": True, "threshold": threshold, "false_notes_removed": 40, "per_recording": rows}
    return report, winner, ids


def test_complete_independent_fresh_evidence_passes():
    report, winner, ids = proof()
    require_fresh_proof(report, winner, "selection", "consumed", "producer", ids)


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate", "replacement", "failed-row", "identity",
    "clock", "timing", "coverage", "recovered", "threshold", "aggregate", "family-sum", "family-count",
    "family-no-gain", "family-swap", "checkpoint", "selection", "consumed", "producer", "deployed", "test-selection"])
def test_failed_incomplete_replaced_or_contradictory_fresh_proof_rejected(mutation):
    report, winner, ids = proof()
    rows = report["strict"]["per_recording"]
    row = rows[0]
    if mutation == "incomplete":
        rows.pop()
    elif mutation == "duplicate":
        row["id"] = rows[1]["id"]
    elif mutation == "replacement":
        row["id"] = "left-v58-fresh-replacement"
    elif mutation in ("failed-row", "identity", "clock"):
        row[{"failed-row": "passes", "identity": "fixed_reference_identities_preserved",
             "clock": "retained_clocks_identical"}[mutation]] = False
    elif mutation in ("timing", "coverage"):
        row[mutation]["passes"] = False
    elif mutation == "recovered":
        row["recovered_notes"] = 1
    elif mutation == "threshold":
        report["strict"]["threshold"] = .98
    elif mutation == "aggregate":
        report["strict"]["false_notes_removed"] = 41
    elif mutation == "family-sum":
        report["family_false_note_gains"]["strict"]["original"] = 21
    elif mutation == "family-count":
        row["corpus"] = "new-first-pass-texture"
    elif mutation == "family-swap":
        row["corpus"], rows[1]["corpus"] = rows[1]["corpus"], row["corpus"]
    elif mutation == "family-no-gain":
        for member in rows:
            if member["corpus"] == "new-first-pass-original":
                member["false_notes_removed"] = 0
        report["strict"]["false_notes_removed"] = 20
        report["family_false_note_gains"]["strict"]["original"] = 0
    elif mutation in ("checkpoint", "selection", "consumed", "producer"):
        report[{"checkpoint": "checkpoint_sha256", "selection": "batch_selection_sha256",
                "consumed": "consumed_regression_sha256", "producer": "producer_sha256"}[mutation]] = "changed"
    elif mutation == "deployed":
        report["new_weights_deployed"] = True
    else:
        report["no_test_used_for_selection"] = False
    with pytest.raises(ValueError, match="fresh"):
        require_fresh_proof(report, winner, "selection", "consumed", "producer", ids)
