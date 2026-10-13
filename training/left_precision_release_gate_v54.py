"""Check complete consumed evidence before any untouched audio is generated."""


def require_consumed_proof(regression, winner, selection_sha, evaluator_sha):
    if (regression["passes"] is not True or regression["recordings"] != 603
            or regression["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or regression["batch_selection_sha256"] != selection_sha
            or regression["evaluator_sha256"] != evaluator_sha
            or regression["no_test_used_for_selection"] is not True
            or regression["no_user_audio_or_scores"] is not True
            or regression["new_weights_deployed"] is not False):
        raise ValueError("Complete preserving consumed regression proof required")
    identities = []
    for name, threshold in (("raw", winner["threshold"]),
                            ("strict", min(.9999, winner["threshold"]+.005))):
        report = regression[name]
        rows = report["per_recording"]
        ids = {r["id"] for r in rows}
        if (report["passes"] is not True or report["threshold"] != threshold
                or report["false_notes_removed"] <= 0 or len(rows) != 603 or len(ids) != 603
                or any(not isinstance(identifier, str) or not identifier for identifier in ids)
                or any(r["passes"] is not True or r["retained_clocks_identical"] is not True
                       or r["recovered_notes"] != 0 for r in rows)
                or sum(r["false_notes_removed"] for r in rows) != report["false_notes_removed"]):
            raise ValueError("Incomplete or nonpreserving per-recording consumed regression proof")
        identities.append(ids)
    if identities[0] != identities[1]:
        raise ValueError("Consumed regression policies used different recordings")
