"""Reject incomplete independent fresh evidence before a V63 runtime export."""


def require_fresh_proof(first, winner, selection_sha, consumed_sha, producer_sha, expected_ids, fresh_protocol_sha, qualification_sha):
    families = {f"left-v69-fresh-{seed}": "new-first-pass-"+("original" if index%2 == 0 else "texture")
                for index, seed in enumerate(range(311101, 311141))}
    if (first["passes"] is not True or first["recordings"] != 40
            or first["first_pass_complete"] is not True
            or first["checkpoint_sha256"] != winner["checkpoint_sha256"]
            or first["batch_selection_sha256"] != selection_sha
            or first["consumed_regression_sha256"] != consumed_sha
            or first["producer_sha256"] != producer_sha
            or first["fresh_protocol_sha256"] != fresh_protocol_sha
            or first["qualification_protocol_sha256"] != qualification_sha
            or first["no_test_used_for_selection"] is not True
            or first["no_user_audio_or_scores"] is not True
            or first["new_weights_deployed"] is not False or len(expected_ids) != 40
            or set(expected_ids) != set(families)):
        raise ValueError("Complete unchanged independent fresh proof required")
    cohorts = []
    for name, threshold in (("raw", winner["threshold"]),
                            ("strict", min(.9999, winner["threshold"]+.005))):
        report = first[name]
        rows = report["per_recording"]
        ids = {row["id"] for row in rows}
        if (report["passes"] is not True or report["threshold"] != threshold
                or len(rows) != 40 or ids != set(expected_ids)
                or report["false_notes_removed"] <= 0
                or report["false_notes_removed"] != sum(row["false_notes_removed"] for row in rows)
                or any(row["passes"] is not True or row["fixed_reference_identities_preserved"] is not True
                       or row["retained_clocks_identical"] is not True or row["timing"]["passes"] is not True
                       or row["coverage"]["passes"] is not True or row["recovered_notes"] != 0
                       or row["false_notes_removed"] < 0 or row["corpus"] != families[row["id"]] for row in rows)):
            raise ValueError("Incomplete or nonpreserving independent fresh rows")
        for family in ("original", "texture"):
            members = [row for row in rows if row["corpus"] == "new-first-pass-"+family]
            gain = sum(row["false_notes_removed"] for row in members)
            if (len(members) != 20 or gain <= 0
                    or first["family_false_note_gains"][name][family] != gain):
                raise ValueError("Separate independent fresh family gain required")
        cohorts.append(ids)
    if cohorts[0] != cohorts[1]:
        raise ValueError("Independent fresh policies used different recordings")
