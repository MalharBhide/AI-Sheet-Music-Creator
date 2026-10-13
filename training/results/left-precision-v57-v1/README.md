# Fresh piano fitting with hard genuine-note protection

V57 adds 72 native-audited, CC0 generated piano MP3 windows from five new parent
compositions to the unchanged earlier fitting corpus. All new parents and their
windows/dynamics are fitting-only; generated positive command-support frames
are UNKNOWN acoustic presence. This is not new human piano evaluation evidence.

Two independently initialized boundary-conditioned models add an auxiliary loss
on the worst 10% of genuine-note losses in each supervised fitting minibatch.
REMOVE examples never receive that protection. No failed candidate checkpoint,
held-out recording, source identity or teacher output is used for fitting.
This is a prospective training hypothesis, not an established accuracy gain.

All existing per-recording identity, pitch, hold, repeated/chord timing and
unsupported-duration gates remain, plus separate overall, real-piano, NRG-CP
and TinySOL wrong-pitch gains at both policies. One frozen validation winner
may enter the complete 603-recording regression and one untouched 40-clip first
pass, then runtime parity. No runner-up, retuning or replacement test clips
after failure. Production remains V43 unless every release gate passes.

Status: fitting preregistered after 1,326 passing tests, six skips and five
warnings. Training is ready; no candidate has been evaluated or released.
