# Prospective training objective question

V58's safest validation settings show normal-policy real-piano and NRG-CP gains,
but no preserving setting gains NRG-CP notes under the stricter policy. These
are maxima across different settings, not a qualifying model or test evidence.

A possible objective mismatch remains: `masked_frame_loss` averages labeled
frame losses, while the immutable deletion veto uses the maximum interior
frame probability. A single false support peak may therefore veto deletion
while contributing little to mean fitting loss. This is a hypothesis from the
loss/gate definitions, not an established cause of the withheld result.

A new preregistered study could add a worst-interior-frame absence loss only for
supervised REMOVE fitting candidates whose entire interior has known-zero
frame targets. Genuine KEEP examples and UNKNOWN frames must never receive
that penalty. Retain all existing genuine-note protection, fixed phase clocks,
normal/stricter thresholds and every release gate. Initialize fresh weights;
never retune this completed V58 study or use failed test labels for fitting.

Separately, 49 later real-piano passages have only symbolic approval for possible
future acquisition. Their audio and MP3 clocks must be verified before fitting.
Neither proposal changes V58's data, results or production weights.
