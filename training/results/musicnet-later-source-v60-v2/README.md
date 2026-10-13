# Bounded parallel acquisition V2

The fixed 49 passages, 20 fitting-only parents and complete original archive
checksums are unchanged. A separate version fetches up to four exact HTTP
206 ranges in parallel and presents them in original byte order to the unchanged
whole-object checksum, gzip/TAR and native waveform-clock verification.
Each chunk is at most 16 MiB; pending plus current chunk buffers are bounded
to four, and each consumer read is at most 1 MiB. Wrong status, offset, total,
length or truncated bytes fail closed. The full original MD5/SHA-256 and
bit-identical historical source crops remain mandatory before completion.

No source audio is transcribed, and no new source enters any fitting/evaluation
cohort. V1's producer, partial plan and evidence are preserved. V1 should only
be stopped after this separate tested transfer has shown live progress.
The result remains outside training until source and MP3 preparation audits pass.

Thirty-five focused transport/native-source tests pass. Complete preflight
checks are running; V2 acquisition has not started.

Complete preregistration preflight: 1,519 tests passed, six skipped and five
warnings. The finite-score guard was corrected and verified before any joint
predictions. Original source studies and source-acquisition V1 are unchanged.
