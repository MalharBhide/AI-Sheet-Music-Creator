# Wrong-pitch bass training against V11

The website remains on the [verified V11 release](bass-articulation-v11.md).
This work addresses unsupported pitches in balanced bass, while retaining its
existing note filtering and trained hold repair. Previous deletions cannot return.
No user uploads or saved scores enter fitting, inference or demonstration jobs.

`current_bass_v11_baseline.py` pins both V10 and V11 arrays, thresholds and
production routing. The V11 cache contains the actual extended durations, rather
than treating historical V10 fragments as current website output. Across 545
recordings, V11 merges 55 boundaries and extends 53 retained intervals. The 40
affected recordings were decoded natively: raw notes/acoustic features matched
their frozen caches exactly, both real production filters matched cached V11
events exactly, and features were recomputed using extended intervals. Unchanged
per-note features do not depend on neighboring note identities and can be reused.

The cache preserves 359 train, 94 validation and 92 consumed regression clips.
Performer/source groups remain separate. Test recordings cannot enter fitting or
selection. The corpus is limited: licensed BabySlakh, licensed GuitarSet and
original synthetic timbres. Shared compositions, synthesized-source limitations
and upstream pretraining overlap remain; this is not a commercial-song benchmark.

The first residual batch fit four boosting heads using 22,378 supervised events.
The context vector contains 52 acoustic features, 20 pitch-relative relationships
and two recomputed baseline confidences (74 total). Its guardian uses 26 base
acoustic features and two confidences (28 total). No labels, recording names or
absolute pitch are inference features. Fixed paired profiles used positive weights
6/12, guardian weight 20, learning rate .04, 600-tree maximum, and declared
100/200/400/500/600 checkpoints. Both pairs' later validation loss worsened, and
no checkpoint passed every preservation gate with a positive gain. These weights
were withheld without regression evaluation, fresh testing, export or deployment.

A separate label-policy experiment aligns fitting with the existing pitch-coverage
gate: every supplied pitch overlap exceeding the gate's `1e-6` second tolerance
receives positive KEEP supervision, including short partial continuations.
Matched attacks and offset-aware assignments also remain positive. Historical
labels, detector features, reference annotations and release gates are unchanged.
This prevents teaching deletion of some partial true-pitch fragments simply
because their support is below the earlier 90% promotion threshold.

This second batch fit four heads using 22,987 supervised events. Validation alone
selected the guarded 500-tree pair with `.005/.025` thresholds. It removed 267
false validation detections with all 4,150 matched attacks, previously matched
offsets and supplied pitch coverage preserved per recording. Frozen evaluation
on the 92 consumed regressions removed 325 false detections but lost seven matched
attacks, offsets and pitch coverage across four recordings. **The winner is
withheld.** No runner-up, test-based threshold retuning, fresh test, export or
deployment occurred. Improving aggregate precision does not justify losing those
correct notes. Frozen plans, checkpoints' hashes, curves and rejection evidence
are archived under `training/results/bass-residual-v11-v1` and
`training/results/bass-residual-coverage-v2`.

The following paragraphs record the pending state at V11. This batch has now
completed and its tested successor was released as V12; see
[bass-residual-v12.md](bass-residual-v12.md) for the final result.

The next bounded batch was running locally. Its frozen acquisition plan adds 48
MP3/Demucs bass crops at 30/60/90 seconds from existing licensed Slakh training
and validation source groups (36 train, 12 validation). No reserved Slakh groups
are decoded. It also adds 80 original generator seeds 261801–261880 (64 train,
16 validation), distinct from every fitting and consumed regression seed used
before this batch. These are new timbre/source examples from the existing
distributions, not independent human-song test evidence.

`continue_bass_v11_training.py` waits for the complete, hash-bound source manifests,
rejects a stopped producer/partial preparation, rebuilds actual V11-duration
evidence, and fits the same declared residual profiles against the enlarged
459-train / 122-validation corpus. It evaluates only a frozen positive validation
winner on the 92 consumed regressions. This runner cannot export, deploy or push
weights. Any candidate still needs fresh stress, portable/native parity, full
integration and a verified idle-queue release. Its live local status is
`.training/bass-expanded-training-v3/status.json`; archived status files are
explicit snapshots and must not be read as completed training evidence.

415 backend/training tests passed with six opt-in integrations skipped for the
new offline framework; three additional continuation safeguards passed. The six
native pipeline cases already passed for the unchanged live V11 code. Ruff and
diff checks pass. Preparation/training use local compute and licensed/original
fixtures only, with no user transcription jobs or score regeneration.
