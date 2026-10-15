# Restore original piano training inputs without changing release gates

Reacquired the exact original 60 NRG-CP progression sources after checking the
publisher CC BY 4.0 grant, archive size, MD5 and SHA-256. The restored source
manifest is identical to the original archived selection and partitions, including
the original exclusion of 1,426 ambiguous same-key clips. The full archive was
removed after verification. The installed sampled piano, rendering library and
license files matched their original hashes. All 180 original derivatives passed
native waveform, key-clock, MP3 alignment, pitch/context and progression-group
checks (144 fitting, 36 validation). These are authored synthetic clocks, not
human-performance labels; the independent real-piano gate remains mandatory.

The six public-domain source editions and partitions also match their original
manifest exactly. Only the original five fitting/validation works were rendered.
All 20 generated fixtures and measured physical recurrence inputs passed native
checks (16 fitting, four validation); the reserved work was not rendered here.

The older baseline-data reader needed a historical route absent at its recorded
preceding commit. The restorer recovered the exact hash-matched body from Git
history, without modifying production or relaxing any guards. Its original
verification passed all 22 source files and 13 inherited model arrays. A full
reader copies only small release/plan witnesses, not large prediction reports
or dataset duplicates, and creates no extra symlinks. Its footprint is about
15 MB. Git history searches are bounded; missing Git in the minimal runtime now
fails cleanly without creating a partial reader. Initial failed checks caught
that error path; those logs and the passing corrected checks are preserved.

The prior NRG-CP audit was preserved when its immutable-output check rejected
overwriting it with the regenerated-cache audit. A separately named native audit
was then completed successfully; no data or model setting was retuned.

Twelve focused and 1,560 full-suite tests pass, with seven full-suite skips and
five warnings. The large original piano and isolated-pitch downloads remain live.
Fitting has not started and no candidate accuracy improvement or deployment is
claimed. All local training data must remain below 25 GB; no user uploads or
saved scores are processed.
