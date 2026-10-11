# Independent left-hand support V39 — withheld

Two new classifiers completed 40 epochs each on 781 licensed training excerpts
and 225 composition/source-disjoint validation excerpts. Coarse sustain,
attack and release CNNs learn directly from physical windows and 100 context
features. They do not reuse the old model's removal logits or guardian scores.
Training targets protect every supplied pitch continuation and genuine attack
or held-note match; KEEP receives 16/64 times the loss weight of REMOVE.

Selection is measured against deployed V38, rather than treating old V32
survivors as observations from the newer website. Removal decisions use complete
V32 evidence, V38 releases run on that same complete input, and only then is the
removal mask applied. Retained clocks are consequently identical to V38.
The original V32 readers run in an isolated, hash-checked source snapshot;
historical guards and website code are unchanged.

Neither candidate qualified. The best preserving diagnostic, small encoder at
epoch 40, removed 3/1 false notes at normal/stricter settings, but **zero real
piano false notes**. At threshold .8 the same checkpoint removed 43 false notes
while failing four recordings' preservation checks. Training loss decreased;
that does not establish a deployable transcription improvement.

No consumed regression, new first pass, export or deployment followed. All 563
consumed recordings remain required for a future validation winner. V38 remains
on the website. No user audio, saved jobs or scores were processed, and engraving
size is unchanged.

The plan, searches, complete diagnostic rows, curves, actual checkpoint hashes
and frozen source witnesses are archived here. Model checkpoints and licensed
audio/feature caches remain local. A subsequent separately declared study adds
physical waveform periodicity, prompted by the validation failures on quiet bass
notes. It must satisfy the same preservation and independent real-piano gates.
