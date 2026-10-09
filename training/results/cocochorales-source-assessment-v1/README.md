# CocoChorales source assessment — metadata only

Publisher: https://magenta.withgoogle.com/datasets/cocochorales
Data credit: Yusong Wu. Research citation: Yusong Wu, Josh Gardner, Ethan
Manilow, Ian Simon, Curtis Hawthorne and Jesse Engel, The Chamber Ensemble
Generator: Limitless High-Quality MIR Data via Generative Modeling (2022).
Dataset license: CC BY 4.0, as declared by the dataset publisher. The separate
repository code license is not used as a substitute for the dataset license.

Two complete metadata archives, original train/1 and valid/1, were downloaded
and matched against the publisher's full MD5 list, with independent SHA256 and
byte counts. Every metadata member was read without extraction, checked for
unsafe paths/links and bounded sizes, and parsed using safe YAML. Seven tests
verify original MIDI grouping and refusal of malformed/unsafe packages.

Each package contains 2,000 records. No original midi_file IDs overlap between
these two packages. This limited inspection does not prove independence of the
entire dataset. Future selection must group all ensemble versions of the same
original MIDI together and preserve fitting/validation/reserved independence.

The corpus contains generatively synthesized four-part chamber ensembles, not
piano recordings. It may provide new harmonic/vibrato/separation-leakage examples,
but cannot replace a real-piano release gate. Exact synthesis timing and pitch
augmentation semantics must be checked when audio and MIDI are acquired.

No audio, MIDI, note-expression or synthesis-parameter packages were acquired.
No new model was fit on this corpus. It is not included in the already frozen
V35 training plan. Only source metadata and checksum witnesses are archived here;
the compressed packages remain local. Future fitting requires a separately
predeclared, fully verified bounded acquisition and physical observation study.
