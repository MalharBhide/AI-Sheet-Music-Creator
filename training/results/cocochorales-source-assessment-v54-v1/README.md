# CocoChorales symbolic source assessment

The [publisher](https://magenta.tensorflow.org/datasets/cocochorales) grants CC BY
4.0 and describes synthesized four-part ensembles. The [format guide](https://github.com/lukewys/chamber-ensemble-generator/blob/master/data_format.md)
provides note-expression frame clocks and parent MIDI IDs. This is a possible
source of authored piano command patterns, not human piano performance truth.

Only training shard 1's note-expression and metadata archives were acquired:
10,623,029 and 2,361,634 bytes. Both match publisher MD5 and object-generation
metadata. No source recordings, F0/pickle files, validation or test shard was
acquired. The 5,114,972,715-byte main audio shard was only checked with HTTP HEAD.
The archive reader handles the publisher's concatenated parallel-bzip streams;
it rejects links, traversal, duplicate paths, changed voices and held-out IDs,
and never extracts members or loads executable serialization.

All 2,000 four-voice clocks pass integer 4ms frame, duration, pitch, rest and
expression-range checks. They contain 188,040 notes and 3,064 holds over four
seconds. Across voices, 1,964 chorales have overlapping commands for one piano
key: those are valid string-ensemble unisons but ambiguous piano key-offs.
No release or voice merge is guessed. The remaining 36 have 3,248 notes,
934 holds over one second and 62 over four seconds; all 36 normalized note
patterns are distinct. Source duration ranges 13.160–35.556 seconds and pitch
42–84, within the piano range. All are publisher training examples.

No selection is cleared for rendering or fitting yet. A future piano derivative
must freeze command/velocity mapping, full-work clocks and source groups, and
reject ambiguous keys rather than fit guessed durations. Original ensemble audio
has per-note pitch corrections; nominal symbolic pitches must not be mistaken
for exact original-audio F0 truth. Future generated piano audio would have its
own known command clock, not human onset/release or new timbre independence.

The full backend/training suite passes 1,180 checks with six opt-in skips and five
dependency warnings. Seventeen focused archive/clock tests pass. No user audio,
saved scores or model weights were processed or changed in this source audit.

Attribution: Yusong Wu, Josh Gardner, Ethan Manilow, Ian Simon, Curtis Hawthorne
and Jesse Engel, *The Chamber Ensemble Generator: Limitless High-Quality MIR
Data via Generative Modeling*, 2022. Dataset CC BY 4.0. Changes here are computed
symbolic counts, clock screens and integrity checks; source media is not distributed.

Other sources checked during V53 fitting:

- The [YCU-PPE-III MIDI card](https://huggingface.co/datasets/anusfoil/ycuppe-midi/blob/main/README.md)
  lists CC BY 4.0 but explicitly says the MIDIs were predicted using Transkun.
  They are excluded as independent human note/hold truth; expert performance
  ratings do not validate individual transcription notes. No files acquired.
- The [2026 MAPS record](https://zenodo.org/records/18160555) still explicitly
  links a noncommercial license. It remains excluded for this potentially
  commercial service; no files acquired.
