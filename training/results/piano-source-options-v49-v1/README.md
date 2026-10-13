# Additional piano-source assessment

This is source research only. No user audio, dataset audio, model inference or
new fitting is involved. The approved website model is unchanged.

- [PDMX v9](https://zenodo.org/records/15571083) has a CC BY 4.0 dataset grant and
  per-score public-domain/CC0 metadata. Its authors identify conflicting licenses
  and recommend the conflict-free subset. The publisher's complete 225,399,738-byte
  catalog passed MD5; all 254,077 rows were scanned. There are 222,820 conflict-free
  rows with valid associated files. This is not a selected or cleared training set.
  Program-zero instrument codes alone do not establish piano intent: 142,561 of
  175,136 clean valid single-track program-zero rows have unknown composers.
  A loose composer substring screen also admits mixed attributions and unrelated
  composers. Use explicit identities and verified work catalogs, compatible
  score rights and conservative connected work/composer partitions. Existing
  MusicNet/Mutopia work identities must be checked before any selection. Shared
  sampled-piano renders would supply synthetic clocks, not human performance.
- [RGTM-PNO](https://zenodo.org/records/13335848) has a CC BY 4.0 grant, but the
  exact publisher MIDI archive verifies to only 70 bytes with zero tar members.
  Its metadata's GitHub link leads to an EDM drum repository. No labeled piano
  pair is available from the checked record; its 7.5GB audio archive was not
  downloaded and cannot be treated as labeled key/hold truth.
- [PianoVAM](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1) is excluded for
  this potentially commercial service. The dataset card specifies CC BY-NC-SA 4.0;
  the [code README](https://github.com/yonghyunk1m/PianoVAM-Code) also specifies
  noncommercial terms. A paper's CC BY grant does not change those dataset terms.
- [SKY-Piano](https://joonhyungbae.github.io/skypiano/) describes per-modality
  release terms. The checked project page did not supply a usable commercial
  audio/MIDI grant, so no files were acquired or cleared for fitting.

Publisher metadata, catalog checksums and screening counts are retained here.
Raw catalogs remain local. These checks do not authorize using every public-domain
label blindly, mixing related works across partitions, or replacing real-piano
release evidence with synthesized data.

The complete MIDI and metadata archives also pass publisher sizes and MD5 hashes.
A streaming inventory checks unique relative paths, file types, sizes and catalog
links without extracting any member. The MIDI archive contains 254,035 files;
metadata contains 254,077. All 222,820 conflict-free valid rows link to both files.
The 42 missing MIDI references are explicitly marked `NA` on invalid rows.
This integrity proof does not clear any composition, arrangement or performance
for fitting. No symbolic notes were rendered, transcribed or used in training.
The audit tool has regression coverage for traversal, links, duplicate members,
unexpected types, missing references and changed download checksums.
The full backend/training suite passes 1,075 tests with six skips and five
dependency warnings. The 14 archive checks also pass separately. Approved
left-presence and chord-release asset hashes remain unchanged.

An exploratory exact-composer-alias metadata screen inspected 148 conflict-free,
valid, single-program-zero CC0 catalog rows. All 148 have matching score license
fields and explicit CC0 URLs; 147 explicitly name one piano, and 140 have exactly
matching catalog/metadata composer text. None are marked blocked or disputed in
the inspected fields. These rows occupy 33 catalog groups, which are not verified
work identities. The counts and complete screening source are retained here.
This is still not a training selection: mismatched identities, score rights,
related arrangements, symbolic clocks and grouping need review before rendering.

Additional checks while V50 fitted:

- [PiJAMA](https://almostimplemented.github.io/PiJAMA/) publishes automatically
  transcribed MIDI and does not distribute its source audio. It is excluded as
  independent human note-error supervision; agreement between transcription
  systems is not independently recorded key truth. No files acquired.
- [PianoCoRe](https://zenodo.org/records/19186016) explicitly releases its original
  and processed data under CC BY-NC-SA 4.0. It is excluded from this potentially
  commercial model. No files acquired.
- [Vienna 4x22](https://datasets.mdw.ac.at/datasets/dataset/98ea25fa-2468-43ff-929b-3c926e163583)
  is already represented in earlier project development. Its existing source,
  clock and partition witnesses must remain authoritative; it cannot be presented
  as new independent data or reassigned across existing development/test cohorts.

PDMX catalog attribution: Phillip Long, Zachary Novack, Julian McAuley and
Taylor Berg-Kirkpatrick, PDMX v9, DOI 10.5281/zenodo.15571083, CC BY 4.0.
Changes here are computed catalog counts and screening summaries; no score or
audio content is distributed. The publisher JSON retains the complete citation.
