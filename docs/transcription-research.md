# Audio-to-score system research

## Supervised melody model — September 13, 2026

The earlier pYIN event filter below is now the comparison baseline. Full-song
vocals use a newly trained temporal decoder, with published dataset provenance,
singer-disjoint splits, held-out real-singing results and a controlled separation
test. Read the [framework](transcription-framework.md),
[training procedure](../training/README.md) and [model card](melody-model-card.md).
The measured improvement applies to vocal recognition; the model does not learn
the complete piano arrangement or establish commercial-product parity.

## Follow-up: vocal continuity and beat evidence — September 13, 2026

Basic Pitch exposes polyphonic pitch, onset and contour estimates and documents its strongest use case as one instrument at a time ([official repository](https://github.com/spotify/basic-pitch)). Its note events alone do not establish which harmonic is the sung fundamental, or whether a vibrato cycle is a new piano attack. Librosa's [pYIN documentation](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html) describes a fundamental-frequency estimator using probabilistic YIN candidates and Viterbi decoding of pitch and voicing. This provides an independent monophonic check using dependencies already installed in the application.

The implemented refinement applies that check only to separated vocals. Strong conflicting pitch evidence rejects an event; matching evidence bounds its outer duration. Continuous same-pitch events merge only when no breath or amplitude re-attack is observed. Uncertain tracking preserves neural events. This is a conservative combination of detectors and acoustic evidence, not a newly trained model or a replacement for source separation. Singing with overlapping voices, consonant changes without clear amplitude attacks, and expressive pitch motion remain limitations.

The accompanying [reproducible benchmark](vocal-quality-benchmark.json) records improvements on seven original synthesized vocal-like phrases, including the failure cases. It does not establish improved accuracy on an annotated real-song corpus. Beat estimation was also revised after a reproducible competing-pulse fixture showed that a single 120 BPM prior could select the wrong rhythmic layer. Candidate beats are now compared against the audio, with robust interval fitting and half/double-time reconciliation across excerpts.

## Findings and engineering decision

A useful transcription product needs to identify the intended musical part, detect notes, interpret rhythm, produce playable notation, and let a musician hear and correct the result. A downloadable PDF proves only that engraving succeeded. It does not establish that the underlying notes, meter, or arrangement are correct.

The strongest lesson from Songscription, Klangio, AnthemScore, and ScoreCloud is their separation of musical tasks and their investment in review. They expose instrument or arrangement choices and provide playback or editing around the generated score. Songscription's own documentation says clean single-instrument recordings are easier than busy mixes; it does not establish a measured success-rate advantage over this application.[^1][^2]

**Recommended architecture:** use a piano-specific model for piano recordings; separate useful stems before analyzing full mixes; offer a distinct melody mode; treat tempo, beat placement, and notation cleanup as explicit stages; and generate playback from the same cleaned musical representation used for MusicXML and PDF. Present full-mix output as a simplified draft unless a validated arrangement model is actually producing it. This is an engineering recommendation based on the evidence below, not a claim of parity with commercial products.

## Why pitch detection alone produces disappointing scores

Three outputs that sound similar in product copy solve different problems:

| Intended output | Musical objective | Consequence for this application |
|---|---|---|
| Direct transcription | Recover a particular instrument's performed notes | Choose a suitable model and, for crowded recordings, a suitable input stem. |
| Lead sheet | Recover the main melody and harmonic outline | Avoid filling the staff with every backing instrument; chord symbols require their own analysis. |
| Piano arrangement | Recast several musical layers for two hands | Select, simplify, and voice musical material; changing every MIDI instrument to piano is insufficient. |

Songscription documents piano covers separately from direct transcription. Klangio's model catalog similarly distinguishes `piano`, `lead`, and `piano_arrangement`.[^1][^3] **Inference:** a commercial pop recording can produce a technically plausible collection of pitches while failing the listener's actual goal. Singing, bass, percussion, synth layers, and overlapping harmonics are not automatically a playable piano part.

Rhythmic interpretation introduces another failure class. Correct timestamps in seconds must become beats, measures, note values, ties, and rests. A wrong first downbeat shifts barlines; a half-speed tempo estimate changes written durations; an excessively fine grid makes performance variation look like complex notation. SheetSage explicitly documents half/double tempo errors and brittle downbeat detection, including manual overrides.[^4] AnthemScore separately exposes beat editing and musical versus performance timing.[^5]

**Inference:** thresholds and note removal can make an unreadable score cleaner without making it more faithful. A simplified score therefore needs an honest label and a retained original transcription so information is not silently lost. Hand assignment, note duration, and rhythmic grouping deserve tests independent of the pitch model.

## Commercial product comparison

| Product | Documented scope | Review and export workflow | Integration implication |
|---|---|---|---|
| Songscription | Target one instrument from a recording; separate piano-cover and simplification features | Score editing, piano roll, speed control; PDF, MIDI, MusicXML, Guitar Pro | Public FAQ directs API/bulk inquiries to the company; no public endpoint contract verified.[^1][^2] |
| Klangio / Piano2Notes / Transcription Studio | Specialized instruments, multiple-part transcription, lead sheets and arrangements | Browser editing; quantized and unquantized MIDI; standard score formats | Public asynchronous REST API with model selection and separate audio-analysis jobs.[^3][^6][^7] |
| AnthemScore | Audio note detection with detailed control over notation and instrument groups | Original/notes comparison, looping, beat/note edits, PDF/MIDI/MusicXML | Desktop processing and web product; current release history removes old CLI options.[^5][^8] |
| ScoreCloud | Songwriter: melody, lyrics, chords from mixed audio. Studio: single-instrument capture and full notation editing | Separated/source playback, MIDI accompaniment, notation correction | Useful distinction between capturing a lead sheet and building a full arrangement.[^9][^10] |
| Spotify Basic Pitch | Lightweight, instrument-agnostic polyphonic audio-to-MIDI | Library/CLI, raw events and MIDI; a separate notation workflow is required | Keep as a portable baseline; documentation recommends one instrument at a time.[^11] |

### Songscription

The FAQ, updated August 30, 2026, lists piano, guitar, bass, double bass, violin, flute, trumpet, saxophone, drums, and vocals. It describes single-instrument extraction from a mix, with full-band/orchestral transcription still on its roadmap. It also acknowledges that noisy or busy audio needs cleanup. Paid plans allow up to 15 minutes per transcription; this is not evidence of unlimited-length processing.[^1]

Its homepage demonstrates the important product interaction: listen to the transcription, change playback speed, and inspect either notation or a piano roll. The roll also supports hand separation and transposition.[^2] Its arranging guide describes adapting a transcription to a specific instrument and difficulty, rather than simply exporting every detected event.[^12]

The company's full-band guide recommends working part by part. It says stems are optional for its proprietary models and can help particularly dense material.[^13] **Inference:** this does not mean a generic model can reproduce Songscription by skipping separation. The vendor's training data, model architecture, and objective benchmark results were not disclosed in the reviewed product documentation. Testimonials and selected demonstrations do not supply a denominator, representative test corpus, or reproducible comparison.

### Klangio, Piano2Notes, and Transcription Studio

Piano2Notes documents polyphonic piano transcription, extraction of piano from ensembles, editing, an interactive roll, and separate quantized/unquantized MIDI exports.[^14] Transcription Studio expands the interface to separate instrument parts and exposes rhythm, tempo, key, and time-signature editing.[^6] Klangio's current support guide distinguishes solo, multi-instrument, rock, ensemble, and arrangement modes. It explicitly says the ensemble mode is not yet suitable for large orchestras and that polyphonic vocals can merge into a single staff.[^15]

For integration, Klangio is the clearest documented commercial candidate. Its API has instrument-specific models, melody/chord extraction, and a piano-arrangement model.[^3] Jobs are submitted asynchronously, then polled or completed through webhooks. API requests require a key and can incur quota costs.[^7] Transcription output is retained for 14 days, and additional export formats may incur additional quota charges.[^16] Beat tracking is a separate endpoint that returns timestamps and positions within measures.[^17]

**Recommendation:** retain a provider interface so a hosted transcription service can be evaluated later against the same local test corpus. Do not silently upload user recordings to a third party or make an undocumented API the default. API availability establishes integration feasibility; it does not establish accuracy, latency, or cost for this workload. Those require a trial with authorized recordings and an actual service account.

### AnthemScore

AnthemScore's documentation is particularly useful for correction design. It supports listening to the original, synthesized notes, or both; separate volume controls; slowed playback; loops; note edits; movable beats; and manual timing changes.[^5] These are mechanisms for finding and repairing transcription errors, rather than evidence that automatic output is always correct.

Version 6.2.0, released September 4, 2026, specifically addresses beat/downbeat and meter detection, rhythm quantization, hand assignment, and arrangement difficulty. It adds chord symbols and expanded score playback.[^8] The older full manual still says chords are not detected and describes features removed by later releases. The dated release history takes precedence for those conflicts. It also records removal of CLI options in version 6.0; old command-line examples should not be assumed to describe a supported current integration.[^8]

**Inference:** the release priorities reinforce that better pitch inference alone does not solve readable notation. Manual bar alignment and hand assignment remain valuable even after a model upgrade.

### ScoreCloud

ScoreCloud Songwriter uses source separation and analysis to produce melody, lyrics, and chord symbols from voice with accompaniment; it can play separated vocals and alternate accompaniment patterns.[^9] The vendor's comparison distinguishes this from Studio, which records/imports a single instrument and builds multi-part arrangements through editing and additional parts. Songwriter also keeps original audio synchronized for comparison.[^10]

ScoreCloud's audio-to-score guide presents automatic output as a draft and recommends comparing it with original audio.[^18] The guide's broad percentage estimate is not accompanied by a shared test set or evaluation method, so it is not used here as measured accuracy. **Recommendation:** adopt the workflow distinction and review loop, rather than borrowing an unsupported percentage for marketing.

## Implementable open-source systems

| System | Supported task and deployment evidence | License evidence | Recommended role |
|---|---|---|---|
| Basic Pitch | Polyphonic notes; Python/JavaScript tooling; TensorFlow, CoreML, TFLite and ONNX variants | Apache-2.0 repository | Portable baseline and general single-instrument option.[^11] |
| ByteDance high-resolution piano transcription | Piano notes and pedal events; CPU/CUDA inference wrapper | Apache 2.0 code declaration; CC BY 4.0 checkpoint | Primary local candidate for actual piano; retain model attribution and test dependencies.[^19][^20][^29] |
| Aria-AMT | Sequence-to-sequence piano transcription; Python 3.11; compiled/batched inference | Apache-2.0 repository; project card distinguishes tools from dataset | Evaluate as another piano backend after deployment benchmarking.[^21][^22] |
| Demucs | Audio source separation, not note transcription; four-stem CPU/GPU workflow | MIT repository license | Separate vocals/bass/accompaniment for full-mix analysis.[^23][^24] |
| MT3 | Multi-instrument note transcription using T5X; pretrained Colab workflow | Apache-2.0 repository | Research comparison; substantial integration work remains.[^25][^26] |
| SheetSage | Pop lead sheets with melody/chords; Linux/Docker workflow | MIT code, but models CC BY-NC-SA 3.0 and additional dependency terms | Research reference; not an unrestricted commercial dependency.[^4] |

### Piano-specific inference

Kong and colleagues' high-resolution model regresses note onset/offset and pedal timing. Their paper reports **96.72% onset F1 on MAESTRO**, versus 94.80% for the compared Onsets and Frames system.[^19] This is a specific research result on piano data. It is neither 96.72% correct sheet music nor a success rate on mastered pop songs, and it does not compare Songscription or this application.

The inference wrapper documents CPU and CUDA operation and FFmpeg for MP3 decoding. It was developed against Python 3.7 and PyTorch 1.4, with other versions incompletely tested.[^20] Upstream is archived and gives 29 GB GPU memory at batch size 12 for **training**; that number is not an inference requirement.[^27] The official Zenodo model record separately identifies the checkpoint license as **CC BY 4.0**, credits Qiuqiang Kong, and dates publication September 17, 2020.[^29] **Recommendation:** retain this model attribution, use a pinned compatibility environment, cache the checkpoint, process bounded overlapping audio windows, and measure CPU time and peak memory.

The published checkpoint is `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`, 171,966,578 bytes, with the publisher's checksum `md5:22b961b77c1878239fec963362097045` and DOI `10.5281/zenodo.4034264`.[^29] These identify the evaluated asset; its license should not be conflated with the code's Apache declaration.

Aria-AMT's README requires Python 3.11 and describes compilation, batching, and optional int8 inference; quantization requires BF16-capable GPUs.[^21] The Aria-MIDI card identifies its transcription engine as Linux/CUDA software and the dataset as CC BY-NC-SA 4.0, while describing pipeline tools as Apache-2.0.[^22] **Recommendation:** do not conflate the generative Aria MIDI model with Aria-AMT, or the dataset license with the code license. No dependable minimum VRAM or target-Mac throughput figure was established from these sources, so a deployment experiment is needed before replacing a working CPU path.

### Separation and multitrack alternatives

Demucs' standard model separates drums, bass, vocals, and an `other` stem. Its experimental six-source model adds piano and guitar, but the maintainers explicitly warn about piano bleed and artifacts. The README describes CPU fallback, segmentation to reduce GPU memory, and approximately 3–7 GB GPU memory depending on settings; transformer segments have a maximum of 7.8 seconds.[^23] **Recommendation:** use established four-stem separation for full mixes, exclude drums from pitched notation, and avoid labeling `other` as an isolated piano. Separation artifacts must be evaluated by listening to the stem as well as the final score.

MT3 jointly transcribes instruments into symbolic events. The original paper frames it as a multi-task model across transcription datasets and calls for more consistent metrics and alignment.[^26] Its repository uses T5X, provides pretrained inference through Colab, and says training is not easily supported.[^25] **Inference:** this is a credible research baseline, but adopting it is not a small dependency swap or a guarantee of engraved multi-staff scores. Model output still needs score interpretation and resource validation.

SheetSage directly targets pop melody/chord lead sheets. Its documented setup downloads roughly 4 GB of Docker image and 100 MB of data; optional Jukebox features need at least 12 GB GPU memory plus roughly 10 GB of downloads. Its models were trained on approximately 24-second segments. Noncommercial model terms and brittle dependencies complicate unrestricted production use.[^4] **Recommendation:** borrow its explicit beat/segment controls and evaluation approach while treating the actual models as a separately licensed research option.

## Proposed application behavior

The following is a design recommendation, not a declaration that every item has already shipped. Implementation status and observed test results belong in the change log and test output.

### Musical modes

Offer three clearly explained choices:

1. **Piano recording:** use the specialized piano engine on the original piano audio. Preserve repeated attacks and distinguish pedal sustain from notated duration when the model supports it.
2. **Full mix:** separate the recording, analyze pitched stems independently, and create a controlled musical draft. Show which layers are included. Do not promise an exact piano cover or orchestral score.
3. **Melody:** prioritize a single clear line. Document whether it uses vocals, an isolated instrument, or a heuristic melody selection, because these can produce different answers.

Expose automatic tempo with an optional manual override. An override should describe a musical interpretation, not silently stretch the original recording. Keep the selected time signature visible. Longer-term, add first-downbeat correction, half/double tempo switches, and segment-level tempo maps for rubato. A fixed BPM can be a practical first step, but cannot faithfully represent every performance.

### One score, several representations

Store original detected events independently from the cleaned score. Apply pitch-range filtering, duplicate suppression, rhythm quantization, voice grouping, and difficulty reduction to a deliberate score representation. Export score MIDI, MusicXML, and PDF from that representation. If raw performance MIDI is also offered, label it distinctly, following the useful quantized/unquantized distinction documented by Klangio.[^14]

The website's default player should sound the score that is displayed. A player driven by unrelated raw model output can disguise errors introduced during notation cleanup. Provide play/pause, seek, elapsed time, playback speed, volume, and an original-audio comparison. Browser playback should start after a user action and keep controls usable while the score loads. For long pieces, schedule a short look-ahead window of notes and render only the visible piano-roll region.

### Website structure and recoverable jobs

Use a recognizable product shell with navigation, a focused upload workspace, helpful mode descriptions, and a results workspace. Give the score and playback controls the visual priority. Make progress explain actual stages: decoding, separating where needed, transcribing, interpreting rhythm, and preparing the score. Keep downloads and retry actions close to the result instead of scattering them across diagnostic panels.

Preserve useful partial outputs and make failed stages explicit. A model failure must not silently become a successful blank score. Persist job identity so a refresh returns to the same result. Keep long-file processing asynchronous with bounded audio buffers, resumable progress, and no arbitrary duration rejection. No implementation can guarantee successful processing of literally any file or unlimited duration: corrupted audio, missing samples, finite storage, and finite compute need precise errors. Valid MP3 decoding reliability and musical transcription accuracy should be tested separately.

## Evaluation and evidence required for accuracy claims

No controlled, reproducible public benchmark found in the reviewed primary product documentation establishes that Songscription has a higher success rate than all alternatives. A comparison must use the same recordings, task definition, target parts, ground truth, and scoring rules. Curated examples, model confidence, completed jobs, and PDF page counts do not measure transcription accuracy.

For note quality, use precision, recall, and F1 matching both pitch and onset, then repeat with offset constraints. `mir_eval` documents default tolerances of 50 ms for onset, 50 cents for pitch, and the larger of 50 ms or 20% of reference duration for offsets. Its dedicated onset-only function ignores pitch, so that function alone cannot establish correct notes.[^28]

**Proposed validation corpus:** clean acoustic and digital piano; repeated notes and pedal; solo voice/instrument; dense mixes with known stems; silence and nonmusical inputs; leading silence and pickups; stable tempo, swing, compound meter, rubato, and tempo changes. Include mono/stereo, variable/constant-bitrate MP3, multiple sample rates, very short clips, and recordings longer than one model window. Use synthetic fixtures for deterministic regressions and annotated real performances for accuracy; neither replaces the other.

Measure notation separately: correct bar alignment, rhythmic readability, voice/hand assignment, excessive ties, omitted melody notes, and agreement between score playback and exported notation. Have a musician compare original and synthesized output without knowing which engine produced it. Report error rates by recording category, peak memory, duration coverage, and processing-time distribution. Only then describe an accuracy improvement quantitatively. Until that evidence exists, the defensible claim is a more appropriate pipeline and a better review workflow.

## Sources

Sources were accessed on **September 10, 2026**, except the official model metadata verified on **September 11, 2026**. “Undated” means the retrieved page did not expose a reliable publication/update date; repository contents and vendor capabilities can change. Titles link to the exact primary sources. Vendor descriptions establish documented features, not independently verified musical quality.

| Ref. | Publisher and exact source | Publication / update date | Access date |
|---|---|---|---|
| 1 | Songscription, [Frequently Asked Questions](https://www.songscription.ai/faq) | Updated 2026-08-30 | 2026-09-10 |
| 2 | Songscription, [Product homepage](https://www.songscription.ai/) | Undated | 2026-09-10 |
| 3 | Klangio, [Model selection for Transcription](https://api-docs.klang.io/docs/advanced-usage/transcription-model-selection) | Undated | 2026-09-10 |
| 4 | Chris Donahue et al., [SheetSage repository and README](https://github.com/chrisdonahue/sheetsage) | Research cited as ISMIR 2022; README undated | 2026-09-10 |
| 5 | Lunaverus, [Full Documentation](https://lunaverus.com/documentation) | Undated; partly superseded by release history | 2026-09-10 |
| 6 | Klangio, [Transcription Studio](https://klang.io/transcription-studio/) | Undated | 2026-09-10 |
| 7 | Klangio, [Basic Job Workflow](https://api-docs.klang.io/docs/getting-started/basic-job-workflow) | Undated | 2026-09-10 |
| 8 | Lunaverus, [AnthemScore Version History](https://lunaverus.com/versionHistory) | Version 6.2.0: 2026-09-04 | 2026-09-10 |
| 9 | Doremir, [ScoreCloud Songwriter](https://scorecloud.com/songwriter/) | Undated | 2026-09-10 |
| 10 | Doremir, [ScoreCloud Songwriter vs. Studio](https://scorecloud.com/learn/songwriter-vs-studio/) | Undated | 2026-09-10 |
| 11 | Spotify, [Basic Pitch repository and README](https://github.com/spotify/basic-pitch) | Research cited as ICASSP 2022; README undated | 2026-09-10 |
| 12 | Andrew Carlins / Songscription, [How to Arrange a Song for Any Instrument](https://www.songscription.ai/blog/arranging-guide) | 2026-06-29 | 2026-09-10 |
| 13 | Andrew Carlins / Songscription, [How to Transcribe a Full Band](https://www.songscription.ai/blog/multi-instrument-transcription-guide) | 2026-06-29 | 2026-09-10 |
| 14 | Klangio, [Piano2Notes](https://piano2notes.klang.io/) | Undated | 2026-09-10 |
| 15 | Klangio, [What Music can Klangio Transcribe?](https://klang.io/help/what-music-transcribe/) | Updated 2026-08-12 | 2026-09-10 |
| 16 | Klangio, [Create a transcription](https://api-docs.klang.io/docs/jobs/transcription-requests) | Undated | 2026-09-10 |
| 17 | Klangio, [Get the beats and downbeats of an audio](https://api-docs.klang.io/docs/jobs/beat-tracking-request) | Undated | 2026-09-10 |
| 18 | Doremir, [How to Convert Audio to Sheet Music](https://scorecloud.com/learn/how-to-convert-audio-to-sheet-music/) | Undated | 2026-09-10 |
| 19 | Qiuqiang Kong et al., [High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times](https://arxiv.org/abs/2010.01815) | 2020-10-05; revised 2021-07-31 | 2026-09-10 |
| 20 | Qiuqiang Kong, [Piano transcription inference](https://github.com/qiuqiangkong/piano_transcription_inference) | Undated README | 2026-09-10 |
| 21 | EleutherAI, [Aria-AMT repository and README](https://github.com/EleutherAI/aria-amt) | Undated README | 2026-09-10 |
| 22 | Louis Bradshaw / EleutherAI, [Aria-MIDI dataset card](https://huggingface.co/datasets/loubb/aria-midi) | Research cited as ICLR 2025; card undated | 2026-09-10 |
| 23 | Meta, [Demucs repository and README](https://github.com/facebookresearch/demucs) | v4 / six-source announcement: 2022-12-07; README undated | 2026-09-10 |
| 24 | Meta, [Demucs LICENSE](https://raw.githubusercontent.com/facebookresearch/demucs/main/LICENSE) | Undated | 2026-09-10 |
| 25 | Magenta, [MT3 repository and README](https://github.com/magenta/mt3) | Research cited as ICLR 2022; README undated | 2026-09-10 |
| 26 | Josh Gardner et al., [MT3: Multi-Task Multitrack Music Transcription](https://arxiv.org/abs/2111.03017) | 2021-11-04; revised 2022-03-15 | 2026-09-10 |
| 27 | ByteDance, [Piano transcription repository and README](https://github.com/bytedance/piano_transcription) | Archived 2025-12-08; README undated | 2026-09-10 |
| 28 | mir_eval contributors, [mir_eval.transcription documentation](https://mir-eval.readthedocs.io/latest/api/transcription.html) | Version 0.8.2; page undated | 2026-09-10 |
| 29 | Qiuqiang Kong / Zenodo, [Official high-resolution piano model metadata](https://zenodo.org/api/records/4034264) | 2020-09-17 | 2026-09-11 |

[^1]: Songscription, [Frequently Asked Questions](https://www.songscription.ai/faq), updated August 30, 2026.
[^2]: Songscription, [Product homepage](https://www.songscription.ai/), undated.
[^3]: Klangio, [Model selection for Transcription](https://api-docs.klang.io/docs/advanced-usage/transcription-model-selection), undated.
[^4]: Chris Donahue et al., [SheetSage README](https://github.com/chrisdonahue/sheetsage), undated; ISMIR 2022 project.
[^5]: Lunaverus, [Full Documentation](https://lunaverus.com/documentation), undated.
[^6]: Klangio, [Transcription Studio](https://klang.io/transcription-studio/), undated.
[^7]: Klangio, [Basic Job Workflow](https://api-docs.klang.io/docs/getting-started/basic-job-workflow), undated.
[^8]: Lunaverus, [AnthemScore Version History](https://lunaverus.com/versionHistory), version 6.2.0 released September 4, 2026.
[^9]: Doremir, [ScoreCloud Songwriter](https://scorecloud.com/songwriter/), undated.
[^10]: Doremir, [ScoreCloud Songwriter vs. Studio](https://scorecloud.com/learn/songwriter-vs-studio/), undated.
[^11]: Spotify, [Basic Pitch README](https://github.com/spotify/basic-pitch), undated.
[^12]: Andrew Carlins / Songscription, [How to Arrange a Song for Any Instrument](https://www.songscription.ai/blog/arranging-guide), June 29, 2026.
[^13]: Andrew Carlins / Songscription, [How to Transcribe a Full Band](https://www.songscription.ai/blog/multi-instrument-transcription-guide), June 29, 2026.
[^14]: Klangio, [Piano2Notes](https://piano2notes.klang.io/), undated.
[^15]: Klangio, [What Music can Klangio Transcribe?](https://klang.io/help/what-music-transcribe/), updated August 12, 2026.
[^16]: Klangio, [Create a transcription](https://api-docs.klang.io/docs/jobs/transcription-requests), undated.
[^17]: Klangio, [Get the beats and downbeats of an audio](https://api-docs.klang.io/docs/jobs/beat-tracking-request), undated.
[^18]: Doremir, [How to Convert Audio to Sheet Music](https://scorecloud.com/learn/how-to-convert-audio-to-sheet-music/), undated.
[^19]: Qiuqiang Kong et al., [High-resolution Piano Transcription with Pedals](https://arxiv.org/abs/2010.01815), October 5, 2020; revised July 31, 2021.
[^20]: Qiuqiang Kong, [Piano transcription inference README](https://github.com/qiuqiangkong/piano_transcription_inference), undated.
[^21]: EleutherAI, [Aria-AMT README](https://github.com/EleutherAI/aria-amt), undated.
[^22]: Louis Bradshaw / EleutherAI, [Aria-MIDI dataset card](https://huggingface.co/datasets/loubb/aria-midi), undated; ICLR 2025 project.
[^23]: Meta, [Demucs README](https://github.com/facebookresearch/demucs), undated; v4 announcement December 7, 2022.
[^24]: Meta, [Demucs LICENSE](https://raw.githubusercontent.com/facebookresearch/demucs/main/LICENSE), undated.
[^25]: Magenta, [MT3 README](https://github.com/magenta/mt3), undated; ICLR 2022 project.
[^26]: Josh Gardner et al., [MT3: Multi-Task Multitrack Music Transcription](https://arxiv.org/abs/2111.03017), November 4, 2021; revised March 15, 2022.
[^27]: ByteDance, [Piano transcription README](https://github.com/bytedance/piano_transcription), archived December 8, 2025.
[^28]: mir_eval contributors, [Transcription evaluation](https://mir-eval.readthedocs.io/latest/api/transcription.html), version 0.8.2, undated.
[^29]: Qiuqiang Kong / Zenodo, [Official model record 4034264](https://zenodo.org/api/records/4034264), published September 17, 2020; accessed September 11, 2026. Metadata license identifier: `cc-by-4.0`.
