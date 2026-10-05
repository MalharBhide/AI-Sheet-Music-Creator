# Original MusicNet assessment

The next corpus candidate is the **original** MusicNet release, not MusicNetEM or
a repackaged mixed-license repository. Publisher record
[5120004](https://zenodo.org/records/5120004) identifies CC BY 4.0. Credit John
Thickstun, Zaid Harchaoui and Sham M. Kakade, University of Washington.

The publisher describes real classical recordings and aligned note/instrument
labels derived through score alignment. These could broaden bass verification
beyond synthesized Slakh and guitar/original timbres. The reported labeling error
rate means label quality and timing need inspection before fitting. This is an
appropriate research candidate, not a guarantee of better commercial-song
transcription. Do not silently substitute MusicNetEM labels or treat a wrapper
repository's license as authority over all its underlying corpora.

Publisher files are pinned as follows:

| File | Bytes | Publisher MD5 |
| --- | ---: | --- |
| `musicnet_metadata.csv` | 43,775 | `1caef62cee9c875235e62aac368b49d8` |
| `musicnet.tar.gz` | 11,097,394,998 | `844764911fa0d5b97c97da944a057590` |
| `musicnet_midis.tar.gz` | 2,601,302 | `b5fa98a113bfc51c8a445def9f24dc7e` |

The metadata file is downloaded and checksum verified: 330 recording rows with
composer, composition, movement, ensemble, source, transcriber, catalog name and
duration. Composer/catalog/composition identities can support a split that keeps
all movements/performances of a work together, rather than splitting individual
clips. This still does not establish performer independence or absence of
upstream detector pretraining overlap.

No MusicNet audio or labels have entered training. The host has about 2 GiB free,
so the 11.1 GB archive cannot be stored or fully extracted here. A future bounded
streaming acquisition can verify the entire publisher checksum while retaining
only a predeclared subset of audio clips and labels. It must keep the full
compressed-byte count/checksum, safe member checks, waveform clocks, source
attribution and untouched source groups; an interrupted/unverified stream cannot
be fitted. Until that exists, continue using the verified current caches.

The metadata/license assessment is archived in
`training/results/musicnet-original-assessment-v1`. No user uploads or scores
were processed during this assessment.
