# Oxford MIDI-test source assessment

The publisher's [Sight to Sound page](https://www.robots.ox.ac.uk/~vgg/research/sighttosound/)
provides eight videos with MIDI captured from a digital piano. The paper describes
one amateur pianist, with MIDI aligned to the recorded audio. Its
[dataset license](https://www.robots.ox.ac.uk/~vgg/research/sighttosound/resources/license_miditest.txt)
is CC BY 4.0 and requests attribution to A. S. Koepke, O. Wiles, Y. Moses and
A. Zisserman, *Sight to sound: An end-to-end approach for visual piano
transcription*, ICASSP 2020.

The 69,276,729-byte original package is verified through complete ZIP CRC checks,
HTTP content length, and SHA256
`0e440b140af7fb5d53d8ec34a17adff26479b2fada537688c36e5386d25a8a1b`.
The server supplies no independent archive checksum; its ETag and Last-Modified
are retained. The bundled and separately published licenses match exactly.
Acquisition rejects unsafe paths, links, encrypted members, corrupt data and
oversized downloads. Six tests cover these checks.

All eight pairs are reserved together as one pianist group. No audio predictions
or MIDI-label interpretation have been performed. This source is not part of the
already declared V33 MusicNet first-pass protocol and cannot be added after seeing
that candidate's results. It is available for a future separately frozen study.
Its timing alignment and release/pedal semantics still need verification before
using it to label training examples. No upstream pretraining independence is
claimed.

The much larger PianoYT source on the same page uses automatically predicted
MIDI. It is not accepted as reliable note truth. Two other real-piano sources
were rechecked but excluded: [MAPS](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/)
and [Saarland Music Data](https://audiolabs-erlangen.com/resources/MIR/SMD/midi)
carry non-commercial licenses and are outside this potentially commercial
project's training scope.

The V33 local study is queued behind the exact original MusicNet acquisition
process. After the verified source is complete it observes only its fitting
partitions, fits the predeclared profiles, and evaluates only a selected frozen
winner. A first-pass run requires the complete consumed regression gate. The
queue never exports or deploys weights and never processes user uploads.
