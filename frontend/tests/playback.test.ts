import assert from "node:assert/strict";
import test from "node:test";
import { firstNoteAt, formatTime, normalizePlayback, noteSchedule, pianoDecay, pianoEnvelopeAt, playbackPosition, ScorePlayer } from "../src/playback.ts";

const note = { pitch: 60, start: 2, end: 6, velocity: 90 };

test("seeking into a held note plays only its remaining duration", () => {
  assert.deepEqual(noteSchedule(note, 4, 1, 100), { start: 100, end: 102 });
  assert.equal(noteSchedule(note, 6, 1, 100), null);
});
test("playback speed changes timing without changing pitch", () => {
  assert.deepEqual(noteSchedule(note, 0, 0.5, 10), { start: 14, end: 22 });
  assert.deepEqual(noteSchedule(note, 0, 2, 10), { start: 11, end: 13 });
  assert.equal(noteSchedule(note, 0, 0, 0), null);
});
test("position is clamped at the start and end of the piece", () => {
  assert.equal(playbackPosition(10, 100, 103, 0.5, 60), 11.5);
  assert.equal(playbackPosition(0, 100.04, 100, 1, 60), 0);
  assert.equal(playbackPosition(59, 0, 3, 2, 60), 60);
});
test("seek locates chords together and handles a long recording", () => {
  const notes = Array.from({ length: 100_000 }, (_, i) => ({ ...note, start: Math.floor(i / 3), end: Math.floor(i / 3) + 1 }));
  assert.equal(firstNoteAt(notes, 12345), 37035);
  assert.equal(firstNoteAt(notes, 12345.5), 37038);
  assert.equal(firstNoteAt(notes, 40000), 100000);
  assert.equal(firstNoteAt([], 10), 0);
});
test("invalid notes are excluded and duration includes the final note release", () => {
  const result = normalizePlayback({ duration: 1, tempo_bpm: 100, notes: [{ ...note, pitch: 300 }, { ...note, end: 0 }, { ...note, velocity: NaN }, { ...note, pitch: 64, start: 0 }] });
  assert.equal(result.notes.length, 2);
  assert.equal(result.notes[0].pitch, 64);
  assert.equal(result.notes[1].velocity, 80);
  assert.equal(result.duration, 6);
  assert.throws(() => normalizePlayback({ duration: Infinity, notes: [] }));
});
test("time labels remain readable beyond an hour", () => {
  assert.equal(formatTime(7), "0:07");
  assert.equal(formatTime(3665), "1:01:05");
  assert.equal(formatTime(NaN), "0:00");
});

test("a piano string decays while held instead of sustaining an organ-like tone", () => {
  const peak = pianoEnvelopeAt(60, 100, 0.006);
  assert.ok(pianoEnvelopeAt(60, 100, 0.2) < peak * 0.6, "the hammer attack dies away quickly");
  assert.ok(pianoEnvelopeAt(60, 100, 2) > peak * 0.01, "a genuine long note still rings");
  assert.ok(pianoEnvelopeAt(60, 100, 10) < peak * 0.001, "there is no fixed sustain floor");
  assert.ok(pianoDecay(36) > pianoDecay(84), "bass strings ring longer than treble strings");
  assert.ok(pianoEnvelopeAt(60, 110, 0.1) > pianoEnvelopeAt(60, 40, 0.1), "velocity controls dynamics");
});

test("releasing a key damps its tail without extending short notes into the next chord", () => {
  const releaseLevel = pianoEnvelopeAt(48, 100, 0.25);
  assert.equal(pianoEnvelopeAt(48, 100, 0.25, 0.25), releaseLevel);
  assert.ok(pianoEnvelopeAt(48, 100, 0.55, 0.25) < releaseLevel * 0.005);
  assert.ok(pianoEnvelopeAt(48, 100, 0.55) > releaseLevel * 0.2, "the same key held down retains a natural decay");
});

test("the audio scheduler bounds voices, resumes held notes, and cancels stale starts", async () => {
  const previousAudioContext = globalThis.AudioContext;
  const previousWindow = globalThis.window;
  const oscillators: { started?: number; stopped?: number; frequency: { value: number } }[] = [];
  let tick: (() => void) | undefined;
  let context: FakeContext;
  let deferResume = false;
  const resumes: (() => void)[] = [];
  type Automation = { kind: string; value: number; time: number; constant?: number };
  const gains: { events: Automation[] }[] = [];
  const parameter = () => ({ value: 0, events: [] as Automation[],
    setValueAtTime(value: number, time: number) { this.events.push({ kind: "set", value, time }); },
    linearRampToValueAtTime(value: number, time: number) { this.events.push({ kind: "ramp", value, time }); },
    setTargetAtTime(value: number, time: number, constant: number) { this.events.push({ kind: "target", value, time, constant }); }
  });
  class FakeContext {
    currentTime = 0;
    sampleRate = 48000;
    destination = {};
    closed = false;
    constructor() { context = this; }
    createGain() { const gain = parameter(); gains.push(gain); return { gain, connect() {}, disconnect() {} }; }
    createDynamicsCompressor() { return { threshold: parameter(), ratio: parameter(), connect() {} }; }
    createBiquadFilter() { return { type: "lowpass", Q: parameter(), frequency: parameter(), connect() {}, disconnect() {} }; }
    createPeriodicWave() { return {}; }
    createOscillator() {
      const oscillator = { frequency: parameter(), started: undefined as number | undefined, stopped: undefined as number | undefined, onended: undefined as (() => void) | undefined,
        setPeriodicWave() {}, connect() {}, disconnect() {},
        start(time: number) { this.started = time; },
        stop(time = 0) { this.stopped = time; }
      };
      oscillators.push(oscillator);
      return oscillator;
    }
    resume() { return deferResume ? new Promise<void>(resolve => resumes.push(resolve)) : Promise.resolve(); }
    close() { this.closed = true; return Promise.resolve(); }
  }
  try {
    globalThis.AudioContext = FakeContext as unknown as typeof AudioContext;
    globalThis.window = { setInterval(callback: () => void) { tick = callback; return 1; }, clearInterval() { tick = undefined; } } as unknown as Window & typeof globalThis;
    const player = new ScorePlayer({ duration: 7200, tempo_bpm: 120, notes: [
      { ...note, start: 0, end: 3 }, { ...note, pitch: 64, start: 1, end: 2 }, { ...note, pitch: 67, start: 3600, end: 3601 }
    ] });
    await player.play(0, 1);
    assert.equal(oscillators.length, 1, "an hour-later note is not instantiated");
    assert.ok(gains.slice(1, 3).flatMap(gain => gain.events).filter(event => event.kind === "target").every(event => event.value === 0),
      "both acoustic components decay toward silence during the note and after release");
    context!.currentTime = 0.9;
    tick?.();
    assert.equal(oscillators.length, 2);
    assert.ok(Math.abs(oscillators[1].started! - 1.04) < 0.001);
    assert.ok(Math.abs(player.pause() - 0.86) < 0.001);
    assert.equal(tick, undefined);
    assert.equal(oscillators[0].stopped, 0);
    const beforeSeek = gains.length;
    await player.play(1.5, 0.5);
    assert.equal(oscillators.length, 4, "both held chord notes resume at a seek point");
    assert.equal(oscillators[2].frequency.value, 440 * 2 ** ((60 - 69) / 12));
    const resumedAmplitude = gains[beforeSeek].events[0].value + gains[beforeSeek + 1].events[0].value;
    assert.ok(Math.abs(resumedAmplitude - pianoEnvelopeAt(60, 90, 3)) < 1e-10,
      "a sought note resumes its elapsed acoustic decay at the chosen playback speed");
    assert.ok(gains.slice(beforeSeek, beforeSeek + 2).every(gain => !gain.events.some(event => event.kind === "ramp")),
      "resuming a held key does not schedule a fresh hammer attack");
    assert.equal(await player.play(7200, 1), false, "seeking to the end does not restart or schedule notes");
    assert.equal(player.position, 7200);
    assert.equal(tick, undefined);

    deferResume = true;
    const pendingStart = player.play(0, 1);
    player.pause();
    resumes.shift()!();
    assert.equal(await pendingStart, false, "pause cancels an in-flight context resume");
    assert.equal(oscillators.length, 4);

    const startBeforeSeek = player.play(0, 1);
    assert.equal(player.seek(12), 12);
    resumes.shift()!();
    assert.equal(await startBeforeSeek, false, "a paused seek cancels pending audio permission/resume");
    assert.equal(oscillators.length, 4, "stale playback cannot create voices after seeking");
    assert.equal(player.pause(), 12, "pausing for the original recording preserves the sought position");
    assert.equal(player.seek(9999), 7200, "seek is clamped to the end");
    assert.equal(player.seek(-1), 0, "seek is clamped to the start");

    const olderSeek = player.play(0, 1);
    const newerSeek = player.play(1.5, 0.5);
    resumes[1]();
    assert.equal(await newerSeek, true);
    resumes[0]();
    assert.equal(await olderSeek, false, "the newest seek wins even when resume promises finish out of order");
    assert.equal(player.position, 1.5);
    assert.equal(oscillators.length, 6);
    resumes.length = 0;

    const startBeforeUnmount = player.play(0, 1);
    player.dispose();
    resumes.shift()!();
    assert.equal(await startBeforeUnmount, false, "disposing a source prevents pending playback");
    assert.equal(context!.closed, true);
    assert.equal(tick, undefined);
  } finally { globalThis.AudioContext = previousAudioContext; globalThis.window = previousWindow; }
});
