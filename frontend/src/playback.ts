export type NoteEvent = { pitch: number; start: number; end: number; velocity: number };
export type ScorePosition = { time: number; page: number; x: number; y: number; height: number };
export type ScoreAudio = { duration: number; tempo_bpm: number; notes: NoteEvent[]; positions?: ScorePosition[] };

export function scorePositionAt(positions: ScorePosition[], time: number): ScorePosition | null {
  let low = 0, high = positions.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (positions[middle].time <= time) low = middle + 1;
    else high = middle;
  }
  return positions[low - 1] ?? null;
}

export function scorePositionNear(positions: ScorePosition[], page: number, x: number, y: number): ScorePosition | null {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  let best: ScorePosition | null = null, distance = Infinity;
  for (const point of positions) {
    if (point.page !== page) continue;
    // Choose the nearest staff system first, then its nearest musical onset.
    const vertical = Math.max(point.y - y, y - point.y - point.height, 0);
    const candidate = vertical * 100 + Math.abs(point.x - x);
    if (candidate < distance) { distance = candidate; best = point; }
  }
  return best;
}

export function normalizePlayback(value: unknown): ScoreAudio {
  if (!value || typeof value !== "object") throw new Error("The score audio could not be read.");
  const data = value as Partial<ScoreAudio>;
  if (!Array.isArray(data.notes) || !Number.isFinite(data.duration) || data.duration! < 0) {
    throw new Error("The score audio could not be read.");
  }
  const notes = data.notes.filter(note => note && Number.isInteger(note.pitch) && note.pitch >= 0 && note.pitch <= 127
    && Number.isFinite(note.start) && Number.isFinite(note.end) && note.start >= 0 && note.end > note.start)
    .map(note => ({ ...note, velocity: Number.isFinite(note.velocity) ? Math.max(1, Math.min(127, note.velocity)) : 80 }))
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  const duration = notes.reduce((end, note) => Math.max(end, note.end), data.duration!);
  const positions = Array.isArray(data.positions) && data.positions.every((p, i, all) => p
    && Number.isFinite(p.time) && p.time >= 0 && p.time <= duration + 0.1
    && Number.isInteger(p.page) && p.page >= 0
    && Number.isFinite(p.x) && p.x >= 0 && p.x <= 1
    && Number.isFinite(p.y) && p.y >= 0 && p.y < 1
    && Number.isFinite(p.height) && p.height > 0 && p.y + p.height <= 1.01
    && (!i || p.time >= all[i - 1].time)) ? data.positions : [];
  return { notes, duration, tempo_bpm: Number.isFinite(data.tempo_bpm) ? data.tempo_bpm! : 120, positions };
}

export function firstNoteAt(notes: NoteEvent[], position: number): number {
  let low = 0;
  let high = notes.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (notes[mid].start < position) low = mid + 1;
    else high = mid;
  }
  return low;
}

export function noteSchedule(note: NoteEvent, position: number, rate: number, audioStart: number) {
  if (!(rate > 0) || note.end <= position) return null;
  return { start: audioStart + Math.max(0, note.start - position) / rate, end: audioStart + (note.end - position) / rate };
}

export function playbackPosition(startPosition: number, startedAt: number, currentTime: number, rate: number, duration: number) {
  return Math.max(0, Math.min(duration, startPosition + Math.max(0, currentTime - startedAt) * rate));
}

export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total % 3600 / 60);
  const remainder = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${remainder}` : `${minutes}:${remainder}`;
}

const PIANO_ATTACK = 0.006;
const HAMMER_DECAY = 0.16;

export function pianoDecay(pitch: number): number {
  // Bass strings ring longer than treble strings. Both eventually fall silent,
  // even while the key remains down; a piano has no constant sustain level.
  return Math.max(0.4, Math.min(3.2, 1.3 * 2 ** ((60 - pitch) / 30)));
}

function pianoRelease(pitch: number): number {
  return Math.max(0.025, Math.min(0.07, 0.045 * 2 ** ((60 - pitch) / 48)));
}

function componentEnvelope(age: number, decay: number): number {
  if (age < 0) return 0;
  return age < PIANO_ATTACK ? age / PIANO_ATTACK : Math.exp(-(age - PIANO_ATTACK) / decay);
}

/** The acoustic envelope used by the player, including key release and seeks. */
export function pianoEnvelopeAt(pitch: number, velocity: number, age: number, heldFor = Infinity): number {
  const heldAge = Math.min(age, heldFor);
  const amplitude = 0.15 * (Math.max(1, Math.min(127, velocity)) / 127) ** 1.35;
  const level = amplitude * (0.68 * componentEnvelope(heldAge, HAMMER_DECAY)
    + 0.32 * componentEnvelope(heldAge, pianoDecay(pitch)));
  return level * Math.exp(-Math.max(0, age - heldFor) / pianoRelease(pitch));
}

/** A bounded look-ahead scheduler: a long score never creates all its voices at once. */
export class ScorePlayer {
  private context: AudioContext;
  private output: GainNode;
  private wave: PeriodicWave;
  private voices = new Set<OscillatorNode>();
  private timer?: number;
  private startPosition = 0;
  private startedAt = 0;
  private rate = 1;
  private index = 0;
  private running = false;
  private disposed = false;
  private generation = 0;
  private score: ScoreAudio;

  constructor(score: ScoreAudio) {
    this.score = score;
    this.context = new AudioContext();
    this.output = this.context.createGain();
    const compressor = this.context.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.ratio.value = 5;
    this.output.gain.value = 0.7;
    this.output.connect(compressor);
    compressor.connect(this.context.destination);
    this.wave = this.context.createPeriodicWave(new Float32Array(13), new Float32Array([0, 1, 0.52, 0.32, 0.18, 0.12, 0.075, 0.045, 0.025, 0.018, 0.012, 0.008, 0.005]));
  }

  get position() {
    return this.running ? playbackPosition(this.startPosition, this.startedAt, this.context.currentTime, this.rate, this.score.duration) : this.startPosition;
  }

  setVolume(volume: number) {
    this.output.gain.setTargetAtTime(Math.max(0, Math.min(1, volume)), this.context.currentTime, 0.015);
  }

  async play(position: number, rate: number) {
    this.pause();
    const generation = this.generation;
    if (this.disposed) return false;
    await this.context.resume();
    // A pause, seek, source change, or original-audio playback may occur while
    // the browser is granting/resuming its audio context. Only the newest start
    // request may create voices.
    if (this.disposed || generation !== this.generation) return false;
    this.startPosition = Math.max(0, Math.min(this.score.duration, position));
    this.rate = rate;
    this.startedAt = this.context.currentTime + 0.04;
    this.running = true;
    this.index = firstNoteAt(this.score.notes, this.startPosition);
    // Resume notes which were already held at the seek point.
    for (let i = 0; i < this.index; i++) {
      if (this.score.notes[i].end > this.startPosition) this.schedule(this.score.notes[i]);
    }
    this.tick();
    if (this.running) this.timer = window.setInterval(() => this.tick(), 40);
    return this.running;
  }

  pause() {
    this.generation++;
    this.startPosition = this.position;
    this.running = false;
    window.clearInterval(this.timer);
    for (const voice of this.voices) {
      try { voice.stop(); } catch { /* Already ended. */ }
      voice.disconnect();
    }
    this.voices.clear();
    return this.startPosition;
  }

  dispose() {
    this.disposed = true;
    this.pause();
    void this.context.close();
  }

  private tick() {
    if (this.position >= this.score.duration) {
      this.startPosition = this.score.duration;
      this.running = false;
      window.clearInterval(this.timer);
      return;
    }
    const horizon = this.position + 0.25 * this.rate;
    while (this.index < this.score.notes.length && this.score.notes[this.index].start <= horizon) {
      this.schedule(this.score.notes[this.index++]);
    }
  }

  private schedule(note: NoteEvent) {
    const timing = noteSchedule(note, this.startPosition, this.rate, this.startedAt);
    if (!timing || timing.end <= this.context.currentTime) return;
    const start = Math.max(this.context.currentTime, timing.start);
    const end = Math.max(start + 0.012, timing.end);
    // A seek or late scheduler tick must resume the string's existing decay,
    // rather than strike every held note again at full volume.
    const onset = this.startedAt + (note.start - this.startPosition) / this.rate;
    const age = Math.max(0, start - onset);
    const oscillator = this.context.createOscillator();
    const hammer = this.context.createGain();
    const body = this.context.createGain();
    const tone = this.context.createBiquadFilter();
    const fade = this.context.createGain();
    oscillator.setPeriodicWave(this.wave);
    const frequency = 440 * 2 ** ((note.pitch - 69) / 12);
    oscillator.frequency.value = frequency;
    const amplitude = 0.15 * (note.velocity / 127) ** 1.35;
    for (const [envelope, weight, decay] of [[hammer, 0.68, HAMMER_DECAY], [body, 0.32, pianoDecay(note.pitch)]] as const) {
      envelope.gain.setValueAtTime(amplitude * weight * componentEnvelope(age, decay), start);
      const attackEnd = Math.min(end, start + Math.max(0, PIANO_ATTACK - age));
      if (attackEnd > start) {
        envelope.gain.linearRampToValueAtTime(amplitude * weight * componentEnvelope(age + attackEnd - start, decay), attackEnd);
      }
      if (attackEnd < end) envelope.gain.setTargetAtTime(0, attackEnd, decay);
      envelope.gain.setTargetAtTime(0, end, pianoRelease(note.pitch));
      oscillator.connect(envelope);
      envelope.connect(tone);
    }
    // A hammer strike starts bright, then its upper harmonics die away. This
    // filter also resumes its softened state when seeking into a held note.
    tone.type = "lowpass";
    tone.Q.value = 0.5;
    const ceiling = this.context.sampleRate * 0.45;
    const mellow = Math.min(ceiling, frequency * 2.2 + 650);
    const bright = Math.min(ceiling, frequency * (5 + note.velocity / 18) + 1800);
    tone.frequency.setValueAtTime(mellow + (bright - mellow) * Math.exp(-age / 0.24), start);
    tone.frequency.setTargetAtTime(mellow, start, 0.24);
    // Fade resumed strings in briefly to avoid a discontinuity/click, without
    // adding a new hammer attack to a note that began before the seek point.
    fade.gain.setValueAtTime(age > 0 ? 0 : 1, start);
    if (age > 0) fade.gain.linearRampToValueAtTime(1, start + 0.008);
    tone.connect(fade);
    fade.connect(this.output);
    this.voices.add(oscillator);
    oscillator.onended = () => {
      this.voices.delete(oscillator);
      oscillator.disconnect(); hammer.disconnect(); body.disconnect(); tone.disconnect(); fade.disconnect();
    };
    oscillator.start(start);
    // A pathological long MIDI note must not leave an inaudible oscillator
    // running for hours after the modeled string has finished ringing.
    const naturalEnd = start + Math.max(0.02, PIANO_ATTACK + 14 * pianoDecay(note.pitch) - age);
    oscillator.stop(Math.min(end + pianoRelease(note.pitch) * 8, naturalEnd));
  }
}
