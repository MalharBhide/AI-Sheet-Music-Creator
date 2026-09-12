export type NoteEvent = { pitch: number; start: number; end: number; velocity: number };
export type ScoreAudio = { duration: number; tempo_bpm: number; notes: NoteEvent[] };

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
  return { notes, duration, tempo_bpm: Number.isFinite(data.tempo_bpm) ? data.tempo_bpm! : 120 };
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
    this.wave = this.context.createPeriodicWave(new Float32Array(9), new Float32Array([0, 1, 0.42, 0.22, 0.12, 0.07, 0.03, 0.02, 0.01]));
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
    const oscillator = this.context.createOscillator();
    const envelope = this.context.createGain();
    oscillator.setPeriodicWave(this.wave);
    oscillator.frequency.value = 440 * 2 ** ((note.pitch - 69) / 12);
    const amplitude = 0.12 * (note.velocity / 127);
    envelope.gain.setValueAtTime(0, start);
    envelope.gain.linearRampToValueAtTime(amplitude, start + 0.008);
    envelope.gain.setTargetAtTime(amplitude * 0.16, start + 0.012, 0.6);
    envelope.gain.setTargetAtTime(0, end, 0.025);
    oscillator.connect(envelope);
    envelope.connect(this.output);
    this.voices.add(oscillator);
    oscillator.onended = () => { this.voices.delete(oscillator); oscillator.disconnect(); envelope.disconnect(); };
    oscillator.start(start);
    oscillator.stop(end + 0.15);
  }
}
