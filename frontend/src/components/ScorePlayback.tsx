import { useEffect, useRef, useState } from "react";
import { Headphones, Loader2, Pause, Play, RotateCcw, Volume2 } from "lucide-react";
import { formatTime, normalizePlayback, ScoreAudio, ScorePlayer } from "../playback";

export default function ScorePlayback({ url, originalUrl, onPositionChange, onScoreLoaded, seekRequest }: {
  url?: string | null; originalUrl?: string | null;
  onPositionChange: (position: number) => void; onScoreLoaded: (score: ScoreAudio | null) => void;
  seekRequest: { time: number } | null;
}) {
  const [score, setScore] = useState<ScoreAudio | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [rate, setRate] = useState(1);
  const [volume, setVolume] = useState(0.7);
  const [retry, setRetry] = useState(0);
  const player = useRef<ScorePlayer | null>(null);
  const originalAudio = useRef<HTMLAudioElement | null>(null);
  const operation = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    setScore(null); onScoreLoaded(null); setError(null); setPosition(0); setIsPlaying(false);
    if (url) fetch(url, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error("Score audio is unavailable. Download the MIDI to listen in your music app.");
      const result = normalizePlayback(await response.json());
      if (!controller.signal.aborted) { setScore(result); onScoreLoaded(result); }
    }).catch(err => {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : "Score audio could not be loaded.");
    });
    return () => { operation.current++; controller.abort(); player.current?.dispose(); player.current = null; };
  }, [url, retry, onScoreLoaded]);

  useEffect(() => { onPositionChange(position); }, [position, onPositionChange]);

  useEffect(() => {
    if (!isPlaying) return;
    let frame = 0;
    function update() {
      const current = player.current?.position ?? 0;
      setPosition(Math.floor(current * 20) / 20);
      if (score && current >= score.duration) { player.current?.pause(); setIsPlaying(false); }
      else frame = requestAnimationFrame(update);
    }
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [isPlaying, score]);

  async function play(at = position, speed = rate) {
    if (!score) return;
    const request = ++operation.current;
    try {
      originalAudio.current?.pause();
      if (!player.current) player.current = new ScorePlayer(score);
      const currentPlayer = player.current;
      currentPlayer.setVolume(volume);
      const started = await currentPlayer.play(at, speed);
      if (request !== operation.current || currentPlayer !== player.current) return;
      setError(null); setIsPlaying(started);
    } catch {
      if (request === operation.current) {
        setIsPlaying(false);
        setError("Audio could not start. Check this browser's sound permissions and try again.");
      }
    }
  }

  function pause() {
    operation.current++;
    setPosition(player.current?.pause() ?? position);
    setIsPlaying(false);
  }

  function seek(next: number) {
    if (score && next >= score.duration) {
      pause(); setPosition(score.duration); return;
    }
    setPosition(next);
    if (isPlaying) void play(next);
  }

  useEffect(() => {
    if (seekRequest && score) seek(seekRequest.time);
  }, [seekRequest]);

  const available = !!score?.notes.length;
  return <div className={`playback-panel ${!url ? "playback-unavailable" : ""}`} aria-label="Score playback">
    <div className="playback-label"><span><Headphones size={16} aria-hidden="true" /> Listen to your score</span></div>
    {error && <div className="playback-error" role="alert">{error} {!score && <button className="text-button" onClick={() => setRetry(value => value + 1)}>Retry</button>}</div>}
    <div className="transport">
      <button className="icon-button restart-button" aria-label="Restart playback" disabled={!available} onClick={() => seek(0)}><RotateCcw size={17} aria-hidden="true" /></button>
      <button className="play-button" aria-label={isPlaying ? "Pause score" : "Play score"} disabled={!available} onClick={() => isPlaying ? pause() : void play(position >= (score?.duration ?? 0) ? 0 : position)}>
        {url && !score && !error ? <Loader2 size={20} className="spin" aria-hidden="true" /> : isPlaying ? <Pause size={21} fill="currentColor" aria-hidden="true" /> : <Play size={21} fill="currentColor" aria-hidden="true" />}
      </button>
      <div className="seek-control"><input aria-label="Playback position" type="range" min={0} max={score?.duration || 1} step={0.01} value={position} disabled={!available} onChange={event => seek(Number(event.target.value))} /><div className="time-labels"><span>{formatTime(position)}</span><span>{score ? formatTime(score.duration) : "—:—"}</span></div></div>
      <label className="speed-control"><span className="sr-only">Playback speed</span><select value={rate} disabled={!available} onChange={event => { const next = Number(event.target.value); setRate(next); if (isPlaying) void play(player.current?.position ?? position, next); }}><option value={0.5}>0.5×</option><option value={0.75}>0.75×</option><option value={1}>1×</option><option value={1.25}>1.25×</option><option value={1.5}>1.5×</option><option value={2}>2×</option></select></label>
      <label className="volume-control"><Volume2 size={17} aria-hidden="true" /><span className="sr-only">Playback volume</span><input type="range" min={0} max={1} step={0.01} value={volume} onChange={event => { const next = Number(event.target.value); setVolume(next); player.current?.setVolume(next); }} /></label>
    </div>
    {!url && <p className="player-hint">Playback becomes available when your score is ready.</p>}
    {score && !score.notes.length && <p className="player-hint">This score contains no playable notes.</p>}
    {originalUrl && <details className="original-audio"><summary>Compare with your original recording</summary><audio ref={originalAudio} controls preload="metadata" src={originalUrl} onPlay={pause}>Your browser does not support audio playback.</audio><p>Original recording and synthesized score may have different timing.</p></details>}
  </div>;
}
