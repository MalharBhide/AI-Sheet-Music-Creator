import { useEffect, useRef, useState } from "react";
import { ArrowRight, AudioLines, Check, ChevronRight, Headphones, Mic2, Music2, Piano, Plus, Radio, X } from "lucide-react";

import { ApiError, apiUrl, fetchExample, fetchHealth, fetchJob, Health, Job, ScoreOptions, uploadAudio } from "./api";
import DownloadPanel from "./components/DownloadPanel";
import JobStatus from "./components/JobStatus";
import ScorePlayback from "./components/ScorePlayback";
import ScorePreview from "./components/ScorePreview";
import UploadDropzone from "./components/UploadDropzone";
import { validateAudioUpload } from "./uploadValidation";
import { ScoreAudio } from "./playback";

const defaultOptions: ScoreOptions = { transcription_mode: "full_mix", detail: "balanced", time_signature: "4/4", grid: "sixteenth" };
const modes = [
  { id: "full_mix", label: "Full song", icon: AudioLines, description: "Main tune and accompaniment, arranged for piano." },
  { id: "piano", label: "Solo piano", icon: Piano, description: "For recordings of piano on its own." },
  { id: "melody", label: "Melody", icon: Mic2, description: "For a voice or instrument playing one tune." }
] as const;
const historyKey = "piano-scribe.recent.v1";
type RecentJob = Pick<Job, "job_id" | "original_filename" | "created_at" | "status">;

function initialJobId(): string | null {
  const id = new URLSearchParams(window.location.search).get("job");
  return id && /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(id) ? id : null;
}
function readHistory(): RecentJob[] {
  try {
    const stored = JSON.parse(localStorage.getItem(historyKey) || "[]");
    return Array.isArray(stored) ? stored.filter(item => typeof item?.job_id === "string" && typeof item?.original_filename === "string" && typeof item?.created_at === "string").slice(0, 8) : [];
  } catch { return []; }
}
function updateUrl(jobId: string | null) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState(null, "", url);
}
function pieceName(filename?: string) { return filename?.replace(/\.[^.]+$/, "") || "Untitled recording"; }

export default function App() {
  const [jobId, setJobId] = useState<string | null>(initialJobId);
  const [job, setJob] = useState<Job | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [options, setOptions] = useState<ScoreOptions>(defaultOptions);
  const [isUploading, setIsUploading] = useState(false);
  const [loadingExample, setLoadingExample] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [healthRetry, setHealthRetry] = useState(0);
  const [pollRetry, setPollRetry] = useState(0);
  const [recent, setRecent] = useState<RecentJob[]>(readHistory);
  const [guideOpen, setGuideOpen] = useState(false);
  const [playbackScore, setPlaybackScore] = useState<ScoreAudio | null>(null);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [seekRequest, setSeekRequest] = useState<{ time: number } | null>(null);
  useEffect(() => { setPlaybackScore(null); setPlaybackPosition(0); setSeekRequest(null); }, [jobId]);
  const submitting = useRef(false);
  const guideRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setConnectionError(null);
    fetchHealth(controller.signal).then(result => {
      if (!controller.signal.aborted) {
        setHealth(result);
        if (!result.ready) setConnectionError("The transcription service is not ready yet. Please reconnect in a moment.");
      }
    }).catch(() => { if (!controller.signal.aborted) setConnectionError("We could not reach the transcription service. Your saved scores will still be here when it reconnects."); });
    return () => controller.abort();
  }, [healthRetry]);

  useEffect(() => {
    if (!file) { setOriginalUrl(null); return; }
    const url = URL.createObjectURL(file);
    setOriginalUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (guideOpen) guideRef.current?.showModal();
    else guideRef.current?.close();
  }, [guideOpen]);

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    let timer: number | undefined;
    let failures = 0;
    setError(null);
    async function poll() {
      try {
        const next = await fetchJob(jobId!, controller.signal);
        if (controller.signal.aborted) return;
        setJob(next); setError(null); failures = 0;
        setRecent(current => {
          const updated = [{ job_id: next.job_id, original_filename: next.original_filename, created_at: next.created_at, status: next.status }, ...current.filter(item => item.job_id !== next.job_id)].slice(0, 8);
          try { localStorage.setItem(historyKey, JSON.stringify(updated)); } catch { /* Browsing without local storage is supported. */ }
          return updated;
        });
        if (next.status !== "done" && next.status !== "failed") timer = window.setTimeout(poll, 1500);
      } catch (err) {
        if (controller.signal.aborted) return;
        failures += 1;
        const missing = err instanceof ApiError && (err.status === 404 || err.status === 422);
        setError(missing ? "This score is no longer available. Upload the recording again to create a new score." : "Connection interrupted. Your recording may still be processing; we are reconnecting.");
        if (!missing) timer = window.setTimeout(poll, Math.min(1000 * 2 ** Math.min(failures, 4), 15000));
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [jobId, pollRetry]);

  function selectFile(next: File) {
    const validation = validateAudioUpload(next, health?.limits.max_upload_mb ?? 0);
    setError(validation);
    if (!validation) setFile(next);
  }

  async function handleUpload(recording: File, settings = options) {
    if (submitting.current || jobId || !health?.ready) return;
    const validation = validateAudioUpload(recording, health.limits.max_upload_mb);
    if (validation) { setError(validation); return; }
    submitting.current = true; setError(null); setIsUploading(true); setUploadProgress(0);
    try {
      const created = await uploadAudio(recording, setUploadProgress, settings);
      setJob(created); setJobId(created.job_id); updateUrl(created.job_id);
    } catch (err) { setError(err instanceof ApiError ? err.message : "Upload failed. Check your connection and try again."); }
    finally { submitting.current = false; setIsUploading(false); }
  }

  async function tryExample() {
    setLoadingExample(true); setError(null);
    try {
      const example = await fetchExample();
      const settings: ScoreOptions = { ...defaultOptions, transcription_mode: "piano", tempo_bpm: 96 };
      setFile(example); setOptions(settings);
      await handleUpload(example, settings);
    } catch (err) { setError(err instanceof Error ? err.message : "The piano example could not be loaded."); }
    finally { setLoadingExample(false); }
  }

  function reset() { setJobId(null); setJob(null); setFile(null); setError(null); updateUrl(null); }
  function openRecent(id: string) {
    // Re-selecting the current score must not clear the result: jobId would stay
    // unchanged, so its completed polling effect would never load it again.
    if (id === jobId) return;
    setJob(null); setFile(null); setJobId(id); setError(null); updateUrl(id);
  }
  const previewUrls = (job?.svg_pages ?? []).map(path => apiUrl(path)!);
  if (!previewUrls.length && job?.download_urls.svg) previewUrls.push(apiUrl(job.download_urls.svg)!);
  const busy = isUploading || loadingExample;
  const done = job?.status === "done";
  const availableDownloads = !!job && Object.values(job.download_urls).some(Boolean);
  const recordingLimits = health ? [health.limits.max_upload_mb > 0 ? `Up to ${health.limits.max_upload_mb} MB` : null, health.limits.max_audio_seconds > 0 ? `${health.limits.max_audio_seconds} seconds maximum` : null].filter(Boolean).join(" · ") : "";

  return <div className={`app-shell ${jobId ? "has-score" : "new-score"}`}>
    <header className="topbar">
      <a className="brand" href="/" onClick={event => { event.preventDefault(); if (!busy) reset(); }} aria-label="Piano Scribe home"><span className="brand-mark"><Piano size={24} aria-hidden="true" /></span><span>Piano<span className="brand-serif">Scribe</span></span></a>
      <div className="header-actions"><button className="help-button" onClick={() => setGuideOpen(true)}>Help</button>{jobId && <button className="button button-outline" onClick={reset} disabled={busy}><Plus size={17} /> New score</button>}</div>
    </header>
    <main className="workspace" id="studio">
      {!jobId && <section className="page-intro"><p className="eyebrow">YOUR MUSIC. YOUR PIANO.</p><h1>Turn a song into<br /><em>something you can play.</em></h1><p className="intro-description">Upload a recording. Get piano sheet music. Play along.</p></section>}
      <div className="studio-grid">
        <aside className="studio-sidebar">
          <section className="setup-card" aria-label={jobId ? "Score downloads" : "Transcription setup"}>
            {jobId ? <>
              <h2 className="sidebar-title">Your score</h2>
              <p className="piece-label">{job ? pieceName(job.original_filename) : "Loading…"}</p>
              {done && <span className="ready-label"><Check size={14} /> Ready to play</span>}
            </> : <>
              <h2 className="sidebar-title">Start with a recording</h2>
              <UploadDropzone file={file} onUpload={selectFile} disabled={busy} />
              {recordingLimits && <p className="recording-limit">{recordingLimits}</p>}
              <fieldset className="mode-fieldset" disabled={busy}><legend>Recording type</legend><div className="mode-options">{modes.map(mode => { const Icon = mode.icon; return <label className={`mode-option ${options.transcription_mode === mode.id ? "selected" : ""}`} key={mode.id}><input type="radio" name="transcription-mode" value={mode.id} checked={options.transcription_mode === mode.id} onChange={() => setOptions(current => ({ ...current, transcription_mode: mode.id }))} /><Icon size={19} aria-hidden="true" /><span>{mode.label}</span></label>; })}</div><p className="mode-description">{modes.find(mode => mode.id === options.transcription_mode)?.description}</p></fieldset>
              <details className="advanced-settings"><summary>More options <Plus size={15} /></summary>
                <label className="detail-setting">Note detail<select value={options.detail} disabled={busy} onChange={event => setOptions(current => ({ ...current, detail: event.target.value as ScoreOptions["detail"] }))}><option value="balanced">Balanced · Easier to read</option><option value="detailed">Detailed · More notes</option></select></label>
                <div className="settings-grid"><label>Tempo (BPM)<input type="number" min={30} max={240} placeholder="Automatic" value={options.tempo_bpm ?? ""} disabled={busy} onChange={event => setOptions(current => ({ ...current, tempo_bpm: event.target.value ? Number(event.target.value) : undefined }))} /></label><label>Time signature<select value={options.time_signature} disabled={busy} onChange={event => setOptions(current => ({ ...current, time_signature: event.target.value as ScoreOptions["time_signature"] }))}><option value="4/4">4/4</option><option value="3/4">3/4</option><option value="6/8">6/8</option></select></label><label className="full-width">Rhythm<select value={options.grid} disabled={busy} onChange={event => setOptions(current => ({ ...current, grid: event.target.value as ScoreOptions["grid"] }))}><option value="sixteenth">Precise · Sixteenth notes</option><option value="eighth">Simple · Eighth notes</option></select></label></div>
              </details>
              <button className="button button-primary create-button" disabled={!file || busy || !health?.ready || !!connectionError || (options.tempo_bpm !== undefined && (options.tempo_bpm < 30 || options.tempo_bpm > 240))} onClick={() => file && void handleUpload(file)}>{isUploading ? "Uploading…" : "Create piano score"}<ArrowRight size={18} /></button>
              <button className="example-button" disabled={busy || !health?.ready || !!connectionError} onClick={() => void tryExample()}>{loadingExample ? "Loading example…" : "Try an example instead"}</button>
            </>}
            {connectionError && <div className="inline-error" role="alert"><p>{connectionError}</p><button className="text-button" onClick={() => setHealthRetry(value => value + 1)}>Reconnect</button></div>}
            {(!done || error) && (job || isUploading || error) && <JobStatus job={job} isUploading={isUploading} uploadProgress={uploadProgress} error={error} />}
            {error && jobId && <button className="text-button" onClick={() => setPollRetry(value => value + 1)}>Retry</button>}
            {job?.status === "failed" && <button className="button button-outline" onClick={reset}>Try another recording</button>}
            {availableDownloads && <DownloadPanel job={job} />}
            {done && <p className="retention-note">Files are available for {health?.limits.retention_hours ?? 24} hours.</p>}
            {job?.analysis?.warnings?.length ? <details className="score-notes"><summary>About this transcription</summary><p>This is an editable first draft. Listen and review the notes before practicing.</p><ul>{job.analysis.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details> : null}
          </section>
          {recent.length > 0 && <details className="recent-card"><summary>Recent scores <span>{recent.length}</span></summary><div className="recent-list">{recent.map(item => <button className={`recent-item ${item.job_id === jobId ? "current" : ""}`} key={item.job_id} disabled={busy} onClick={() => openRecent(item.job_id)}><Music2 size={17} /><span><strong>{pieceName(item.original_filename)}</strong><small>{new Date(item.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}{item.status === "failed" ? " · Needs attention" : item.status !== "done" ? " · In progress" : ""}</small></span><ChevronRight size={15} /></button>)}</div></details>}
        </aside>
        <section className="score-workspace" aria-label="Sheet music and playback">
          <div className="score-heading"><div><p className="eyebrow">{jobId ? "PIANO SCORE" : "A PLACE FOR YOUR MUSIC"}</p><h2>{job ? pieceName(job.original_filename) : "Your sheet music"}</h2></div>{done && job?.analysis?.tempo_bpm && <span className="score-tempo">♩ {Math.round(job.analysis.tempo_bpm)}</span>}</div>
          {(done || job?.status === "failed") && job?.download_urls.playback && <ScorePlayback key={`playback-${jobId}`} url={apiUrl(job.download_urls.playback)} originalUrl={originalUrl} onPositionChange={setPlaybackPosition} onScoreLoaded={setPlaybackScore} seekRequest={seekRequest} />}
          <ScorePreview key={`preview-${jobId || "new"}`} status={job?.status} previewUrls={previewUrls} pdfUrl={apiUrl(job?.download_urls.pdf)} error={job?.error || (!job ? error : null)} position={playbackPosition} positions={playbackScore?.positions ?? []} onSeek={time => { setPlaybackPosition(time); setSeekRequest({ time }); }} />
        </section>
      </div>
    </main>
    <footer className="site-footer"><span>PianoScribe</span><span>From listening to playing.</span></footer>
    <dialog className="guide-dialog" ref={guideRef} onCancel={() => setGuideOpen(false)} onClick={event => { if (event.target === event.currentTarget) setGuideOpen(false); }}><button className="icon-button dialog-close" aria-label="Close transcription guide" onClick={() => setGuideOpen(false)}><X size={20} /></button><p className="eyebrow">A NOTE BEFORE THE FIRST NOTE</p><h2>A good score starts<br />with a good recording.</h2><p>Piano Scribe creates an editable first draft. Listen to the result and compare it with your recording before relying on the notes.</p><div className="guide-section"><Piano size={23} /><div><h3>Solo piano → transcription</h3><p>A clear solo recording gives the system its best chance to identify the notes you played. Heavy reverb, background noise, and overlapping instruments make this harder.</p></div></div><div className="guide-section"><Radio size={23} /><div><h3>Full song → piano arrangement</h3><p>A full mix contains voices, drums, bass, and other instruments. The result is a piano reduction of detected musical material, not a guaranteed note-for-note piano score.</p></div></div><div className="guide-section"><Headphones size={23} /><div><h3>Listen, then refine</h3><p>Use the piano player to hear the actual notes in your score, slow it down, and compare with the original. Download MusicXML for edits in a notation app or MIDI for your music software.</p></div></div><p className="guide-footnote">Long recordings are processed in sections. Processing time depends on recording length, complexity, and the available hardware. Files are retained for {health?.limits.retention_hours ?? 24} hours; recent history is stored only in this browser.</p><button className="button button-primary" onClick={() => setGuideOpen(false)}>Back to the studio <ArrowRight size={16} /></button></dialog>
  </div>;
}
