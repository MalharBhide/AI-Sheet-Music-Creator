import { useEffect, useRef, useState } from "react";
import { ArrowRight, AudioLines, Check, ChevronRight, CircleHelp, Clock3, FileAudio2, Headphones, Mic2, Music2, Piano, Plus, Radio, Sparkles, X } from "lucide-react";

import { ApiError, apiUrl, fetchExample, fetchHealth, fetchJob, Health, Job, ScoreOptions, uploadAudio } from "./api";
import DownloadPanel from "./components/DownloadPanel";
import JobStatus from "./components/JobStatus";
import ScorePlayback from "./components/ScorePlayback";
import ScorePreview from "./components/ScorePreview";
import UploadDropzone from "./components/UploadDropzone";
import { validateAudioUpload } from "./uploadValidation";

const defaultOptions: ScoreOptions = { transcription_mode: "full_mix", detail: "balanced", time_signature: "4/4", grid: "sixteenth" };
const modes = [
  { id: "full_mix", label: "Full song", icon: AudioLines, description: "Create a playable piano arrangement from a song with multiple instruments. Dense mixes and vocals may need more correction." },
  { id: "piano", label: "Solo piano", icon: Piano, description: "Transcribe the notes in a solo piano recording. A clear recording with little reverb gives the most faithful result." },
  { id: "melody", label: "Melody", icon: Mic2, description: "Focus on a single melodic line. Best for a clear solo instrument or an isolated melody." }
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
  function openRecent(id: string) { setJob(null); setFile(null); setJobId(id); setError(null); updateUrl(id); }
  const previewUrls = (job?.svg_pages ?? []).map(path => apiUrl(path)!);
  if (!previewUrls.length && job?.download_urls.svg) previewUrls.push(apiUrl(job.download_urls.svg)!);
  const busy = isUploading || loadingExample;
  const done = job?.status === "done";
  const availableDownloads = !!job && Object.values(job.download_urls).some(Boolean);
  const recordingLimits = health ? [health.limits.max_upload_mb > 0 ? `Up to ${health.limits.max_upload_mb} MB` : null, health.limits.max_audio_seconds > 0 ? `${health.limits.max_audio_seconds} seconds maximum` : null].filter(Boolean).join(" · ") : "";

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" onClick={event => { event.preventDefault(); if (!busy) reset(); }} aria-label="Piano Scribe home"><span className="brand-mark"><Piano size={24} strokeWidth={1.6} aria-hidden="true" /></span><span>piano<span className="brand-serif">scribe</span><span className="brand-dot">.</span></span></a>
      <nav className="header-nav" aria-label="Main navigation"><a className="nav-active" href="#studio">Studio</a><button onClick={() => setGuideOpen(true)}>How it works <ChevronRight size={13} aria-hidden="true" /></button></nav>
      <div className="header-actions"><span className={`service-status ${connectionError ? "offline" : ""}`}><i />{connectionError ? "Service unavailable" : health?.ready ? "Ready when you are" : "Connecting…"}</span><button className="button button-small button-outline" onClick={reset} disabled={busy}><Plus size={15} aria-hidden="true" /> New transcription</button></div>
    </header>

    <main className="workspace" id="studio">
      <section className="page-intro">
        <div><p className="eyebrow"><span /> YOUR MUSIC, IN A NEW FORM</p><h1>{done ? <>Listen. Refine. <em>Make it yours.</em></> : <>From sound <em>to score.</em></>}</h1><p className="intro-description">{done ? "Your first draft is ready. Play it back, check the notes, and take it to the piano." : "Turn a recording into piano sheet music you can read, play, and make your own."}</p></div>
        <button className="intro-guide" onClick={() => setGuideOpen(true)}><span className="guide-icon"><Headphones size={22} strokeWidth={1.5} /></span><span>A little guidance goes a long way.<strong>Get the best from your recording <ArrowRight size={14} /></strong></span></button>
      </section>

      <div className="studio-grid">
        <aside className="studio-sidebar">
          <section className="setup-card" aria-label="Transcription setup">
            <div className="section-title"><span className="section-number">01</span><h2>{jobId ? "Your recording" : "Create a transcription"}</h2></div>
            {jobId ? <div className="recording-card"><span><FileAudio2 size={22} aria-hidden="true" /></span><div><strong>{job ? pieceName(job.original_filename) : "Loading recording…"}</strong><p>{job?.original_filename.split(".").pop()?.toUpperCase() || "AUDIO"} RECORDING</p></div></div> : <>
              <UploadDropzone file={file} onUpload={selectFile} disabled={busy} />
              <p className="recording-limit">{recordingLimits || "Long recordings welcome. Larger files take more time."}</p>
              <fieldset className="mode-fieldset" disabled={busy}><legend>What are we listening to?</legend><div className="mode-options">{modes.map(mode => { const Icon = mode.icon; return <label className={`mode-option ${options.transcription_mode === mode.id ? "selected" : ""}`} key={mode.id}><input type="radio" name="transcription-mode" value={mode.id} checked={options.transcription_mode === mode.id} onChange={() => setOptions(current => ({ ...current, transcription_mode: mode.id }))} /><Icon size={19} strokeWidth={1.7} aria-hidden="true" /><span>{mode.label}</span>{options.transcription_mode === mode.id && <Check className="mode-check" size={11} aria-hidden="true" />}</label>; })}</div><p className="mode-description">{modes.find(mode => mode.id === options.transcription_mode)?.description}</p></fieldset>
              <div className="detail-setting"><span className="field-label">Score detail</span><div className="segmented-control" role="group" aria-label="Score detail"><button disabled={busy} aria-pressed={options.detail === "balanced"} className={options.detail === "balanced" ? "selected" : ""} onClick={() => setOptions(current => ({ ...current, detail: "balanced" }))}>Balanced</button><button disabled={busy} aria-pressed={options.detail === "detailed"} className={options.detail === "detailed" ? "selected" : ""} onClick={() => setOptions(current => ({ ...current, detail: "detailed" }))}>More detail</button></div><p className="field-help">{options.detail === "balanced" ? "A cleaner, more readable starting point." : "Keep more detected notes, including quieter details."}</p></div>
              <details className="advanced-settings"><summary>Tempo & rhythm <Plus size={14} aria-hidden="true" /></summary><div className="settings-grid"><label>Tempo <span>(BPM)</span><input type="number" min={30} max={240} placeholder="Auto-detect" value={options.tempo_bpm ?? ""} disabled={busy} onChange={event => setOptions(current => ({ ...current, tempo_bpm: event.target.value ? Number(event.target.value) : undefined }))} /></label><label>Time signature<select value={options.time_signature} disabled={busy} onChange={event => setOptions(current => ({ ...current, time_signature: event.target.value as ScoreOptions["time_signature"] }))}><option value="4/4">4/4 · Common time</option><option value="3/4">3/4 · Waltz time</option><option value="6/8">6/8 · Compound time</option></select></label><label className="full-width">Smallest rhythmic value<select value={options.grid} disabled={busy} onChange={event => setOptions(current => ({ ...current, grid: event.target.value as ScoreOptions["grid"] }))}><option value="sixteenth">Sixteenth notes · More precise</option><option value="eighth">Eighth notes · Simpler rhythms</option></select></label></div><p className="field-help">Time signature is your choice; it is not automatically detected.</p></details>
              <button className="button button-primary create-button" disabled={!file || busy || !health?.ready || !!connectionError || (options.tempo_bpm !== undefined && (options.tempo_bpm < 30 || options.tempo_bpm > 240))} onClick={() => file && void handleUpload(file)}><Sparkles size={17} aria-hidden="true" />{isUploading ? "Uploading recording…" : "Create sheet music"}<ArrowRight size={17} aria-hidden="true" /></button>
              <div className="example-action"><span>Just exploring?</span><button disabled={busy || !health?.ready || !!connectionError} onClick={() => void tryExample()}>{loadingExample ? "Loading example…" : "Try a piano example"} <ArrowRight size={12} aria-hidden="true" /></button></div>
            </>}
            {connectionError && <div className="inline-error" role="alert"><p>{connectionError}</p><button className="text-button" onClick={() => setHealthRetry(value => value + 1)}>Reconnect</button></div>}
            {(job || isUploading || error) && <JobStatus job={job} isUploading={isUploading} uploadProgress={uploadProgress} error={error} />}
            {error && jobId && <button className="text-button" onClick={() => setPollRetry(value => value + 1)}>Retry score status</button>}
            {job?.status === "failed" && <button className="button button-outline" onClick={reset}>Try another recording</button>}
            {availableDownloads && <DownloadPanel job={job} />}
            {done && <p className="retention-note">Download your files within {health?.limits.retention_hours ?? 24} hours. Keep a MusicXML copy to edit your score later.</p>}
          </section>

          {job?.analysis?.warnings?.length ? <section className="review-card"><div className="minor-heading"><CircleHelp size={16} /><h2>Notes for your review</h2></div><ul>{job.analysis.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></section> : null}
          <section className="recent-card" aria-label="Recent transcriptions"><div className="minor-heading"><Clock3 size={15} aria-hidden="true" /><h2>Recent transcriptions</h2><span>THIS DEVICE</span></div>{recent.length ? <div className="recent-list">{recent.slice(0, 4).map(item => <button className={`recent-item ${item.job_id === jobId ? "current" : ""}`} key={item.job_id} disabled={busy} onClick={() => openRecent(item.job_id)}><Music2 size={17} aria-hidden="true" /><span><strong>{pieceName(item.original_filename)}</strong><small>{new Date(item.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })} <span>·</span> {item.status === "done" ? "Score created" : item.status === "failed" ? "Needs attention" : "In progress"}</small></span><ChevronRight size={14} aria-hidden="true" /></button>)}</div> : <p className="recent-empty">Your recordings will appear here, so you can pick up where you left off.</p>}</section>
        </aside>

        <section className="score-workspace" aria-label="Sheet music and playback">
          <div className="score-heading"><div><span className="section-number">02</span><h2>{job ? pieceName(job.original_filename) : "Your sheet music"}</h2>{done && <span className="draft-badge">FIRST DRAFT</span>}</div><span className="score-instrument"><Piano size={15} aria-hidden="true" /> Piano</span></div>
          {done && job?.analysis && <div className="score-metadata">{job.analysis.tempo_bpm ? <span>♩ = {Math.round(job.analysis.tempo_bpm)} BPM</span> : null}{job.analysis.key_signature ? <span>{job.analysis.key_signature}</span> : null}{typeof job.analysis.note_count === "number" ? <span>{job.analysis.note_count.toLocaleString()} notes</span> : null}<span>{previewUrls.length ? `${previewUrls.length} ${previewUrls.length === 1 ? "page" : "pages"}` : "Piano score"}</span></div>}
          <ScorePreview status={job?.status} previewUrls={previewUrls} pdfUrl={apiUrl(job?.download_urls.pdf)} error={job?.error || (!job ? error : null)} />
          <ScorePlayback key={jobId || "new"} url={apiUrl(job?.download_urls.playback)} originalUrl={originalUrl} />
        </section>
      </div>
      <section className="workflow-strip" aria-label="The transcription process"><div><span>01</span><p><strong>Bring a recording</strong>Choose the mode that fits your audio.</p></div><ChevronRight size={17} aria-hidden="true" /><div><span>02</span><p><strong>Find the notes</strong>We shape a first draft of your music.</p></div><ChevronRight size={17} aria-hidden="true" /><div><span>03</span><p><strong>Make it your own</strong>Listen, download, and keep creating.</p></div></section>
    </main>
    <footer className="site-footer"><span className="footer-brand">pianoscribe.</span><p>For the music you want to play.</p><button onClick={() => setGuideOpen(true)}>A note on transcription <ArrowRight size={13} aria-hidden="true" /></button></footer>

    <dialog className="guide-dialog" ref={guideRef} onCancel={() => setGuideOpen(false)} onClick={event => { if (event.target === event.currentTarget) setGuideOpen(false); }}><button className="icon-button dialog-close" aria-label="Close transcription guide" onClick={() => setGuideOpen(false)}><X size={20} /></button><p className="eyebrow">A NOTE BEFORE THE FIRST NOTE</p><h2>A good score starts<br />with a good recording.</h2><p>Piano Scribe creates an editable first draft. Listen to the result and compare it with your recording before relying on the notes.</p><div className="guide-section"><Piano size={23} /><div><h3>Solo piano → transcription</h3><p>A clear solo recording gives the system its best chance to identify the notes you played. Heavy reverb, background noise, and overlapping instruments make this harder.</p></div></div><div className="guide-section"><Radio size={23} /><div><h3>Full song → piano arrangement</h3><p>A full mix contains voices, drums, bass, and other instruments. The result is a piano reduction of detected musical material, not a guaranteed note-for-note piano score.</p></div></div><div className="guide-section"><Headphones size={23} /><div><h3>Listen, then refine</h3><p>Use the piano player to hear the actual notes in your score, slow it down, and compare with the original. Download MusicXML for edits in a notation app or MIDI for your music software.</p></div></div><p className="guide-footnote">Long recordings are processed in sections. Processing time depends on recording length, complexity, and the available hardware. Files are retained for {health?.limits.retention_hours ?? 24} hours; recent history is stored only in this browser.</p><button className="button button-primary" onClick={() => setGuideOpen(false)}>Back to the studio <ArrowRight size={16} /></button></dialog>
  </div>;
}
