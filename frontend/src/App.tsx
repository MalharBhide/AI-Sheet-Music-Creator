import { useEffect, useRef, useState } from "react";
import { FileMusic, RotateCcw } from "lucide-react";

import { ApiError, apiUrl, fetchHealth, fetchJob, Health, Job, uploadAudio } from "./api";
import DownloadPanel from "./components/DownloadPanel";
import JobStatus from "./components/JobStatus";
import ScorePreview from "./components/ScorePreview";
import UploadDropzone from "./components/UploadDropzone";

function initialJobId(): string | null {
  const id = new URLSearchParams(window.location.search).get("job");
  return id && /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(id) ? id : null;
}

function updateUrl(jobId: string | null) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState(null, "", url);
}

export default function App() {
  const [jobId, setJobId] = useState<string | null>(initialJobId);
  const [job, setJob] = useState<Job | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [healthRetry, setHealthRetry] = useState(0);
  const [pollRetry, setPollRetry] = useState(0);
  const submitting = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    setConnectionError(null);
    fetchHealth(controller.signal).then(result => {
      if (!controller.signal.aborted) {
        setHealth(result);
        if (!result.ready) setConnectionError("The server needs setup before it can transcribe audio. See the README.");
      }
    }).catch(() => {
      if (!controller.signal.aborted) setConnectionError("Could not connect to the server. Check that it is running.");
    });
    return () => controller.abort();
  }, [healthRetry]);

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
        setJob(next);
        setError(null);
        failures = 0;
        if (next.status !== "done" && next.status !== "failed") timer = window.setTimeout(poll, 1500);
      } catch (err) {
        if (controller.signal.aborted) return;
        failures += 1;
        const missing = err instanceof ApiError && (err.status === 404 || err.status === 422);
        setError(missing ? err.message : "Connection interrupted. Your job may still be processing.");
        if (!missing && failures < 5) timer = window.setTimeout(poll, Math.min(1000 * 2 ** failures, 15000));
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [jobId, pollRetry]);

  async function handleUpload(file: File) {
    if (submitting.current || jobId || !health?.ready) return;
    setError(null);
    if (!/\.(wav|mp3|flac|ogg|m4a|aac|aiff?)$/i.test(file.name)) {
      setError("Choose a WAV, MP3, FLAC, OGG, M4A, AAC, or AIFF recording."); return;
    }
    if (!file.size || file.size > health.limits.max_upload_mb * 1024 * 1024) {
      setError("Choose a non-empty file up to " + health.limits.max_upload_mb + " MB."); return;
    }
    submitting.current = true;
    setIsUploading(true);
    try {
      const created = await uploadAudio(file);
      setJob(created);
      setJobId(created.job_id);
      updateUrl(created.job_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed. Check your connection and try again.");
    } finally {
      submitting.current = false;
      setIsUploading(false);
    }
  }

  function reset() {
    setJobId(null); setJob(null); setError(null);
    updateUrl(null);
  }

  const previewUrls = (job?.svg_pages ?? []).map(path => apiUrl(path)!);
  if (!previewUrls.length && job?.download_urls.svg) previewUrls.push(apiUrl(job.download_urls.svg)!);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <FileMusic aria-hidden="true" size={22} />
          <span>AI Sheet Music Creator</span>
        </div>
        {jobId && (
          <button className="icon-button" type="button" onClick={reset} aria-label="Start over">
            <RotateCcw aria-hidden="true" size={18} />
          </button>
        )}
      </header>

      <main className="workspace">
        <section className="control-panel" aria-label="Audio upload and job status">
          <div className="intro-copy">
            <h1>Make piano sheet music from audio</h1>
            <p>Upload a recording, then download the generated score as a PDF.</p>
          </div>
          <UploadDropzone onUpload={handleUpload} disabled={isUploading || !!jobId || !health?.ready || !!connectionError} />
          {health && <p className="recording-note">Up to {health.limits.max_upload_mb} MB · {health.limits.max_audio_seconds} seconds</p>}
          {connectionError && <div className="connection-error" role="alert"><p>{connectionError}</p><button className="retry-button" onClick={() => setHealthRetry(value => value + 1)}>Reconnect</button></div>}
          {jobId && !job && !error && <p className="recording-note" role="status">Loading your saved job…</p>}
          <JobStatus job={job} isUploading={isUploading} error={error} />
          {error && jobId && <button className="retry-button" onClick={() => setPollRetry(value => value + 1)}>Retry job status</button>}
          <DownloadPanel job={job} />
          {job?.status === "done" && <p className="recording-note">This is a first draft. Check the notes and rhythm, and download your files within {health?.limits.retention_hours ?? 24} hours.</p>}
        </section>

        <section className="preview-panel" aria-label="Sheet music preview">
          <ScorePreview status={job?.status} previewUrls={previewUrls} error={job?.error || error} />
        </section>
      </main>
    </div>
  );
}
