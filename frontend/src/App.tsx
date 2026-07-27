import { useEffect, useMemo, useState } from "react";
import { FileMusic, RotateCcw } from "lucide-react";

import { apiUrl, fetchJob, Job, uploadAudio } from "./api";
import DownloadPanel from "./components/DownloadPanel";
import JobStatus from "./components/JobStatus";
import ScorePreview from "./components/ScorePreview";
import UploadDropzone from "./components/UploadDropzone";

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        setJob(await fetchJob(job.job_id));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not refresh the job.");
      }
    }, 2000);

    return () => window.clearInterval(interval);
  }, [job]);

  const previewUrl = useMemo(() => apiUrl(job?.download_urls.svg), [job]);

  async function handleUpload(file: File) {
    setError(null);
    setIsUploading(true);
    setJob(null);

    try {
      setJob(await uploadAudio(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  function reset() {
    setJob(null);
    setError(null);
    setIsUploading(false);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <FileMusic aria-hidden="true" size={22} />
          <span>AI Sheet Music Creator</span>
        </div>
        {job && (
          <button className="icon-button" type="button" onClick={reset} aria-label="Start over">
            <RotateCcw aria-hidden="true" size={18} />
          </button>
        )}
      </header>

      <main className="workspace">
        <section className="control-panel" aria-label="Audio upload and job status">
          <UploadDropzone onUpload={handleUpload} disabled={isUploading || !!job} />
          <JobStatus job={job} isUploading={isUploading} error={error} />
          <DownloadPanel job={job} />
        </section>

        <section className="preview-panel" aria-label="Sheet music preview">
          <ScorePreview
            status={job?.status}
            previewUrl={previewUrl}
            error={job?.error || error}
          />
        </section>
      </main>
    </div>
  );
}

