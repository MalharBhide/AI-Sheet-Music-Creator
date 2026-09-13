import { Job } from "../api";

export default function JobStatus({ job, isUploading, uploadProgress, error }: {
  job: Job | null; isUploading: boolean; uploadProgress: number; error: string | null;
}) {
  const failed = job?.status === "failed" || !!error;
  const processing = job && job.status !== "done" && job.status !== "failed";
  const progress = Math.max(0, Math.min(100, isUploading ? uploadProgress : job?.progress ?? 0));
  const labels = { queued: "Waiting to begin…", preprocessing: "Preparing your recording…", transcribing: "Finding the notes…", scoring: "Writing your score…", rendering: "Finishing your score…", done: "Ready to play", failed: "Something went wrong" };
  return <div className={`status-panel ${failed ? "is-error" : ""}`} aria-live="polite">
    <p>{error || job?.error || (isUploading ? `Uploading · ${uploadProgress}%` : job ? labels[job.status] : "Waiting…")}</p>
    {(isUploading || processing) && <progress max={100} value={progress} aria-label={isUploading ? "Upload progress" : "Processing progress"} />}
    {processing && <small>Longer recordings take a few minutes. You can come back later.</small>}
  </div>;
}
