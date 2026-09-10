import { CheckCircle2, CircleDashed, Loader2, XCircle } from "lucide-react";

import { Job, JobStatus as Status } from "../api";

const steps: { status: Status; label: string }[] = [
  { status: "queued", label: "Queued" },
  { status: "preprocessing", label: "Audio" },
  { status: "transcribing", label: "MIDI" },
  { status: "scoring", label: "Notation" },
  { status: "rendering", label: "Render" },
  { status: "done", label: "Done" }
];

export default function JobStatus({
  job,
  isUploading,
  uploadProgress,
  error
}: {
  job: Job | null;
  isUploading: boolean;
  uploadProgress: number;
  error: string | null;
}) {
  const currentIndex = job ? steps.findIndex((step) => step.status === job.status) : -1;
  const failed = job?.status === "failed" || (!job && !!error);
  const processing = job && job.status !== "done" && job.status !== "failed";
  const progress = Math.max(0, Math.min(100, isUploading ? uploadProgress : job?.progress ?? 0));

  return (
    <div className="status-panel" aria-live="polite">
      <div className="panel-heading">
        {failed ? (
          <XCircle aria-hidden="true" size={18} />
        ) : job?.status === "done" ? (
          <CheckCircle2 aria-hidden="true" size={18} />
        ) : isUploading || job ? (
          <Loader2 aria-hidden="true" className="spin" size={18} />
        ) : (
          <CircleDashed aria-hidden="true" size={18} />
        )}
        <h2>Status</h2>
      </div>

      <div className="status-text">
        {error || (failed
          ? job?.error || "The score could not be generated."
          : isUploading
            ? uploadProgress === 100 ? "Upload received. Adding your recording to the queue…" : `Uploading ${uploadProgress}%`
            : job
              ? labelFor(job.status)
              : "Waiting")}
      </div>

      {(isUploading || processing) && <progress className="job-progress" max={100} value={progress} aria-label={isUploading ? "Upload progress" : "Processing progress"} />}
      {processing && <p className="processing-note">{progress}% · You can return to this page later; processing continues on the server.</p>}

      {(isUploading || job) && <div className="step-row">
        {steps.map((step, index) => (
          <div
            className={`step ${index <= currentIndex ? "is-active" : ""} ${
              step.status === job?.status ? "is-current" : ""
            }`}
            key={step.status}
          >
            <span className="step-dot" />
            <span className="step-label">{step.label}</span>
          </div>
        ))}
      </div>}
    </div>
  );
}

function labelFor(status: Status): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "preprocessing":
      return "Preparing audio";
    case "transcribing":
      return "Transcribing notes";
    case "scoring":
      return "Writing notation";
    case "rendering":
      return "Rendering score";
    case "done":
      return "Ready";
    case "failed":
      return "Failed";
  }
}
