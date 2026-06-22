import { FileMusic, Loader2, Music } from "lucide-react";

import { JobStatus } from "../api";

export default function ScorePreview({
  status,
  previewUrl,
  error
}: {
  status?: JobStatus;
  previewUrl: string | null;
  error?: string | null;
}) {
  if (status === "done" && previewUrl) {
    return (
      <div className="score-frame">
        <img src={previewUrl} alt="Generated piano sheet music" />
      </div>
    );
  }

  if (status === "failed" || error) {
    return (
      <div className="empty-preview is-error">
        <FileMusic aria-hidden="true" size={34} />
        <p>{error || "The score could not be generated."}</p>
      </div>
    );
  }

  if (status) {
    return (
      <div className="empty-preview">
        <Loader2 aria-hidden="true" className="spin" size={34} />
        <p>{statusLabel(status)}</p>
      </div>
    );
  }

  return (
    <div className="empty-preview">
      <Music aria-hidden="true" size={34} />
      <p>Upload audio to generate piano sheet music.</p>
    </div>
  );
}

function statusLabel(status: JobStatus): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "preprocessing":
      return "Preparing audio";
    case "transcribing":
      return "Listening for piano notes";
    case "scoring":
      return "Building the score";
    case "rendering":
      return "Rendering preview";
    case "done":
      return "Ready";
    case "failed":
      return "Failed";
  }
}
