import { useEffect, useState } from "react";
import { FileMusic, Loader2, Music } from "lucide-react";

import { JobStatus } from "../api";

export default function ScorePreview({
  status, previewUrls, error
}: {
  status?: JobStatus;
  previewUrls: string[];
  error?: string | null;
}) {
  const [page, setPage] = useState(0);
  const firstPage = previewUrls[0];
  useEffect(() => setPage(0), [firstPage]);

  if (status === "done" && previewUrls.length) {
    const pageNumber = Math.min(page, previewUrls.length - 1);
    return (
      <div className="preview-pages">
        <div className="score-frame">
          <img src={previewUrls[pageNumber] + "?preview=true"} alt={"Generated piano sheet music, page " + (pageNumber + 1)} />
        </div>
        {previewUrls.length > 1 && <nav className="page-controls" aria-label="Score pages">
          <button disabled={pageNumber === 0} onClick={() => setPage(value => value - 1)}>Previous</button>
          <span>Page {pageNumber + 1} of {previewUrls.length}</span>
          <button disabled={pageNumber === previewUrls.length - 1} onClick={() => setPage(value => value + 1)}>Next</button>
        </nav>}
      </div>
    );
  }

  if (status === "failed" || error) {
    return <div className="empty-preview is-error"><FileMusic aria-hidden="true" size={34} /><p>{error || "The score could not be generated."}</p></div>;
  }
  if (status) {
    const labels: Record<JobStatus, string> = {
      queued: "Queued", preprocessing: "Preparing audio", transcribing: "Listening for piano notes",
      scoring: "Building the score", rendering: "Rendering preview", done: "Ready", failed: "Failed"
    };
    return <div className="empty-preview"><Loader2 aria-hidden="true" className="spin" size={34} /><p>{labels[status]}</p></div>;
  }
  return <div className="empty-preview"><Music aria-hidden="true" size={34} /><h2>Your score preview</h2><p>Choose an audio file to create a piano score.</p></div>;
}
