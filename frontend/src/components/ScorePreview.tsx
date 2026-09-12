import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, FileMusic, Loader2, Minus, Music2, Plus } from "lucide-react";
import { JobStatus } from "../api";

export default function ScorePreview({ status, previewUrls, pdfUrl, error }: {
  status?: JobStatus; previewUrls: string[]; pdfUrl?: string | null; error?: string | null;
}) {
  const [page, setPage] = useState(0);
  const [zoom, setZoom] = useState(100);
  const [imageError, setImageError] = useState(false);
  const firstPage = previewUrls[0];
  useEffect(() => { setPage(0); setZoom(100); setImageError(false); }, [firstPage]);
  useEffect(() => setImageError(false), [page]);

  if (previewUrls.length && !imageError) {
    const pageNumber = Math.min(page, previewUrls.length - 1);
    return <div className="preview-pages"><div className="preview-toolbar"><nav className="page-controls" aria-label="Score pages"><button className="icon-button" disabled={pageNumber === 0} aria-label="Previous score page" onClick={() => setPage(value => value - 1)}><ArrowLeft size={15} /></button><span>Page <strong>{pageNumber + 1}</strong> of {previewUrls.length}</span><button className="icon-button" disabled={pageNumber === previewUrls.length - 1} aria-label="Next score page" onClick={() => setPage(value => value + 1)}><ArrowRight size={15} /></button></nav><div className="zoom-controls"><button className="icon-button" aria-label="Zoom out" disabled={zoom <= 60} onClick={() => setZoom(value => value - 20)}><Minus size={15} /></button><span>{zoom}%</span><button className="icon-button" aria-label="Zoom in" disabled={zoom >= 160} onClick={() => setZoom(value => value + 20)}><Plus size={15} /></button></div></div><div className="score-frame"><img style={{ width: `${zoom}%`, maxWidth: "none" }} src={previewUrls[pageNumber] + "?preview=true"} alt={"Generated piano sheet music, page " + (pageNumber + 1)} onError={() => setImageError(true)} /></div></div>;
  }
  if (pdfUrl) return <object className="pdf-preview" data={pdfUrl + "?preview=true"} type="application/pdf" aria-label="Generated piano sheet music PDF"><div className="empty-preview"><FileMusic size={34} aria-hidden="true" /><p>Your PDF is ready. <a href={pdfUrl}>Download the sheet music</a> to view it.</p></div></object>;
  if (status === "failed" || error) return <div className="empty-preview is-error"><span className="preview-state-icon"><FileMusic size={29} aria-hidden="true" /></span><h2>Let's give that another listen.</h2><p>{error || "We could not create a score from this recording. Try a clearer recording or another transcription mode."}</p></div>;
  if (status && status !== "done") {
    const labels: Record<string, string> = { queued: "Your recording is in the queue.", preprocessing: "Getting your recording ready.", transcribing: "Listening for the notes.", scoring: "Giving the music its shape.", rendering: "Putting the finishing notes on paper." };
    return <div className="empty-preview is-processing"><span className="preview-state-icon"><Loader2 className="spin" size={28} aria-hidden="true" /></span><p className="eyebrow">A LITTLE PATIENCE, A LITTLE MUSIC</p><h2>{labels[status]}</h2><p>Your score will appear here. You can return to this page while the transcription continues.</p><div className="processing-staff" aria-hidden="true">{[0, 1, 2, 3, 4].map(line => <i key={line} />)}</div></div>;
  }
  if (status === "done") return <div className="empty-preview"><FileMusic size={30} /><h2>Your files are ready.</h2><p>A visual preview is unavailable for this score. Use the available downloads to open it in your music app.</p></div>;
  return <div className="empty-preview idle-preview"><div className="manuscript" aria-hidden="true"><span className="manuscript-caption">A LITTLE SPACE FOR YOUR NEXT IDEA</span>{[0, 1, 2].map(staff => <div className="staff-system" key={staff}>{[0, 1].map(hand => <div className="staff" key={hand}>{[0, 1, 2, 3, 4].map(line => <i key={line} />)}</div>)}</div>)}</div><div className="empty-preview-message"><span className="preview-state-icon"><Music2 size={29} strokeWidth={1.4} aria-hidden="true" /></span><h2>Your next idea,<br /><em>on paper.</em></h2><p>Upload a recording to see your score here.<br />Then press play and hear it come to life.</p></div><span className="preview-caption"><span /> PDF · MUSICXML · MIDI</span></div>;
}
