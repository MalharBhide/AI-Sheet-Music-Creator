import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, FileMusic, Loader2, Minus, Music2, Plus } from "lucide-react";
import { JobStatus } from "../api";
import { ScorePosition, scorePositionAt, scorePositionNear } from "../playback";

export default function ScorePreview({ status, previewUrls, pdfUrl, error, position, positions, onSeek }: {
  status?: JobStatus; previewUrls: string[]; pdfUrl?: string | null; error?: string | null;
  position: number; positions: ScorePosition[];
  onSeek: (time: number) => void;
}) {
  const [page, setPage] = useState(0);
  const [zoom, setZoom] = useState(100);
  const [imageError, setImageError] = useState(false);
  const [follow, setFollow] = useState(true);
  const frame = useRef<HTMLDivElement>(null);
  const paper = useRef<HTMLDivElement>(null);
  const firstPage = previewUrls[0];
  const validMap = positions.length > 0 && positions.every(item => item.page < previewUrls.length);
  const current = validMap ? scorePositionAt(positions, position) : null;
  const pageNumber = Math.min(follow && current ? current.page : page, Math.max(0, previewUrls.length - 1));
  const currentY = current?.y;
  const currentHeight = current?.height;
  const currentPage = current?.page;
  const currentX = current?.x;

  useEffect(() => { setPage(0); setZoom(100); setImageError(false); setFollow(true); }, [firstPage]);
  useEffect(() => { setImageError(false); }, [pageNumber]);
  // Scroll only as the music enters a different system, or the page/zoom changes.
  // The player uses its audio clock; this view never estimates time from a page.
  useEffect(() => {
    const viewport = frame.current, sheet = paper.current;
    if (!follow || !viewport || !sheet || currentY === undefined) return;
    function reveal() {
      const top = sheet!.offsetTop + currentY! * sheet!.clientHeight;
      const bottom = top + (currentHeight ?? 0) * sheet!.clientHeight;
      if (top < viewport!.scrollTop + 28 || bottom > viewport!.scrollTop + viewport!.clientHeight - 28) {
        viewport!.scrollTo({ top: Math.max(0, top - 60), behavior: "instant" as ScrollBehavior });
      }
    }
    const observer = new ResizeObserver(reveal);
    observer.observe(sheet);
    reveal();
    return () => observer.disconnect();
  }, [follow, currentY, currentHeight, currentPage, zoom, pageNumber]);

  useEffect(() => {
    const viewport = frame.current, sheet = paper.current;
    if (!follow || !viewport || !sheet || currentX === undefined) return;
    const left = sheet.offsetLeft + currentX * sheet.clientWidth;
    if (left < viewport.scrollLeft + 24 || left > viewport.scrollLeft + viewport.clientWidth - 36) {
      viewport.scrollTo({ left: Math.max(0, left - viewport.clientWidth * .35) });
    }
  }, [follow, currentX, currentPage, zoom, pageNumber]);

  function browse(next: number) {
    setFollow(false); setPage(next);
    frame.current?.scrollTo({ top: 0, left: 0 });
  }

  if (previewUrls.length && !imageError) {
    return <div className="preview-pages">
      <div className="preview-toolbar">
        <nav className="page-controls" aria-label="Score pages">
          <button className="icon-button" disabled={pageNumber === 0} aria-label="Previous score page" onClick={() => browse(pageNumber - 1)}><ArrowLeft size={17} /></button>
          <span>{pageNumber + 1} / {previewUrls.length}</span>
          <button className="icon-button" disabled={pageNumber === previewUrls.length - 1} aria-label="Next score page" onClick={() => browse(pageNumber + 1)}><ArrowRight size={17} /></button>
        </nav>
        {validMap && <button className={`follow-button ${follow ? "active" : ""}`} aria-pressed={follow} onClick={() => { if (follow) setPage(pageNumber); setFollow(value => !value); }}>Follow playback</button>}
        <div className="zoom-controls">
          <button className="icon-button" aria-label="Zoom out" disabled={zoom <= 60} onClick={() => setZoom(value => value - 20)}><Minus size={16} /></button>
          <span>{zoom}%</span>
          <button className="icon-button" aria-label="Zoom in" disabled={zoom >= 180} onClick={() => setZoom(value => value + 20)}><Plus size={16} /></button>
        </div>
      </div>
      <div className="score-frame" ref={frame}>
        <div className={`score-paper ${validMap ? "seekable" : ""}`} ref={paper} style={{ width: `${zoom}%` }}
          role={validMap ? 'button' : undefined} tabIndex={validMap ? 0 : undefined}
          aria-label={validMap ? 'Seek in sheet music. Click a note, or use left and right arrow keys.' : undefined}
          onClick={event => {
            if (!validMap) return;
            const box = event.currentTarget.getBoundingClientRect();
            const target = scorePositionNear(positions, pageNumber, (event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height);
            if (target) { setFollow(true); onSeek(target.time); }
          }}
          onKeyDown={event => {
            if (!validMap || !['ArrowLeft', 'ArrowRight', 'Enter', ' '].includes(event.key)) return;
            event.preventDefault();
            const candidates = positions.filter(p => p.page === pageNumber);
            const target = event.key === 'ArrowRight' ? candidates.find(p => p.time > position + .001)
              : event.key === 'ArrowLeft' ? [...candidates].reverse().find(p => p.time < position - .001)
              : candidates[0];
            if (target) { setFollow(true); onSeek(target.time); }
          }}>
          <img src={previewUrls[pageNumber] + "?preview=true"} alt={`Piano sheet music, page ${pageNumber + 1}`} onError={() => setImageError(true)} />
          {current && current.page === pageNumber && <div className="score-cursor" data-time={current.time} aria-label="Current playback position in the score" role="img" style={{ left: `${current.x * 100}%`, top: `${current.y * 100}%`, height: `${current.height * 100}%` }} />}
        </div>
      </div>
      {validMap && <p className="preview-notice">Click a note to jump to that moment.</p>}
      {!validMap && <p className="preview-notice">Page following is unavailable for this score. You can still listen and browse the pages.</p>}
    </div>;
  }
  if (pdfUrl) return <object className="pdf-preview" data={pdfUrl + "?preview=true"} type="application/pdf" aria-label="Piano sheet music PDF"><div className="empty-preview"><FileMusic size={34} /><p><a href={pdfUrl}>Download the PDF</a> to view your score.</p></div></object>;
  if (status === "failed" || error) return <div className="empty-preview is-error"><FileMusic size={32} /><h2>We couldn’t create this score.</h2><p>{error || "Try a clearer recording or a different mode."}</p></div>;
  if (status && status !== "done") return <div className="empty-preview is-processing"><Loader2 className="spin" size={30} /><h2>Your music is on its way.</h2><p>You can leave this page and return when it’s ready.</p></div>;
  if (status === "done") return <div className="empty-preview"><FileMusic size={32} /><h2>Your files are ready.</h2><p>Download your score to open it in a music app.</p></div>;
  return <div className="empty-preview idle-preview"><div className="preview-art" aria-hidden="true"><div className="demo-staff">𝄞<span>♪</span><span>♩</span><span>♫</span></div></div><Music2 size={24} /><h2>A new way to play<br /><em>your favorite music.</em></h2><p>Your piano score will appear here.<br />Press play to follow along, one note at a time.</p></div>;
}
