import { Download } from "lucide-react";
import { apiUrl, Job } from "../api";

export default function DownloadPanel({ job }: { job: Job | null }) {
  if (!job) return null;
  const pdf = apiUrl(job.download_urls.pdf);
  const formats = [
    { key: "musicxml" as const, label: "MusicXML", description: "Edit the notes" },
    { key: "midi" as const, label: "MIDI", description: "Open in a music app" },
    { key: "svg" as const, label: "SVG", description: "First page" },
    { key: "svg_zip" as const, label: "All SVG pages", description: "ZIP download" }
  ];
  return <div className="downloads" aria-label="Download your score">
    {pdf && <a className="button button-primary download-pdf" href={pdf}><Download size={18} /> Download sheet music<span>PDF</span></a>}
    <details className="more-formats" open={!pdf}><summary>Other formats</summary><div className="download-grid">{formats.map(format => {
      const href = apiUrl(job.download_urls[format.key]);
      return href ? <a className="format-download" href={href} key={format.key}><span><strong>{format.label}</strong><small>{format.description}</small></span><Download size={15} /></a> : null;
    })}</div></details>
  </div>;
}
