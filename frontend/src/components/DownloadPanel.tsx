import { ArrowDownToLine, FileAudio, FileCode2, FileText } from "lucide-react";
import { apiUrl, Job } from "../api";

export default function DownloadPanel({ job }: { job: Job | null }) {
  if (!job) return null;
  const pdf = apiUrl(job.download_urls.pdf);
  const formats = [
    { key: "musicxml" as const, label: "MusicXML", description: "Edit your score", icon: FileCode2 },
    { key: "midi" as const, label: "MIDI", description: "Open in your DAW", icon: FileAudio }
  ];
  return <div className="downloads" aria-label="Download your score"><div className="minor-heading"><ArrowDownToLine size={15} aria-hidden="true" /><h2>Take your music with you</h2></div>{pdf && <a className="button button-primary download-pdf" href={pdf}><FileText size={17} aria-hidden="true" /> Download PDF <ArrowDownToLine size={17} aria-hidden="true" /></a>}<div className="download-grid">{formats.map(item => { const href = apiUrl(job.download_urls[item.key]); const Icon = item.icon; return href ? <a className="format-download" href={href} key={item.key}><Icon size={18} aria-hidden="true" /><span><strong>{item.label}</strong><small>{item.description}</small></span><ArrowDownToLine size={13} aria-hidden="true" /></a> : null; })}</div>{(job.download_urls.svg || job.download_urls.svg_zip) && <details className="advanced-downloads"><summary>Vector score files</summary><div>{job.download_urls.svg && <a href={apiUrl(job.download_urls.svg)!}>First page (SVG)</a>}{job.download_urls.svg_zip && <a href={apiUrl(job.download_urls.svg_zip)!}>All pages (ZIP)</a>}</div></details>}</div>;
}
