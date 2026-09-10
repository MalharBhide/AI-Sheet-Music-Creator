import { Download, FileArchive, FileAudio, FileCode2, FileText } from "lucide-react";

import { apiUrl, Job } from "../api";

type DownloadItem = {
  key: "pdf" | "musicxml" | "midi" | "svg" | "svg_zip";
  label: string;
  icon: typeof FileText;
};

const items: DownloadItem[] = [
  { key: "pdf", label: "Download sheet music (PDF)", icon: FileText },
  { key: "musicxml", label: "MusicXML", icon: FileCode2 },
  { key: "midi", label: "MIDI", icon: FileAudio },
  { key: "svg", label: "SVG page 1", icon: FileCode2 },
  { key: "svg_zip", label: "All SVG pages", icon: FileArchive }
];

export default function DownloadPanel({ job }: { job: Job | null }) {
  const hasExtraFiles = items.slice(1).some(item => !!job?.download_urls[item.key]);
  const hasPdf = !!job?.download_urls.pdf;

  return (
    <div className="downloads" aria-label="Downloads">
      <div className="panel-heading">
        <Download aria-hidden="true" size={18} />
        <h2>Your sheet music</h2>
      </div>
      <div className="download-grid">
        {items.slice(0, 1).map((item) => {
          const Icon = item.icon;
          const href = apiUrl(job?.download_urls[item.key]);
          return (
            <a
              className={`download-button ${!href ? "is-disabled" : ""}`}
              href={href || undefined}
              key={item.key}
              aria-disabled={!href}
            >
              <Icon aria-hidden="true" size={17} />
              <span>{item.label}</span>
            </a>
          );
        })}
      </div>
      {hasExtraFiles && (
        <details className="advanced-downloads" open={!hasPdf}>
          <summary>{hasPdf ? "More file formats" : "Available files"}</summary>
          <div className="download-grid compact-download-grid">
            {items.slice(1).filter(item => !!job?.download_urls[item.key]).map((item) => {
              const Icon = item.icon;
              const href = apiUrl(job?.download_urls[item.key]);
              return (
                <a className={`download-button ${!href ? "is-disabled" : ""}`} href={href || undefined} aria-disabled={!href} key={item.key}>
                  <Icon aria-hidden="true" size={17} />
                  <span>{item.label}</span>
                </a>
              );
            })}
          </div>
        </details>
      )}
    </div>
  );
}
