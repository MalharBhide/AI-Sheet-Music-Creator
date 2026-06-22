import { Download, FileArchive, FileAudio, FileCode2, FileText } from "lucide-react";

import { apiUrl, Job } from "../api";

type DownloadItem = {
  key: "pdf" | "musicxml" | "midi" | "svg";
  label: string;
  icon: typeof FileText;
};

const items: DownloadItem[] = [
  { key: "pdf", label: "PDF", icon: FileText },
  { key: "musicxml", label: "MusicXML", icon: FileCode2 },
  { key: "midi", label: "MIDI", icon: FileAudio },
  { key: "svg", label: "SVG", icon: FileArchive }
];

export default function DownloadPanel({ job }: { job: Job | null }) {
  const isReady = job?.status === "done";

  return (
    <div className="downloads" aria-label="Downloads">
      <div className="panel-heading">
        <Download aria-hidden="true" size={18} />
        <h2>Downloads</h2>
      </div>
      <div className="download-grid">
        {items.map((item) => {
          const Icon = item.icon;
          const href = apiUrl(job?.download_urls[item.key]);
          return (
            <a
              className={`download-button ${!isReady || !href ? "is-disabled" : ""}`}
              href={isReady && href ? href : undefined}
              key={item.key}
              aria-disabled={!isReady || !href}
            >
              <Icon aria-hidden="true" size={17} />
              <span>{item.label}</span>
            </a>
          );
        })}
      </div>
    </div>
  );
}

