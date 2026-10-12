import { FileMusic, Headphones, MousePointer2, Music2 } from "lucide-react";
import { Job } from "../api";

const modeLabels: Record<string, string> = { full_mix: "Full song arrangement", piano: "Solo piano", melody: "Single melody" };

export default function ScoreDetails({ job, pages, canPlay }: { job: Job; pages: number; canPlay: boolean }) {
  const tempo = job.analysis?.tempo_bpm;
  const mode = modeLabels[job.analysis?.transcription_mode ?? ""];
  const key = job.analysis?.key_signature;
  return <section className="score-details" aria-labelledby="score-details-heading">
    <div className="score-details-heading"><span className="feature-icon"><Music2 size={19} aria-hidden="true" /></span><h2 id="score-details-heading">Inside your score</h2></div>
    <dl className="score-facts">
      {mode && <div><dt>Recording type</dt><dd>{mode}</dd></div>}
      {typeof tempo === "number" && Number.isFinite(tempo) && tempo > 0 && <div><dt>Tempo</dt><dd>{Math.round(tempo)} BPM</dd></div>}
      {typeof key === "string" && key.trim() && <div><dt>Key</dt><dd>{key}</dd></div>}
      {pages > 0 && <div><dt>Sheet music</dt><dd>{pages} {pages === 1 ? "page" : "pages"}</dd></div>}
    </dl>
    <details className="practice-guide"><summary>A little help with practice</summary><ul>
      {canPlay && <li><Headphones size={16} aria-hidden="true" /><span>Start with playback at 0.75× to hear the rhythm more clearly.</span></li>}
      {canPlay && pages > 0 && <li><MousePointer2 size={16} aria-hidden="true" /><span>When score following is available, click a note to return to a passage.</span></li>}
      <li><FileMusic size={16} aria-hidden="true" /><span>Review this first draft, then download MusicXML to adjust notes in your notation app.</span></li>
    </ul></details>
  </section>;
}
