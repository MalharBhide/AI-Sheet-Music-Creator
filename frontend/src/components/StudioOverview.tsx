import { ArrowRight, AudioLines, FileMusic, Headphones, Music2, Piano } from "lucide-react";
import { ScoreOptions } from "../api";
import RecordingExplorer from "./RecordingExplorer";

const steps = [
  { icon: AudioLines, title: "Bring your recording", description: "Choose an audio file and tell us whether it’s a full song, solo piano, or a melody." },
  { icon: Headphones, title: "Listen and follow along", description: "Hear your piano score, slow down playback, and click the sheet music to jump to a passage." },
  { icon: FileMusic, title: "Make it your own", description: "Print a PDF, edit the notes in a notation app with MusicXML, or take MIDI into your music software." }
];

export default function StudioOverview({ retentionHours, mode, onChooseMode, disabled }: { retentionHours: number; mode: ScoreOptions["transcription_mode"]; onChooseMode: (mode: ScoreOptions["transcription_mode"]) => void; disabled: boolean }) {
  return <div className="studio-overview">
    <RecordingExplorer mode={mode} onChoose={onChooseMode} disabled={disabled} />
    <section className="workflow-section" id="how-it-works" aria-labelledby="workflow-heading">
      <div className="section-heading"><div><p className="eyebrow">FROM SOUND TO SCORE</p><h2 id="workflow-heading">Your next practice session starts here.</h2></div><a href="#studio" className="section-link">Open the studio <ArrowRight size={16} /></a></div>
      <div className="workflow-grid">{steps.map((step, index) => <article className="workflow-card" key={step.title}><div className="workflow-card-top"><span className="feature-icon"><step.icon size={23} /></span><span className="step-number">0{index + 1}</span></div><h3>{step.title}</h3><p>{step.description}</p></article>)}</div>
    </section>
    <section className="formats-section" id="formats" aria-labelledby="formats-heading"><div><p className="eyebrow">KEEP PLAYING, YOUR WAY</p><h2 id="formats-heading">A score that goes with you.</h2><p>Your transcription is a starting point. Listen, review, and refine it in the tools you already use.</p></div><div className="format-features"><div><FileMusic size={21} /><span><strong>PDF</strong><small>Print and practice</small></span></div><div><Music2 size={21} /><span><strong>MusicXML</strong><small>Edit your sheet music</small></span></div><div><Piano size={21} /><span><strong>MIDI</strong><small>Work with the notes</small></span></div></div></section>
    <section className="faq-section" id="questions" aria-labelledby="questions-heading"><div><p className="eyebrow">BEFORE YOU BEGIN</p><h2 id="questions-heading">A few helpful notes.</h2></div><div className="faq-list"><details><summary>Which recording type should I choose?</summary><p>Choose Solo piano for piano on its own, Full song for a piano arrangement of a mixed recording, or Melody for a voice or instrument playing one tune. Clear recordings with less noise and reverb are easier to transcribe.</p></details><details><summary>Can I change the notes afterward?</summary><p>Download MusicXML to edit your score in a notation app, or MIDI to edit the notes in music software. The website lets you listen to the transcription and follow its position in the sheet music.</p></details><details><summary>Where can I find my previous scores?</summary><p>Recent scores are saved in this browser. Download your files to keep them: server files are available for {retentionHours} hours. A recent-score entry does not extend that availability.</p></details></div></section>
  </div>;
}
