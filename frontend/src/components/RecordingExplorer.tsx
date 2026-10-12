import { ArrowRight, AudioLines, Check, Mic2, Piano } from "lucide-react";
import { ScoreOptions } from "../api";

const recordings = [
  { id: "full_mix", icon: AudioLines, label: "A full song", title: "Bring a favorite song to the piano.", description: "Start with a recording that includes vocals and instruments. Create a piano arrangement of the main tune and accompaniment.", output: "Melody + piano accompaniment", advice: "A clear mix with an audible melody gives you a better starting point.", tag: "ARRANGE" },
  { id: "piano", icon: Piano, label: "Solo piano", title: "From the recording to the music stand.", description: "Capture a piano performance as a score you can listen to, review, and take into your next practice session.", output: "Two-hand piano sheet music", advice: "Use a recording with little room noise or reverb so attacks and holds are easier to hear.", tag: "TRANSCRIBE" },
  { id: "melody", icon: Mic2, label: "A single melody", title: "Give that tune a place on the page.", description: "Start with one voice or instrument playing a tune. Focus on a single musical line, ready to explore at the piano.", output: "A single melody line", advice: "An isolated voice or instrument works best. Strong backing parts can obscure the tune.", tag: "CAPTURE" }
] as const;

export default function RecordingExplorer({ mode, onChoose, disabled }: { mode: ScoreOptions["transcription_mode"]; onChoose: (mode: ScoreOptions["transcription_mode"]) => void; disabled: boolean }) {
  const selected = recordings.find(recording => recording.id === mode) ?? recordings[0];
  const Icon = selected.icon;
  return <section className="recording-explorer" aria-labelledby="recording-heading" id="recording-types">
    <div className="section-heading"><div><p className="eyebrow">FIND YOUR STARTING POINT</p><h2 id="recording-heading">Different recordings. One piano studio.</h2></div></div>
    <div className="explorer-layout">
      <div className="explorer-options" role="group" aria-label="Explore recording types">{recordings.map(recording => <button key={recording.id} className={`explorer-option ${mode === recording.id ? "selected" : ""}`} disabled={disabled} aria-pressed={mode === recording.id} onClick={() => onChoose(recording.id)}><recording.icon size={21} aria-hidden="true" /><span>{recording.label}</span>{mode === recording.id ? <Check size={16} aria-hidden="true" /> : <ArrowRight size={16} aria-hidden="true" />}</button>)}</div>
      <div className="explorer-detail" aria-live="polite" aria-atomic="true">
        <div className="explorer-detail-top"><span className="feature-icon"><Icon size={24} aria-hidden="true" /></span><span className="explorer-tag">{selected.tag}</span></div>
        <h3>{selected.title}</h3><p>{selected.description}</p>
        <div className="explorer-output"><span>YOUR STARTING SCORE</span><strong>{selected.output}</strong></div>
        <p className="explorer-advice">{selected.advice}</p>
        <a className="section-link" href="#studio">Set up {selected.label.toLowerCase()} <ArrowRight size={16} aria-hidden="true" /></a>
      </div>
    </div>
  </section>;
}
