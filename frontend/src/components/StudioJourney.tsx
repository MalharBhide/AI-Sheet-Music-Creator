import { Check } from "lucide-react";

const steps = ["Add a recording", "Create your score", "Listen & download"];

export default function StudioJourney({ stage }: { stage: 0 | 1 | 2 }) {
  return <ol className="studio-journey" aria-label="Score creation steps">{steps.map((label, index) => <li key={label} className={index === stage ? "current-step" : index < stage ? "complete-step" : ""} aria-current={index === stage ? "step" : undefined}><span className="journey-number" aria-hidden="true">{index < stage ? <Check size={13} /> : index + 1}</span><span>{label}</span></li>)}</ol>;
}
