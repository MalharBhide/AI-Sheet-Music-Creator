import { ArrowDown, ArrowRight, AudioLines, Headphones, Music2 } from "lucide-react";

const waveform = [12, 20, 14, 32, 42, 24, 16, 30, 48, 36, 20, 14, 28, 40, 26, 16, 22, 46, 32, 18, 12, 26, 38, 22, 14, 30, 44, 28, 16];

export default function StudioIntro() {
  return <section className="page-intro" aria-labelledby="intro-heading">
    <div className="intro-copy">
      <p className="eyebrow"><span className="intro-dot" /> YOUR MUSIC. YOUR PIANO.</p>
      <h1 id="intro-heading">Hear it.<br /><em>Bring it to the keys.</em></h1>
      <p className="intro-description">Turn your recordings into piano sheet music.<br />Listen, explore the notes, and make your next practice session your own.</p>
      <div className="intro-actions"><a href="#studio" className="button button-primary">Create a piano score <ArrowRight size={17} /></a><a href="#how-it-works" className="intro-secondary">See how it works <ArrowDown size={15} /></a></div>
      <div className="intro-formats"><span><AudioLines size={14} /> Audio to piano</span><span><Headphones size={14} /> Follow-along playback</span><span><Music2 size={14} /> PDF, MIDI & MusicXML</span></div>
    </div>
    <div className="intro-illustration" aria-hidden="true">
      <div className="illustration-orbit" />
      <div className="sound-card"><span className="sound-icon"><AudioLines size={21} /></span><div className="sound-wave">{waveform.map((height, index) => <i key={index} style={{ height }} />)}</div><span className="sound-caption">Your recording</span></div>
      <div className="illustrated-score"><div className="illustration-title"><span>Made for your piano</span><Music2 size={18} /></div>
        <svg viewBox="0 0 400 160" fill="none">
          {[38, 49, 60, 71, 82, 107, 118, 129, 140, 151].map(y => <path key={y} d={`M14 ${y}H386`} stroke="#dce4dc" />)}
          <path d="M22 38V151M127 38V151M255 38V151M382 38V151" stroke="#c4d2c8" />
          <text x="27" y="78" fill="#627e70" fontSize="58" fontFamily="serif">𝄞</text><text x="28" y="143" fill="#627e70" fontSize="36" fontFamily="serif">𝄢</text>
          {[[82, 71], [111, 60], [154, 65], [185, 49], [218, 55], [280, 60], [315, 71], [352, 60]].map(([x, y]) => <g key={x}><ellipse cx={x} cy={y} rx="6" ry="4.5" transform={`rotate(-20 ${x} ${y})`} fill="#355a48" /><path d={`M${x + 5} ${y}V${y - 28}`} stroke="#355a48" strokeWidth="1.6" /></g>)}
          {[[83, 140], [154, 129], [218, 135], [280, 140], [352, 129]].map(([x, y]) => <g key={x}><ellipse cx={x} cy={y} rx="6" ry="4.5" transform={`rotate(-20 ${x} ${y})`} fill="#739181" /><path d={`M${x + 5} ${y}V${y - 27}`} stroke="#739181" strokeWidth="1.6" /></g>)}
        </svg>
        <div className="illustration-caption"><span className="intro-dot" /> From listening to playing</div>
      </div>
      <div className="illustrated-keyboard">{Array.from({ length: 15 }, (_, index) => <span key={index} className={[2, 5, 9].includes(index) ? "pressed-key" : ""}>{![2, 6, 9, 13, 14].includes(index) && <i />}</span>)}</div>
    </div>
  </section>;
}
