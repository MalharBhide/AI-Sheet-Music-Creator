import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { ArrowUp, Check, FileAudio2 } from "lucide-react";

const acceptedAudioTypes = [
  "audio/mpeg",
  "audio/wav",
  "audio/x-wav",
  "audio/mp4",
  "audio/aac",
  "audio/flac",
  "audio/ogg"
].join(",");

export default function UploadDropzone({
  onUpload,
  disabled,
  file
}: {
  onUpload: (file: File) => void;
  disabled?: boolean;
  file?: File | null;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  function openPicker() {
    if (!disabled) {
      inputRef.current?.click();
    }
  }

  function handleInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
      event.target.value = "";
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file && !disabled) {
      onUpload(file);
    }
  }

  return (
    <div
      className={`dropzone ${isDragging ? "is-dragging" : ""} ${disabled ? "is-disabled" : ""}`}
      onClick={openPicker}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      role="button"
      aria-disabled={disabled}
      aria-label={file ? "Change audio recording" : "Choose an audio recording"}
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openPicker();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={`${acceptedAudioTypes},.mp3,.wav,.m4a,.aac,.flac,.ogg,.aif,.aiff`}
        onChange={handleInput}
        disabled={disabled}
      />
      <span className="upload-symbol">{file ? <FileAudio2 aria-hidden="true" size={26} /> : <ArrowUp aria-hidden="true" size={26} />}</span>
      <div className="dropzone-copy">
        <h2>{file ? file.name : "Drop your audio here"}</h2>
        <p>{file ? `${file.size >= 1024 * 1024 ? (file.size / (1024 * 1024)).toFixed(1) + " MB" : Math.max(1, Math.round(file.size / 1024)) + " KB"} · Click to replace` : <>or <span>browse files</span> from your device</>}</p>
      </div>
      {file ? <span className="file-selected"><Check size={12} /> READY TO UPLOAD</span> : <span className="file-types">MP3 · WAV · M4A · FLAC + more</span>}
    </div>
  );
}
