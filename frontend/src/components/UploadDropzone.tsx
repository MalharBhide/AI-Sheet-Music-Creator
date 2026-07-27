import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

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
  disabled
}: {
  onUpload: (file: File) => void;
  disabled?: boolean;
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
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          openPicker();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={`${acceptedAudioTypes},.mp3,.wav,.m4a,.aac,.flac,.ogg`}
        onChange={handleInput}
        disabled={disabled}
      />
      <UploadCloud aria-hidden="true" size={30} />
      <div>
        <h1>Upload Audio</h1>
        <p>MP3, WAV, M4A, AAC, FLAC, or OGG</p>
      </div>
    </div>
  );
}

