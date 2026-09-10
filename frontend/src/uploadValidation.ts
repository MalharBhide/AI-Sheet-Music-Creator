export function validateAudioUpload(file: { name: string; size: number }, maxUploadMb: number): string | null {
  if (!/\.(wav|mp3|flac|ogg|m4a|aac|aiff?)$/i.test(file.name)) {
    return "Choose a WAV, MP3, FLAC, OGG, M4A, AAC, or AIFF recording.";
  }
  if (!file.size) return "Choose a non-empty audio recording.";
  // The API uses zero to disable its optional upload limit.
  if (maxUploadMb > 0 && file.size > maxUploadMb * 1024 * 1024) {
    return "Choose a file up to " + maxUploadMb + " MB.";
  }
  return null;
}
