export type JobStatus =
  | "queued"
  | "preprocessing"
  | "transcribing"
  | "scoring"
  | "rendering"
  | "done"
  | "failed";

export type GeneratedFiles = {
  midi?: string | null;
  musicxml?: string | null;
  pdf?: string | null;
  svg?: string | null;
};

export type Job = {
  job_id: string;
  original_filename: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  error?: string | null;
  files: GeneratedFiles;
  download_urls: GeneratedFiles;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export async function uploadAudio(file: File): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData
  });

  return parseJsonResponse<Job>(response);
}

export async function fetchJob(jobId: string): Promise<Job> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  return parseJsonResponse<Job>(response);
}

export function apiUrl(path?: string | null): string | null {
  if (!path) {
    return null;
  }

  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  const base = API_BASE.endsWith("/api") ? API_BASE.slice(0, -4) : "";
  return `${base}${path}`;
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.detail || "Request failed.";
    throw new Error(message);
  }
  return payload as T;
}

