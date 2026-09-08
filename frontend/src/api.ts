export type JobStatus = "queued" | "preprocessing" | "transcribing" | "scoring" | "rendering" | "done" | "failed";

export type GeneratedFiles = {
  midi?: string | null;
  musicxml?: string | null;
  pdf?: string | null;
  svg?: string | null;
  svg_zip?: string | null;
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
  svg_pages: string[];
};

export type Health = {
  ready: boolean;
  limits: { max_upload_mb: number; max_audio_seconds: number; retention_hours: number };
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (init.signal?.aborted) controller.abort();
  init.signal?.addEventListener("abort", abort, { once: true });
  const timeout = window.setTimeout(abort, init.method === "POST" ? 120000 : 15000);
  try {
    const response = await fetch(API_BASE + path, { ...init, signal: controller.signal });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message = typeof payload?.detail === "string"
        ? payload.detail
        : "Request failed. Check the file and try again.";
      throw new ApiError(message, response.status);
    }
    return payload as T;
  } finally {
    window.clearTimeout(timeout);
    init.signal?.removeEventListener("abort", abort);
  }
}

export async function uploadAudio(file: File): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);
  return request<Job>("/upload", { method: "POST", body: formData });
}

export const fetchJob = (jobId: string, signal?: AbortSignal) =>
  request<Job>("/jobs/" + encodeURIComponent(jobId), { signal });
export const fetchHealth = (signal?: AbortSignal) => request<Health>("/health", { signal });

export function apiUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = API_BASE.endsWith("/api") ? API_BASE.slice(0, -4) : "";
  return base + path;
}
