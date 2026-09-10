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
  progress: number;
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
  const timeout = window.setTimeout(abort, 15000);
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

export function uploadAudio(file: File, onProgress?: (percent: number) => void): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API_BASE + "/upload");
    // Large recordings and slow connections have no overall upload deadline.
    xhr.timeout = 0;
    xhr.responseType = "json";
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) onProgress?.(Math.round(event.loaded / event.total * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response?.job_id) {
        resolve(xhr.response as Job);
      } else {
        const detail = xhr.response?.detail;
        reject(new ApiError(typeof detail === "string" ? detail : "Upload failed. Check the file and try again.", xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError("Upload interrupted. Check your connection and try again.", 0));
    xhr.onabort = () => reject(new ApiError("Upload cancelled.", 0));
    xhr.send(formData);
  });
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
