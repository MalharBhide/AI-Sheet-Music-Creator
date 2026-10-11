export type JobStatus = "queued" | "preprocessing" | "transcribing" | "scoring" | "rendering" | "done" | "failed";

export type GeneratedFiles = {
  midi?: string | null;
  musicxml?: string | null;
  pdf?: string | null;
  svg?: string | null;
  svg_zip?: string | null;
  playback?: string | null;
};

export type ScoreOptions = {
  transcription_mode: "piano" | "full_mix" | "melody";
  detail: "balanced" | "detailed";
  tempo_bpm?: number;
  time_signature: "4/4" | "3/4" | "6/8";
  grid: "eighth" | "sixteenth";
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
  analysis?: {
    tempo_bpm?: number;
    key_signature?: string;
    engine?: string;
    note_count?: number;
    warnings?: string[];
    transcription_mode?: string;
  } | null;
};

export type Health = {
  ready: boolean;
  limits: { max_upload_mb: number; max_audio_seconds: number; retention_hours: number };
};

const API_BASE = (import.meta.env?.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

async function request<T>(path: string, read: (response: Response) => Promise<T>, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (init.signal?.aborted) controller.abort();
  init.signal?.addEventListener("abort", abort, { once: true });
  const timeout = globalThis.setTimeout(abort, 15000);
  try {
    const response = await fetch(API_BASE + path, { ...init, signal: controller.signal });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const message = typeof payload?.detail === "string"
        ? payload.detail
        : "Request failed. Check the file and try again.";
      throw new ApiError(message, response.status);
    }
    return await read(response);
  } finally {
    globalThis.clearTimeout(timeout);
    init.signal?.removeEventListener("abort", abort);
  }
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new ApiError("The server returned an unreadable response. Please try again.", response.status);
  }
  return payload as T;
}

function isJob(value: unknown): value is Job {
  if (!value || typeof value !== "object") return false;
  const job = value as Job;
  const urls = job.download_urls;
  return typeof job.job_id === "string" && !!job.job_id
    && typeof job.original_filename === "string" && typeof job.created_at === "string"
    && ["queued", "preprocessing", "transcribing", "scoring", "rendering", "done", "failed"].includes(job.status)
    && Number.isFinite(job.progress) && job.progress >= 0 && job.progress <= 100
    && !!urls && typeof urls === "object" && !Array.isArray(urls)
    && Object.values(urls).every(url => url == null || typeof url === "string")
    && Array.isArray(job.svg_pages) && job.svg_pages.every(url => typeof url === "string")
    && (job.error == null || typeof job.error === "string")
    && (job.analysis == null || (typeof job.analysis === "object" && !Array.isArray(job.analysis)
      && (job.analysis.warnings == null || (Array.isArray(job.analysis.warnings)
        && job.analysis.warnings.every(warning => typeof warning === "string")))));
}

async function readJob(response: Response): Promise<Job> {
  const payload = await readJson<unknown>(response);
  if (!isJob(payload)) throw new ApiError("The server returned an invalid score response. Please try again.", response.status);
  return payload;
}

async function readHealth(response: Response): Promise<Health> {
  const payload = await readJson<Health>(response);
  if (typeof payload.ready !== "boolean" || !payload.limits
    || ![payload.limits.max_upload_mb, payload.limits.max_audio_seconds, payload.limits.retention_hours]
      .every(value => Number.isFinite(value) && value >= 0)) {
    throw new ApiError("The server returned an invalid service response. Please reconnect.", response.status);
  }
  return payload;
}

export function uploadAudio(file: File, onProgress?: (percent: number) => void, options?: ScoreOptions): Promise<Job> {
  const formData = new FormData();
  formData.append("file", file);
  if (options) Object.entries(options).forEach(([key, value]) => {
    if (value !== undefined) formData.append(key, String(value));
  });
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
      if (xhr.status >= 200 && xhr.status < 300 && isJob(xhr.response)) {
        resolve(xhr.response);
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
  request("/jobs/" + encodeURIComponent(jobId), readJob, { signal });
export const fetchHealth = (signal?: AbortSignal) => request("/health", readHealth, { signal });

export function fetchExample(signal?: AbortSignal): Promise<File> {
  return request("/example", async response =>
    new File([await response.blob()], "Piano study.wav", { type: "audio/wav" }), { signal });
}

export function apiUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const base = API_BASE.endsWith("/api") ? API_BASE.slice(0, -4) : "";
  return base + path;
}
