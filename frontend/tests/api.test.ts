import assert from "node:assert/strict";
import test from "node:test";
import { ApiError, apiUrl, fetchExample, fetchHealth, fetchJob } from "../src/api.ts";

test("API errors retain their status and server-provided explanation", async t => {
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({ detail: "This job has expired." }), { status: 404 }));
  await assert.rejects(fetchJob("expired"), error => error instanceof ApiError
    && error.status === 404 && error.message === "This job has expired.");
});

test("successful HTTP responses with unreadable JSON fail clearly", async t => {
  for (const body of ["<!doctype html>", "null", "42", '"unexpected"']) {
    t.mock.method(globalThis, "fetch", async () => new Response(body));
    await assert.rejects(fetchHealth(), /unreadable response/);
  }
});

test("incomplete service responses cannot reach the upload form", async t => {
  for (const payload of [{}, { ready: true }, { ready: true, limits: {} },
    { ready: true, limits: { max_upload_mb: -1, max_audio_seconds: 0, retention_hours: 24 } }]) {
    t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify(payload)));
    await assert.rejects(fetchHealth(), /invalid service response/);
  }
  const healthy = { ready: true, limits: { max_upload_mb: 0, max_audio_seconds: 0, retention_hours: 24 } };
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify(healthy)));
  assert.deepEqual(await fetchHealth(), healthy);
});

test("malformed score responses cannot crash downloads, previews, or warning lists", async t => {
  const job = { job_id: "job", original_filename: "recording.mp3", created_at: "2026-10-11T00:00:00Z",
    status: "done", progress: 100, download_urls: { pdf: "/api/downloads/job/pdf" }, svg_pages: [], analysis: { warnings: [] } };
  for (const payload of [{}, { ...job, download_urls: null }, { ...job, download_urls: { pdf: 42 } },
    { ...job, svg_pages: "score.svg" }, { ...job, status: "unknown" }, { ...job, progress: null },
    { ...job, analysis: { warnings: "warning" } }, { ...job, error: { message: "bad" } }]) {
    t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify(payload)));
    await assert.rejects(fetchJob("job"), /invalid score response/);
  }
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify(job)));
  assert.deepEqual(await fetchJob("job"), job);
});

test("job requests encode IDs and propagate cancellation", async t => {
  const controller = new AbortController();
  t.mock.method(globalThis, "fetch", async (url, init) => {
    assert.equal(url, "/api/jobs/a%2Fb");
    return new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(new DOMException("Cancelled", "AbortError")), { once: true });
    });
  });
  const pending = fetchJob("a/b", controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: "AbortError" });
});

test("example download times out even when its response body stalls", async t => {
  let expire: (() => void) | undefined;
  let cleared = false;
  t.mock.method(globalThis, "setTimeout", (callback, delay) => {
    assert.equal(delay, 15000);
    expire = callback;
    return 123;
  });
  t.mock.method(globalThis, "clearTimeout", id => { assert.equal(id, 123); cleared = true; });
  t.mock.method(globalThis, "fetch", async (_url, init) => ({
    ok: true,
    blob: () => new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(new DOMException("Timed out", "AbortError")), { once: true });
      expire!();
    })
  }));
  await assert.rejects(fetchExample(), { name: "AbortError" });
  assert.equal(cleared, true);
});

test("example download produces a file and preview URLs preserve remote hosts", async t => {
  t.mock.method(globalThis, "fetch", async url => {
    assert.equal(url, "/api/example");
    return new Response(new Uint8Array([82, 73, 70, 70]));
  });
  const file = await fetchExample();
  assert.equal(file.name, "Piano study.wav");
  assert.equal(file.type, "audio/wav");
  assert.equal(file.size, 4);
  assert.equal(apiUrl(null), null);
  assert.equal(apiUrl("/api/downloads/job/pdf"), "/api/downloads/job/pdf");
  assert.equal(apiUrl("https://example.com/score.svg"), "https://example.com/score.svg");
});
