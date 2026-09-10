import assert from "node:assert/strict";
import test from "node:test";

import { validateAudioUpload } from "../src/uploadValidation.ts";

const megabyte = 1024 * 1024;

for (const size of [1, 26 * megabyte, 1024 * megabyte + 1]) {
  test(`zero limit accepts a non-empty MP3 of ${size} bytes`, () => {
    // Use file metadata to cover large recordings without allocating their audio.
    assert.equal(validateAudioUpload({ name: "long recording.mp3", size }, 0), null);
  });
}

for (const limit of [0, 25]) {
  test(`empty recordings are rejected with an upload limit of ${limit} MB`, () => {
    assert.equal(validateAudioUpload({ name: "empty.mp3", size: 0 }, limit),
      "Choose a non-empty audio recording.");
  });
}

test("a positive limit rejects a file one byte too large", () => {
  assert.equal(validateAudioUpload({ name: "song.mp3", size: 25 * megabyte + 1 }, 25),
    "Choose a file up to 25 MB.");
});

for (const size of [25 * megabyte - 1, 25 * megabyte]) {
  test(`a positive limit accepts a file of ${size} bytes at or below its cap`, () => {
    assert.equal(validateAudioUpload({ name: "song.mp3", size }, 25), null);
  });
}

for (const type of ["audio/mpeg", "application/octet-stream", ""]) {
  test(`uppercase MP3 uploads accept MIME type ${type || "unspecified"}`, () => {
    const file = new File(["MP3 fixture"], "RECORDING.MP3", { type });
    assert.equal(validateAudioUpload(file, 0), null);
  });
}

for (const name of ["recording", "recording.exe", "recording.mp3.exe"]) {
  test(`unsupported filename ${name} reports supported audio formats`, () => {
    assert.equal(validateAudioUpload({ name, size: 100 }, 0),
      "Choose a WAV, MP3, FLAC, OGG, M4A, AAC, or AIFF recording.");
  });
}
