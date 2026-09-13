import assert from "node:assert/strict";
import test from "node:test";
import { normalizePlayback, scorePositionAt, scorePositionNear } from "../src/playback.ts";

const positions = [
  { time: 0, page: 0, x: .2, y: .1, height: .1 },
  { time: .5, page: 0, x: .5, y: .1, height: .1 },
  { time: 3, page: 0, x: .2, y: .5, height: .1 },
  { time: 5, page: 1, x: .25, y: .15, height: .1 },
  { time: 6, page: 1, x: .8, y: .15, height: .1 },
];
test("following uses exact segment times through held notes, rests, and page changes", () => {
  assert.equal(scorePositionAt(positions, -.1), null);
  assert.equal(scorePositionAt(positions, .49), positions[0]);
  assert.equal(scorePositionAt(positions, 2.99), positions[1]);
  assert.equal(scorePositionAt(positions, 3), positions[2]);
  assert.equal(scorePositionAt(positions, 5), positions[3]);
  assert.equal(scorePositionAt(positions, 20), positions[4]);
  assert.equal(scorePositionAt(positions, .2), positions[0], "seeking backwards restores the first page");
  assert.equal(scorePositionAt([], 0), null);
});
test("clicks select the closest musical onset on the clicked page and staff system", () => {
  assert.equal(scorePositionNear(positions, 0, .49, .14), positions[1]);
  assert.equal(scorePositionNear(positions, 0, .49, .56), positions[2]);
  assert.equal(scorePositionNear(positions, 1, .24, .18), positions[3]);
  assert.equal(scorePositionNear(positions, 1, .79, .18), positions[4]);
  assert.equal(scorePositionNear(positions, 2, .2, .2), null);
  assert.equal(scorePositionNear(positions, 0, NaN, .2), null);
});
test("page following fails gracefully for old scores or corrupt position maps", () => {
  const payload = { duration: 10, tempo_bpm: 120, notes: [] };
  assert.deepEqual(normalizePlayback(payload).positions, []);
  assert.deepEqual(normalizePlayback({ ...payload, positions }).positions, positions);
  for (const invalid of [{ ...positions[0], x: 5 }, { ...positions[0], page: -1 }, { ...positions[0], time: Infinity }, { ...positions[0], y: .99 }]) {
    assert.deepEqual(normalizePlayback({ ...payload, positions: [invalid] }).positions, []);
  }
  assert.deepEqual(normalizePlayback({ ...payload, positions: [...positions].reverse() }).positions, []);
});
