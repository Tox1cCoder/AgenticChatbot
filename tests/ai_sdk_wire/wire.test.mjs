// Parses backend-generated SSE fixtures with the pinned AI SDK.
//
// tests/test_ai_sdk_wire_fixtures.py writes the fixtures and manifest.json
// into AI_SDK_FIXTURE_DIR; run it first. `processResponseStream` is used only
// in this harness and stays pinned to the SDK version in package-lock.json.
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { DefaultChatTransport, readUIMessageStream } from 'ai';

async function parseWire(wire) {
  const bytes = new TextEncoder().encode(wire);
  const body = new ReadableStream({
    start(controller) { controller.enqueue(bytes); controller.close(); },
  });
  const chunks = [];
  for await (const chunk of new DefaultChatTransport().processResponseStream(body)) {
    chunks.push(chunk);
  }
  return chunks;
}

// What a vanilla SDK consumer ends with: text parts only, custom data ignored.
async function vanillaText(chunks) {
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  let last;
  for await (const message of readUIMessageStream({ stream })) last = message;
  return (last?.parts ?? [])
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('');
}

function loadManifest() {
  const dir = process.env.AI_SDK_FIXTURE_DIR;
  assert.ok(dir, 'AI_SDK_FIXTURE_DIR is not set; run tests/test_ai_sdk_wire_fixtures.py first');
  const manifestPath = join(dir, 'manifest.json');
  assert.ok(existsSync(manifestPath), `missing fixture manifest: ${manifestPath}`);
  const entries = JSON.parse(readFileSync(manifestPath, 'utf8'));
  assert.ok(Array.isArray(entries) && entries.length > 0, 'fixture manifest is empty');
  return { dir, entries };
}

test('parser rejects a raw heartbeat frame', async () => {
  await assert.rejects(parseWire('data: {"type":"heartbeat"}\n\n'));
});

test('keepalive comments and [DONE] produce no chunks', async () => {
  assert.equal((await parseWire(': keepalive\n\ndata: [DONE]\n\n')).length, 0);
});

const { dir, entries } = loadManifest();

for (const entry of entries) {
  test(`fixture ${entry.file} is ${entry.expect}ed by the SDK parser`, async () => {
    const wire = readFileSync(join(dir, entry.file), 'utf8');
    if (entry.expect === 'reject') {
      await assert.rejects(parseWire(wire));
      return;
    }
    assert.equal(entry.expect, 'accept', `unknown expectation ${entry.expect}`);
    const chunks = await parseWire(wire);
    assert.ok(chunks.length > 0, 'an accepted fixture must carry chunks');
    const seen = new Set(chunks.map((chunk) => chunk.type));
    for (const required of entry.requires ?? []) {
      assert.ok(seen.has(required), `${required} did not survive SDK parsing`);
    }
    if (entry.sdkText !== undefined) {
      // The SDK never applies data-answer-reconcile itself; the client must.
      assert.equal(await vanillaText(chunks), entry.sdkText);
    }
  });
}
