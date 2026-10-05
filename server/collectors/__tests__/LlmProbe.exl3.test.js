/**
 * Unit tests for EXL3 tools/serve_openai.py detection and /health tok/s.
 */
import { test } from "node:test";
import { strict as assert } from "node:assert";
import { LlmProbe } from "../LlmProbe.js";

function jsonRes(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

function notFound() {
  return {
    ok: false,
    status: 404,
    json: async () => ({}),
    text: async () => "",
  };
}

test("_healthLooksLikeExl3: backend field or ok+busy", () => {
  assert.equal(LlmProbe._healthLooksLikeExl3({ backend: "exl3" }), true);
  assert.equal(LlmProbe._healthLooksLikeExl3({ ok: true, busy: false }), true);
  assert.equal(LlmProbe._healthLooksLikeExl3({ status: "ok" }), false);
  assert.equal(LlmProbe._healthLooksLikeExl3(null), false);
});

test("_detectServerType: owned_by exl3 → exl3", async () => {
  const probe = new LlmProbe({ lanIp: "10.0.0.1" }, 8888);
  probe._fetch = async (url) => {
    const u = String(url);
    if (u.endsWith("/slots")) return notFound();
    if (u.endsWith("/v1/models")) {
      return jsonRes({
        data: [{ id: "qwen3.8-27b-exl3-3.5bpw-wm", owned_by: "exl3" }],
      });
    }
    return notFound();
  };
  await probe._detectServerType();
  assert.equal(probe.serverIsOpenAI, true);
  assert.equal(probe.backendType, "exl3");
});

test("_detectServerType: OpenAI models + /health ok+busy → exl3 (not vllm)", async () => {
  const probe = new LlmProbe({ lanIp: "10.0.0.1" }, 8888);
  probe._fetch = async (url) => {
    const u = String(url);
    if (u.endsWith("/slots")) return notFound();
    if (u.endsWith("/v1/models")) {
      return jsonRes({
        data: [{ id: "qwen3.8-27b-exl3-3.5bpw-wm", owned_by: "local" }],
      });
    }
    if (u.endsWith("/metrics")) return notFound();
    if (u.endsWith("/get_server_info") || u.endsWith("/server_info")) return notFound();
    if (u.endsWith("/health")) {
      return jsonRes({
        ok: true,
        busy: true,
        backend: "exl3",
        prompt_tokens_total: 10,
        completion_tokens_total: 20,
      });
    }
    return notFound();
  };
  await probe._detectServerType();
  assert.equal(probe.backendType, "exl3");
});

test("_applyExl3Health: counter diffs → tok/s; idle → 0", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe._applyExl3Health(
    {
      ok: true,
      busy: true,
      backend: "exl3",
      prompt_tokens_total: 100,
      completion_tokens_total: 50,
      context_length: 65536,
    },
    2
  );
  assert.equal(probe.totalOutputTokens, 50);
  assert.equal(probe.contextLength, 65536);
  assert.equal(probe.slotsActive, 1);

  probe._applyExl3Health(
    {
      ok: true,
      busy: false,
      backend: "exl3",
      prompt_tokens_total: 100,
      completion_tokens_total: 50,
      context_length: 65536,
    },
    2
  );
  assert.equal(probe.generationTps, 0);
  assert.equal(probe.prefillTps, 0);
  assert.equal(probe.slotsActive, 0);

  probe._applyExl3Health(
    {
      ok: true,
      busy: true,
      backend: "exl3",
      prompt_tokens_total: 180,
      completion_tokens_total: 90,
      context_length: 65536,
    },
    2
  );
  assert.equal(probe.generationTps, 20);
  assert.equal(probe.prefillTps, 40);
});

test("probe: exl3 path does not mislabel as vllm", async () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.serverIsOpenAI = true;
  probe.backendType = "exl3";
  probe.authOpen = true;
  probe._lastDetectAt = Date.now();
  probe.lastProbeTime = Date.now() - 2000;
  probe.lastTokenCounts = { input: 100, output: 50 };
  probe._fetch = async (url) => {
    const u = String(url);
    if (u.endsWith("/v1/models")) {
      return jsonRes({
        data: [
          {
            id: "qwen3.8-27b-exl3-3.5bpw-wm",
            owned_by: "exl3",
            max_model_len: 65536,
          },
        ],
      });
    }
    if (u.endsWith("/health")) {
      return jsonRes({
        ok: true,
        busy: true,
        backend: "exl3",
        prompt_tokens_total: 140,
        completion_tokens_total: 90,
        context_length: 65536,
      });
    }
    return notFound();
  };
  const snap = await probe.probe();
  assert.equal(snap.backend, "exl3");
  assert.equal(snap.generationTps, 20);
  assert.equal(snap.prefillTps, 20);
  assert.equal(snap.available, true);
});

// ─── TensorFold /health + /metrics tiles ──────────────────────────────────

test("_applyExl3Health: TensorFold /health fills kv pool, prefix cache and MTP tiles", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe._applyExl3Health(
    {
      ok: true,
      backend: "tensorfold",
      busy: true,
      requests_running: 1,
      prompt_tokens_total: 20000,
      completion_tokens_total: 500,
      cached_tokens_total: 19000,
      drafted_total: 1000,
      accepted_total: 620,
      pool_tokens: 2500608,
      pool_free_tokens: 1536000,
      streams: { decoding: 1, prefilling: 0, max: 4 },
      context_length: 1048576,
    },
    1
  );
  // cached / (cached + uncached) = 19000 / 20000
  assert.equal(probe.prefixCacheHitRate, 0.95);
  // (pool_tokens - pool_free_tokens) / pool_tokens
  assert.equal(probe.kvCacheUsage, 0.3857);
  // accepted / drafted
  assert.equal(probe.mtpAcceptanceRate, 0.62);
  // first sample only seeds the split baseline
  assert.equal(probe.cachedPrefillTps, 0);
  assert.equal(probe.uncachedPrefillTps, 0);
  // no TensorFold source: stays null rather than a fabricated number
  assert.equal(probe.ttftP95Seconds, null);
  assert.equal(probe.requestsWaiting, null);

  // Second sample moves the cached/uncached prefill split.
  probe._applyExl3Health(
    {
      ok: true,
      backend: "tensorfold",
      busy: true,
      requests_running: 1,
      prompt_tokens_total: 22000,
      completion_tokens_total: 600,
      cached_tokens_total: 21000,
      drafted_total: 1100,
      accepted_total: 682,
      pool_tokens: 2500608,
      pool_free_tokens: 1536000,
      streams: { decoding: 1, prefilling: 0, max: 4 },
      context_length: 1048576,
    },
    1
  );
  assert.equal(probe.cachedPrefillTps, 2000);
  assert.equal(probe.uncachedPrefillTps, 0);
  assert.equal(probe.prefixCacheHitRate, 0.9545);
});

test("_applyTensorfoldMetrics: TTFT / request-latency p95 and waiting queue", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  const body = [
    "tensorfold:requests_waiting 2",
    'tensorfold:time_to_first_token_seconds_bucket{le="0.5"} 10',
    'tensorfold:time_to_first_token_seconds_bucket{le="1.0"} 96',
    'tensorfold:time_to_first_token_seconds_bucket{le="+Inf"} 100',
    "tensorfold:time_to_first_token_seconds_count 100",
    'tensorfold:request_latency_seconds_bucket{le="2"} 50',
    'tensorfold:request_latency_seconds_bucket{le="4"} 100',
    'tensorfold:request_latency_seconds_bucket{le="+Inf"} 100',
    "tensorfold:request_latency_seconds_count 100",
  ].join("\n");
  probe._applyTensorfoldMetrics(body);
  assert.equal(probe.requestsWaiting, 2);
  // p95 target 95 inside le=1.0: 0.5 + 0.5*((95-10)/(96-10)) = 0.9942
  assert.equal(probe.ttftP95Seconds, 0.994);
  // p95 target 95 inside le=4: 2 + 2*((95-50)/(100-50)) = 3.8
  assert.equal(probe.e2eP95Seconds, 3.8);
});

test("_applyTensorfoldMetrics: empty body clears histograms but not an absent queue series", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8888);
  probe.requestsWaiting = 5;
  probe._applyTensorfoldMetrics("");
  assert.equal(probe.ttftP95Seconds, null);
  assert.equal(probe.e2eP95Seconds, null);
  assert.equal(probe.requestsWaiting, 5, "a missing requests_waiting series must not zero the queue");
});
