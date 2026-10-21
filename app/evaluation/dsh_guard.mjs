// Loaded before the pinned dsh CLI. All supported inference fetches are admitted
// and reconciled by the shared Python budget service before the CLI sees output.
// This is an adapter for a trusted CLI, not an OS sandbox for arbitrary plugins.
const originalFetch = globalThis.fetch.bind(globalThis);
const bridge = process.env.RESEARCH_BUDGET_BRIDGE;
const token = process.env.RESEARCH_BUDGET_TOKEN;
if (!bridge || !token) throw new Error("research_budget_bridge_missing");

async function control(path, data) {
  const response = await originalFetch(`${bridge}/${path}`, {
    method: "POST", redirect: "error",
    headers: {"Content-Type": "application/json", "Authorization": `Bearer ${token}`},
    body: JSON.stringify(data), signal: AbortSignal.timeout(10000),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.code || "research_budget_denied");
  return body;
}

async function boundedText(response) {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > 2 * 1024 * 1024) throw new Error("research_response_limit");
      chunks.push(Buffer.from(value));
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  return Buffer.concat(chunks).toString("utf8");
}

globalThis.fetch = async function guardedFetch(input, init) {
  const request = new Request(input, init);
  const url = new URL(request.url);
  if (url.origin !== "https://api.deepseek.com" || url.search || url.hash) {
    throw new Error("research_network_destination_blocked");
  }
  if (request.method === "GET" && url.pathname === "/models") {
    await control("catalog", {});
    return originalFetch(request, {redirect: "error", signal: AbortSignal.any([
      request.signal, AbortSignal.timeout(10000),
    ])});
  }
  if (request.method !== "POST" || !["/anthropic/v1/messages", "/v1/messages"].includes(url.pathname)) {
    throw new Error("research_network_operation_blocked");
  }
  const text = await request.clone().text();
  if (Buffer.byteLength(text) > 256 * 1024) throw new Error("research_input_limit");
  const incoming = JSON.parse(text);
  // Helper tasks are text-in/text-out. Do not expose local or server-side tools,
  // file uploads, session-log extensions, or arbitrary plugin request fields.
  const payload = {};
  for (const key of ["model", "messages", "system", "max_tokens", "temperature", "stream"]) {
    if (Object.hasOwn(incoming, key)) payload[key] = incoming[key];
  }
  payload.thinking = {type: "disabled"};
  const admission = await control("reserve", {payload});
  let settled = false;
  try {
    const requestHeaders = new Headers(request.headers);
    requestHeaders.delete("content-length");
    const response = await originalFetch(request.url, {
      method: "POST", redirect: "error", headers: requestHeaders,
      body: JSON.stringify(payload), signal: AbortSignal.any([
        request.signal, AbortSignal.timeout(60000),
      ]),
    });
    // Buffer a bounded response before returning it, so tool calls cannot run
    // before accounting and content checks have completed.
    const body = await boundedText(response);
    const result = await control("complete", {
      id: admission.id, status: response.status,
      content_type: response.headers.get("content-type") || "", body,
    });
    settled = true;
    if (!result.accepted) throw new Error(result.code || "research_cli_response_rejected");
    const headers = new Headers(response.headers);
    for (const name of ["content-length", "content-encoding", "transfer-encoding"]) headers.delete(name);
    return new Response(body, {status: response.status, headers});
  } catch (error) {
    if (!settled) await control("unknown", {id: admission.id}).catch(() => {});
    throw new Error("research_cli_request_failed", {cause: error});
  }
};
