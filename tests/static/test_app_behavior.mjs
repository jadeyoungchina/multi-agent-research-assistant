import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function response(value, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "request failed",
    json: async () => value,
  };
}

class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  add(value) {
    this.values.add(value);
  }

  toggle(value, force) {
    if (force) {
      this.values.add(value);
    } else {
      this.values.delete(value);
    }
  }
}

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.value = "";
    this.files = [];
    this.hidden = false;
    this.checked = false;
    this._text = "";
  }

  get firstChild() {
    return this.children[0] ?? null;
  }

  get textContent() {
    return this._text + this.children.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    this.children.splice(this.children.indexOf(child), 1);
    return child;
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  querySelectorAll() {
    return this.checkboxes ?? [];
  }

  focus() {}

  setAttribute() {}

  removeAttribute() {}
}

class FakeEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeEventSource.OPEN;
    this.listeners = new Map();
    this.closed = false;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  close() {
    this.closed = true;
    this.readyState = FakeEventSource.CLOSED;
  }

  async emitWorkflow(event) {
    await this.listeners.get("workflow")({ data: JSON.stringify(event) });
  }
}

async function flush() {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve();
  }
}

async function loadDashboard(fetch) {
  FakeEventSource.instances = [];
  const elements = new Map();
  for (const id of [
    "upload-form",
    "document-input",
    "document-list",
    "research-form",
    "question-input",
    "research-documents",
    "timeline",
    "report",
    "status-message",
    "retry-research",
    "citation-dialog",
    "citation-content",
    "citation-close",
  ]) {
    elements.set(`#${id}`, new FakeElement());
  }
  const context = vm.createContext({
    document: {
      querySelector: (selector) => elements.get(selector),
      createElement: (tagName) => new FakeElement(tagName),
    },
    EventSource: FakeEventSource,
    FormData,
    Blob,
    File,
    fetch,
    console,
    JSON,
    Array,
    Set,
    Promise,
    encodeURIComponent,
  });
  const script = await readFile("app/static/app.js", "utf8");
  vm.runInContext(
    `${script}\nglobalThis.dashboard = { startResearch, refreshDocuments };`,
    context,
  );
  await flush();
  return { dashboard: context.dashboard, elements };
}

async function testNewerResearchSubmissionOwnsTheUi() {
  const posts = [deferred(), deferred()];
  const getResults = [deferred()];
  let postIndex = 0;
  const { dashboard, elements } = await loadDashboard((path, options = {}) => {
    if (path === "/api/documents") {
      return Promise.resolve(response([]));
    }
    if (path === "/api/research" && options.method === "POST") {
      return posts[postIndex++].promise;
    }
    if (path === "/api/research/old") {
      return getResults[0].promise;
    }
    throw new Error(`unexpected request: ${path}`);
  });
  elements.get("#research-documents").checkboxes = [{ checked: true, value: "doc" }];
  elements.get("#question-input").value = "older question";
  const olderSubmission = dashboard.startResearch();
  elements.get("#question-input").value = "newer question";
  const newerSubmission = dashboard.startResearch();

  posts[1].resolve(response({ run_id: "new", status: "queued" }, 202));
  await newerSubmission;
  const newSource = FakeEventSource.instances[0];
  posts[0].resolve(response({ run_id: "old", status: "queued" }, 202));
  await olderSubmission;

  assert.equal(newSource.closed, false, "an older POST must not close the newer stream");
  assert.equal(FakeEventSource.instances.length, 1, "an older POST must not open a stream");

  const oldResult = loadDashboard((path, options = {}) => {
    if (path === "/api/documents") {
      return Promise.resolve(response([]));
    }
    if (path === "/api/research" && options.method === "POST") {
      return Promise.resolve(response({ run_id: "old", status: "queued" }, 202));
    }
    if (path === "/api/research/old") {
      return getResults[0].promise;
    }
    if (path === "/api/research/new" && options.method === "POST") {
      return Promise.resolve(response({ run_id: "new", status: "queued" }, 202));
    }
    throw new Error(`unexpected request: ${path}`);
  });
  const old = await oldResult;
  old.elements.get("#research-documents").checkboxes = [{ checked: true, value: "doc" }];
  old.elements.get("#question-input").value = "old question";
  await old.dashboard.startResearch();
  const oldFinished = FakeEventSource.instances[0].emitWorkflow({
    stage: "writer",
    event_type: "finished",
    payload: {},
  });
  old.elements.get("#question-input").value = "new question";
  const replacementSubmission = old.dashboard.startResearch();
  getResults[0].resolve(response({
    run: { status: "completed", evidence_sufficient: true },
    report: { title: "OLD REPORT", summary: "stale", findings: [], limitations: [], citations: [], evidence_sufficient: true },
  }));
  await replacementSubmission;
  await oldFinished;
  await flush();

  assert.equal(old.elements.get("#report").textContent.includes("OLD REPORT"), false, "a stale final GET must not render during a newer submission");
}

async function testClosedEventSourceFetchesFinalStatus() {
  let resultRequests = 0;
  const { dashboard, elements } = await loadDashboard((path, options = {}) => {
    if (path === "/api/documents") {
      return Promise.resolve(response([]));
    }
    if (path === "/api/research" && options.method === "POST") {
      return Promise.resolve(response({ run_id: "closed", status: "queued" }, 202));
    }
    if (path === "/api/research/closed") {
      resultRequests += 1;
      return Promise.resolve(response({
        run: { status: "completed", evidence_sufficient: true },
        report: { title: "Recovered", summary: "final", findings: [], limitations: [], citations: [], evidence_sufficient: true },
      }));
    }
    throw new Error(`unexpected request: ${path}`);
  });
  elements.get("#research-documents").checkboxes = [{ checked: true, value: "doc" }];
  elements.get("#question-input").value = "closed stream";
  await dashboard.startResearch();
  const source = FakeEventSource.instances[0];

  source.readyState = FakeEventSource.CONNECTING;
  source.onerror();
  assert.equal(resultRequests, 0, "native reconnect must not fetch final status");
  assert.match(elements.get("#status-message").textContent, /重新连接/);

  source.readyState = FakeEventSource.CLOSED;
  source.onerror();
  await flush();
  assert.equal(resultRequests, 1, "a closed stream must recover through final status");
  assert.match(elements.get("#report").textContent, /Recovered/);
}

async function testClosedEventSourceWithoutTerminalResultOffersRetry() {
  const { dashboard, elements } = await loadDashboard((path, options = {}) => {
    if (path === "/api/documents") {
      return Promise.resolve(response([]));
    }
    if (path === "/api/research" && options.method === "POST") {
      return Promise.resolve(response({ run_id: "unknown", status: "queued" }, 202));
    }
    if (path === "/api/research/unknown") {
      return Promise.resolve(response({
        run: { status: "running", evidence_sufficient: null },
        report: null,
      }));
    }
    throw new Error(`unexpected request: ${path}`);
  });
  elements.get("#research-documents").checkboxes = [{ checked: true, value: "doc" }];
  elements.get("#question-input").value = "terminal state unavailable";
  await dashboard.startResearch();
  const source = FakeEventSource.instances[0];
  source.readyState = FakeEventSource.CLOSED;
  source.onerror();
  await flush();

  assert.equal(elements.get("#retry-research").hidden, false);
  assert.match(elements.get("#status-message").textContent, /请重试/);
}

async function testNewerDocumentRefreshOwnsTheList() {
  const olderResponse = deferred();
  const newerResponse = deferred();
  let requests = 0;
  const { dashboard, elements } = await loadDashboard((path) => {
    assert.equal(path, "/api/documents");
    if (requests++ === 0) {
      return Promise.resolve(response([]));
    }
    return requests === 2 ? olderResponse.promise : newerResponse.promise;
  });
  const olderRefresh = dashboard.refreshDocuments();
  const newerRefresh = dashboard.refreshDocuments();
  newerResponse.resolve(response([{ id: "new", filename: "new.txt", status: "ready", page_count: 1 }]));
  await flush();
  olderResponse.resolve(response([{ id: "old", filename: "old.txt", status: "ready", page_count: 1 }]));
  await Promise.all([olderRefresh, newerRefresh]);

  assert.match(elements.get("#research-documents").textContent, /new.txt/);
  assert.doesNotMatch(elements.get("#research-documents").textContent, /old.txt/);
}

async function testUploadsNormalizeSupportedMimeTypesAndPreserveContents() {
  let uploaded;
  const { elements } = await loadDashboard((path, options = {}) => {
    assert.equal(path, "/api/documents");
    if (options.method === "POST") {
      uploaded = options.body;
      return Promise.resolve(response({ documents: [] }, 201));
    }
    return Promise.resolve(response([]));
  });
  const cases = [
    ["notes.md", "", "text/markdown", "# Untyped Markdown"],
    ["notes.MARKDOWN", "", "text/markdown", "# Uppercase Markdown"],
    ["generic.md", "application/octet-stream", "text/markdown", "# Generic Markdown"],
    ["typed.md", "text/markdown", "text/markdown", "# Typed Markdown"],
    ["paper.pdf", "application/pdf", "application/pdf", "%PDF-1.7"],
    ["notes.txt", "text/plain", "text/plain", "Plain text"],
    ["untyped.pdf", "", "application/pdf", "%PDF-1.7"],
    ["untyped.txt", "", "text/plain", "Untyped text"],
    ["unsupported.bin", "application/octet-stream", "application/octet-stream", "Unknown"],
  ];
  elements.get("#document-input").files = cases.map(([name, type, , content]) =>
    new File([content], name, { type }),
  );

  await elements.get("#upload-form").listeners.get("submit")({ preventDefault() {} });

  assert.ok(uploaded instanceof FormData, "the submit handler must send multipart form data");
  const files = uploaded.getAll("files");
  assert.equal(files.length, cases.length);
  for (let index = 0; index < cases.length; index += 1) {
    const [name, , expectedType, content] = cases[index];
    assert.equal(files[index].name, name, "normalization must preserve the filename");
    assert.equal(files[index].type, expectedType, `${name} must be sent with its supported MIME type`);
    assert.equal(await files[index].text(), content, "normalization must preserve the content");
  }
}

await testNewerResearchSubmissionOwnsTheUi();
await testClosedEventSourceFetchesFinalStatus();
await testClosedEventSourceWithoutTerminalResultOffersRetry();
await testNewerDocumentRefreshOwnsTheList();
await testUploadsNormalizeSupportedMimeTypesAndPreserveContents();
