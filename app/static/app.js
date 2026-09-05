const uploadForm = document.querySelector("#upload-form");
const documentInput = document.querySelector("#document-input");
const documentList = document.querySelector("#document-list");
const researchForm = document.querySelector("#research-form");
const questionInput = document.querySelector("#question-input");
const researchDocuments = document.querySelector("#research-documents");
const timeline = document.querySelector("#timeline");
const report = document.querySelector("#report");
const statusMessage = document.querySelector("#status-message");
const retryResearchButton = document.querySelector("#retry-research");
const citationDialog = document.querySelector("#citation-dialog");
const citationContent = document.querySelector("#citation-content");
const citationClose = document.querySelector("#citation-close");

let activeEventSource = null;
let activeRunId = null;
let currentCitations = [];
let documents = [];

async function apiFetch(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ error: { message: response.statusText } }));
    throw new Error(body.error?.message || "请求失败");
  }
  return response.status === 204 ? null : response.json();
}

function appendText(parent, tagName, text, className = "") {
  const node = document.createElement(tagName);
  node.textContent = text;
  if (className) {
    node.className = className;
  }
  parent.appendChild(node);
  return node;
}

function clearChildren(parent) {
  while (parent.firstChild) {
    parent.removeChild(parent.firstChild);
  }
}

function showStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("is-error", isError);
}

function closeEventStream() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  activeRunId = null;
}

function selectedDocumentIds() {
  return Array.from(
    researchDocuments.querySelectorAll('input[type="checkbox"]:checked'),
    (input) => input.value,
  );
}

function documentStatus(documentRecord) {
  const pages = documentRecord.page_count === 1 ? "1 页" : `${documentRecord.page_count} 页`;
  return `${documentRecord.status} · ${pages}`;
}

function renderDocuments() {
  const selectedIds = new Set(selectedDocumentIds());
  clearChildren(documentList);
  clearChildren(researchDocuments);

  if (documents.length === 0) {
    appendText(documentList, "p", "尚未上传资料。", "empty-state");
    appendText(researchDocuments, "p", "上传资料后可在这里选择。", "empty-state");
    return;
  }

  for (const documentRecord of documents) {
    const item = document.createElement("div");
    item.className = "document-item";
    const description = document.createElement("div");
    appendText(description, "strong", documentRecord.filename);
    appendText(description, "p", documentStatus(documentRecord));
    item.appendChild(description);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.dataset.documentId = documentRecord.id;
    deleteButton.textContent = "删除";
    deleteButton.addEventListener("click", deleteDocument);
    item.appendChild(deleteButton);
    documentList.appendChild(item);

    const option = document.createElement("label");
    option.className = "research-document-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "document_ids";
    checkbox.value = documentRecord.id;
    checkbox.checked = selectedIds.has(documentRecord.id);
    option.appendChild(checkbox);
    appendText(option, "span", documentRecord.filename);
    researchDocuments.appendChild(option);
  }
}

async function refreshDocuments() {
  try {
    documents = await apiFetch("/api/documents");
    renderDocuments();
  } catch (error) {
    showStatus(`无法加载资料：${error.message}`, true);
  }
}

async function uploadDocuments(event) {
  event.preventDefault();
  if (documentInput.files.length === 0) {
    showStatus("请先选择至少一个文件。", true);
    documentInput.focus();
    return;
  }

  const formData = new FormData();
  for (const file of documentInput.files) {
    formData.append("files", file);
  }

  try {
    showStatus("正在上传资料…");
    await apiFetch("/api/documents", { method: "POST", body: formData });
    documentInput.value = "";
    await refreshDocuments();
    showStatus("资料上传完成。");
  } catch (error) {
    showStatus(`上传失败：${error.message}`, true);
  }
}

async function deleteDocument(event) {
  const documentId = event.currentTarget.dataset.documentId;
  if (!documentId) {
    return;
  }

  try {
    await apiFetch(`/api/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    });
    await refreshDocuments();
    showStatus("资料已删除。");
  } catch (error) {
    showStatus(`删除失败：${error.message}`, true);
  }
}

function resetResearchOutput() {
  closeEventStream();
  clearChildren(timeline);
  clearChildren(report);
  currentCitations = [];
  retryResearchButton.hidden = true;
}

async function startResearch(event) {
  if (event) {
    event.preventDefault();
  }
  const question = questionInput.value.trim();
  const documentIds = selectedDocumentIds();
  if (question.length < 3) {
    showStatus("研究问题至少需要 3 个字符。", true);
    questionInput.focus();
    return;
  }
  if (documentIds.length === 0) {
    showStatus("请至少选择一份研究资料。", true);
    researchDocuments.focus();
    return;
  }

  resetResearchOutput();
  try {
    showStatus("已创建研究任务，正在连接进度流…");
    const created = await apiFetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_ids: documentIds }),
    });
    openEventStream(created.run_id);
  } catch (error) {
    showStatus(`创建研究失败：${error.message}`, true);
    retryResearchButton.hidden = false;
  }
}

function workflowMessage(event) {
  const payload = event.payload || {};
  const detail =
    payload.message ||
    payload.error_message ||
    payload.reason ||
    payload.error_code ||
    payload.status ||
    "";
  return detail ? `${event.stage}：${event.event_type}（${detail}）` : `${event.stage}：${event.event_type}`;
}

function renderWorkflowEvent(event) {
  const item = document.createElement("li");
  item.className = `timeline-item is-${event.event_type}`;
  appendText(item, "strong", workflowMessage(event));
  if (event.event_type === "finished" && event.payload?.evidence_sufficient === false) {
    appendText(item, "p", "证据不足：报告保留了当前限制。", "evidence-warning");
  }
  timeline.appendChild(item);

  if (event.event_type === "retry") {
    showStatus("研究正在根据审查意见重试…");
  } else if (event.event_type === "failed") {
    showStatus("研究执行失败，可检查进度后重试。", true);
    retryResearchButton.hidden = false;
  }
}

function openEventStream(runId) {
  closeEventStream();
  activeRunId = runId;
  const source = new EventSource(`/api/research/${encodeURIComponent(runId)}/events`);
  activeEventSource = source;
  source.addEventListener("workflow", async (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch (_error) {
      showStatus("收到无法读取的进度事件。", true);
      return;
    }
    renderWorkflowEvent(event);
    if (event.event_type === "finished" || event.event_type === "failed") {
      source.close();
      if (activeEventSource === source) {
        activeEventSource = null;
      }
      await fetchResearchResult(runId);
    }
  });
  source.onerror = () => {
    if (activeEventSource === source) {
      showStatus("进度连接暂时中断，正在安全地重新连接…", true);
    }
  };
}

async function fetchResearchResult(runId) {
  try {
    const result = await apiFetch(`/api/research/${encodeURIComponent(runId)}`);
    if (activeRunId === runId || activeRunId === null) {
      renderResearchResult(result);
    }
  } catch (error) {
    showStatus(`无法读取研究结果：${error.message}`, true);
    retryResearchButton.hidden = false;
  }
}

function renderResearchResult(result) {
  clearChildren(report);
  if (result.run.status === "failed") {
    appendText(report, "h2", "研究失败");
    appendText(
      report,
      "p",
      result.run.error_message || result.run.error_code || "研究任务未能完成。",
      "evidence-warning",
    );
    retryResearchButton.hidden = false;
    return;
  }
  if (result.run.status === "completed" && result.report) {
    renderReport(result.report, result.run);
    showStatus(result.run.evidence_sufficient === false ? "研究已完成，但证据不足。" : "研究已完成。");
  }
}

function addCitationButtons(parent, evidenceIds) {
  for (const evidenceId of evidenceIds) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation-button";
    button.dataset.evidenceId = evidenceId;
    button.textContent = `查看引用 ${evidenceId}`;
    button.addEventListener("click", openCitation);
    parent.appendChild(button);
  }
}

function renderReport(researchReport, run) {
  currentCitations = Array.isArray(researchReport.citations) ? researchReport.citations : [];
  if (run.evidence_sufficient === false || researchReport.evidence_sufficient === false) {
    appendText(report, "p", "证据不足", "evidence-warning");
  }
  appendText(report, "h2", researchReport.title || "研究报告");
  appendText(report, "p", researchReport.summary || "未提供摘要。", "report-card");

  for (const finding of researchReport.findings || []) {
    const card = document.createElement("section");
    card.className = "report-card";
    appendText(card, "h3", finding.heading || "研究发现");
    appendText(card, "p", finding.narrative || "未提供说明。");
    addCitationButtons(card, finding.evidence_ids || []);
    report.appendChild(card);
  }

  const limitations = researchReport.limitations || [];
  if (limitations.length > 0) {
    const limitationCard = document.createElement("section");
    limitationCard.className = "report-card";
    appendText(limitationCard, "h3", "限制");
    const list = document.createElement("ul");
    list.className = "limitations-list";
    for (const limitation of limitations) {
      appendText(list, "li", limitation);
    }
    limitationCard.appendChild(list);
    report.appendChild(limitationCard);
  }

  if (currentCitations.length > 0) {
    const citationsCard = document.createElement("section");
    citationsCard.className = "report-card";
    appendText(citationsCard, "h3", "引用证据");
    addCitationButtons(citationsCard, currentCitations.map((citation) => citation.evidence_id));
    report.appendChild(citationsCard);
  }
}

function openCitation(event) {
  const evidenceId = event.currentTarget.dataset.evidenceId;
  const citation = currentCitations.find((item) => item.evidence_id === evidenceId);
  if (!citation) {
    showStatus("找不到该引用的证据详情。", true);
    return;
  }
  clearChildren(citationContent);
  appendText(citationContent, "p", `证据 ID：${citation.evidence_id}`);
  appendText(citationContent, "p", `文件：${citation.filename}`);
  const location = citation.page_number === null || citation.page_number === undefined
    ? `片段：${citation.chunk_index}`
    : `页码：${citation.page_number}；片段：${citation.chunk_index}`;
  appendText(citationContent, "p", location);
  appendText(citationContent, "p", citation.excerpt, "citation-excerpt");
  if (typeof citationDialog.showModal === "function") {
    citationDialog.showModal();
  } else {
    citationDialog.setAttribute("open", "");
  }
  citationClose.focus();
}

function closeCitation() {
  if (typeof citationDialog.close === "function") {
    citationDialog.close();
  } else {
    citationDialog.removeAttribute("open");
  }
}

uploadForm.addEventListener("submit", uploadDocuments);
researchForm.addEventListener("submit", startResearch);
retryResearchButton.addEventListener("click", startResearch);
citationClose.addEventListener("click", closeCitation);
refreshDocuments();
