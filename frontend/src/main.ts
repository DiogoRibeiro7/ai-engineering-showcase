import "./styles.css";
import {
  ApiError,
  postQuery,
  streamQuery,
  type AgentAnswer,
  type Citation,
  type StreamMetadata,
} from "./api";

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`Missing element: #${id}`);
  }
  return node as T;
}

const form = el<HTMLFormElement>("query-form");
const questionInput = el<HTMLTextAreaElement>("question");
const streamToggle = el<HTMLInputElement>("stream-toggle");
const submitBtn = el<HTMLButtonElement>("submit-btn");
const statusBox = el<HTMLElement>("status");
const answerPanel = el<HTMLElement>("answer-panel");
const answerBox = el<HTMLElement>("answer");
const actionsBox = el<HTMLElement>("actions");
const metaPanel = el<HTMLElement>("meta-panel");
const metaList = el<HTMLDListElement>("meta");
const sourcesPanel = el<HTMLElement>("sources-panel");
const sourcesList = el<HTMLOListElement>("sources");

function setStatus(message: string, kind: "info" | "error" | "none"): void {
  if (kind === "none") {
    statusBox.hidden = true;
    statusBox.textContent = "";
    return;
  }
  statusBox.hidden = false;
  statusBox.textContent = message;
  statusBox.className = `status status--${kind}`;
}

function clearResults(): void {
  answerPanel.hidden = true;
  metaPanel.hidden = true;
  sourcesPanel.hidden = true;
  answerBox.textContent = "";
  actionsBox.textContent = "";
  metaList.textContent = "";
  sourcesList.textContent = "";
}

function renderActions(actions: string[]): void {
  actionsBox.textContent = "";
  if (actions.length === 0) {
    return;
  }
  const heading = document.createElement("h3");
  heading.className = "actions__title";
  heading.textContent = "Recommended actions";
  const list = document.createElement("ul");
  for (const action of actions) {
    const item = document.createElement("li");
    item.textContent = action;
    list.appendChild(item);
  }
  actionsBox.appendChild(heading);
  actionsBox.appendChild(list);
}

function addMeta(term: string, value: string): void {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  metaList.appendChild(dt);
  metaList.appendChild(dd);
}

interface MetaView {
  provider?: string;
  latencyMs?: number;
  route: string;
  confidence: number;
}

function renderMeta(view: MetaView): void {
  metaList.textContent = "";
  if (view.provider !== undefined) {
    addMeta("Provider", view.provider);
  }
  if (view.latencyMs !== undefined) {
    addMeta("Latency", `${view.latencyMs.toFixed(1)} ms`);
  }
  addMeta("Route", view.route);
  addMeta("Confidence", view.confidence.toFixed(3));
  metaPanel.hidden = false;
}

function renderSources(citations: Citation[]): void {
  sourcesList.textContent = "";
  if (citations.length === 0) {
    sourcesPanel.hidden = true;
    return;
  }
  for (const citation of citations) {
    const item = document.createElement("li");
    item.className = "source";

    const head = document.createElement("div");
    head.className = "source__head";
    const id = document.createElement("span");
    id.className = "source__id";
    id.textContent = `[${citation.citation_id}] ${citation.document_id}`;
    const tag = document.createElement("span");
    tag.className = "source__tag";
    tag.textContent = `${citation.source} · score ${citation.score.toFixed(3)}`;
    head.appendChild(id);
    head.appendChild(tag);

    const quote = document.createElement("blockquote");
    quote.className = "source__quote";
    quote.textContent = citation.quote;

    item.appendChild(head);
    item.appendChild(quote);
    sourcesList.appendChild(item);
  }
  sourcesPanel.hidden = false;
}

function renderAnswer(result: AgentAnswer): void {
  answerBox.textContent = result.answer;
  answerPanel.hidden = false;
  renderActions(result.recommended_actions);
  renderMeta({
    route: result.route,
    confidence: result.confidence,
  });
  renderSources(result.citations);
}

function setBusy(busy: boolean): void {
  submitBtn.disabled = busy;
  submitBtn.textContent = busy ? "Working…" : "Ask";
}

async function runNonStreaming(question: string): Promise<void> {
  setStatus("Querying…", "info");
  const response = await postQuery({ question });
  renderAnswer(response.result);
  setStatus("", "none");
}

async function runStreaming(question: string): Promise<void> {
  setStatus("Streaming…", "info");
  answerPanel.hidden = false;
  answerBox.textContent = "";
  let streamed = "";
  await streamQuery(
    { question },
    {
      onContent: (text: string) => {
        streamed += text;
        answerBox.textContent = streamed;
      },
      onMetadata: (metadata: StreamMetadata) => {
        renderActions(metadata.recommended_actions);
        renderMeta({
          provider: metadata.provider,
          latencyMs: metadata.latency_ms,
          route: metadata.route,
          confidence: metadata.confidence,
        });
        renderSources(metadata.citations);
      },
    },
  );
  setStatus("", "none");
}

form.addEventListener("submit", (event: SubmitEvent) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question.length < 3) {
    setStatus("Please enter a question of at least 3 characters.", "error");
    return;
  }
  clearResults();
  setBusy(true);
  const run = streamToggle.checked ? runStreaming(question) : runNonStreaming(question);
  run
    .catch((error: unknown) => {
      const message = error instanceof ApiError ? error.message : String(error);
      setStatus(`Request failed: ${message}`, "error");
    })
    .finally(() => {
      setBusy(false);
    });
});
