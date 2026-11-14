const messagesEl = document.getElementById("chat-messages");
const inputEl = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");

function getSelectedSources() {
  const boxes = document.querySelectorAll(".source-checkbox");
  if (boxes.length === 0) return null;
  const checked = Array.from(boxes)
    .filter((b) => b.checked)
    .map((b) => b.value);
  if (checked.length === boxes.length) return null; // no filter = search everything
  return checked;
}

function scopeToSingleSource(source) {
  document.querySelectorAll(".source-checkbox").forEach((b) => {
    b.checked = b.value === source;
  });
}

function addUserMessage(text) {
  const empty = document.getElementById("empty-state");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addAssistantPlaceholder() {
  const div = document.createElement("div");
  div.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  div.appendChild(bubble);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function renderCitations(container, citations) {
  if (!citations || citations.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "citations";
  citations.forEach((c) => {
    const pct = Math.round(c.confidence * 100);
    const div = document.createElement("div");
    div.className = "citation";

    const head = document.createElement("div");
    head.className = "citation-head";
    const link = document.createElement("a");
    link.href = `/files/${encodeURIComponent(c.source)}#page=${c.page}`;
    link.target = "_blank";
    link.textContent = `${c.source} · Page ${c.page}`;
    head.appendChild(link);

    const snippet = document.createElement("div");
    snippet.className = "citation-snippet";
    snippet.textContent = `"${c.snippet}"`;

    const confRow = document.createElement("div");
    confRow.className = "confidence-row";
    confRow.innerHTML = `<div class="meter"><div class="meter-fill" style="width:${pct}%"></div></div><span class="confidence-label">${pct}%</span>`;

    div.appendChild(head);
    div.appendChild(snippet);
    div.appendChild(confRow);
    wrap.appendChild(div);
  });
  container.appendChild(wrap);
}

function renderSuggestions(container, suggestions, question) {
  if (!suggestions || suggestions.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "suggestions";
  suggestions.forEach((s) => {
    const chip = document.createElement("button");
    chip.className = "suggestion-chip";
    chip.dataset.source = s.source;
    chip.dataset.question = question;
    chip.textContent = `${s.source} · Page ${s.page}`;
    wrap.appendChild(chip);
  });
  container.appendChild(wrap);
}

async function ask(presetQuestion, presetSources) {
  const question = (presetQuestion ?? inputEl.value).trim();
  if (!question) return;
  if (presetQuestion === undefined) inputEl.value = "";
  sendBtn.disabled = true;

  addUserMessage(question);
  const assistantDiv = addAssistantPlaceholder();
  const bubble = assistantDiv.querySelector(".msg-bubble");
  let started = false;

  try {
    const resp = await fetch("/chat/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        sources: presetSources !== undefined ? presetSources : getSelectedSources(),
      }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.kind === "citations") {
          if (event.mode === "refused") bubble.classList.add("unanswered");
          if (event.mode === "general") {
            const badge = document.createElement("div");
            badge.className = "mode-badge general";
            badge.textContent = "General knowledge — not from your documents";
            assistantDiv.appendChild(badge);
          }
          if (event.mode === "document_list") {
            const badge = document.createElement("div");
            badge.className = "mode-badge neutral";
            badge.textContent = "From your Document Library";
            assistantDiv.appendChild(badge);
          }
          renderSuggestions(assistantDiv, event.suggestions, question);
          renderCitations(assistantDiv, event.citations);
        } else if (event.kind === "token") {
          if (!started) {
            bubble.textContent = "";
            started = true;
          }
          bubble.textContent += event.text;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (event.kind === "error") {
          bubble.classList.remove("unanswered");
          bubble.classList.add("error-bubble");
          bubble.textContent = event.text;
          started = true;
        }
      }
    }
  } catch (err) {
    bubble.textContent = "Something went wrong talking to the server.";
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", () => ask());
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});
resetBtn.addEventListener("click", async () => {
  await fetch("/chat/reset", { method: "POST" });
  location.reload();
});

// Suggestion chips are created dynamically (live) or server-rendered (history
// reload), so a single delegated listener on the message list covers both.
messagesEl.addEventListener("click", (e) => {
  const chip = e.target.closest(".suggestion-chip");
  if (!chip) return;
  const { source, question } = chip.dataset;
  scopeToSingleSource(source);
  ask(question, [source]);
});
