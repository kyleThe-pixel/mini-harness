/* mini-harness frontend: tabs, model list, SSE streaming, file browser. */
const $ = (id) => document.getElementById(id);
const state = {
  key: localStorage.getItem("harness_key") || "",
  models: [],
  running: false,
};

function apiKey() { return $("apiKey").value.trim() || state.key; }
function headers() { return { "Content-Type": "application/json" }; }

/* ---------- tabs ---------- */
document.querySelectorAll(".tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "files") loadFiles();
  });
});

/* ---------- api key + models ---------- */
$("apiKey").value = state.key;
$("saveKey").addEventListener("click", () => {
  state.key = $("apiKey").value.trim();
  localStorage.setItem("harness_key", state.key);
  $("saveKey").textContent = state.key ? "saved ✓" : "save";
  setTimeout(() => ($("saveKey").textContent = "save"), 1500);
  loadModels();
});

async function loadModels() {
  try {
    const r = await fetch("/api/models?key=" + encodeURIComponent(apiKey()));
    const d = await r.json();
    state.models = d.models || [];
    const opts = state.models
      .map((m) => `<option value="${m.id}">${m.label}</option>`)
      .join("");
    $("chatModel").innerHTML = opts;
    document.querySelectorAll(".agent-model").forEach((sel) => {
      const cur = sel.value;
      sel.innerHTML = opts;
      if (cur) sel.value = cur;
      else if (d.default) sel.value = d.default;
    });
    if (d.default && !$("chatModel").value) $("chatModel").value = d.default;
  } catch (e) {
    console.warn("model list failed", e);
  }
}
$("refreshModels").addEventListener("click", loadModels);

/* ---------- SSE helper (POST with streaming body) ---------- */
async function streamPost(url, body, onEvent) {
  const resp = await fetch(url, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const t = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${t.slice(0, 200)}`);
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const p of parts) {
      const line = p.trim();
      if (line.startsWith("data:")) {
        try { onEvent(JSON.parse(line.slice(5).trim())); }
        catch (e) { console.warn("bad event", line); }
      }
    }
  }
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ---------- mission ---------- */
const DEFAULT_AGENTS = [
  { name: "lead", role: "team lead — plans, assigns, synthesizes", model: "" },
  { name: "coder", role: "software engineer — writes and runs code", model: "" },
  { name: "reviewer", role: "code reviewer — checks work, finds bugs", model: "" },
];

function agentRow(spec) {
  const div = document.createElement("div");
  div.className = "agent-row";
  div.innerHTML = `
    <input class="agent-name" value="${esc(spec.name)}" placeholder="name">
    <input class="agent-role" value="${esc(spec.role)}" placeholder="role">
    <select class="agent-model"></select>
    <button class="rm" title="remove">✕</button>`;
  const sel = div.querySelector(".agent-model");
  sel.innerHTML = state.models.map((m) => `<option value="${m.id}">${m.label}</option>`).join("");
  if (spec.model) sel.value = spec.model;
  div.querySelector(".rm").addEventListener("click", () => div.remove());
  return div;
}
function addAgent(spec) { $("agentRows").appendChild(agentRow(spec || { name: "", role: "", model: "" })); }
$("addAgent").addEventListener("click", () => addAgent());

const cards = {};
function cardFor(name, role, model) {
  if (cards[name]) return cards[name];
  const div = document.createElement("div");
  div.className = "agent-card";
  div.innerHTML = `<div class="head"><span><span class="dot"></span><span class="name">${esc(name)}</span></span>
    <span class="model">${esc(model)}</span></div>
    <div class="role" style="color:var(--muted);font-size:11px;margin-bottom:6px">${esc(role)}</div>
    <div class="feed"></div>`;
  $("teamBoard").appendChild(div);
  cards[name] = { el: div, feed: div.querySelector(".feed"), dot: div.querySelector(".dot") };
  return cards[name];
}
function setDot(name, cls) {
  const c = cards[name]; if (!c) return;
  c.dot.className = "dot" + (cls ? " " + cls : "");
}
function feedHtml(ev) {
  if (ev.type === "tool_call")
    return `<div class="tool">🔧 ${esc(ev.tool)} <span style="color:var(--muted)">${esc(JSON.stringify(ev.args)).slice(0, 160)}</span></div>`;
  if (ev.type === "tool_result")
    return `<div class="toolres">${ev.ok ? "✓" : "✗"} ${esc(ev.tool)}: ${esc(ev.text).slice(0, 500)}</div>`;
  if (ev.type === "agent_message")
    return `<div class="msg">💬 → <b>${esc(ev.to)}</b>: ${esc(ev.text).slice(0, 400)}</div>`;
  if (ev.type === "error") return `<div class="tool" style="color:var(--red)">⚠ ${esc(ev.text)}</div>`;
  return "";
}

$("launch").addEventListener("click", async () => {
  if (state.running) return;
  const goal = $("goal").value.trim();
  if (!goal) { alert("Describe the mission goal first."); return; }
  const specs = [...document.querySelectorAll(".agent-row")].map((r) => ({
    name: r.querySelector(".agent-name").value.trim() || "agent",
    role: r.querySelector(".agent-role").value.trim() || "team member",
    model: r.querySelector(".agent-model").value,
  }));
  if (!specs.length) { alert("Add at least one agent."); return; }

  state.running = true;
  $("launch").disabled = true;
  $("teamBoard").innerHTML = "";
  Object.keys(cards).forEach((k) => delete cards[k]);
  $("missionSummary").classList.add("hidden");
  $("missionStatus").textContent = "mission running…";
  specs.forEach((s) => cardFor(s.name, s.role, s.model));

  let tokenBuf = {};
  try {
    await streamPost("/api/mission", {
      goal, agents: specs, key: apiKey(),
      max_rounds: parseInt($("maxRounds").value) || 4,
    }, (ev) => {
      if (ev.type === "token" && ev.agent) {
        tokenBuf[ev.agent] = tokenBuf[ev.agent] || document.createElement("span");
        if (!tokenBuf[ev.agent].parentNode && cards[ev.agent])
          cards[ev.agent].feed.appendChild(tokenBuf[ev.agent]);
        tokenBuf[ev.agent].textContent += ev.text;
        tokenBuf[ev.agent].className = "tok";
      } else if (ev.type === "agent_start") {
        setDot(ev.agent, "working");
        tokenBuf[ev.agent] = null;
      } else if (ev.type === "agent_done") {
        setDot(ev.agent, "done");
        if (cards[ev.agent] && ev.final)
          cards[ev.agent].feed.insertAdjacentHTML("beforeend",
            `<div class="final">${esc(ev.final).slice(0, 1200)}</div>`);
      } else if (ev.type === "mission_done") {
        $("missionSummary").innerHTML = "<b>Mission summary</b>\n" + esc(ev.summary);
        $("missionSummary").classList.remove("hidden");
      } else if (ev.type === "round_start") {
        /* round separators are implicit in agent cards */
      } else if (ev.agent && cards[ev.agent]) {
        const html = feedHtml(ev);
        if (html) cards[ev.agent].feed.insertAdjacentHTML("beforeend", html);
      } else if (ev.type === "error") {
        $("missionStatus").textContent = "error: " + ev.text;
      }
    });
  } catch (e) {
    $("missionStatus").textContent = "error: " + e.message;
  }
  $("missionStatus").textContent = "mission finished";
  state.running = false;
  $("launch").disabled = false;
});

/* ---------- solo chat ---------- */
function chatMsg(who, bodyHtml) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.innerHTML = `<div class="who">${who === "user" ? "you" : "agent"}</div><div class="body">${bodyHtml}</div>`;
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
  return div.querySelector(".body");
}
$("clearChat").addEventListener("click", () => ($("chatLog").innerHTML = ""));
$("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("chatInput").value.trim();
  if (!text || state.running) return;
  $("chatInput").value = "";
  chatMsg("user", esc(text));
  state.running = true;
  const body = chatMsg("agent", "");
  try {
    await streamPost("/api/chat", { message: text, model: $("chatModel").value, key: apiKey() }, (ev) => {
      if (ev.type === "token") body.textContent += ev.text;
      else if (ev.type === "tool_call")
        body.innerHTML += `<div class="tool">🔧 ${esc(ev.tool)} ${esc(JSON.stringify(ev.args)).slice(0, 200)}</div>`;
      else if (ev.type === "error") body.innerHTML += `<div class="tool" style="color:var(--red)">⚠ ${esc(ev.text)}</div>`;
      $("chatLog").scrollTop = $("chatLog").scrollHeight;
    });
  } catch (err) {
    body.innerHTML += `<div class="tool" style="color:var(--red)">⚠ ${esc(err.message)}</div>`;
  }
  state.running = false;
});

/* ---------- files ---------- */
function renderTree(nodes, parent) {
  nodes.forEach((n) => {
    const div = document.createElement("div");
    div.className = "ft-node " + (n.dir ? "dir" : "file");
    div.textContent = (n.dir ? "📁 " : "📄 ") + n.name;
    parent.appendChild(div);
    if (n.dir && n.children) {
      const kids = document.createElement("div");
      kids.className = "ft-children";
      parent.appendChild(kids);
      renderTree(n.children, kids);
    } else if (!n.dir) {
      div.addEventListener("click", () => loadFile(n.path));
    }
  });
}
async function loadFiles() {
  const r = await fetch("/api/files?key=" + encodeURIComponent(apiKey()));
  const d = await r.json();
  $("fileTree").innerHTML = "";
  renderTree(d.tree || [], $("fileTree"));
}
async function loadFile(path) {
  $("filePath").textContent = path;
  $("fileContent").textContent = "loading…";
  const r = await fetch("/api/file?path=" + encodeURIComponent(path) + "&key=" + encodeURIComponent(apiKey()));
  const d = await r.json();
  $("fileContent").textContent = d.content || d.error || "(empty)";
}
$("reloadFiles").addEventListener("click", loadFiles);

/* ---------- init ---------- */
DEFAULT_AGENTS.forEach(addAgent);
loadModels();
