// ── Synapta Real-Time Client ──
// Connects to the Python backend via WebSocket for REAL inference.
// Every token comes from the actual Nemotron-30B model with actual routing.

const COLORS = { code: "#00d4aa", math: "#818cf8", science: "#f59e0b", none: "#9ca3af" };
const LABELS = { code: "Code ⚡", math: "Math 🔢", science: "Science 🔬", none: "Base 🧠" };

let ws = null;
let currentMode = "routed";
let selectedExpert = "code";
let generating = false;
let swapCount = 0;
let tokenIdx = 0;
let genStartTime = 0;
let serverReady = false;

const WS_URL = `ws://${location.hostname || 'localhost'}:${location.port || '8765'}/ws/generate`;

// ── WebSocket Connection ──
function connectWebSocket() {
  updateStatus("connecting");

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log("✅ WebSocket connected to", WS_URL);
    // Check server status via REST
    fetch(`http://${location.hostname || 'localhost'}:${location.port || '8765'}/api/status`)
      .then(r => r.json())
      .then(data => {
        if (data.ready) {
          serverReady = true;
          updateStatus("ready", data);
        } else {
          updateStatus("loading", data);
          // Poll until ready
          pollReady();
        }
      })
      .catch(() => updateStatus("connected"));
  };

  ws.onclose = () => {
    console.log("❌ WebSocket disconnected");
    serverReady = false;
    updateStatus("disconnected");
    // Reconnect after 3s
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    serverReady = false;
    updateStatus("error");
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };
}

function pollReady() {
  const interval = setInterval(() => {
    fetch(`http://${location.hostname || 'localhost'}:${location.port || '8765'}/api/status`)
      .then(r => r.json())
      .then(data => {
        if (data.ready) {
          serverReady = true;
          updateStatus("ready", data);
          clearInterval(interval);
        }
      })
      .catch(() => {});
  }, 2000);
}

function updateStatus(state, data) {
  const badge = document.getElementById("statusBadge");
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  if (!badge) return;

  switch (state) {
    case "ready":
      dot.style.background = "var(--accent-code)";
      dot.style.animation = "pulse 2s ease-in-out infinite";
      text.textContent = `Live — Nemotron-30B (${data?.vram_gb || '?'}GB VRAM)`;
      badge.style.borderColor = "rgba(0,212,170,0.3)";
      break;
    case "loading":
      dot.style.background = "#f59e0b";
      dot.style.animation = "pulse 1s ease-in-out infinite";
      text.textContent = "Model loading...";
      badge.style.borderColor = "rgba(245,158,11,0.3)";
      break;
    case "connecting":
      dot.style.background = "#818cf8";
      dot.style.animation = "pulse 1s ease-in-out infinite";
      text.textContent = "Connecting...";
      badge.style.borderColor = "rgba(129,140,248,0.3)";
      break;
    case "disconnected":
    case "error":
      dot.style.background = "#ef4444";
      dot.style.animation = "none";
      text.textContent = "Server offline — start demo/server.py";
      badge.style.borderColor = "rgba(239,68,68,0.3)";
      break;
    default:
      dot.style.background = "#818cf8";
      text.textContent = "Connected";
  }
}

// ── Handle server messages ──
function handleMessage(msg) {
  switch (msg.type) {
    case "token":
      appendToken(msg.text, msg.domain, msg.index);
      updateMetrics(msg.index, msg.speed);
      break;

    case "swap":
      fireSwap(msg.from, msg.to, msg.at_token, msg.weights);
      break;

    case "route":
      // Initial routing decision
      updateRouterDisplay(msg.domain, msg.weights);
      break;

    case "done":
      finishGeneration(msg);
      break;

    case "error":
      showError(msg.message);
      finishGeneration(msg);
      break;
  }
}

function appendToken(text, domain, index) {
  const output = document.getElementById("outputArea");

  // Remove cursor
  const cursor = output.querySelector(".cursor-blink");
  if (cursor) cursor.remove();

  const span = document.createElement("span");
  span.className = "token";
  const color = currentMode === "routed" ? (COLORS[domain] || COLORS.none) :
                currentMode === "naked" ? COLORS.none :
                COLORS[selectedExpert];
  span.style.color = color;
  span.textContent = text;
  span.title = `Token ${index} — ${LABELS[domain] || domain}`;
  output.appendChild(span);

  // Re-add cursor
  const cur = document.createElement("span");
  cur.className = "cursor-blink";
  output.appendChild(cur);

  // Update router display
  if (currentMode === "routed") {
    document.getElementById("routerName").textContent = LABELS[domain] || domain;
    document.getElementById("routerName").style.color = COLORS[domain] || COLORS.none;
    document.getElementById("routerBar").style.background = COLORS[domain] || COLORS.none;
  }

  output.scrollTop = output.scrollHeight;
}

function updateMetrics(totalTokens, speed) {
  tokenIdx = totalTokens;
  const elapsed = ((performance.now() - genStartTime) / 1000).toFixed(1);
  document.getElementById("mTokens").textContent = totalTokens;
  document.getElementById("mTime").textContent = elapsed + "s";
  document.getElementById("mSpeed").textContent = speed || "0";
}

function updateRouterDisplay(domain, weights) {
  document.getElementById("routerName").textContent = LABELS[domain] || domain;
  document.getElementById("routerName").style.color = COLORS[domain] || COLORS.none;
  document.getElementById("routerBar").style.background = COLORS[domain] || COLORS.none;

  // Show weights if available
  if (weights) {
    const timeline = document.getElementById("swapTimeline");
    const placeholder = timeline.querySelector('.swap-placeholder');
    if (placeholder) placeholder.textContent = `Initial route: ${LABELS[domain]} (${(Math.max(...Object.values(weights)) * 100).toFixed(0)}% confidence)`;
  }
}

function fireSwap(from, to, atToken, weights) {
  swapCount++;
  document.getElementById("mSwaps").textContent = swapCount;

  const timeline = document.getElementById("swapTimeline");
  const placeholder = timeline.querySelector('.swap-placeholder');
  if (placeholder) placeholder.remove();

  const el = document.createElement("div");
  el.className = "swap-event";
  el.innerHTML = `
    <span class="swap-token-idx">@${atToken}</span>
    <span class="swap-domain" style="background:${COLORS[from]}22;color:${COLORS[from]}">${LABELS[from] || from}</span>
    <span class="swap-arrow">→</span>
    <span class="swap-domain" style="background:${COLORS[to]}22;color:${COLORS[to]}">${LABELS[to] || to}</span>
  `;
  el.style.animation = 'fadeInUp 0.3s ease';
  timeline.appendChild(el);
  timeline.scrollTop = timeline.scrollHeight;
}

function showError(message) {
  const output = document.getElementById("outputArea");
  const cursor = output.querySelector(".cursor-blink");
  if (cursor) cursor.remove();

  const errDiv = document.createElement("div");
  errDiv.style.cssText = "color:#ef4444;padding:12px;background:rgba(239,68,68,0.08);border-radius:8px;margin-top:8px;font-size:13px;";
  errDiv.textContent = `⚠️ ${message}`;
  output.appendChild(errDiv);
}

// ── Generate ──
function generate() {
  if (generating) return;
  const prompt = document.getElementById("promptInput").value.trim();
  if (!prompt) return;

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showError("Not connected to server. Run: python3 demo/server.py");
    return;
  }

  if (!serverReady) {
    showError("Model is still loading. Please wait...");
    return;
  }

  generating = true;
  swapCount = 0;
  tokenIdx = 0;
  genStartTime = performance.now();

  document.getElementById("btnGenerate").disabled = true;
  document.getElementById("btnGenerate").textContent = "Generating...";

  const output = document.getElementById("outputArea");
  output.innerHTML = '<span class="cursor-blink"></span>';

  document.getElementById("mTokens").textContent = "0";
  document.getElementById("mSpeed").textContent = "0";
  document.getElementById("mSwaps").textContent = "0";
  document.getElementById("mTime").textContent = "0s";
  document.getElementById("swapTimeline").innerHTML = '<div class="swap-placeholder">Routing...</div>';

  // Determine mode
  let mode, adapter;
  if (currentMode === "routed") {
    mode = "routed";
    adapter = null;
  } else if (currentMode === "naked") {
    mode = "naked";
    adapter = null;
  } else {
    mode = "single";
    adapter = selectedExpert;
  }

  // Send to server
  ws.send(JSON.stringify({
    prompt: prompt,
    mode: mode,
    adapter: adapter,
    max_tokens: 512,
  }));
}

function finishGeneration(msg) {
  generating = false;
  document.getElementById("btnGenerate").disabled = false;
  document.getElementById("btnGenerate").textContent = "Generate →";

  const cursor = document.getElementById("outputArea").querySelector(".cursor-blink");
  if (cursor) cursor.remove();

  if (msg && msg.total_tokens) {
    document.getElementById("mTokens").textContent = msg.total_tokens;
    document.getElementById("mTime").textContent = msg.elapsed_s + "s";
    document.getElementById("mSpeed").textContent = msg.speed || "0";
    document.getElementById("mSwaps").textContent = msg.swaps || swapCount;
  }
}

// ── UI Actions ──
function setMode(btn) {
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentMode = btn.dataset.mode;

  const expertSelector = document.getElementById("expertSelector");

  if (currentMode === "single") {
    document.getElementById("outputMode").textContent = `Single Expert: ${LABELS[selectedExpert]}`;
    if (expertSelector) expertSelector.style.display = "flex";
    document.getElementById("routerName").textContent = LABELS[selectedExpert];
    document.getElementById("routerName").style.color = COLORS[selectedExpert];
    document.getElementById("routerBar").style.background = COLORS[selectedExpert];
  } else if (currentMode === "naked") {
    document.getElementById("outputMode").textContent = "Naked Base Model (no adapters)";
    if (expertSelector) expertSelector.style.display = "none";
    document.getElementById("routerName").textContent = "Base 🧠";
    document.getElementById("routerName").style.color = COLORS.none;
    document.getElementById("routerBar").style.background = COLORS.none;
  } else {
    document.getElementById("outputMode").textContent = "Synapta-Routed Mode";
    if (expertSelector) expertSelector.style.display = "none";
    document.getElementById("routerName").textContent = "Auto-Routing";
    document.getElementById("routerName").style.color = "var(--accent-code)";
    document.getElementById("routerBar").style.background = "var(--accent-code)";
  }
}

function selectExpert(btn) {
  document.querySelectorAll(".expert-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  selectedExpert = btn.dataset.expert;
  document.getElementById("outputMode").textContent = `Single Expert: ${LABELS[selectedExpert]}`;
  document.getElementById("routerName").textContent = LABELS[selectedExpert];
  document.getElementById("routerName").style.color = COLORS[selectedExpert];
  document.getElementById("routerBar").style.background = COLORS[selectedExpert];
}

function setPrompt(btn) {
  const icon = btn.querySelector(".example-icon");
  const text = btn.textContent.replace(icon ? icon.textContent : "", "").trim();
  document.getElementById("promptInput").value = text;
}

// ── Keyboard shortcut ──
document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate();
});

// ── Animate benchmark bars on scroll ──
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.querySelectorAll('.bench-bar').forEach(bar => {
        bar.style.width = bar.style.getPropertyValue('--pct');
      });
    }
  });
}, { threshold: 0.2 });

document.querySelectorAll('.bench-card').forEach(card => {
  card.querySelectorAll('.bench-bar').forEach(bar => { bar.style.width = '0%'; });
  observer.observe(card);
});

// ── Smooth nav scroll ──
document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

// ── Connect on load ──
document.addEventListener("DOMContentLoaded", () => {
  connectWebSocket();
});
