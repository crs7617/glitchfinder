// ═══════════════════════════════════════════════════════════════════
// Glitchfinder — vanilla JS, ES modules, no dependencies
// ═══════════════════════════════════════════════════════════════════

const DATA_DIR = "data/";
const PAGE_SIZE = 50;
const MAP_MIN_TOKENS = 100;

// ── State ──────────────────────────────────────────────────────
let index = null;
let currentData = null;
let filtered = [];
let page = 0;
let selectedRow = -1;
let scoreMin = 0, scoreMax = 1;

// ── DOM refs ───────────────────────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ── Theme ──────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem("gf-theme");
  if (saved) {
    document.documentElement.dataset.theme = saved;
  } else {
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }
  $("#theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme;
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("gf-theme", next);
    if (currentData) drawMap();
  });
}

// ── Init ───────────────────────────────────────────────────────
async function init() {
  initTheme();

  try {
    const resp = await fetch(DATA_DIR + "index.json");
    if (!resp.ok) throw new Error(resp.statusText);
    index = await resp.json();
  } catch (e) {
    console.error("Failed to load index.json", e);
    showState("error");
    return;
  }

  const sel = $("#model-select");
  sel.innerHTML = '<option value="">Choose model\u2026</option>';
  for (const m of index.models) {
    const opt = document.createElement("option");
    opt.value = m.file;
    const short = m.repo_id.split("/").pop();
    opt.textContent = short + " \u2014 " + m.vocab_size.toLocaleString() + " tokens";
    sel.appendChild(opt);
  }

  // Restore from URL hash
  const hash = location.hash.slice(1);
  if (hash) {
    const p = new URLSearchParams(hash);
    if (p.get("m")) {
      sel.value = p.get("m");
      await loadModel(p.get("m"));
      if (p.get("q")) $("#search-input").value = p.get("q");
      if (p.get("s")) $("#sort-select").value = p.get("s");
      if (p.get("sc")) $("#script-filter").value = p.get("sc");
      applyFilters();
    }
  }

  // Events
  sel.addEventListener("change", () => loadModel(sel.value));
  $("#search-input").addEventListener("input", debounce(() => { page = 0; applyFilters(); }, 200));
  $("#sort-select").addEventListener("change", () => { page = 0; applyFilters(); });
  $("#script-filter").addEventListener("change", () => { page = 0; applyFilters(); });
  for (const id of ["f-special", "f-unreachable", "f-verified"]) {
    $("#" + id).addEventListener("change", () => { page = 0; applyFilters(); });
  }
  $("#prev-btn").addEventListener("click", () => { page--; renderTable(); });
  $("#next-btn").addEventListener("click", () => { page++; renderTable(); });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-backdrop").addEventListener("click", closeDrawer);
  $("#compare-select").addEventListener("change", (e) => loadCompare(e.target.value));

  // Keyboard shortcuts
  document.addEventListener("keydown", onKeydown);

  // Canvas resize
  window.addEventListener("resize", debounce(() => { if (currentData) drawMap(); }, 150));
}

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// ── Keyboard ───────────────────────────────────────────────────
function onKeydown(e) {
  if (e.key === "/" && !e.ctrlKey && !e.metaKey) {
    if (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA") return;
    e.preventDefault();
    $("#search-input").focus();
  }
  if (e.key === "Escape") {
    closeDrawer();
    $("#search-input").blur();
  }
  if (!currentData || !filtered.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const dir = e.key === "ArrowDown" ? 1 : -1;
    selectedRow = Math.max(0, Math.min(filtered.length - 1, selectedRow + dir));
    const neededPage = Math.floor(selectedRow / PAGE_SIZE);
    if (neededPage !== page) { page = neededPage; renderTable(); }
    highlightRow(selectedRow);
    const tok = filtered[selectedRow];
    if (tok) showDrawer(tok);
  }
}

function highlightRow(idx) {
  for (const tr of $$("#tbody tr")) tr.removeAttribute("aria-selected");
  const localIdx = idx - page * PAGE_SIZE;
  const rows = $$("#tbody tr");
  if (rows[localIdx]) {
    rows[localIdx].setAttribute("aria-selected", "true");
    rows[localIdx].scrollIntoView({ block: "nearest" });
  }
}

// ── State management ───────────────────────────────────────────
function showState(which) {
  for (const s of ["empty", "loading", "error"]) {
    const el = $("#state-" + s);
    if (el) el.hidden = s !== which;
  }
  const sections = ["model-bar", "map-section", "table-section", "chart-section", "compare-section"];
  if (which) sections.forEach(id => { const el = $("#" + id); if (el) el.hidden = true; });
}

// ── Load model ─────────────────────────────────────────────────
async function loadModel(file) {
  if (!file) { currentData = null; showState("empty"); updateHero(null); return; }
  showState("loading");
  try {
    const resp = await fetch(DATA_DIR + file);
    if (!resp.ok) throw new Error(resp.statusText);
    currentData = await resp.json();
  } catch (e) {
    console.error("Failed to load model data", e);
    showState("error");
    return;
  }

  // Compute score range
  const scores = currentData.tokens.map(t => t.score);
  scoreMin = Math.min(...scores);
  scoreMax = Math.max(...scores);

  showState(null);
  updateHero(currentData);
  renderModelBar();
  populateScriptFilter();
  populateCompareSelect(file);
  page = 0;
  applyFilters();
  drawMap();
  renderChart();

  for (const id of ["model-bar", "map-section", "table-section", "chart-section", "compare-section"]) {
    $("#" + id).hidden = false;
  }
  updateHash();
}

// ── Hero specimens ─────────────────────────────────────────────
function updateHero(data) {
  const el = $("#hero-specimens");
  if (!data) {
    el.innerHTML = '<p class="hero__specimens-empty">Select a model to see specimens.</p>';
    return;
  }
  // Pick 5 interesting tokens: prefer verified, then non-flagged low-rank, then flagged
  const verified = data.tokens.filter(t => t.verified);
  const nonFlagged = data.tokens.filter(t => !t.special && !t.unreachable && !t.partial && !t.verified);
  const flagged = data.tokens.filter(t => t.unreachable && t.decoded && t.decoded.trim());
  const picks = [];
  for (const src of [verified, nonFlagged, flagged]) {
    for (const t of src) {
      if (picks.length >= 5) break;
      if (!picks.find(p => p.id === t.id) && t.decoded) picks.push(t);
    }
  }

  let html = "";
  picks.forEach((t, i) => {
    const flag = t.verified ? '<span class="specimen__flag specimen__flag--verified">verified glitch</span>'
      : t.unreachable ? '<span class="specimen__flag specimen__flag--unreachable">unreachable</span>'
      : t.special ? '<span class="specimen__flag specimen__flag--special">special</span>'
      : "";
    html += `<div class="specimen" style="--delay:${i * 0.7}s">
      <span class="specimen__tok">${escTok(t.decoded)}</span>
      <span class="specimen__meta">id ${t.id} &middot; ${t.score.toFixed(4)}</span>
      ${flag}
    </div>`;
  });
  el.innerHTML = html;
}

// ── Model bar ──────────────────────────────────────────────────
function renderModelBar() {
  const m = currentData.metadata;
  const vs = m.verification_summary;
  const items = [
    ["Model", m.repo_id],
    ["Arch", m.architecture],
    ["Vocab", m.vocab_size.toLocaleString()],
    ["Hidden", m.hidden_size],
    ["Tied", m.tied ? "yes" : "no"],
    ["Indicator", m.indicator.replace("_", " ")],
  ];
  if (vs) items.push(["Verified", vs.confirmed_glitches + "/" + vs.total_verified + " glitches"]);

  $("#model-meta").innerHTML = items.map(([k, v]) =>
    `<span class="meta-pair"><strong>${esc(k)}</strong><span>${esc(String(v))}</span></span>`
  ).join("");
}

// ── Vocabulary map ─────────────────────────────────────────────
function drawMap() {
  if (!currentData || currentData.metadata.vocab_size < MAP_MIN_TOKENS) {
    $("#map-section").hidden = true;
    return;
  }

  const vocabSize = currentData.metadata.vocab_size;
  const canvas = $("#vocab-map");
  const wrap = $("#map-wrap");
  const dpr = window.devicePixelRatio || 1;

  // Compute grid dimensions
  const wrapWidth = wrap.clientWidth;
  const cols = Math.ceil(Math.sqrt(vocabSize * (wrapWidth / 400)));
  const cellSize = Math.max(1, Math.floor(wrapWidth / cols));
  const actualCols = Math.floor(wrapWidth / cellSize);
  const rows = Math.ceil(vocabSize / actualCols);
  const height = rows * cellSize;

  canvas.width = wrapWidth * dpr;
  canvas.height = height * dpr;
  canvas.style.height = height + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  // Build score lookup (token_id -> score percentile)
  const scoreLookup = new Float32Array(vocabSize);
  const flagLookup = new Uint8Array(vocabSize); // 0=normal, 1=flagged, 2=verified
  for (const t of currentData.tokens) {
    const pct = scoreMax > scoreMin ? (t.score - scoreMin) / (scoreMax - scoreMin) : 0.5;
    scoreLookup[t.id] = pct;
    if (t.verified) flagLookup[t.id] = 2;
    else if (t.special || t.unreachable || t.partial) flagLookup[t.id] = 1;
  }

  // Get theme colors
  const style = getComputedStyle(document.documentElement);
  const coldColor = parseColor(style.getPropertyValue("--map-cold").trim());
  const warmColor = parseColor(style.getPropertyValue("--map-warm").trim());
  const bgColor = style.getPropertyValue("--bg").trim();

  // Draw cells
  const imageData = ctx.createImageData(wrapWidth * dpr, height * dpr);
  const pixels = imageData.data;

  for (let id = 0; id < vocabSize; id++) {
    const col = id % actualCols;
    const row = Math.floor(id / actualCols);
    const pct = scoreLookup[id] || 0.5;
    const flag = flagLookup[id];

    // Interpolate color based on percentile (lower score = warmer = more suspicious)
    const t = 1 - pct; // invert so suspicious tokens are warm
    let r = coldColor[0] + (warmColor[0] - coldColor[0]) * t;
    let g = coldColor[1] + (warmColor[1] - coldColor[1]) * t;
    let b = coldColor[2] + (warmColor[2] - coldColor[2]) * t;

    // Brighten flagged/verified
    if (flag === 2) { r = 59; g = 154; b = 106; } // verified green
    else if (flag === 1 && t > 0.6) { r = Math.min(255, r * 1.3); }

    // Fill cell pixels
    const x0 = Math.round(col * cellSize * dpr);
    const y0 = Math.round(row * cellSize * dpr);
    const x1 = Math.round((col + 1) * cellSize * dpr);
    const y1 = Math.round((row + 1) * cellSize * dpr);
    const w = wrapWidth * dpr;

    for (let py = y0; py < y1 && py < height * dpr; py++) {
      for (let px = x0; px < x1 && px < wrapWidth * dpr; px++) {
        const idx = (py * w + px) * 4;
        pixels[idx] = r | 0;
        pixels[idx + 1] = g | 0;
        pixels[idx + 2] = b | 0;
        pixels[idx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Store grid info for interaction
  canvas.dataset.cols = actualCols;
  canvas.dataset.cellSize = cellSize;
  canvas.dataset.vocabSize = vocabSize;

  // Events
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    const id = row * actualCols + col;
    const tip = $("#map-tooltip");

    if (id >= 0 && id < vocabSize) {
      const tok = currentData.tokens.find(t => t.id === id);
      if (tok) {
        tip.innerHTML = `<strong>#${id}</strong> ${escTok(tok.decoded)}<br>score: ${tok.score.toFixed(6)}${tok.verified ? "<br>verified glitch" : ""}`;
      } else {
        tip.innerHTML = `<strong>#${id}</strong><br>not in top 2000`;
      }
      tip.hidden = false;
      tip.style.left = (e.clientX + 12) + "px";
      tip.style.top = (e.clientY + 12) + "px";
    } else {
      tip.hidden = true;
    }
  };
  canvas.onmouseleave = () => { $("#map-tooltip").hidden = true; };
  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / cellSize);
    const row = Math.floor((e.clientY - rect.top) / cellSize);
    const id = row * actualCols + col;
    const tok = currentData.tokens.find(t => t.id === id);
    if (tok) showDrawer(tok);
  };
}

function parseColor(hex) {
  if (hex.startsWith("#")) hex = hex.slice(1);
  return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
}

// ── Script filter ──────────────────────────────────────────────
function populateScriptFilter() {
  const scripts = new Set();
  for (const t of currentData.tokens) {
    for (const s of (t.scripts || [])) scripts.add(s);
  }
  const sel = $("#script-filter");
  sel.innerHTML = '<option value="">All scripts</option>';
  for (const s of [...scripts].sort()) {
    const opt = document.createElement("option");
    opt.value = s; opt.textContent = s;
    sel.appendChild(opt);
  }
}

// ── Compare ────────────────────────────────────────────────────
function populateCompareSelect(currentFile) {
  const sel = $("#compare-select");
  sel.innerHTML = '<option value="">Choose second model\u2026</option>';
  for (const m of index.models) {
    if (m.file === currentFile) continue;
    const opt = document.createElement("option");
    opt.value = m.file;
    opt.textContent = m.repo_id.split("/").pop();
    sel.appendChild(opt);
  }
  $("#compare-area").innerHTML = "";
}

async function loadCompare(file) {
  const area = $("#compare-area");
  if (!file || !currentData) { area.innerHTML = ""; return; }
  let cmpData;
  try {
    const resp = await fetch(DATA_DIR + file);
    cmpData = await resp.json();
  } catch (e) { return; }

  // Small multiples: for each model, show a mini canvas colored by score
  const models = [
    { name: currentData.metadata.repo_id.split("/").pop(), tokens: currentData.tokens, vocab: currentData.metadata.vocab_size },
    { name: cmpData.metadata.repo_id.split("/").pop(), tokens: cmpData.tokens, vocab: cmpData.metadata.vocab_size },
  ];

  // Also find overlap
  const set1 = new Set(currentData.tokens.map(t => t.id));
  const overlap = cmpData.tokens.filter(t => set1.has(t.id));

  let html = `<p class="compare-stat">${overlap.length} tokens appear in both top-2000 lists.</p>`;
  html += '<div class="compare-grid">';
  for (const m of models) {
    html += `<div class="compare-card">
      <div class="compare-card__title">${esc(m.name)} (${m.vocab.toLocaleString()} vocab)</div>
      <canvas class="compare-card__canvas" data-model="${esc(m.name)}"></canvas>
      <div class="compare-stat">${m.tokens.filter(t => t.special || t.unreachable || t.partial).length} flagged in top 2000</div>
    </div>`;
  }
  html += "</div>";
  area.innerHTML = html;

  // Draw mini maps
  for (const m of models) {
    const canvas = area.querySelector(`canvas[data-model="${esc(m.name)}"]`);
    drawMiniMap(canvas, m.tokens, m.vocab);
  }
}

function drawMiniMap(canvas, tokens, vocabSize) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.parentElement.clientWidth;
  const cols = Math.ceil(Math.sqrt(vocabSize * (width / 200)));
  const cellSize = Math.max(1, Math.floor(width / cols));
  const actualCols = Math.floor(width / cellSize);
  const rows = Math.ceil(vocabSize / actualCols);
  const height = rows * cellSize;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = height + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const style = getComputedStyle(document.documentElement);
  const cold = parseColor(style.getPropertyValue("--map-cold").trim());
  const warm = parseColor(style.getPropertyValue("--map-warm").trim());

  const scores = tokens.map(t => t.score);
  const sMin = Math.min(...scores);
  const sMax = Math.max(...scores);

  const lookup = new Float32Array(vocabSize).fill(0.5);
  for (const t of tokens) {
    lookup[t.id] = sMax > sMin ? (t.score - sMin) / (sMax - sMin) : 0.5;
  }

  const imageData = ctx.createImageData(width * dpr, height * dpr);
  const px = imageData.data;

  for (let id = 0; id < vocabSize; id++) {
    const c = id % actualCols;
    const r = Math.floor(id / actualCols);
    const pct = 1 - lookup[id];
    const cr = cold[0] + (warm[0] - cold[0]) * pct;
    const cg = cold[1] + (warm[1] - cold[1]) * pct;
    const cb = cold[2] + (warm[2] - cold[2]) * pct;

    const x0 = Math.round(c * cellSize * dpr);
    const y0 = Math.round(r * cellSize * dpr);
    const x1 = Math.round((c + 1) * cellSize * dpr);
    const y1 = Math.round((r + 1) * cellSize * dpr);
    const w = width * dpr;

    for (let py = y0; py < y1 && py < height * dpr; py++) {
      for (let ppx = x0; ppx < x1 && ppx < width * dpr; ppx++) {
        const idx = (py * w + ppx) * 4;
        px[idx] = cr | 0; px[idx + 1] = cg | 0; px[idx + 2] = cb | 0; px[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

// ── Chart ──────────────────────────────────────────────────────
function renderChart() {
  const stats = currentData.script_stats || {};
  const RENAME = { "Common": "Punctuation & digits", "Unknown": "Unassigned / private use", "Control": "Control characters" };
  const MIN_COUNT = 30;

  let entries = Object.entries(stats).map(([name, v]) => ({
    name: RENAME[name] || name,
    total: v.total,
    flagged: v.flagged,
    rate: v.total > 0 ? v.flagged / v.total : 0,
  }));

  // Group small scripts into Other
  let otherTotal = 0, otherFlagged = 0;
  entries = entries.filter(e => {
    if (e.total < MIN_COUNT) { otherTotal += e.total; otherFlagged += e.flagged; return false; }
    return true;
  });
  if (otherTotal > 0) {
    entries.push({ name: "Other (< 30 tokens)", total: otherTotal, flagged: otherFlagged, rate: otherTotal > 0 ? otherFlagged / otherTotal : 0 });
  }

  entries.sort((a, b) => b.rate - a.rate);

  const maxRate = Math.max(...entries.map(e => e.rate), 0.01);

  let html = "";
  for (const e of entries) {
    const pct = (e.rate / maxRate * 100).toFixed(1);
    const ratePct = (e.rate * 100).toFixed(1);
    html += `<div class="chart-row">
      <span class="chart-row__label" title="${esc(e.name)}">${esc(e.name)}</span>
      <div class="chart-row__bar-wrap">
        <div class="chart-row__bar" style="width:${pct}%"></div>
      </div>
      <span class="chart-row__nums">${ratePct}% (${e.flagged}/${e.total})</span>
    </div>`;
  }

  const footnote = entries.find(e => e.name.startsWith("Other"));
  if (footnote) {
    html += `<p class="chart-footnote">* "Other" groups ${otherTotal} tokens across scripts with fewer than 30 tokens each.</p>`;
  }

  $("#chart-area").innerHTML = html;
}

// ── Filter & sort ──────────────────────────────────────────────
function applyFilters() {
  if (!currentData) return;
  const q = ($("#search-input").value || "").trim().toLowerCase();
  const sc = $("#script-filter").value;
  const fS = $("#f-special").checked;
  const fU = $("#f-unreachable").checked;
  const fV = $("#f-verified").checked;

  filtered = currentData.tokens.filter(t => {
    if (q) {
      const idMatch = String(t.id) === q;
      const textMatch = (t.decoded || "").toLowerCase().includes(q);
      if (!idMatch && !textMatch) return false;
    }
    if (sc && !(t.scripts || []).includes(sc)) return false;
    if (fS && !t.special) return false;
    if (fU && !t.unreachable) return false;
    if (fV && !t.verified) return false;
    return true;
  });

  const sort = $("#sort-select").value;
  filtered.sort((a, b) => {
    if (sort === "rank-asc") return a.rank - b.rank;
    if (sort === "score-asc") return a.score - b.score;
    if (sort === "score-desc") return b.score - a.score;
    if (sort === "id-asc") return a.id - b.id;
    return 0;
  });

  selectedRow = -1;
  renderTable();
  updateHash();
}

// ── Table rendering ────────────────────────────────────────────
function renderTable() {
  const total = filtered.length;
  const pages = Math.ceil(total / PAGE_SIZE) || 1;
  page = Math.max(0, Math.min(page, pages - 1));

  const start = page * PAGE_SIZE;
  const slice = filtered.slice(start, start + PAGE_SIZE);

  $("#count-badge").textContent = "(" + total.toLocaleString() + ")";
  $("#page-info").textContent = (page + 1) + " / " + pages;
  $("#prev-btn").disabled = page === 0;
  $("#next-btn").disabled = page >= pages - 1;

  const tbody = $("#tbody");
  let html = "";
  for (const t of slice) {
    const flags = [];
    if (t.special) flags.push('<span class="flag flag--special">special</span>');
    if (t.unreachable) flags.push('<span class="flag flag--unreachable">unreachable</span>');
    if (t.verified) flags.push('<span class="flag flag--verified flag--check">verified</span>');

    // Score bar width (0-100% relative to data range)
    const pct = scoreMax > scoreMin ? ((t.score - scoreMin) / (scoreMax - scoreMin) * 100) : 50;

    html += `<tr data-id="${t.id}" tabindex="-1">
      <td class="cell--num">${t.rank}</td>
      <td class="cell--num">${t.id}</td>
      <td class="cell--tok">${escTok(t.decoded)}</td>
      <td class="cell--score"><span class="score-bar"><span class="score-bar__track"><span class="score-bar__fill" style="width:${pct.toFixed(0)}%"></span></span>${t.score.toFixed(4)}</span></td>
      <td>${flags.join("")}</td>
      <td style="font-size:0.75rem;color:var(--fg-dim)">${(t.scripts || []).join(", ")}</td>
    </tr>`;
  }
  tbody.innerHTML = html;

  // Row click -> drawer
  for (const tr of tbody.querySelectorAll("tr")) {
    tr.addEventListener("click", () => {
      const id = parseInt(tr.dataset.id);
      const tok = currentData.tokens.find(t => t.id === id);
      if (tok) showDrawer(tok);
    });
  }
}

// ── Drawer ─────────────────────────────────────────────────────
function showDrawer(t) {
  const flags = [];
  if (t.special) flags.push("special");
  if (t.unreachable) flags.push("unreachable");
  if (t.partial) flags.push("partial-utf8");

  let verifiedHtml;
  if (t.verified !== undefined) {
    verifiedHtml = t.verified
      ? `<span class="glitch-label">CONFIRMED GLITCH</span> (max prob ${(t.max_prob * 100).toFixed(4)}%)`
      : "Not a glitch";
  } else {
    verifiedHtml = "Unverified (indicator only)";
  }

  const tokenUrl = new URL(location.href);
  tokenUrl.hash = `m=${$("#model-select").value}&tid=${t.id}`;

  const dl = [
    ["Token ID", t.id],
    ["Rank", t.rank],
    ["Decoded", `<code>${escTok(t.decoded)}</code>`],
    ["Hex bytes", `<code>${t.hex || "\u2014"}</code>`],
    ["Score", t.score.toFixed(8)],
    ["Scripts", (t.scripts || []).join(", ") || "\u2014"],
    ["Flags", flags.join(", ") || "none"],
    ["Verification", verifiedHtml],
  ];

  let html = `<h3>Token ${t.id}</h3>`;
  html += '<dl>';
  for (const [label, value] of dl) {
    html += `<div class="detail-row"><dt>${label}</dt><dd>${value}</dd></div>`;
  }
  html += '</dl>';
  html += `<div class="drawer__actions"><button class="drawer__copy-btn" id="copy-link-btn">Copy link to this token</button></div>`;

  const content = $("#drawer-content");
  content.innerHTML = html;
  $("#drawer").hidden = false;
  $("#drawer-backdrop").hidden = false;

  // Copy link
  content.querySelector("#copy-link-btn").addEventListener("click", () => {
    navigator.clipboard.writeText(tokenUrl.href).then(() => {
      content.querySelector("#copy-link-btn").textContent = "Copied!";
      setTimeout(() => { content.querySelector("#copy-link-btn").textContent = "Copy link to this token"; }, 1500);
    });
  });
}

function closeDrawer() {
  $("#drawer").hidden = true;
  $("#drawer-backdrop").hidden = true;
}

// ── URL hash ───────────────────────────────────────────────────
function updateHash() {
  const m = $("#model-select").value;
  if (!m) { history.replaceState(null, "", location.pathname); return; }
  const p = new URLSearchParams();
  p.set("m", m);
  const q = $("#search-input").value;
  if (q) p.set("q", q);
  const s = $("#sort-select").value;
  if (s !== "rank-asc") p.set("s", s);
  const sc = $("#script-filter").value;
  if (sc) p.set("sc", sc);
  history.replaceState(null, "", "#" + p.toString());
}

// ── Escape helpers ─────────────────────────────────────────────
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function escTok(s) {
  if (!s) return '<span class="tok-empty">empty</span>';
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp === 0x0a) out += '<span class="tok-ws">LF</span>';
    else if (cp === 0x0d) out += '<span class="tok-ws">CR</span>';
    else if (cp === 0x09) out += '<span class="tok-ws">TAB</span>';
    else if (cp === 0x00) out += '<span class="tok-ws">NUL</span>';
    else if (cp < 0x20 || cp === 0x7f) out += `<span class="tok-ws">x${cp.toString(16).padStart(2, "0")}</span>`;
    else if (cp === 0x20) out += '<span class="tok-ws">\u00b7</span>';
    else if (cp === 0xa0) out += '<span class="tok-ws">NBSP</span>';
    else out += esc(ch);
  }
  return out;
}

// ── Boot ───────────────────────────────────────────────────────
init();
