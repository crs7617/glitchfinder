// Glitchfinder Explorer — vanilla JS, no dependencies
(function () {
    "use strict";

    const PAGE_SIZE = 50;
    const DATA_DIR = "data/";
    let index = null;
    let currentData = null;
    let filtered = [];
    let page = 0;
    let compareData = null;

    // DOM refs
    const $ = (s) => document.querySelector(s);
    const modelSel = $("#model-select");
    const searchIn = $("#search-input");
    const sortSel = $("#sort-select");
    const scriptFilt = $("#script-filter");
    const fSpecial = $("#f-special");
    const fUnreach = $("#f-unreachable");
    const fVerified = $("#f-verified");
    const tbody = $("#tbody");
    const countBadge = $("#count-badge");
    const pageInfo = $("#page-info");
    const prevBtn = $("#prev-btn");
    const nextBtn = $("#next-btn");
    const drawer = $("#drawer");
    const drawerContent = $("#drawer-content");
    const drawerClose = $("#drawer-close");
    const compareSel = $("#compare-select");

    // ── Init ──────────────────────────────────────────────────────
    async function init() {
        try {
            const resp = await fetch(DATA_DIR + "index.json");
            index = await resp.json();
        } catch (e) {
            console.error("Failed to load index.json", e);
            return;
        }

        // Populate model selector
        modelSel.innerHTML = '<option value="">-- Select model --</option>';
        for (const m of index.models) {
            const opt = document.createElement("option");
            opt.value = m.file;
            opt.textContent = m.repo_id + " (" + m.vocab_size.toLocaleString() + " vocab)";
            modelSel.appendChild(opt);
        }

        // Read initial model from URL hash
        const hash = location.hash.slice(1);
        if (hash) {
            const params = new URLSearchParams(hash);
            const model = params.get("m");
            if (model) {
                modelSel.value = model;
                await loadModel(model);
                if (params.get("q")) { searchIn.value = params.get("q"); }
                if (params.get("s")) { sortSel.value = params.get("s"); }
                if (params.get("sc")) { scriptFilt.value = params.get("sc"); }
                applyFilters();
            }
        }

        // Events
        modelSel.addEventListener("change", () => loadModel(modelSel.value));
        searchIn.addEventListener("input", debounce(() => { page = 0; applyFilters(); }, 200));
        sortSel.addEventListener("change", () => { page = 0; applyFilters(); });
        scriptFilt.addEventListener("change", () => { page = 0; applyFilters(); });
        fSpecial.addEventListener("change", () => { page = 0; applyFilters(); });
        fUnreach.addEventListener("change", () => { page = 0; applyFilters(); });
        fVerified.addEventListener("change", () => { page = 0; applyFilters(); });
        prevBtn.addEventListener("click", () => { page--; render(); });
        nextBtn.addEventListener("click", () => { page++; render(); });
        drawerClose.addEventListener("click", () => drawer.hidden = true);
        compareSel.addEventListener("change", () => loadCompare(compareSel.value));
    }

    function debounce(fn, ms) {
        let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
    }

    // ── Load model ────────────────────────────────────────────────
    async function loadModel(file) {
        if (!file) { currentData = null; hideAll(); return; }
        try {
            const resp = await fetch(DATA_DIR + file);
            currentData = await resp.json();
        } catch (e) {
            console.error("Failed to load model data", e);
            return;
        }

        showModelInfo();
        populateScriptFilter();
        populateCompareSelect(file);
        showChart();
        page = 0;
        applyFilters();
        updateHash();

        $("#model-info").hidden = false;
        $("#chart-section").hidden = false;
        $("#compare-section").hidden = false;
        $("#table-section").hidden = false;
    }

    function hideAll() {
        $("#model-info").hidden = true;
        $("#chart-section").hidden = true;
        $("#compare-section").hidden = true;
        $("#table-section").hidden = true;
    }

    // ── Model info ────────────────────────────────────────────────
    function showModelInfo() {
        const m = currentData.metadata;
        const vs = m.verification_summary;
        let html = "";
        const items = [
            ["Model", m.repo_id],
            ["Architecture", m.architecture],
            ["Vocab size", m.vocab_size.toLocaleString()],
            ["Hidden dim", m.hidden_size],
            ["Tied", m.tied ? "Yes" : "No"],
            ["Indicator", m.indicator],
        ];
        if (vs) {
            items.push(["Verified glitches", vs.confirmed_glitches + "/" + vs.total_verified]);
        }
        for (const [k, v] of items) {
            html += '<div class="meta-item"><strong>' + esc(k) + "</strong>" + esc(String(v)) + "</div>";
        }
        $("#model-meta").innerHTML = html;
    }

    // ── Script filter ─────────────────────────────────────────────
    function populateScriptFilter() {
        const scripts = new Set();
        for (const t of currentData.tokens) {
            for (const s of (t.scripts || [])) scripts.add(s);
        }
        scriptFilt.innerHTML = '<option value="">All</option>';
        for (const s of [...scripts].sort()) {
            const opt = document.createElement("option");
            opt.value = s; opt.textContent = s;
            scriptFilt.appendChild(opt);
        }
    }

    // ── Compare selector ──────────────────────────────────────────
    function populateCompareSelect(currentFile) {
        compareSel.innerHTML = '<option value="">-- select --</option>';
        for (const m of index.models) {
            if (m.file === currentFile) continue;
            const opt = document.createElement("option");
            opt.value = m.file;
            opt.textContent = m.repo_id;
            compareSel.appendChild(opt);
        }
        $("#compare-area").innerHTML = "";
    }

    async function loadCompare(file) {
        if (!file || !currentData) { $("#compare-area").innerHTML = ""; return; }
        try {
            const resp = await fetch(DATA_DIR + file);
            compareData = await resp.json();
        } catch (e) { return; }
        showComparison();
    }

    function showComparison() {
        if (!currentData || !compareData) return;
        // Find tokens in both datasets by token_id
        const map2 = new Map();
        for (const t of compareData.tokens) map2.set(t.id, t);

        const both = [];
        for (const t of currentData.tokens) {
            const t2 = map2.get(t.id);
            if (t2) both.push({ id: t.id, decoded: t.decoded, score1: t.score, score2: t2.score, rank1: t.rank, rank2: t2.rank });
        }
        both.sort((a, b) => a.rank1 - b.rank1);

        let html = "<p>" + both.length + " tokens appear in both top-2000 lists.</p>";
        if (both.length > 0) {
            html += "<table><thead><tr><th>ID</th><th>Token</th><th>Rank 1</th><th>Score 1</th><th>Rank 2</th><th>Score 2</th></tr></thead><tbody>";
            for (const t of both.slice(0, 100)) {
                html += "<tr><td>" + t.id + "</td><td class='tok-cell'>" + escTok(t.decoded) + "</td>"
                    + "<td>" + t.rank1 + "</td><td>" + t.score1.toFixed(6) + "</td>"
                    + "<td>" + t.rank2 + "</td><td>" + t.score2.toFixed(6) + "</td></tr>";
            }
            html += "</tbody></table>";
            if (both.length > 100) html += "<p>Showing first 100 of " + both.length + ".</p>";
        }
        $("#compare-area").innerHTML = html;
    }

    // ── Chart ─────────────────────────────────────────────────────
    function showChart() {
        const stats = currentData.script_stats || {};
        const entries = Object.entries(stats).sort((a, b) => b[1].total - a[1].total).slice(0, 20);
        if (!entries.length) { $("#chart-section").hidden = true; return; }

        const maxVal = Math.max(...entries.map(([, v]) => v.total));
        let html = '<div class="chart-legend">'
            + '<span class="leg-total">Total</span>'
            + '<span class="leg-flagged">Flagged</span>'
            + '<span class="leg-verified">Verified</span></div>';

        for (const [script, v] of entries) {
            const tw = (v.total / maxVal * 100).toFixed(1);
            const fw = (v.flagged / maxVal * 100).toFixed(1);
            const vw = ((v.verified_glitch || 0) / maxVal * 100).toFixed(1);
            html += '<div class="chart-bar-group">'
                + '<span class="chart-label" title="' + esc(script) + '">' + esc(script) + '</span>'
                + '<div class="chart-bars">'
                + '<div class="chart-bar total" style="width:' + tw + '%"></div>'
                + '<div class="chart-bar flagged" style="width:' + fw + '%"></div>'
                + '<div class="chart-bar verified" style="width:' + vw + '%"></div>'
                + '</div>'
                + '<span class="chart-count">' + v.total + '</span></div>';
        }
        $("#chart-area").innerHTML = html;
    }

    // ── Filter & sort ─────────────────────────────────────────────
    function applyFilters() {
        if (!currentData) return;
        let tokens = currentData.tokens;
        const q = searchIn.value.trim().toLowerCase();
        const sc = scriptFilt.value;

        filtered = tokens.filter((t) => {
            if (q) {
                const idMatch = String(t.id) === q;
                const textMatch = (t.decoded || "").toLowerCase().includes(q);
                if (!idMatch && !textMatch) return false;
            }
            if (sc && !(t.scripts || []).includes(sc)) return false;
            if (fSpecial.checked && !t.special) return false;
            if (fUnreach.checked && !t.unreachable) return false;
            if (fVerified.checked && !t.verified) return false;
            return true;
        });

        // Sort
        const sort = sortSel.value;
        filtered.sort((a, b) => {
            if (sort === "rank-asc") return a.rank - b.rank;
            if (sort === "score-asc") return a.score - b.score;
            if (sort === "score-desc") return b.score - a.score;
            if (sort === "id-asc") return a.id - b.id;
            return 0;
        });

        render();
        updateHash();
    }

    // ── Render table page ─────────────────────────────────────────
    function render() {
        const total = filtered.length;
        const pages = Math.ceil(total / PAGE_SIZE) || 1;
        if (page >= pages) page = pages - 1;
        if (page < 0) page = 0;

        const start = page * PAGE_SIZE;
        const slice = filtered.slice(start, start + PAGE_SIZE);

        countBadge.textContent = total.toLocaleString();
        pageInfo.textContent = "Page " + (page + 1) + " of " + pages;
        prevBtn.disabled = page === 0;
        nextBtn.disabled = page >= pages - 1;

        // Build rows
        let html = "";
        for (const t of slice) {
            const flags = [];
            if (t.special) flags.push('<span class="flag flag-special">special</span>');
            if (t.unreachable) flags.push('<span class="flag flag-unreachable">unreachable</span>');
            if (t.verified) flags.push('<span class="flag flag-verified">verified</span>');

            html += '<tr data-id="' + t.id + '">'
                + "<td>" + t.rank + "</td>"
                + "<td>" + t.id + "</td>"
                + '<td class="tok-cell">' + escTok(t.decoded) + "</td>"
                + "<td>" + t.score.toFixed(6) + "</td>"
                + "<td>" + flags.join("") + "</td>"
                + "<td>" + (t.scripts || []).join(", ") + "</td>"
                + "</tr>";
        }
        tbody.innerHTML = html;

        // Row click -> drawer
        for (const tr of tbody.querySelectorAll("tr")) {
            tr.addEventListener("click", () => {
                const id = parseInt(tr.dataset.id);
                const tok = currentData.tokens.find((t) => t.id === id);
                if (tok) showDrawer(tok);
            });
        }
    }

    // ── Token detail drawer ───────────────────────────────────────
    function showDrawer(t) {
        let html = "<h3>Token " + t.id + "</h3>";
        html += '<div class="detail-row"><strong>Rank:</strong> ' + t.rank + "</div>";
        html += '<div class="detail-row"><strong>Decoded:</strong> <code>' + escTok(t.decoded) + "</code></div>";
        html += '<div class="detail-row"><strong>Hex bytes:</strong> ' + (t.hex || "—") + "</div>";
        html += '<div class="detail-row"><strong>Score:</strong> ' + t.score.toFixed(8) + "</div>";
        html += '<div class="detail-row"><strong>Scripts:</strong> ' + (t.scripts || []).join(", ") + "</div>";

        const flags = [];
        if (t.special) flags.push("special");
        if (t.unreachable) flags.push("unreachable");
        if (t.partial) flags.push("partial-utf8");
        html += '<div class="detail-row"><strong>Flags:</strong> ' + (flags.join(", ") || "none") + "</div>";

        if (t.verified !== undefined) {
            html += '<div class="detail-row"><strong>Verified:</strong> ' + (t.verified ? "GLITCH" : "not glitch") + "</div>";
            html += '<div class="detail-row"><strong>Max prob:</strong> ' + (t.max_prob !== undefined ? t.max_prob.toFixed(8) : "—") + "</div>";
        } else {
            html += '<div class="detail-row"><strong>Verified:</strong> unverified (indicator only)</div>';
        }

        drawerContent.innerHTML = html;
        drawer.hidden = false;
    }

    // ── URL hash state ────────────────────────────────────────────
    function updateHash() {
        if (!modelSel.value) { history.replaceState(null, "", location.pathname); return; }
        const p = new URLSearchParams();
        p.set("m", modelSel.value);
        if (searchIn.value) p.set("q", searchIn.value);
        if (sortSel.value !== "rank-asc") p.set("s", sortSel.value);
        if (scriptFilt.value) p.set("sc", scriptFilt.value);
        history.replaceState(null, "", "#" + p.toString());
    }

    // ── Escape helpers ────────────────────────────────────────────
    function esc(s) {
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    function escTok(s) {
        // Escape HTML, then visualize whitespace/control chars
        let out = "";
        for (const ch of s) {
            const cp = ch.codePointAt(0);
            if (cp < 0x20 || cp === 0x7f) {
                // Control char -> visible label
                const names = { 0: "NUL", 7: "BEL", 8: "BS", 9: "TAB", 10: "LF", 13: "CR", 27: "ESC" };
                const label = names[cp] || "x" + cp.toString(16).padStart(2, "0");
                out += '<span style="opacity:.5;font-size:.7em">[' + label + ']</span>';
            } else if (cp === 0x20) {
                out += '<span style="opacity:.3">\u00b7</span>';
            } else if (cp === 0xa0) {
                out += '<span style="opacity:.3">NBSP</span>';
            } else {
                out += esc(ch);
            }
        }
        return out || '<span style="opacity:.4">(empty)</span>';
    }

    // ── Boot ──────────────────────────────────────────────────────
    init();
})();
