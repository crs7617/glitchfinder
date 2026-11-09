// Glitchfinder Explorer — vanilla JS, no framework
(function () {
    "use strict";

    const PAGE_SIZE = 100;
    let currentData = null;
    let filteredTokens = [];
    let currentPage = 0;
    let loadedModels = {}; // name → data

    // DOM refs
    const modelSelect = document.getElementById("model-select");
    const modelSelect2 = document.getElementById("model-select-2");
    const fileInput = document.getElementById("file-input");
    const loadBtn = document.getElementById("load-btn");
    const searchInput = document.getElementById("search-input");
    const sortSelect = document.getElementById("sort-select");
    const scriptFilter = document.getElementById("script-filter");
    const flagSpecial = document.getElementById("flag-special");
    const flagUnreachable = document.getElementById("flag-unreachable");
    const flagPartial = document.getElementById("flag-partial");
    const flagGlitch = document.getElementById("flag-glitch");
    const metadataSection = document.getElementById("metadata");
    const metadataContent = document.getElementById("metadata-content");
    const scriptChart = document.getElementById("script-chart");
    const chartContainer = document.getElementById("chart-container");
    const resultsSection = document.getElementById("results");
    const resultCount = document.getElementById("result-count");
    const tbody = document.getElementById("token-tbody");
    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");
    const pageInfo = document.getElementById("page-info");
    const compareRow = document.getElementById("compare-row");
    const compareBtn = document.getElementById("compare-btn");
    const comparisonSection = document.getElementById("comparison");
    const compareTbody = document.getElementById("compare-tbody");

    // Load JSON file
    loadBtn.addEventListener("click", function () {
        const file = fileInput.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                const data = JSON.parse(e.target.result);
                const name = data.metadata?.repo_id || file.name;
                loadedModels[name] = data;
                addModelOption(name);
                modelSelect.value = name;
                activateModel(data);
            } catch (err) {
                alert("Failed to parse JSON: " + err.message);
            }
        };
        reader.readAsText(file);
    });

    modelSelect.addEventListener("change", function () {
        const name = this.value;
        if (name && loadedModels[name]) {
            activateModel(loadedModels[name]);
        }
    });

    function addModelOption(name) {
        if ([...modelSelect.options].some(o => o.value === name)) return;
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        modelSelect.appendChild(opt);

        const opt2 = opt.cloneNode(true);
        modelSelect2.appendChild(opt2);

        if (Object.keys(loadedModels).length >= 2) {
            compareRow.style.display = "flex";
        }
    }

    function activateModel(data) {
        currentData = data;
        currentPage = 0;
        showMetadata(data.metadata);
        populateScriptFilter(data.script_stats);
        showScriptChart(data.script_stats);
        applyFilters();
        metadataSection.style.display = "block";
        scriptChart.style.display = "block";
        resultsSection.style.display = "block";
    }

    function showMetadata(meta) {
        metadataContent.innerHTML = [
            `<div class="meta-item"><strong>Repo:</strong> ${esc(meta.repo_id)}</div>`,
            `<div class="meta-item"><strong>Arch:</strong> ${esc(meta.architecture)}</div>`,
            `<div class="meta-item"><strong>Vocab:</strong> ${meta.vocab_size.toLocaleString()}</div>`,
            `<div class="meta-item"><strong>Hidden:</strong> ${meta.hidden_size}</div>`,
            `<div class="meta-item"><strong>Tied:</strong> ${meta.tie_word_embeddings}</div>`,
            `<div class="meta-item"><strong>Indicator:</strong> ${esc(meta.indicator)}</div>`,
        ].join("");
    }

    function populateScriptFilter(stats) {
        scriptFilter.innerHTML = '<option value="">All scripts</option>';
        for (const name of Object.keys(stats).sort()) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = `${name} (${stats[name].total})`;
            scriptFilter.appendChild(opt);
        }
    }

    function showScriptChart(stats) {
        const entries = Object.entries(stats).sort((a, b) => b[1].total - a[1].total);
        const maxTotal = Math.max(...entries.map(e => e[1].total), 1);
        chartContainer.innerHTML = entries.map(([name, s]) => {
            const totalPct = (s.total / maxTotal * 100).toFixed(1);
            const flaggedPct = (s.flagged / maxTotal * 100).toFixed(1);
            return `<div class="bar-row">
                <span class="bar-label" title="${esc(name)}">${esc(name)}</span>
                <div class="bar-container">
                    <div class="bar-total" style="width:${totalPct}%"></div>
                    <div class="bar-flagged" style="width:${flaggedPct}%"></div>
                </div>
                <span class="bar-count">${s.total} / ${s.flagged}</span>
            </div>`;
        }).join("");
    }

    // Filtering and sorting
    function applyFilters() {
        if (!currentData) return;
        const search = searchInput.value.toLowerCase();
        const script = scriptFilter.value;
        const onlySpecial = flagSpecial.checked;
        const onlyUnreachable = flagUnreachable.checked;
        const onlyPartial = flagPartial.checked;
        const onlyGlitch = flagGlitch.checked;

        filteredTokens = currentData.tokens.filter(t => {
            if (search && !t.decoded.toLowerCase().includes(search)) return false;
            if (script && !(t.unicode_scripts || []).includes(script)) return false;
            if (onlySpecial && !t.is_special) return false;
            if (onlyUnreachable && !t.is_unreachable) return false;
            if (onlyPartial && !t.is_partial_utf8) return false;
            if (onlyGlitch && !(t.verification && t.verification.is_glitch)) return false;
            return true;
        });

        sortTokens();
        currentPage = 0;
        renderTable();
    }

    function sortTokens() {
        const mode = sortSelect.value;
        filteredTokens.sort((a, b) => {
            switch (mode) {
                case "score-asc": return a.score - b.score;
                case "score-desc": return b.score - a.score;
                case "id-asc": return a.token_id - b.token_id;
                case "id-desc": return b.token_id - a.token_id;
                default: return a.rank - b.rank;
            }
        });
    }

    function renderTable() {
        const start = currentPage * PAGE_SIZE;
        const end = Math.min(start + PAGE_SIZE, filteredTokens.length);
        const page = filteredTokens.slice(start, end);
        const totalPages = Math.ceil(filteredTokens.length / PAGE_SIZE) || 1;

        resultCount.textContent = `(${filteredTokens.length} tokens)`;
        pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
        prevBtn.disabled = currentPage === 0;
        nextBtn.disabled = end >= filteredTokens.length;

        tbody.innerHTML = page.map(t => {
            const flags = [];
            if (t.is_special) flags.push('<span class="flag-badge flag-special">special</span>');
            if (t.is_unreachable) flags.push('<span class="flag-badge flag-unreachable">unreachable</span>');
            if (t.is_partial_utf8) flags.push('<span class="flag-badge flag-partial">partial</span>');

            let verif = "";
            if (t.verification) {
                if (t.verification.is_glitch) {
                    verif = `<span class="flag-badge flag-glitch">GLITCH (${t.verification.max_prob.toFixed(4)})</span>`;
                } else if (t.verification.max_prob >= 0) {
                    verif = `<span class="flag-badge flag-ok">OK (${t.verification.max_prob.toFixed(4)})</span>`;
                } else {
                    verif = '<span class="flag-badge" style="background:#555">unverified</span>';
                }
            }

            return `<tr>
                <td>${t.rank}</td>
                <td>${t.token_id}</td>
                <td title="${esc(t.decoded)}">${esc(t.decoded)}</td>
                <td>${t.score.toFixed(6)}</td>
                <td>${flags.join("") || "-"}</td>
                <td>${(t.unicode_scripts || []).join(", ") || "-"}</td>
                <td>${verif || "-"}</td>
            </tr>`;
        }).join("");
    }

    // Event listeners for filters
    searchInput.addEventListener("input", applyFilters);
    sortSelect.addEventListener("change", applyFilters);
    scriptFilter.addEventListener("change", applyFilters);
    flagSpecial.addEventListener("change", applyFilters);
    flagUnreachable.addEventListener("change", applyFilters);
    flagPartial.addEventListener("change", applyFilters);
    flagGlitch.addEventListener("change", applyFilters);
    prevBtn.addEventListener("click", () => { currentPage--; renderTable(); });
    nextBtn.addEventListener("click", () => { currentPage++; renderTable(); });

    // Comparison
    compareBtn.addEventListener("click", function () {
        const name2 = modelSelect2.value;
        if (!name2 || !loadedModels[name2] || !currentData) return;
        const data2 = loadedModels[name2];

        // Find tokens flagged in both (top 2% by rank in both)
        const cutoff1 = Math.ceil(currentData.tokens.length * 0.02);
        const cutoff2 = Math.ceil(data2.tokens.length * 0.02);
        const top1 = new Map(currentData.tokens.slice(0, cutoff1).map(t => [t.token_id, t]));
        const top2 = new Map(data2.tokens.slice(0, cutoff2).map(t => [t.token_id, t]));

        const common = [];
        for (const [id, t1] of top1) {
            if (top2.has(id)) {
                common.push({ token_id: id, decoded: t1.decoded, score1: t1.score, score2: top2.get(id).score });
            }
        }

        compareTbody.innerHTML = common.map(c =>
            `<tr><td>${c.token_id}</td><td>${esc(c.decoded)}</td><td>${c.score1.toFixed(6)}</td><td>${c.score2.toFixed(6)}</td></tr>`
        ).join("") || "<tr><td colspan='4'>No common tokens found</td></tr>";

        comparisonSection.style.display = "block";
    });

    function esc(str) {
        const d = document.createElement("div");
        d.textContent = str || "";
        return d.innerHTML;
    }
})();
