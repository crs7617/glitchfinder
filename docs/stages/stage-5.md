# Stage 5: report.py + cli.py — Output & Orchestration

## Files

### `src/glitchfinder/report.py`
**What it does:** Transforms raw analysis results into a structured JSON report.

**Key functions:**
- `_sanitize_filename(name)` — Converts repo IDs like `meta-llama/Llama-2-7b` to safe filenames (`meta-llama--Llama-2-7b`). Without this, saving would fail on Windows (/ is illegal in filenames).
- `compute_script_stats(records, scores, verification)` — Groups tokens by Unicode script, counts total/flagged/verified_glitch per script. Powers the per-script bar chart in the site.
- `build_report(info, records, scores, verification, indicator_name, ranked_ids)` — Assembles the full JSON structure: metadata block, script_stats, and a tokens array ordered by rank. Each token entry has rank, id, decoded string, score, all flags, scripts, and optional verification results.
- `save_report(report, output_dir, model_name)` — Creates output directory if needed, writes JSON with `ensure_ascii=False` (preserves Unicode). Returns the path.

**What breaks if removed:** No output. The CLI runs the whole pipeline but has nowhere to write results.

### `src/glitchfinder/cli.py`
**What it does:** The entry point. Wires all modules together in the correct order with memory management.

**Key functions:**
- `run_analysis(repo_id, ...)` — The 5-step pipeline: (1) load embeddings, (2) analyze vocab, (3) compute indicators + rank + select candidates, (4) free embeddings then verify, (5) build and save report. The memory lifecycle is critical: embeddings are deleted before the verification model is loaded.
- `main()` — Argparse with `analyze` subcommand. Flags: `--cache-dir`, `--output-dir`, `--no-verify`, `--top-n`, `--dtype`.

**What breaks if removed:** No CLI. The `glitchfinder` command doesn't work. Modules can still be used as a library.

### `tests/test_report.py`
9 tests: filename sanitization (slash, special chars, simple), report structure validation, required fields in token entries, verification data inclusion, file creation, directory auto-creation, script stats grouping.

### `tests/test_cli.py`
3 tests: basic arg parsing, all options parsing, no-command exits with help.

**What tests prove:** Report JSON has the right structure. Filename sanitization is safe. CLI args parse correctly.
**What tests don't prove:** The full pipeline end-to-end (would need real model download). Memory lifecycle (embeddings freed before model load). Report size/performance with large vocabularies.

## One thing that could still be wrong
The CLI chooses cosine_distance for tied models and l2_norm for untied. But `cosine_distance_indicator` is called with `emb.e_in` (input embeddings) for tied models, not `e_out`. This is technically correct because tied means e_in == e_out, but semantically the function parameter is named `e_out`. Could confuse future readers.

---

## Questions

1. The report includes ALL tokens ordered by rank, even the 30k+ non-suspicious ones. For large vocab models this creates huge JSON files (50MB+). Should we truncate to `--top-n` in the report, or keep the full list for the explorer's search/filter features?

2. `run_analysis` picks the indicator based solely on `tie_word_embeddings`. Some models claim to be untied but actually share weights (quantization artifacts). Should we add a check — e.g., compare `np.allclose(e_in, e_out)` — to detect this?

3. The CLI has no `--indicator` flag — it auto-selects. Should we expose the choice (l2_norm, cosine_distance, euclidean_distance) as a user option, or is auto-selection based on tied/untied the right default?

<details><summary>Suggested answers</summary>

1. Truncate to `--top-n` in the report. The explorer only needs suspicious tokens, and a 50MB JSON defeats the purpose of a lightweight static site. Non-suspicious tokens aren't interesting. Add a note in the metadata about total vocab size vs reported count.

2. Good idea but low priority. `np.allclose` on two (vocab_size, hidden_dim) arrays is cheap. Add it as a warning, not a behavior change — if they're close, log "embeddings appear tied despite config" and use cosine distance.

3. Auto-selection is correct as the default. The paper's methodology is clear about which indicator to use when. Exposing it as a flag adds complexity for advanced users but shouldn't be the default path. Could add `--indicator` as an advanced option later.

</details>
