# Stage 6: site/ — Static HTML Explorer

## Files

### `site/index.html`
**What it does:** The main page. Layout with controls (model selector, file loader, search, sort, script filter, flag checkboxes), metadata panel, per-script bar chart, paginated token table, and comparison section. No framework — pure semantic HTML.

**What breaks if removed:** No web UI. Users can still use the CLI and read JSON directly.

### `site/style.css`
**What it does:** Dark theme UI. Styled controls, table, bar chart (CSS-width-based bars for flagged/total per script), pagination, flag badges color-coded by type (purple=special, red=unreachable, orange=partial, bright red=glitch, green=OK).

### `site/app.js`
**What it does:** All client-side logic in an IIFE.

**Key functionality:**
- **File loading:** FileReader API reads a local JSON file, parses it, populates the model selector.
- **Filtering:** Search by decoded string, filter by Unicode script dropdown, checkboxes for special/unreachable/partial/verified-glitch-only. All filters compose (AND logic).
- **Sorting:** By score (asc/desc) or token ID (asc/desc).
- **Bar chart:** CSS-width bars showing total vs flagged token counts per Unicode script, scaled to the max script count.
- **Pagination:** 100 tokens per page with prev/next buttons.
- **Model comparison:** Load 2 models, click Compare to find tokens in the top 2% of both. Shows common flagged tokens side by side.
- **Escaping:** All user-controlled strings go through `esc()` (creates a text node then reads innerHTML) to prevent XSS.

**What tests prove and don't prove:** No automated tests. This is a static site — testing requires manual browser interaction. The XSS-safe `esc()` function is the main security concern; it uses DOM-based escaping which is correct.

## One thing that could still be wrong
The model comparison uses top 2% by rank position in the token array, but the token array order depends on how the report was built (may or may not be rank-ordered). If a report is built without `ranked_ids`, the comparison would use arbitrary ordering. The app should sort by score before selecting the top 2%.

---

## Questions

1. The explorer loads JSON via FileReader (user picks a local file). An alternative is to scan a `results/` directory via fetch. But that requires either a file server or a manifest file listing available JSONs. Is the file-picker approach sufficient, or should we add a `results/manifest.json` that the site fetches?

2. The bar chart uses CSS widths for bars. For scripts with very few tokens (e.g., Cherokee with 2 tokens out of 32k), the bar is invisible. Should we add a minimum width or log scale?

3. The comparison feature finds tokens in the top 2% of both models. The 2% threshold is hardcoded. Should it be configurable in the UI, or is 2% a reasonable fixed default?

<details><summary>Suggested answers</summary>

1. File-picker is sufficient for now. A manifest.json would require a build step or server. The file-picker works offline, which is valuable for a tool that handles potentially sensitive model analysis. Could add manifest support later as an enhancement.

2. Add a minimum width of 2px. Log scale would distort the visual comparison. The count label already shows exact numbers, so a barely-visible bar with a "2/0" label is informative enough.

3. 2% is a reasonable fixed default that matches the indicator stage's `top_fraction`. Making it configurable adds UI complexity with minimal benefit — users who want different thresholds can adjust the report's `--top-n` instead.

</details>
