# Site Redesign

## What changed
Complete rewrite of `site/index.html`, `site/style.css`, `site/app.js`.
Added self-hosted fonts in `site/fonts/`.

## Design decisions (see also docs/DECISIONS.md)
- **Typography**: Source Serif 4 (display/body), JetBrains Mono (tokens/numbers)
- **Color**: near-black ground (#0c0c0e), warm off-white text (#e8e4df), signal-red
  accent (#ff3b30). Light theme: warm paper (#f5f3ef).
- **Layout**: left-aligned, no centered hero, scientific-instrument aesthetic
- **Architecture**: ES modules, no build step, all relative paths

## Features
1. **Hero**: left-aligned claim, 5 specimen tokens from loaded model data (never
   hardcoded), subtle CSS glitch animation, prefers-reduced-motion respected
2. **Vocabulary map**: canvas grid of all tokens by ID, colored by indicator
   percentile, devicePixelRatio-aware, hover tooltip, click opens drawer
3. **Explorer**: paginated table, sticky header, score mini-bars, whitespace shown
   as chips (LF, TAB, CR, NBSP), flags as badges, verified as check mark
4. **Script chart**: flagged RATE per writing system (sorted), scripts < 30 tokens
   grouped as "Other", renamed Common/Unknown/Control to plain language
5. **Compare**: small-multiple mini canvases on same scale
6. **Method & Limits**: honest prose with exact verified counts
7. **Polish**: keyboard (/ search, Esc close, arrows move), URL hash state,
   theme toggle persisted to localStorage, copy-link button, empty/loading/error
   states, sr-only H1, proper landmarks and heading hierarchy

## What breaks if removed
The entire user-facing interface.

## What tests prove and don't prove
**Prove**: Screenshots confirm layout, sections render, no JS errors.
**Don't prove**: Real user interaction, screen reader compatibility beyond basic checks.

## One thing that could still be wrong
The vocabulary map for very large vocabs (150k+) requires many pixel writes and may
be slow on low-end devices. Could benefit from WebGL or offscreen canvas.
