# Glitchfinder Progress

| Stage | Status | Commit | Tests | Note |
|-------|--------|--------|-------|------|
| 0 | done | a75f8fc | no tests (scaffolding) | pyproject, .gitignore, __init__, results dir |
| 1 | done | 6271f55 | 13 passed | models.py: config parsing, tensor name map, embedding loading |
| 2 | done | ff22118 | 16 passed | tokenizer_pass.py: byte-fallback, scripts, reachability, reference selection |
| 3 | done | 0931bed | 15 passed | indicators.py: L2 norm, cosine/euclidean distance, ranking, candidate selection |
| 4 | done | 5ec6ae3 | 9 passed | verify.py: size gating, 3 prompt templates, probability extraction |
| 5 | done | bd7c027 | 12 passed | report.py + cli.py: JSON output, argparse, 5-step pipeline |
| 6 | done | 1f85904 | no tests (static site) | HTML explorer: search, sort, filter, chart, comparison |
| 7 | done | efeb966 | no tests (docs) | README, EXPLAIN.md, stage notes, PROGRESS |
| Step 1 | done | 7b45700 | 70 passed | Loader: NoSafetensorsError, local-dir loading, offline e2e test |
| Step 2 | done | 21884a1 | 77 passed | Windows UTF-8 safety, --show-top flag, _safe_token_repr |
| Step 3 | done | 44a5e83 | 77 passed | Real run on Qwen3-0.6B, bf16 fix, Qwen3 arch support |
| Step 4 | done | 7a8a9f1 | 77 passed | Paper check: indicators match arXiv 2405.05417 exactly |
| Step 5 | done | aded21c | 77 passed | Verification: Qwen3-0.6B 19/19 glitches confirmed |
| Step 6 | done | c55c227 | 77 passed | 5 models analyzed, GPT-2 tensor fix, batch script |
| Step 7 | done | eb5c0b8 | 77 passed | Site data: 5 models, 1.75 MB total, index.json manifest |
| Step 8 | done | 08fbb81 | 77 passed | Site rebuild: dark/light, search, filters, drawer, compare |
| Step 9 | done | 28787f3 | 77 passed | Chart PNG: per-script flagged tokens by model |
| Step 10 | done | (this) | 77 passed | README, Pages workflow, stage docs, final PROGRESS |
