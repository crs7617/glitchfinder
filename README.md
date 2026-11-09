# glitchfinder

Detect under-trained ("glitch") tokens in open-weight LLMs by analyzing embedding matrices and tokenizer metadata.

```
  Tokens by Unicode Script (example: GPT-2)
  ──────────────────────────────────────────
  Latin       ████████████████████████████░░  38012 total, 42 flagged
  Common      ██████████░░░░░░░░░░░░░░░░░░░░  8234 total, 15 flagged
  CJK         █████░░░░░░░░░░░░░░░░░░░░░░░░░  2891 total, 203 flagged
  Cyrillic    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   847 total, 89 flagged
  Arabic      █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   312 total, 67 flagged
  Unknown     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    61 total, 61 flagged
```

Based on the methodology from ["Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models"](https://arxiv.org/abs/2405.05417) (Rumbelow & Watkins, 2024). Independent implementation — no code from cohere-ai/magikarp.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
```

This installs CPU-only PyTorch automatically via the configured index.

## Usage

```bash
# Analyze a model (indicators only, no verification)
uv run glitchfinder analyze openai-community/gpt2 --no-verify

# Analyze with verification (model must be <=1B params)
uv run glitchfinder analyze openai-community/gpt2

# Control output
uv run glitchfinder analyze meta-llama/Llama-2-7b-hf \
    --no-verify --top-n 500 --output-dir results
```

Results are saved as JSON in `results/`.

## How It Works

1. **Load embeddings** — reads only the embedding tensors from safetensors files (never loads the full model)
2. **Analyze tokenizer** — classifies each token: special, unreachable, partial UTF-8, Unicode script
3. **Compute indicators** — L2 norm (untied models) or cosine distance from reference centroid (tied models)
4. **Verify candidates** — loads the full model (<=1B only) and tests if candidates can be predicted
5. **Generate report** — JSON with per-token scores, flags, verification, and per-script statistics

See [docs/EXPLAIN.md](docs/EXPLAIN.md) for a detailed explanation of each step.

## Requirements and Limitations

- **Safetensors only.** Models must publish `.safetensors` weights (most modern models do).
- **CPU only, 8GB RAM.** Embedding matrices must fit in memory. Very large vocabs (128k+) with large hidden dims may not fit.
- **Verification: <=1B parameters (float32).** Larger models produce candidates labeled "unverified."
- **Indicators are heuristics.** Low L2 norm or proximity to the garbage centroid is a strong signal, not a guarantee. Verification confirms; without it, results are candidates only.
- **Three prompt templates.** Verification coverage is not exhaustive.

## Testing

```bash
uv run pytest -q
```

## Project Structure

```
src/glitchfinder/
    __init__.py          Package root
    models.py            Embedding extraction from safetensors
    tokenizer_pass.py    Token classification heuristics
    indicators.py        Statistical indicators (pure functions)
    verify.py            Model-based verification
    report.py            JSON report generation
    cli.py               CLI entry point and pipeline orchestration
site/                    Static HTML explorer (open index.html)
docs/                    Explanations and stage notes
results/                 Output directory for JSON reports
```

## Citation

This tool implements the methodology described in:

> Rumbelow, J., & Watkins, M. (2024). Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models. *arXiv:2405.05417*.

## License

This is an educational/research project. Not affiliated with or endorsed by the paper's authors.
