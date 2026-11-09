# glitchfinder
Detects under-trained ("glitch") tokens in open-weight LLMs from embedding
matrices + tokenizer, verifies on small models, publishes a static explorer.

## Constraints
- Windows, 8 GB RAM, CPU only. NEVER load a full model for indicator work:
  read only the embedding tensor(s) via safetensors safe_open.
- Verification (running the model) only for models <= ~1.7B.

## Commands
- test: `uv run pytest -q`
- run:  `uv run glitchfinder analyze <hf_repo_id>`

## Rules
- Indicator functions are pure (arrays in, scores out), heavily commented.
- All outputs go to results/<model>.json. Never commit weights.
- Don't invent paper formulas; read arXiv 2405.05417 and ask me if unclear.

## Workflow
- Work in stages. After each stage: run `uv run pytest -q`, show me the output, then commit.
- Commit only that stage's files by explicit path (never `git add -A` or `git add .`). Check `git status` and .gitignore first. Never commit weights, .venv, caches, or result files over 5 MB.
- Commit message: "stage N: <what>" plus a 2-3 line body on why.
- After each commit, explain in plain language: what each file and function does, why it exists, what breaks if it's removed, what the tests prove and what they don't, and one thing that could still be wrong. Then ask me 3 questions about it, one at a time, and wait for my answers.
- After the questions, STOP. Don't start the next stage until I say "go".
- Local commits only. Never push, never force anything.