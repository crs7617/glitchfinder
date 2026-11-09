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
- Run ALL remaining stages (1-6) autonomously, in order, without waiting for me.
- At each checkpoint: run `uv run pytest -q`; stage and commit ONLY that stage's
  files by explicit path ("stage N: <what>" + 2-3 line body); run `git status`
  and assign any leftover uncommitted file to a stage or delete it.
- After each commit, write docs/stages/stage-N.md: what each file/function does,
  why it exists, what breaks if removed, what the tests prove and don't prove,
  one thing that could still be wrong, then 3 questions for me with answers
  hidden at the bottom of the file. Then continue immediately to the next stage.
- Maintain docs/PROGRESS.md: stage, status, commit hash, test result, 1-line note.
- STOP and report only if: tests still fail after 2 fix attempts; a decision
  would change scope; a command is destructive; RAM use exceeds ~3 GB; or a
  download exceeds ~2 GB. Otherwise keep going.
- Time-box: if a stage runs past ~45 minutes, simplify it, log the cut in
  PROGRESS.md, and move on.
- Local commits only. Never push. Never commit weights, .venv, caches, or files
  over 5 MB.
- Scope: pipeline on 3-5 ungated models; verification (transformers+torch CPU,
  float32, <=1B only) on 1-2; label the rest "candidates, unverified".
- Finish with: README (chart first, credit arXiv 2405.05417, honest limits),
  a per-script chart script, and a final summary of what's done and what's cut.