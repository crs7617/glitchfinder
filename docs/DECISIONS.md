# Decisions Log

## Step 1: Loader Fix
- **Decision**: Instead of trying to support .bin weights (pytorch_model.bin), we fail
  gracefully with a clear "no safetensors" message. safetensors is the only supported
  format. Rationale: the project explicitly uses `safe_open` and never loads full models
  for indicator work. Supporting .bin would require torch.load which contradicts the
  memory-safety design.
- **Decision**: Replace `sshleifer/tiny-gpt2` references with offline synthetic test
  fixtures. This avoids network dependency in tests and the 404 issue entirely.
- **Decision**: Batch runs (multi-model) will catch `NoSafetensorsError` and skip,
  logging a warning.

## Step 2: Windows UTF-8
- **Decision**: All file writes use `encoding="utf-8"`. Token printing uses
  `repr()` for non-printable/partial-byte tokens to avoid console encoding crashes.

## Step 3: Real Run
- **Decision**: Use Qwen/Qwen3-0.6B as primary test model (has safetensors, ungated).

## Step 4: Paper Check
- **Decision**: Will quote arXiv 2405.05417 Section 3 in docs/EXPLAIN.md and verify
  tied vs untied indicator selection matches.
