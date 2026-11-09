# Step 5: Verification on Qwen3-0.6B

## What changed
- Ran full verification of all 19 non-flagged candidates from the top 2000
- Merged verification results back into the main report JSON
- All 19/19 confirmed as glitches (max_prob < 1%)

## Results
| Token ID | Token | max_prob | Notes |
|----------|-------|----------|-------|
| 45972 | `webElementX` | 0.000000 | Selenium artifact |
| 75171 | `\tNdrFcShort` | 0.000010 | Tab-prefixed code fragment |
| 151661 | `<\|fim_suffix\|>` | 0.000001 | FIM token not in special list |
| 151660 | `<\|fim_middle\|>` | 0.000001 | FIM token not in special list |
| 151202 | U+FC19 (Arabic) | 0.000000 | Rare Arabic ligature |
| 123306 | U+2A917 (CJK) | 0.000001 | Rare CJK ideograph |
| 151619 | U+133E7 (Egyptian) | 0.000002 | Egyptian hieroglyph |

Remaining 1981 tokens are flagged (unreachable/special) — labeled "candidates, unverified."

## Timing
- Model load: ~90s
- Per-token verification: ~50s average (3 prompts x 4 forward passes each)
- Total: ~20 minutes for 19 tokens

## What breaks if removed
The report would have no verified glitches — all tokens would be "candidates, unverified."

## What tests prove and don't prove
**Prove**: The verification pipeline runs end-to-end on a real model.
**Don't prove**: That the 1981 flagged tokens are truly glitches (they're structural flags only).

## One thing that could still be wrong
The 3-prompt verification might miss tokens that are predictable in specific contexts
not covered by our templates (e.g., code completion, specific languages).

---

### Quiz

1. Why were `<|fim_suffix|>` and `<|fim_middle|>` detected as glitches?
2. What is the "1% threshold" and where does it come from?
3. Why don't we verify the 1981 flagged tokens?

<details>
<summary>Answers</summary>

1. They're in the tokenizer but not in the model's special_ids list, so our code doesn't
   flag them as "special." The model was likely not trained with FIM objectives, so it
   never learned to predict these tokens. They have near-zero probability in all prompts.
2. A token is classified as a glitch if its maximum probability across all verification
   prompts is below 1% (0.01). This comes from arXiv 2405.05417 Section 2.3.
3. They're already known to be problematic by structure (unreachable, special, partial
   UTF-8). Verifying them would waste time and wouldn't add information — we already
   know the model can't meaningfully use them.
</details>
