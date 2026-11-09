# Stage 2: tokenizer_pass.py — Token Heuristics

## Files

### `src/glitchfinder/tokenizer_pass.py`
**What it does:** Loads a HuggingFace tokenizer and classifies every token in the vocabulary by structural properties — before looking at embeddings at all. This produces the `TokenRecord` list that all downstream stages use for filtering and reference token selection.

**Key functions:**
- `_extract_raw_bytes(token_str)` — Converts a token string to raw bytes. Handles byte-fallback tokenizers (Llama/Gemma) where individual bytes are represented as `<0xAB>` patterns. Without this, we couldn't distinguish real text tokens from byte tokens.
- `_detect_unicode_scripts(text)` — Approximates Unicode script assignment using `unicodedata.name()`. Maps first word of the Unicode name (e.g., "LATIN", "CJK", "ARABIC") to script labels. Used for per-script statistics in the report.
- `analyze_vocab(repo_id, vocab_size)` — The main function. Iterates all token IDs, for each one: decodes it, checks if it's special, extracts raw bytes, tests UTF-8 validity, checks roundtrip reachability (encode(decode(id)) == [id]?), detects scripts. Returns a list of TokenRecord.
- `find_reference_token_ids(records)` — Selects tokens that are known-unused (special+unreachable, empty decode) to serve as the "garbage centroid" for the cosine distance indicator. Needs >= 10; falls back to partial UTF-8 tokens if not enough.

**What breaks if removed:** The indicator stage (stage 3) needs reference token IDs for tied models. The report stage needs TokenRecord flags. The verification stage needs records to exclude already-flagged tokens.

### `tests/test_tokenizer_pass.py`
16 tests covering:
- Byte extraction: normal tokens, single/multi byte-fallback `<0xAB>`, hex case insensitivity, empty string, Unicode
- Script detection: Latin, CJK, Arabic, Cyrillic, mixed, empty, punctuation (Common)
- Reference token selection: enough special+unreachable, fallback to partial UTF-8, empty decode inclusion

**What tests prove:** Byte-fallback parsing is correct. Script detection works for major scripts. Reference selection meets the >= 10 threshold with appropriate fallbacks.
**What tests don't prove:** Integration with a real tokenizer (would need network). The `is_unreachable` logic with real tokenizer roundtrip behavior. SentencePiece leading-space handling.

## One thing that could still be wrong
The Unicode script detection is a heuristic based on the first word of `unicodedata.name()`. This misclassifies some characters — e.g., "DIGIT NINE" maps to "Common" via the category fallback, but "SUPERSCRIPT TWO" would map to the unknown "Superscript" script instead of "Common". A proper implementation would use the `regex` package's `\p{Script=Latin}` support.

---

## Questions

1. The `is_unreachable` check does `encode(decode([id])) != [id]`. For merge-based tokenizers (BPE), a token like "ing" is reachable but only as part of "running" — it would still roundtrip as `["ing"]`. Is this the right definition, or should we also test reachability from natural text?

2. `find_reference_token_ids` requires >= 10 reference tokens. For models with very small vocabularies (e.g., character-level models with ~256 tokens), we might not find 10. Should we lower the threshold, or is character-level out of scope?

3. The script detection maps "FULLWIDTH" and "HALFWIDTH" as separate scripts. CJK models often have many fullwidth Latin characters. Should these be merged into their base scripts (fullwidth A → Latin), or is keeping them separate useful for the per-script chart?

<details><summary>Suggested answers</summary>

1. Roundtrip is the right definition for this purpose. We're looking for tokens the model literally cannot produce via normal tokenization — these are the strongest "known unused" signals. Natural-text reachability is a much harder question and not needed for reference selection.

2. Character-level models are out of scope. They don't have the BPE/SentencePiece structure that creates glitch tokens. Document this in EXPLAIN.md.

3. Keep them separate. Fullwidth tokens in CJK models are a distinct category — they often include tokens that exist for legacy encoding compatibility and are plausible glitch candidates. The per-script chart is more informative with them separated.

</details>
