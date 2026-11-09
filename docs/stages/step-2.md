# Step 2: Windows UTF-8 Safety

## What changed
- **cli.py**: Added `_ensure_utf8_stdout()` that reconfigures stdout/stderr to UTF-8
  with `errors="replace"` so printing CJK/Arabic/etc. tokens never crashes on Windows
  consoles. Added `_safe_token_repr()` that escapes control characters and non-printable
  bytes. Added `--show-top N` flag to print top suspicious tokens to console.
- **test_cli.py**: 7 new tests for `_safe_token_repr` and updated arg parsing tests.

## Why it exists
Windows consoles default to cp1252 or similar encodings. Printing a CJK token like
`\u4e16\u754c` would raise `UnicodeEncodeError`. The reconfigure ensures UTF-8, and
the repr fallback escapes anything truly unprintable.

## What breaks if removed
- Running on Windows with default console encoding would crash when printing non-Latin tokens
- No way to inspect results from the command line without opening the JSON

## What tests prove and don't prove
**Prove**: printable chars pass through, control chars are escaped, CJK is preserved.
**Don't prove**: actual Windows console behavior (tests run in UTF-8 terminal).

## One thing that could still be wrong
`sys.stdout.reconfigure` might not work on all Python builds (e.g. embedded interpreters).

---

### Quiz

1. Why use `errors="replace"` instead of `errors="strict"` for stdout?
2. Why does `_safe_token_repr` not use Python's built-in `repr()`?
3. What Windows encoding typically causes the crash this step prevents?

<details>
<summary>Answers</summary>

1. `"replace"` substitutes unencodable characters with `?` instead of raising an
   exception. This is safer for a CLI tool that might encounter any Unicode codepoint.
2. `repr()` wraps the string in quotes and escapes differently. We want the token
   content to be readable (CJK stays as-is) while only escaping true control characters.
3. cp1252 (Windows-1252), the default ANSI code page for Western European Windows.
</details>
