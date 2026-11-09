"""Token-level heuristics from tokenizer metadata."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class TokenRecord:
    token_id: int
    decoded_str: str
    raw_bytes: bytes
    is_special: bool
    is_partial_utf8: bool
    is_unreachable: bool
    unicode_scripts: set[str] = field(default_factory=set)
    is_defined: bool = True  # all codepoints are defined Unicode characters


# ── Byte-fallback helpers ────────────────────────────────────────────────────

_BYTE_TOKEN_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _extract_raw_bytes(token_str: str) -> bytes:
    """Convert a token string (possibly with byte-fallback patterns) to raw bytes.

    Byte-fallback tokenizers (Llama, Gemma) represent individual bytes as
    <0xAB> tokens. This function detects that pattern and converts to actual
    bytes. For normal tokens, it just encodes as UTF-8.
    """
    # Check if the entire token is a single byte-fallback pattern
    m = _BYTE_TOKEN_RE.fullmatch(token_str)
    if m:
        return bytes([int(m.group(1), 16)])

    # Check if the token is a sequence of byte-fallback patterns
    parts = _BYTE_TOKEN_RE.findall(token_str)
    if parts and _BYTE_TOKEN_RE.sub("", token_str) == "":
        return bytes(int(h, 16) for h in parts)

    # Normal token
    try:
        return token_str.encode("utf-8")
    except UnicodeEncodeError:
        return token_str.encode("utf-8", errors="replace")


def _detect_unicode_scripts(text: str) -> set[str]:
    """Return the set of Unicode script names for all characters in text.

    Uses unicodedata to categorize each character. Characters that are
    Common (punctuation, digits) or Unknown are reported as such.
    """
    scripts = set()
    for ch in text:
        try:
            # unicodedata doesn't have a script() function in stdlib,
            # so we approximate using the character's Unicode category
            # and name. For a proper implementation we'd need the
            # `unicategories` or `regex` package, but we can get a
            # reasonable approximation from the Unicode name.
            name = unicodedata.name(ch, "")
            if not name:
                scripts.add("Unknown")
                continue

            # Heuristic: extract script from Unicode name
            # e.g., "LATIN SMALL LETTER A" → "Latin"
            #        "CJK UNIFIED IDEOGRAPH-4E00" → "CJK"
            #        "ARABIC LETTER BEH" → "Arabic"
            first_word = name.split()[0]
            script_map = {
                "LATIN": "Latin",
                "CJK": "CJK",
                "ARABIC": "Arabic",
                "CYRILLIC": "Cyrillic",
                "GREEK": "Greek",
                "HEBREW": "Hebrew",
                "DEVANAGARI": "Devanagari",
                "HIRAGANA": "Hiragana",
                "KATAKANA": "Katakana",
                "HANGUL": "Hangul",
                "THAI": "Thai",
                "GEORGIAN": "Georgian",
                "ARMENIAN": "Armenian",
                "ETHIOPIC": "Ethiopic",
                "TIBETAN": "Tibetan",
                "BENGALI": "Bengali",
                "TAMIL": "Tamil",
                "TELUGU": "Telugu",
                "KANNADA": "Kannada",
                "MALAYALAM": "Malayalam",
                "MYANMAR": "Myanmar",
                "KHMER": "Khmer",
                "LAO": "Lao",
                "MONGOLIAN": "Mongolian",
                "SINHALA": "Sinhala",
                "GUJARATI": "Gujarati",
                "GURMUKHI": "Gurmukhi",
                "ORIYA": "Oriya",
                "JAVANESE": "Javanese",
                "SUNDANESE": "Sundanese",
                "CHEROKEE": "Cherokee",
                "CANADIAN": "Canadian",
                "TAGALOG": "Tagalog",
                "FULLWIDTH": "Fullwidth",
                "HALFWIDTH": "Halfwidth",
            }
            script = script_map.get(first_word)
            if script:
                scripts.add(script)
            else:
                cat = unicodedata.category(ch)
                if cat.startswith("L"):
                    scripts.add(first_word.title())
                elif cat.startswith(("P", "S", "Z", "N")):
                    scripts.add("Common")
                elif cat.startswith("C"):
                    scripts.add("Control")
                else:
                    scripts.add("Common")
        except Exception:
            scripts.add("Unknown")
    return scripts


def _is_defined_char(ch: str) -> bool:
    """Check if a character is a defined Unicode codepoint."""
    return unicodedata.category(ch) != "Cn"


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze_vocab(
    repo_id: str,
    vocab_size: int,
    cache_dir: Optional[str] = None,
) -> list[TokenRecord]:
    """Analyze all tokens in the vocabulary using the tokenizer.

    For each token, we determine:
    - decoded string representation
    - whether it's a special token
    - whether it represents partial/invalid UTF-8
    - whether it's "unreachable" (encoding the decoded form doesn't roundtrip)
    - which Unicode scripts appear in the decoded form
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    special_ids = set(tokenizer.all_special_ids)
    records: list[TokenRecord] = []

    for token_id in range(vocab_size):
        # Decode the token
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        is_special = token_id in special_ids

        # Get the raw token string (before detokenization)
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if raw_token is None:
            raw_token = ""

        # Extract raw bytes
        raw_bytes = _extract_raw_bytes(raw_token)

        # Check if it's partial/invalid UTF-8
        is_partial_utf8 = False
        try:
            raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            is_partial_utf8 = True

        # Check unreachability: does encode(decode(id)) roundtrip?
        is_unreachable = False
        if decoded.strip() == "" and not is_special:
            is_unreachable = True
        else:
            try:
                re_encoded = tokenizer.encode(decoded, add_special_tokens=False)
                if re_encoded != [token_id]:
                    is_unreachable = True
            except Exception:
                is_unreachable = True

        # Unicode script detection
        scripts = _detect_unicode_scripts(decoded) if decoded else set()

        # Check if all characters are defined Unicode codepoints
        is_defined = all(_is_defined_char(ch) for ch in decoded) if decoded else True

        records.append(TokenRecord(
            token_id=token_id,
            decoded_str=decoded,
            raw_bytes=raw_bytes,
            is_special=is_special,
            is_partial_utf8=is_partial_utf8,
            is_unreachable=is_unreachable,
            unicode_scripts=scripts,
            is_defined=is_defined,
        ))

    return records


def find_reference_token_ids(records: list[TokenRecord]) -> list[int]:
    """Find token IDs suitable as reference points for glitch detection.

    Reference tokens are known-unused tokens: special tokens that are also
    unreachable, or tokens with empty decode. These serve as a baseline
    for comparison — glitch tokens should look similar to these in the
    embedding space.

    We need at least 10 reference tokens. If we can't find enough from
    special+unreachable, we fall back to partial UTF-8 tokens.
    """
    # Primary: special and unreachable tokens, or empty decode
    refs = [
        r.token_id for r in records
        if (r.is_special and r.is_unreachable) or r.decoded_str.strip() == ""
    ]

    if len(refs) >= 10:
        return refs

    # Fallback: add partial UTF-8 tokens
    partial = [r.token_id for r in records if r.is_partial_utf8]
    refs = list(set(refs + partial))

    if len(refs) >= 10:
        return refs[:max(10, len(refs))]

    # Last resort: all unreachable tokens
    unreachable = [r.token_id for r in records if r.is_unreachable]
    refs = list(set(refs + unreachable))
    return refs
