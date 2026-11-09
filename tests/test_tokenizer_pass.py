"""Tests for tokenizer_pass.py — token heuristics."""

from unittest.mock import MagicMock

import pytest

from glitchfinder.tokenizer_pass import (
    _extract_raw_bytes,
    _detect_unicode_scripts,
    TokenRecord,
    find_reference_token_ids,
)


class TestExtractRawBytes:
    """Test byte extraction from token strings."""

    def test_normal_token(self):
        assert _extract_raw_bytes("hello") == b"hello"

    def test_single_byte_fallback(self):
        """<0xAB> pattern should be converted to actual byte."""
        result = _extract_raw_bytes("<0xAB>")
        assert result == bytes([0xAB])

    def test_multi_byte_fallback(self):
        result = _extract_raw_bytes("<0xE2><0x80><0x99>")
        assert result == bytes([0xE2, 0x80, 0x99])

    def test_hex_case_insensitive(self):
        result = _extract_raw_bytes("<0xab>")
        assert result == bytes([0xAB])

    def test_empty_string(self):
        assert _extract_raw_bytes("") == b""

    def test_unicode_token(self):
        result = _extract_raw_bytes("cafe\u0301")  # café with combining accent
        assert result == "cafe\u0301".encode("utf-8")


class TestDetectUnicodeScripts:
    """Test Unicode script detection."""

    def test_latin(self):
        scripts = _detect_unicode_scripts("hello")
        assert "Latin" in scripts

    def test_cjk(self):
        scripts = _detect_unicode_scripts("\u4e16\u754c")  # 世界
        assert "CJK" in scripts

    def test_mixed(self):
        scripts = _detect_unicode_scripts("hello\u4e16")
        assert "Latin" in scripts
        assert "CJK" in scripts

    def test_empty(self):
        scripts = _detect_unicode_scripts("")
        assert len(scripts) == 0

    def test_punctuation(self):
        scripts = _detect_unicode_scripts(".,!")
        # Punctuation is Common
        assert "Common" in scripts

    def test_arabic(self):
        scripts = _detect_unicode_scripts("\u0627\u0644")
        assert "Arabic" in scripts

    def test_cyrillic(self):
        scripts = _detect_unicode_scripts("\u041f\u0440")
        assert "Cyrillic" in scripts


class TestFindReferenceTokenIds:
    """Test reference token selection."""

    def _make_record(self, tid, special=False, unreachable=False,
                     partial=False, decoded=" "):
        return TokenRecord(
            token_id=tid,
            decoded_str=decoded,
            raw_bytes=b"",
            is_special=special,
            is_partial_utf8=partial,
            is_unreachable=unreachable,
        )

    def test_enough_special_unreachable(self):
        """When we have >=10 special+unreachable tokens, use those."""
        records = []
        # 15 special+unreachable tokens
        for i in range(15):
            records.append(self._make_record(i, special=True, unreachable=True))
        # 85 normal tokens
        for i in range(15, 100):
            records.append(self._make_record(i, decoded="word"))
        refs = find_reference_token_ids(records)
        assert len(refs) >= 10
        assert all(r < 15 for r in refs)

    def test_fallback_to_partial_utf8(self):
        """When not enough special+unreachable, include partial UTF-8."""
        records = []
        # Only 3 special+unreachable
        for i in range(3):
            records.append(self._make_record(i, special=True, unreachable=True))
        # 10 partial UTF-8
        for i in range(3, 13):
            records.append(self._make_record(i, partial=True, decoded="x"))
        # Normal
        for i in range(13, 100):
            records.append(self._make_record(i, decoded="word"))
        refs = find_reference_token_ids(records)
        assert len(refs) >= 10

    def test_empty_decode_included(self):
        """Tokens with empty decode should be reference candidates."""
        records = []
        for i in range(12):
            records.append(self._make_record(i, decoded=""))
        for i in range(12, 100):
            records.append(self._make_record(i, decoded="word"))
        refs = find_reference_token_ids(records)
        assert len(refs) >= 10
