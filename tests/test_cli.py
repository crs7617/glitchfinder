"""Tests for cli.py — argument parsing and orchestration."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from glitchfinder.cli import main, _safe_token_repr


class TestArgParsing:
    """Test CLI argument parsing."""

    def test_analyze_basic(self):
        """Basic analyze command should parse correctly."""
        with patch("glitchfinder.cli.run_analysis") as mock_run:
            with patch("sys.argv", ["glitchfinder", "analyze", "test/model"]):
                main()
            mock_run.assert_called_once_with(
                repo_id="test/model",
                cache_dir=None,
                output_dir="results",
                no_verify=False,
                top_n=2000,
                dtype="float32",
                show_top=0,
            )

    def test_analyze_with_options(self):
        with patch("glitchfinder.cli.run_analysis") as mock_run:
            with patch("sys.argv", [
                "glitchfinder", "analyze", "meta-llama/Llama-2-7b",
                "--no-verify", "--top-n", "500", "--dtype", "bfloat16",
                "--output-dir", "out", "--show-top", "30",
            ]):
                main()
            mock_run.assert_called_once_with(
                repo_id="meta-llama/Llama-2-7b",
                cache_dir=None,
                output_dir="out",
                no_verify=True,
                top_n=500,
                dtype="bfloat16",
                show_top=30,
            )

    def test_no_command_prints_help(self):
        with patch("sys.argv", ["glitchfinder"]):
            with pytest.raises(SystemExit):
                main()


class TestSafeTokenRepr:
    """Test console-safe token representation."""

    def test_printable_unchanged(self):
        assert _safe_token_repr("hello") == "hello"

    def test_newline_escaped(self):
        assert _safe_token_repr("a\nb") == "a\\nb"

    def test_tab_escaped(self):
        assert _safe_token_repr("\t") == "\\t"

    def test_null_byte_escaped(self):
        assert _safe_token_repr("\x00") == "\\u0000"

    def test_cjk_preserved(self):
        # CJK characters are printable, should pass through
        assert _safe_token_repr("\u4e16\u754c") == "\u4e16\u754c"

    def test_control_char_escaped(self):
        assert _safe_token_repr("\x01\x02") == "\\u0001\\u0002"

    def test_mixed(self):
        result = _safe_token_repr("hello\x00world\n")
        assert result == "hello\\u0000world\\n"
