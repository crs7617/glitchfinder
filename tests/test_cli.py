"""Tests for cli.py — argument parsing and orchestration."""

import sys
from unittest.mock import patch, MagicMock

import pytest

from glitchfinder.cli import main


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
            )

    def test_analyze_with_options(self):
        with patch("glitchfinder.cli.run_analysis") as mock_run:
            with patch("sys.argv", [
                "glitchfinder", "analyze", "meta-llama/Llama-2-7b",
                "--no-verify", "--top-n", "500", "--dtype", "bfloat16",
                "--output-dir", "out",
            ]):
                main()
            mock_run.assert_called_once_with(
                repo_id="meta-llama/Llama-2-7b",
                cache_dir=None,
                output_dir="out",
                no_verify=True,
                top_n=500,
                dtype="bfloat16",
            )

    def test_no_command_prints_help(self):
        with patch("sys.argv", ["glitchfinder"]):
            with pytest.raises(SystemExit):
                main()
