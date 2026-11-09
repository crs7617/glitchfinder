"""Tests for report.py — JSON report generation."""

import json
import tempfile
from pathlib import Path

import pytest

from glitchfinder.report import (
    _sanitize_filename,
    build_report,
    save_report,
    compute_script_stats,
)
from glitchfinder.models import ModelInfo
from glitchfinder.tokenizer_pass import TokenRecord
from glitchfinder.verify import VerificationResult


def _make_record(tid, decoded="tok", special=False, unreachable=False,
                 partial=False, scripts=None):
    return TokenRecord(
        token_id=tid,
        decoded_str=decoded,
        raw_bytes=decoded.encode("utf-8"),
        is_special=special,
        is_partial_utf8=partial,
        is_unreachable=unreachable,
        unicode_scripts=scripts or {"Latin"},
    )


def _make_info():
    return ModelInfo(
        repo_id="test/model",
        vocab_size=10,
        hidden_size=64,
        tie_word_embeddings=True,
        architecture="GPT2LMHeadModel",
    )


class TestSanitizeFilename:
    def test_slash_replacement(self):
        assert _sanitize_filename("meta-llama/Llama-2-7b") == "meta-llama--Llama-2-7b"

    def test_special_chars(self):
        result = _sanitize_filename("model:v1.0")
        assert ":" not in result

    def test_simple_name(self):
        assert _sanitize_filename("gpt2") == "gpt2"


class TestBuildReport:
    def test_report_structure(self):
        info = _make_info()
        records = [_make_record(i) for i in range(10)]
        scores = {i: float(i) for i in range(10)}

        report = build_report(info, records, scores, None, "l2_norm")

        assert "metadata" in report
        assert "script_stats" in report
        assert "tokens" in report
        assert report["metadata"]["repo_id"] == "test/model"
        assert report["metadata"]["indicator"] == "l2_norm"
        assert len(report["tokens"]) == 10

    def test_token_entries_have_required_fields(self):
        info = _make_info()
        records = [_make_record(i) for i in range(5)]
        scores = {i: float(i) * 0.1 for i in range(5)}

        report = build_report(info, records, scores, None, "cosine_distance")
        tok = report["tokens"][0]

        required = ["rank", "token_id", "decoded", "score",
                     "is_special", "is_unreachable", "is_partial_utf8"]
        for field in required:
            assert field in tok, f"Missing field: {field}"

    def test_verification_included(self):
        info = _make_info()
        records = [_make_record(i) for i in range(5)]
        scores = {i: float(i) for i in range(5)}
        verif = {0: VerificationResult(0, 0.005, True, [0.001, 0.003, 0.005])}

        report = build_report(info, records, scores, verif, "l2_norm")
        tok0 = [t for t in report["tokens"] if t["token_id"] == 0][0]

        assert "verification" in tok0
        assert tok0["verification"]["is_glitch"] is True
        assert tok0["verification"]["max_prob"] == 0.005


class TestSaveReport:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = {"metadata": {}, "tokens": []}
            path = save_report(report, tmpdir, "test/model")
            assert path.exists()
            assert path.name == "test--model.json"

            with open(path) as f:
                data = json.load(f)
            assert data == report

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sub" / "dir"
            save_report({"data": 1}, str(out), "m")
            assert out.exists()


class TestComputeScriptStats:
    def test_basic_stats(self):
        records = [
            _make_record(0, scripts={"Latin"}, special=True),
            _make_record(1, scripts={"Latin"}),
            _make_record(2, scripts={"CJK"}),
            _make_record(3, scripts={"CJK"}, unreachable=True),
        ]
        stats = compute_script_stats(records)
        assert stats["Latin"]["total"] == 2
        assert stats["Latin"]["flagged"] == 1
        assert stats["CJK"]["total"] == 2
        assert stats["CJK"]["flagged"] == 1
