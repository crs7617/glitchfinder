"""Command-line interface for glitchfinder."""

from __future__ import annotations

import argparse
import gc
import io
import sys

import numpy as np


def _safe_token_repr(s: str) -> str:
    """Return a console-safe representation of a token string.

    Escapes non-printable characters and partial-byte sequences so
    printing never crashes on Windows consoles with limited encodings.
    """
    out = []
    for ch in s:
        if ch.isprintable() and ch != "\x7f":
            out.append(ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(f"\\u{ord(ch):04x}")
    return "".join(out)


def _ensure_utf8_stdout():
    """Reconfigure stdout/stderr to UTF-8 with error replacement.

    On Windows, the default console encoding (cp1252, etc.) crashes when
    printing CJK, Arabic, or other non-Latin token strings.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_analysis(
    repo_id: str,
    cache_dir: str | None = None,
    output_dir: str = "results",
    no_verify: bool = False,
    top_n: int = 2000,
    dtype: str = "float32",
    show_top: int = 0,
) -> None:
    """Run the full glitchfinder pipeline."""
    from . import models, tokenizer_pass, indicators, verify, report

    # Step 1: Load embeddings
    print(f"[1/5] Loading embeddings from {repo_id}...")
    try:
        emb = models.load_embeddings(repo_id, cache_dir=cache_dir)
    except models.NoSafetensorsError as e:
        print(f"  ERROR: {e}")
        return
    info = emb.info
    print(f"  Vocab: {info.vocab_size}, Hidden: {info.hidden_size}, "
          f"Tied: {info.tie_word_embeddings}, Arch: {info.architecture}")

    # Step 2: Analyze vocabulary
    print(f"[2/5] Analyzing tokenizer vocabulary...")
    records = tokenizer_pass.analyze_vocab(repo_id, info.vocab_size, cache_dir=cache_dir)
    n_special = sum(1 for r in records if r.is_special)
    n_unreachable = sum(1 for r in records if r.is_unreachable)
    n_partial = sum(1 for r in records if r.is_partial_utf8)
    print(f"  Special: {n_special}, Unreachable: {n_unreachable}, Partial UTF-8: {n_partial}")

    # Step 3: Compute indicators
    print(f"[3/5] Computing indicators...")
    if info.tie_word_embeddings:
        # For tied models: use cosine distance from reference tokens
        ref_ids = tokenizer_pass.find_reference_token_ids(records)
        print(f"  Using {len(ref_ids)} reference tokens for cosine distance indicator")
        scores_array = indicators.cosine_distance_indicator(emb.e_in, ref_ids)
        indicator_name = "cosine_distance"
        lower_is_glitchy = True  # closer to ref = more suspicious
    else:
        # For untied models: use L2 norm on input embeddings
        scores_array = indicators.l2_norm_indicator(emb.e_in)
        indicator_name = "l2_norm"
        lower_is_glitchy = True  # lower norm = more suspicious

    # Build scores dict and rank
    scores = {i: float(scores_array[i]) for i in range(len(scores_array))}
    ranked_ids = indicators.rank_candidates(scores_array, lower_is_glitchy=lower_is_glitchy)

    # Select verification candidates
    candidates = indicators.select_verification_candidates(ranked_ids, records)
    # Limit to top_n
    candidates = candidates[:top_n]
    print(f"  {len(candidates)} verification candidates selected")

    # Free embedding memory before verification
    del emb
    del scores_array
    gc.collect()

    # Step 4: Verification
    verification = {}
    if not no_verify:
        print(f"[4/5] Verifying candidates...")
        results = verify.verify_candidates(
            repo_id, candidates, info, dtype=dtype, cache_dir=cache_dir
        )
        verification = {r.token_id: r for r in results}
        n_glitch = sum(1 for r in results if r.is_glitch)
        n_unverified = sum(1 for r in results if r.max_prob < 0)
        print(f"  Verified glitches: {n_glitch}, Unverified: {n_unverified}")
    else:
        print(f"[4/5] Skipping verification (--no-verify)")

    # Step 5: Build and save report
    print(f"[5/5] Building report...")
    rpt = report.build_report(
        info=info,
        records=records,
        scores=scores,
        verification=verification if verification else None,
        indicator_name=indicator_name,
        ranked_ids=ranked_ids.tolist()[:top_n],
    )
    out_path = report.save_report(rpt, output_dir, repo_id)
    print(f"  Report saved to {out_path}")

    # Print top tokens if requested
    if show_top > 0 and rpt.get("tokens"):
        print(f"\n  Top {show_top} suspicious tokens:")
        print(f"  {'Rank':>5} {'ID':>7} {'Score':>10} {'Flags':>20} Token")
        print(f"  {'-'*5} {'-'*7} {'-'*10} {'-'*20} {'-'*30}")
        for tok in rpt["tokens"][:show_top]:
            flags = []
            if tok["is_special"]:
                flags.append("special")
            if tok["is_unreachable"]:
                flags.append("unreachable")
            if tok["is_partial_utf8"]:
                flags.append("partial-utf8")
            if tok.get("verification", {}).get("is_glitch"):
                flags.append("VERIFIED")
            flag_str = ",".join(flags) if flags else "-"
            safe_decoded = _safe_token_repr(tok["decoded"])
            print(f"  {tok['rank']:>5} {tok['token_id']:>7} {tok['score']:>10.6f} {flag_str:>20} {safe_decoded!r}")

        # Also show 10 random ordinary tokens for comparison
        import random
        ordinary = [t for t in rpt["tokens"] if not t["is_special"]
                    and not t["is_unreachable"] and not t["is_partial_utf8"]]
        if ordinary:
            sample = random.sample(ordinary, min(10, len(ordinary)))
            sample.sort(key=lambda t: t["rank"])
            print(f"\n  10 random ordinary tokens for comparison:")
            print(f"  {'Rank':>5} {'ID':>7} {'Score':>10} Token")
            print(f"  {'-'*5} {'-'*7} {'-'*10} {'-'*30}")
            for tok in sample:
                safe_decoded = _safe_token_repr(tok["decoded"])
                print(f"  {tok['rank']:>5} {tok['token_id']:>7} {tok['score']:>10.6f} {safe_decoded!r}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="glitchfinder",
        description="Detect under-trained (glitch) tokens in open-weight LLMs",
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze subcommand
    analyze = subparsers.add_parser("analyze", help="Analyze a model for glitch tokens")
    analyze.add_argument("repo_id", help="HuggingFace model repo ID")
    analyze.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    analyze.add_argument("--output-dir", default="results", help="Output directory")
    analyze.add_argument("--no-verify", action="store_true", help="Skip model verification")
    analyze.add_argument("--top-n", type=int, default=2000, help="Max candidates to report")
    analyze.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Dtype for verification model",
    )
    analyze.add_argument(
        "--show-top", type=int, default=0,
        help="Print top N suspicious tokens to console",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        _ensure_utf8_stdout()
        run_analysis(
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            no_verify=args.no_verify,
            top_n=args.top_n,
            dtype=args.dtype,
            show_top=args.show_top,
        )
    else:
        parser.print_help()
        sys.exit(1)
