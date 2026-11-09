"""Build trimmed site data from results/*.json files.

Outputs:
  site/data/<model>.json  — top 2000 tokens per model, trimmed fields
  site/data/index.json    — manifest listing all models with summary stats
"""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
SITE_DATA_DIR = Path("site/data")
MAX_TOKENS = 2000


def _sanitize_name(repo_id: str) -> str:
    """Convert repo_id to a safe filename."""
    import re
    name = repo_id.replace("/", "--")
    return re.sub(r"[^\w\-.]", "_", name)


def trim_report(report: dict) -> dict:
    """Trim a report to the top MAX_TOKENS tokens with essential fields only."""
    tokens = report.get("tokens", [])[:MAX_TOKENS]
    trimmed_tokens = []
    for t in tokens:
        entry = {
            "rank": t["rank"],
            "id": t["token_id"],
            "decoded": t.get("decoded", ""),
            "hex": t.get("raw_bytes_hex", ""),
            "score": round(t["score"], 6),
            "special": t.get("is_special", False),
            "unreachable": t.get("is_unreachable", False),
            "partial": t.get("is_partial_utf8", False),
            "scripts": t.get("unicode_scripts", []),
        }
        if "verification" in t:
            entry["verified"] = t["verification"]["is_glitch"]
            entry["max_prob"] = round(t["verification"]["max_prob"], 8)
        trimmed_tokens.append(entry)

    meta = report.get("metadata", {})
    return {
        "metadata": {
            "repo_id": meta.get("repo_id", ""),
            "vocab_size": meta.get("vocab_size", 0),
            "hidden_size": meta.get("hidden_size", 0),
            "architecture": meta.get("architecture", ""),
            "tied": meta.get("tie_word_embeddings", True),
            "indicator": meta.get("indicator", ""),
            "verification_summary": meta.get("verification_summary"),
        },
        "script_stats": report.get("script_stats", {}),
        "tokens": trimmed_tokens,
    }


def build_index(models: list[dict]) -> dict:
    """Build an index manifest from model summaries."""
    return {
        "models": models,
        "generated_by": "glitchfinder build_site_data.py",
    }


def main():
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_summaries = []
    total_size = 0

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        if "verification" in json_file.name:
            continue  # skip auxiliary verification files

        print(f"Processing {json_file.name}...")
        with open(json_file, encoding="utf-8") as f:
            report = json.load(f)

        trimmed = trim_report(report)
        meta = trimmed["metadata"]

        # Count flags
        tokens = trimmed["tokens"]
        n_verified = sum(1 for t in tokens if t.get("verified", False))
        n_special = sum(1 for t in tokens if t.get("special", False))
        n_unreachable = sum(1 for t in tokens if t.get("unreachable", False))
        n_partial = sum(1 for t in tokens if t.get("partial", False))

        out_name = _sanitize_name(meta["repo_id"]) + ".json"
        out_path = SITE_DATA_DIR / out_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=True, separators=(",", ":"))

        file_size = out_path.stat().st_size
        total_size += file_size

        model_summaries.append({
            "repo_id": meta["repo_id"],
            "file": out_name,
            "vocab_size": meta["vocab_size"],
            "architecture": meta["architecture"],
            "tied": meta["tied"],
            "indicator": meta["indicator"],
            "token_count": len(tokens),
            "verified_glitches": n_verified,
            "special": n_special,
            "unreachable": n_unreachable,
            "partial": n_partial,
            "file_size_kb": round(file_size / 1024, 1),
        })

        print(f"  -> {out_path} ({file_size/1024:.1f} KB, {len(tokens)} tokens)")

    # Write index
    index = build_index(model_summaries)
    index_path = SITE_DATA_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=True)

    print(f"\nTotal site data: {total_size/1024:.1f} KB ({total_size/1024/1024:.2f} MB)")
    print(f"Index: {index_path}")
    print(f"Models: {len(model_summaries)}")

    if total_size > 5 * 1024 * 1024:
        print("WARNING: Total size exceeds 5 MB limit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
