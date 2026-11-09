"""Run glitchfinder analysis on a list of models."""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Reconfigure stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from glitchfinder.cli import run_analysis
from glitchfinder.models import NoSafetensorsError

# Models to analyze. All ungated, all have safetensors.
# verify_ok: True if <=1B params and we want to run verification
# All models run with --no-verify first (indicator pass only).
# Verification is done separately on select small models via verify_batch.py.
MODELS = [
    {"repo_id": "Qwen/Qwen3-0.6B",              "note": "152k vocab, tied, verified separately"},
    {"repo_id": "Qwen/Qwen2.5-0.5B",             "note": "152k vocab, tied"},
    {"repo_id": "HuggingFaceTB/SmolLM2-360M",    "note": "49k vocab, tied"},
    {"repo_id": "openai-community/gpt2",          "note": "50k vocab, tied"},
    {"repo_id": "microsoft/phi-1_5",              "note": "51k vocab, untied"},
]


def main():
    skip_existing = "--skip-existing" in sys.argv
    models = MODELS

    # Allow filtering by model name
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        names = sys.argv[1:]
        models = [m for m in MODELS if m["repo_id"] in names]

    for i, m in enumerate(models):
        repo = m["repo_id"]
        no_verify = True  # always --no-verify in batch; verify separately
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(models)}] {repo}")
        if m.get("note"):
            print(f"  Note: {m['note']}")
        print(f"{'='*60}")

        if skip_existing:
            from glitchfinder.report import _sanitize_filename
            out_path = os.path.join("results", _sanitize_filename(repo) + ".json")
            if os.path.exists(out_path):
                print(f"  Skipping (already exists: {out_path})")
                continue

        t0 = time.time()
        try:
            run_analysis(
                repo_id=repo,
                no_verify=no_verify,
                show_top=30,
            )
        except NoSafetensorsError as e:
            print(f"  SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print("All models processed.")


if __name__ == "__main__":
    main()
