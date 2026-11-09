"""Generate a chart PNG from site/data showing flagged tokens by script and model."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SITE_DATA = Path("site/data")
OUT_PATH = Path("docs/chart.png")


def main():
    with open(SITE_DATA / "index.json", encoding="utf-8") as f:
        index = json.load(f)

    # Collect per-model script stats
    models = []
    all_scripts = set()
    for m in index["models"]:
        with open(SITE_DATA / m["file"], encoding="utf-8") as f:
            data = json.load(f)
        stats = data.get("script_stats", {})
        # Short model name
        name = m["repo_id"].split("/")[-1]
        models.append({"name": name, "stats": stats})
        all_scripts.update(stats.keys())

    # Pick top N scripts by total flagged across all models
    script_totals = {}
    for s in all_scripts:
        script_totals[s] = sum(
            m["stats"].get(s, {}).get("flagged", 0) for m in models
        )
    top_scripts = sorted(script_totals, key=script_totals.get, reverse=True)[:12]

    # Build matrix: models x scripts (flagged count)
    model_names = [m["name"] for m in models]
    matrix = np.zeros((len(models), len(top_scripts)))
    for i, m in enumerate(models):
        for j, s in enumerate(top_scripts):
            matrix[i, j] = m["stats"].get(s, {}).get("flagged", 0)

    # Stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_scripts)))

    left = np.zeros(len(models))
    bars = []
    for j, s in enumerate(top_scripts):
        bar = ax.barh(model_names, matrix[:, j], left=left, color=colors[j], label=s, height=0.6)
        bars.append(bar)
        left += matrix[:, j]

    ax.set_xlabel("Flagged tokens in top 2000")
    ax.set_title("Flagged tokens by Unicode script and model")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.invert_yaxis()
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
    print(f"Chart saved to {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    plt.close()


if __name__ == "__main__":
    main()
