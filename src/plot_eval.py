"""
Aggregate comparison plot across evaluation settings.

Reads `results.csv` from any number of run directories (typically one each for
zero_shot / few_shot / supervised) and renders a single figure where every
(task, metric, dataset) triple lives in its own subplot with one grouped bar
per (method, setting) combination. This is the final "testing metrics"
comparison the project report hangs off of.

Usage:
    python -m src.plot_eval \
        --runs outputs/testing/20260419_020246_v1 \
               outputs/testing/<fewshot_run> \
               outputs/testing/<supervised_run> \
        --out outputs/testing/final_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SETTING_ORDER = ["zero_shot", "few_shot", "supervised"]
SETTING_COLORS = {
    "zero_shot": "#4C72B0",
    "few_shot": "#DD8452",
    "supervised": "#55A868",
}

NC_METRICS = [("acc", "Accuracy"), ("macro_f1", "Macro-F1")]
LP_METRICS = [("auc", "AUC"), ("ap", "AP")]


def load_runs(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for rd in run_dirs:
        csv = rd / "results.csv"
        if not csv.exists():
            print(f"  skip (no results.csv): {rd}")
            continue
        df = pd.read_csv(csv)
        if "setting" not in df.columns:
            # legacy CSV (pre-setting column): it's always zero_shot
            df["setting"] = "zero_shot"
        df["run_dir"] = str(rd)
        frames.append(df)
        print(f"  loaded {len(df):4d} rows from {rd}")
    if not frames:
        raise FileNotFoundError("no results.csv found in any --runs dir")
    return pd.concat(frames, ignore_index=True)


def _plot_metric_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    metric: str,
    title: str,
) -> None:
    """One subplot: x=method, grouped bars per setting."""
    settings = [s for s in SETTING_ORDER if s in sub["setting"].unique()]
    methods = sorted(sub["method"].unique())
    if not methods or not settings:
        ax.set_title(title + "  (no data)")
        ax.axis("off")
        return

    x = np.arange(len(methods))
    width = 0.8 / max(len(settings), 1)

    for i, s in enumerate(settings):
        vals = []
        for m in methods:
            cell = sub[(sub["method"] == m) & (sub["setting"] == s)][metric]
            vals.append(float(cell.iloc[0]) if len(cell) else np.nan)
        offset = (i - (len(settings) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width,
               color=SETTING_COLORS.get(s, None), label=s)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)


def make_comparison(df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(df["dataset"].unique())
    tasks = sorted(df["task"].unique())

    task_metrics = {
        "node_classification": NC_METRICS,
        "link_prediction": LP_METRICS,
    }

    # row per (task, metric), col per dataset
    rows = sum(len(task_metrics.get(t, [])) for t in tasks)
    cols = len(datasets)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False,
    )

    r = 0
    for task in tasks:
        for key, label in task_metrics.get(task, []):
            for c, dname in enumerate(datasets):
                ax = axes[r][c]
                sub = df[(df["task"] == task) & (df["dataset"] == dname)]
                _plot_metric_panel(
                    ax, sub, key, f"{task[:2].upper()} {label} — {dname}",
                )
            r += 1

    # one shared legend at the top
    handles, labels = [], []
    for s, col in SETTING_COLORS.items():
        if s in df["setting"].unique():
            handles.append(plt.Rectangle((0, 0), 1, 1, color=col))
            labels.append(s)
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.005), ncol=len(handles),
                   frameon=False, fontsize=10)
    fig.suptitle("NetFM — final comparison across evaluation settings",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_leaderboard(df: pd.DataFrame, out_path: Path) -> None:
    """Markdown / plain-text leaderboard grouped by (task, dataset, setting)."""
    lines: list[str] = ["# NetFM — combined leaderboard", ""]
    for (task, dataset), g in df.groupby(["task", "dataset"]):
        lines.append(f"\n## {task}  —  {dataset}")
        if task == "node_classification":
            cols = ["setting", "method", "acc", "top5_acc", "macro_f1", "weighted_f1"]
            sort = "acc"
        else:
            cols = ["setting", "method", "auc", "ap", "hits_50", "hits_100", "mrr"]
            sort = "auc"
        keep = [c for c in cols if c in g.columns]
        sub = g[keep].sort_values([sort], ascending=False)
        lines.append(sub.to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate NetFM evaluation plots")
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more run dirs (e.g. outputs/testing/<id>)")
    p.add_argument("--out", type=str, default="outputs/testing/final_comparison.png")
    p.add_argument("--leaderboard", type=str,
                   default="outputs/testing/final_leaderboard.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(r) for r in args.runs]
    df = load_runs(run_dirs)
    make_comparison(df, Path(args.out))
    make_leaderboard(df, Path(args.leaderboard))


if __name__ == "__main__":
    main()
