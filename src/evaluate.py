"""
Downstream evaluation runner for NetFM + baselines.

Three evaluation settings (see architecture.md §5.2):
  * zero_shot  — frozen encoder + LogisticRegression / dot-product
  * few_shot   — fine-tune netfm (and subsample labels for others)
                 on K labels per class; netfm becomes encoder+head joint
  * supervised — train a fresh supervised GCN end-to-end on full labels;
                 frozen baselines are still reported as a reference

Plots are organised as `plots/<task>/<dataset>/<method>.png` so every
(task, dataset) pair has its own folder.

Usage:
    python -m src.evaluate \
        --checkpoint outputs/training/<run_id>/encoder.pt \
        --setting few_shot --k-per-class 10 \
        --run-name v1_fewshot
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.baselines import (
    ALL_EMBEDDERS,
    embed_netfm,
    embed_node2vec,
    embed_random,
    embed_structural,
    embed_structural as embed_structural_for_sup,
    embed_svd,
    supervised_link_prediction,
    supervised_node_classification,
)
from src.data import HELDOUT_DATASETS, load_dataset
from src.features import DEFAULT_SVD_DIM
from src.finetune import (
    few_shot_subsample,
    finetune_netfm_link_prediction,
    finetune_netfm_node_classification,
)
from src.tasks import (
    LinkPredResult,
    NodeClsResult,
    make_edge_split,
    make_node_split,
    plot_link_prediction,
    plot_node_classification,
    result_to_row,
    run_link_prediction,
    run_node_classification,
)


SETTINGS = ("zero_shot", "few_shot", "supervised")

CSV_COLS = [
    "setting", "method", "task", "dataset",
    "acc", "top5_acc", "macro_f1", "weighted_f1",
    "auc", "ap", "hits_50", "hits_100", "mrr",
    "train_size", "test_size", "num_classes",
]


# ---------------------------------------------------------------------------
# Per-method embedding
# ---------------------------------------------------------------------------

def get_embeddings(
    method: str,
    graph,
    checkpoint: Path | None,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    if method == "random":
        return embed_random(graph.num_nodes, d=256, seed=seed)
    if method == "structural":
        return embed_structural(graph)
    if method == "svd":
        return embed_svd(graph)
    if method == "netfm":
        if checkpoint is None:
            raise ValueError("NetFM requires --checkpoint")
        return embed_netfm(graph, checkpoint, device)
    if method == "node2vec":
        return embed_node2vec(graph, device)
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Per-task dispatch
# ---------------------------------------------------------------------------

def can_node_classify(graph) -> bool:
    y = graph.node_labels
    return y is not None and y.ndim == 1


def can_link_predict(graph) -> bool:
    return graph.edge_index is not None and graph.edge_index.size(1) > 10


def _plot_path(plots_dir: Path, task: str, dataset: str, method: str) -> Path:
    d = plots_dir / task / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{method}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.setting not in SETTINGS:
        raise ValueError(f"--setting must be one of {SETTINGS}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{run_id}_{args.run_name}"
    run_dir = Path(args.outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"run_id: {run_id}")
    print(f"run dir: {run_dir}")
    print(f"setting: {args.setting}")
    print(f"device: {device}")

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLS).writeheader()

    for dname in datasets:
        print(f"\n{'='*70}\nDATASET: {dname}\n{'='*70}")
        graph = load_dataset(dname)
        labels = graph.node_labels.cpu().numpy() if graph.node_labels is not None else None

        do_ncls = can_node_classify(graph)
        do_lp = can_link_predict(graph)

        ncls_split = (
            make_node_split(graph.num_nodes, seed=args.seed) if do_ncls else None
        )
        if do_lp:
            lp_pos_src, lp_pos_dst, lp_neg = make_edge_split(
                graph.edge_index, graph.num_nodes,
                held_frac=args.held_frac, seed=args.seed,
            )
        else:
            lp_pos_src = lp_pos_dst = lp_neg = None

        # Few-shot subsampling (labels for NC). Done once per dataset, shared.
        if args.setting == "few_shot" and do_ncls:
            tr_full = ncls_split[0]
            tr_sub = few_shot_subsample(tr_full, labels, args.k_per_class, seed=args.seed)
            print(f"few-shot: {len(tr_full)} train labels -> {len(tr_sub)} "
                  f"({args.k_per_class}/class)")
            ncls_train = tr_sub
        else:
            ncls_train = ncls_split[0] if do_ncls else None
        ncls_test = ncls_split[2] if do_ncls else None

        # Per-method loop (skipped entirely for supervised setting)
        if args.setting != "supervised":
            for method in methods:
                print(f"\n[{args.setting}][{dname}] method={method}")
                t0 = time.time()

                # For few-shot + netfm, skip frozen embed and go straight to FT
                if args.setting == "few_shot" and method == "netfm":
                    if checkpoint is None:
                        print("  SKIP: netfm few-shot needs --checkpoint")
                        continue

                    if do_ncls:
                        try:
                            res, _ = finetune_netfm_node_classification(
                                graph, checkpoint, labels, ncls_train, ncls_test,
                                device, epochs=args.ft_epochs_nc,
                                lr=args.ft_lr, weight_decay=args.ft_wd,
                                head_hidden=args.ft_head_hidden,
                                verbose=args.verbose,
                            )
                            emb_struct = embed_structural_for_sup(graph)
                            plot_node_classification(
                                res, emb_struct, labels,
                                _plot_path(plots_dir, "node_classification", dname, method),
                                title=f"Node classification — {dname} — {method} (few-shot FT)",
                            )
                            _append_row(csv_path, result_to_row(
                                method, "node_classification", dname, res,
                                setting=args.setting,
                            ))
                            print(f"  ncls-ft acc={res.accuracy:.4f}  macro_f1={res.macro_f1:.4f}")
                        except Exception as e:
                            print(f"  ncls-ft FAILED: {e}")

                    if do_lp:
                        try:
                            res, _ = finetune_netfm_link_prediction(
                                graph, checkpoint, lp_pos_src, lp_pos_dst, lp_neg,
                                device, epochs=args.ft_epochs_lp,
                                lr=args.ft_lr, weight_decay=args.ft_wd,
                                verbose=args.verbose,
                            )
                            plot_link_prediction(
                                res,
                                _plot_path(plots_dir, "link_prediction", dname, method),
                                title=f"Link prediction — {dname} — {method} (few-shot FT)",
                            )
                            _append_row(csv_path, result_to_row(
                                method, "link_prediction", dname, res,
                                setting=args.setting,
                            ))
                            print(f"  lp-ft auc={res.auc:.4f}  ap={res.ap:.4f}")
                        except Exception as e:
                            print(f"  lp-ft FAILED: {e}")
                    continue

                # Frozen-embedding path (zero_shot always, few_shot for non-netfm)
                try:
                    emb = get_embeddings(method, graph, checkpoint, device, args.seed)
                    print(f"  embedded in {time.time() - t0:.1f}s  shape={emb.shape}")
                except Exception as e:
                    print(f"  SKIP embedding: {e}")
                    continue

                max_std = float(emb.std(axis=0).max())
                if max_std < 1e-10:
                    print(f"  SKIP: zero-variance embedding (max std = {max_std:.2e})")
                    continue

                if do_ncls:
                    try:
                        res: NodeClsResult = run_node_classification(
                            emb, labels, train_idx=ncls_train, test_idx=ncls_test,
                            seed=args.seed,
                        )
                        plot_node_classification(
                            res, emb, labels,
                            _plot_path(plots_dir, "node_classification", dname, method),
                            title=f"Node classification — {dname} — {method} ({args.setting})",
                        )
                        _append_row(csv_path, result_to_row(
                            method, "node_classification", dname, res,
                            setting=args.setting,
                        ))
                        print(f"  ncls acc={res.accuracy:.4f}  macro_f1={res.macro_f1:.4f}")
                    except Exception as e:
                        print(f"  ncls FAILED: {e}")

                if do_lp:
                    try:
                        res: LinkPredResult = run_link_prediction(
                            emb, graph.edge_index, graph.num_nodes,
                            seed=args.seed,
                            pos_src_override=lp_pos_src,
                            pos_dst_override=lp_pos_dst,
                            neg_override=lp_neg,
                        )
                        plot_link_prediction(
                            res,
                            _plot_path(plots_dir, "link_prediction", dname, method),
                            title=f"Link prediction — {dname} — {method} ({args.setting})",
                        )
                        _append_row(csv_path, result_to_row(
                            method, "link_prediction", dname, res,
                            setting=args.setting,
                        ))
                        print(f"  lp auc={res.auc:.4f}  ap={res.ap:.4f}  hits@50={res.hits_at_50:.4f}")
                    except Exception as e:
                        print(f"  lp FAILED: {e}")

        # Supervised GCN ceiling (setting=supervised only, or --include-supervised)
        if args.setting == "supervised" or args.include_supervised:
            if do_ncls:
                print(f"\n[{args.setting}][{dname}] method=supervised_gcn (NC)")
                try:
                    tr_full, _, te = ncls_split
                    preds, probs = supervised_node_classification(
                        graph, labels, tr_full, te, device, verbose=args.verbose,
                    )
                    from sklearn.metrics import (
                        accuracy_score, f1_score, confusion_matrix,
                        top_k_accuracy_score,
                    )
                    y_true = labels[te]
                    y_pred = preds[te]
                    num_classes = int(max(labels.max() + 1, y_pred.max() + 1))
                    try:
                        top5 = top_k_accuracy_score(
                            y_true, probs[te], k=5,
                            labels=np.arange(num_classes),
                        )
                    except Exception:
                        top5 = float("nan")
                    res = NodeClsResult(
                        accuracy=accuracy_score(y_true, y_pred),
                        top5_accuracy=float(top5),
                        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
                        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
                        per_class_f1=f1_score(y_true, y_pred, average=None, zero_division=0,
                                              labels=np.arange(num_classes)),
                        confusion=confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)),
                        num_classes=num_classes,
                        y_true=y_true,
                        y_pred=y_pred,
                        train_size=len(tr_full),
                        test_size=len(te),
                    )
                    sup_emb = embed_structural(graph)
                    plot_node_classification(
                        res, sup_emb, labels,
                        _plot_path(plots_dir, "node_classification", dname, "supervised_gcn"),
                        title=f"Node classification — {dname} — supervised_gcn (ceiling)",
                    )
                    _append_row(csv_path, result_to_row(
                        "supervised_gcn", "node_classification", dname, res,
                        setting=args.setting,
                    ))
                    print(f"  sup-gcn ncls acc={res.accuracy:.4f}  macro_f1={res.macro_f1:.4f}")
                except Exception as e:
                    print(f"  sup-gcn ncls FAILED: {e}")

            if do_lp:
                print(f"\n[{args.setting}][{dname}] method=supervised_gcn (LP)")
                try:
                    edge_set = set(zip(
                        graph.edge_index[0].tolist(), graph.edge_index[1].tolist()
                    ))
                    held = set(zip(lp_pos_src.tolist(), lp_pos_dst.tolist()))
                    held |= {(b, a) for a, b in held}
                    keep = np.array(
                        [[a, b] for (a, b) in edge_set if (a, b) not in held]
                    ).T
                    orig_ei = graph.edge_index
                    graph.edge_index = torch.from_numpy(keep).long()
                    emb_sup = supervised_link_prediction(
                        graph, keep, (lp_pos_src, lp_pos_dst), lp_neg,
                        device, verbose=args.verbose,
                    )
                    graph.edge_index = orig_ei

                    res = run_link_prediction(
                        emb_sup, graph.edge_index, graph.num_nodes,
                        seed=args.seed,
                        pos_src_override=lp_pos_src,
                        pos_dst_override=lp_pos_dst,
                        neg_override=lp_neg,
                    )
                    plot_link_prediction(
                        res,
                        _plot_path(plots_dir, "link_prediction", dname, "supervised_gcn"),
                        title=f"Link prediction — {dname} — supervised_gcn (ceiling)",
                    )
                    _append_row(csv_path, result_to_row(
                        "supervised_gcn", "link_prediction", dname, res,
                        setting=args.setting,
                    ))
                    print(f"  sup-gcn lp auc={res.auc:.4f}  ap={res.ap:.4f}")
                except Exception as e:
                    print(f"  sup-gcn lp FAILED: {e}")

    _write_summary(csv_path, run_dir / "summary.txt")
    print(f"\nDone. Run dir: {run_dir}")


def _append_row(csv_path: Path, row: dict) -> None:
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writerow(row)


def _write_summary(csv_path: Path, out_path: Path) -> None:
    """Pretty-print a leaderboard grouped by (task, dataset)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    lines: list[str] = []
    for (task, dataset), sub in df.groupby(["task", "dataset"]):
        lines.append(f"\n{task}  —  {dataset}")
        lines.append("─" * 70)
        if task == "node_classification":
            cols = ["method", "acc", "top5_acc", "macro_f1", "weighted_f1"]
        else:
            cols = ["method", "auc", "ap", "hits_50", "hits_100", "mrr"]
        sub = sub[cols].sort_values(cols[1], ascending=False)
        lines.append(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    text = "\n".join(lines)
    print(text)
    out_path.write_text(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NetFM + baselines on downstream tasks")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to NetFM encoder.pt (required if 'netfm' is in --methods)")
    p.add_argument("--methods", type=str,
                   default=",".join(ALL_EMBEDDERS),
                   help="Comma-separated list of methods")
    p.add_argument("--datasets", type=str,
                   default=",".join(HELDOUT_DATASETS),
                   help="Comma-separated list of held-out datasets")
    p.add_argument("--setting", type=str, default="zero_shot", choices=list(SETTINGS),
                   help="Evaluation setting (zero_shot / few_shot / supervised)")
    p.add_argument("--k-per-class", type=int, default=10,
                   help="Labels per class for few-shot NC (default 10)")
    p.add_argument("--ft-epochs-nc", type=int, default=100,
                   help="NetFM fine-tune epochs for node classification")
    p.add_argument("--ft-epochs-lp", type=int, default=50,
                   help="NetFM fine-tune epochs for link prediction")
    p.add_argument("--ft-lr", type=float, default=5e-4)
    p.add_argument("--ft-wd", type=float, default=5e-4)
    p.add_argument("--ft-head-hidden", type=int, default=0,
                   help=">0 = use 2-layer MLP head with this hidden size")
    p.add_argument("--held-frac", type=float, default=0.1,
                   help="Fraction of edges held out for link prediction")
    p.add_argument("--include-supervised", action="store_true",
                   help="Also train supervised GCN baselines alongside the chosen setting")
    p.add_argument("--outputs-root", type=str, default="outputs/testing")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
