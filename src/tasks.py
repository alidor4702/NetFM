"""
NetFM downstream task evaluation.

Each task takes a pre-computed embedding matrix [N, d] and produces metrics +
a visualisation. The same entry points are used by NetFM, baselines, and any
other method we want to compare — input/output contract is identical.

Tasks:
  run_node_classification  → fits logistic regression on train split
  run_link_prediction      → scores held-out edges vs random negatives by dot product
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_node_split(
    num_nodes: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random 60/20/20 split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_tr = int(train_frac * num_nodes)
    n_val = int(val_frac * num_nodes)
    return perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]


def make_edge_split(
    edge_index: torch.Tensor,
    num_nodes: int,
    held_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hide `held_frac` of (undirected) edges as positives; sample equal number of
    random non-edges as negatives. Returns (pos_src, pos_dst, neg_pairs)
    where neg_pairs is shape [M, 2].
    """
    rng = np.random.default_rng(seed)
    ei = edge_index.cpu().numpy()
    # Keep only one direction per undirected edge to avoid double-counting
    mask = ei[0] < ei[1]
    ei_u = ei[:, mask]  # [2, E_undir]
    E = ei_u.shape[1]
    n_held = max(1, int(round(held_frac * E)))
    idx = rng.permutation(E)[:n_held]
    pos_src, pos_dst = ei_u[0, idx], ei_u[1, idx]

    neg = np.empty((n_held, 2), dtype=np.int64)
    edge_set = set(map(tuple, ei.T.tolist()))
    k = 0
    while k < n_held:
        cand = rng.integers(0, num_nodes, size=(2 * (n_held - k), 2))
        for i in range(cand.shape[0]):
            s, t = int(cand[i, 0]), int(cand[i, 1])
            if s == t:
                continue
            if (s, t) in edge_set or (t, s) in edge_set:
                continue
            neg[k] = (s, t)
            k += 1
            if k == n_held:
                break
    return pos_src, pos_dst, neg


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class NodeClsResult:
    accuracy: float
    top5_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: np.ndarray
    confusion: np.ndarray
    num_classes: int
    y_true: np.ndarray
    y_pred: np.ndarray
    train_size: int
    test_size: int


def run_node_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    max_iter: int = 500,
    seed: int = 0,
    C: float = 1.0,
) -> NodeClsResult:
    """
    Fit logistic regression on train embeddings and report metrics on test.
    If splits aren't provided, makes a 60/20/20 split (val unused here).
    """
    n = embeddings.shape[0]
    if train_idx is None or test_idx is None:
        train_idx, _, test_idx = make_node_split(n, seed=seed)

    clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
    clf.fit(embeddings[train_idx], labels[train_idx])
    y_true = labels[test_idx]
    y_pred = clf.predict(embeddings[test_idx])

    num_classes = int(max(labels.max() + 1, y_pred.max() + 1))
    probs = clf.predict_proba(embeddings[test_idx])
    try:
        top5 = top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(num_classes))
    except Exception:
        top5 = float("nan")

    return NodeClsResult(
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
        train_size=len(train_idx),
        test_size=len(test_idx),
    )


@dataclass
class LinkPredResult:
    auc: float
    ap: float
    hits_at_50: float
    hits_at_100: float
    mrr: float
    pos_scores: np.ndarray
    neg_scores: np.ndarray
    num_pos: int
    num_neg: int


def _hits_at_k(pos_scores: np.ndarray, neg_scores: np.ndarray, k: int) -> float:
    """Fraction of positives that rank in the top-k over a pool of (pos ∪ neg)."""
    if len(pos_scores) == 0:
        return 0.0
    # For each positive, count how many negatives outrank it.
    neg_sorted = np.sort(neg_scores)[::-1]
    # rank = number of negs >= this pos + 1 (worst-case on ties)
    # Hits@k = fraction of positives whose rank <= k
    ranks = np.searchsorted(-neg_sorted, -pos_scores, side="left") + 1
    return float((ranks <= k).mean())


def _mrr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) == 0:
        return 0.0
    neg_sorted = np.sort(neg_scores)[::-1]
    ranks = np.searchsorted(-neg_sorted, -pos_scores, side="left") + 1
    return float((1.0 / ranks).mean())


def run_link_prediction(
    embeddings: np.ndarray,
    edge_index: torch.Tensor,
    num_nodes: int,
    held_frac: float = 0.1,
    seed: int = 0,
    pos_dst_override: np.ndarray | None = None,
    pos_src_override: np.ndarray | None = None,
    neg_override: np.ndarray | None = None,
) -> LinkPredResult:
    """
    Dot-product scoring over held-out positive edges vs random negatives.
    Overrides let us reuse the same split across methods for fair comparison.
    """
    if pos_src_override is None:
        pos_src, pos_dst, neg = make_edge_split(edge_index, num_nodes, held_frac, seed)
    else:
        pos_src, pos_dst, neg = pos_src_override, pos_dst_override, neg_override

    pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(axis=1)
    neg_scores = (embeddings[neg[:, 0]] * embeddings[neg[:, 1]]).sum(axis=1)

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    return LinkPredResult(
        auc=roc_auc_score(labels, scores),
        ap=average_precision_score(labels, scores),
        hits_at_50=_hits_at_k(pos_scores, neg_scores, 50),
        hits_at_100=_hits_at_k(pos_scores, neg_scores, 100),
        mrr=_mrr(pos_scores, neg_scores),
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        num_pos=len(pos_scores),
        num_neg=len(neg_scores),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _tsne_2d(embeddings: np.ndarray, seed: int = 0, n_sample: int = 3000):
    """t-SNE to 2D on at most n_sample rows (for speed). Returns coords + mask."""
    n = embeddings.shape[0]
    if n > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, n_sample, replace=False)
    else:
        idx = np.arange(n)
    ts = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    coords = ts.fit_transform(embeddings[idx])
    return coords, idx


def plot_node_classification(
    result: NodeClsResult,
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=14)

    # 1. Summary stats
    ax = axes[0, 0]
    ax.axis("off")
    txt = (
        f"Samples (train/test): {result.train_size:,} / {result.test_size:,}\n"
        f"Classes: {result.num_classes}\n\n"
        f"Accuracy:       {result.accuracy:.4f}\n"
        f"Top-5 accuracy: {result.top5_accuracy:.4f}\n"
        f"Macro F1:       {result.macro_f1:.4f}\n"
        f"Weighted F1:    {result.weighted_f1:.4f}\n"
    )
    ax.text(0.05, 0.5, txt, family="monospace", fontsize=11, va="center")
    ax.set_title("Summary")

    # 2. Confusion matrix (normalised rows)
    ax = axes[0, 1]
    cm = result.confusion.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum
    im = ax.imshow(cm_norm, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Confusion (row-normalised)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3. Per-class F1 (sorted ascending — worst classes on the left)
    ax = axes[0, 2]
    f1 = result.per_class_f1
    order = np.argsort(f1)
    ax.bar(range(len(f1)), f1[order], color="C0")
    ax.axhline(result.macro_f1, color="red", linestyle="--",
               label=f"macro-F1={result.macro_f1:.2f}")
    ax.set_title("Per-class F1 (ascending)")
    ax.set_xlabel("class (reordered)")
    ax.set_ylabel("F1")
    ax.legend(fontsize=9)

    # 4. t-SNE of test-set embeddings coloured by true class
    ax = axes[1, 0]
    coords, idx = _tsne_2d(embeddings)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels[idx],
                         s=4, cmap="tab20", alpha=0.7)
    ax.set_title(f"t-SNE of embeddings (n={len(idx)})")
    ax.set_xticks([]); ax.set_yticks([])

    # 5. Pred-vs-true label histogram (top-K most frequent classes)
    ax = axes[1, 1]
    counts_true = np.bincount(result.y_true, minlength=result.num_classes)
    counts_pred = np.bincount(result.y_pred, minlength=result.num_classes)
    top = np.argsort(counts_true)[::-1][:20]
    width = 0.4
    xs = np.arange(len(top))
    ax.bar(xs - width / 2, counts_true[top], width, label="true", color="C0")
    ax.bar(xs + width / 2, counts_pred[top], width, label="pred", color="C1")
    ax.set_title("Top-20 class counts (true vs pred)")
    ax.set_xlabel("class id")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)

    # 6. Per-class F1 vs class size
    ax = axes[1, 2]
    sizes = np.bincount(result.y_true, minlength=result.num_classes)
    ax.scatter(sizes, f1, s=18, alpha=0.7, color="C2")
    ax.set_xscale("log")
    ax.set_title("F1 vs class size")
    ax.set_xlabel("#true test samples")
    ax.set_ylabel("F1")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_link_prediction(
    result: LinkPredResult,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14)

    # 1. Summary
    ax = axes[0, 0]
    ax.axis("off")
    txt = (
        f"Positives / negatives: {result.num_pos:,} / {result.num_neg:,}\n\n"
        f"ROC-AUC:   {result.auc:.4f}\n"
        f"AP (PR):   {result.ap:.4f}\n"
        f"Hits @50:  {result.hits_at_50:.4f}\n"
        f"Hits @100: {result.hits_at_100:.4f}\n"
        f"MRR:       {result.mrr:.4f}\n"
    )
    ax.text(0.05, 0.5, txt, family="monospace", fontsize=11, va="center")
    ax.set_title("Summary")

    # 2. ROC curve
    ax = axes[0, 1]
    scores = np.concatenate([result.pos_scores, result.neg_scores])
    labels = np.concatenate(
        [np.ones(result.num_pos), np.zeros(result.num_neg)]
    )
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, color="C0", label=f"AUC={result.auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_title("ROC curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()

    # 3. PR curve
    ax = axes[1, 0]
    prec, rec, _ = precision_recall_curve(labels, scores)
    ax.plot(rec, prec, color="C2", label=f"AP={result.ap:.3f}")
    ax.set_title("Precision-Recall curve")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend()

    # 4. Score distribution
    ax = axes[1, 1]
    bins = np.linspace(
        min(result.pos_scores.min(), result.neg_scores.min()),
        max(result.pos_scores.max(), result.neg_scores.max()),
        50,
    )
    ax.hist(result.neg_scores, bins=bins, alpha=0.6, color="C3",
            label="negatives", density=True)
    ax.hist(result.pos_scores, bins=bins, alpha=0.6, color="C0",
            label="positives", density=True)
    ax.set_title("Score distribution (dot product)")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def result_to_row(
    method: str, task: str, dataset: str, res, setting: str = "zero_shot",
) -> dict:
    """Flatten a result dataclass to a CSV-friendly row."""
    if isinstance(res, NodeClsResult):
        return {
            "setting": setting,
            "method": method, "task": task, "dataset": dataset,
            "acc": res.accuracy, "top5_acc": res.top5_accuracy,
            "macro_f1": res.macro_f1, "weighted_f1": res.weighted_f1,
            "auc": "", "ap": "", "hits_50": "", "hits_100": "", "mrr": "",
            "train_size": res.train_size, "test_size": res.test_size,
            "num_classes": res.num_classes,
        }
    if isinstance(res, LinkPredResult):
        return {
            "setting": setting,
            "method": method, "task": task, "dataset": dataset,
            "acc": "", "top5_acc": "", "macro_f1": "", "weighted_f1": "",
            "auc": res.auc, "ap": res.ap,
            "hits_50": res.hits_at_50, "hits_100": res.hits_at_100, "mrr": res.mrr,
            "train_size": "", "test_size": "", "num_classes": "",
        }
    raise TypeError(f"unknown result type: {type(res)}")
