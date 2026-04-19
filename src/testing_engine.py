"""
Backend for the Model Testing tab in the NetFM visualizer.

Frontend-agnostic: produces data structures that a Qt view can render as an
animated sequence. The three public entry points are:

  list_models(outputs_dir)                 -> list[ModelEntry]
  build_nc_test(method, dataset, ...)      -> NodeClsTest
  build_lp_test(method, dataset, ...)      -> LinkPredTest

Each test carries three "frames" that the UI walks through: original, hidden,
predicted. Frames reuse the same (coords, keep_ids) from a GraphBundle so
animations are just recolorings of the same scatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src import baselines
from src.data import HELDOUT_DATASETS, load_dataset
from src.tasks import (
    LinkPredResult,
    NodeClsResult,
    make_edge_split,
    make_node_split,
    run_link_prediction,
    run_node_classification,
)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """A selectable testing method. Either a baseline or a trained NetFM run."""
    key: str                 # unique key used internally
    label: str               # human-readable label shown in UI
    method: str              # "random" | "structural" | "svd" | "node2vec" | "netfm"
    checkpoint: Optional[Path] = None  # only set for netfm


def list_models(outputs_dir: Path | str = "outputs/training") -> list[ModelEntry]:
    """Enumerate all available testing methods.

    Baselines are always included. Each `outputs/training/<run>/encoder.pt`
    becomes a separate selectable NetFM model.
    """
    out: list[ModelEntry] = [
        ModelEntry(key="random",     label="Random (no learning)",    method="random"),
        ModelEntry(key="structural", label="Structural features only", method="structural"),
        ModelEntry(key="svd",        label="SVD features only",        method="svd"),
        ModelEntry(key="node2vec",   label="Node2Vec (self-trained)",  method="node2vec"),
    ]
    root = Path(outputs_dir)
    if root.exists():
        for run in sorted(root.iterdir()):
            ckpt = run / "encoder.pt"
            if ckpt.exists():
                out.append(ModelEntry(
                    key=f"netfm:{run.name}",
                    label=f"NetFM · {run.name}",
                    method="netfm",
                    checkpoint=ckpt,
                ))
    return out


def available_datasets() -> list[str]:
    """Held-out datasets — the ones we test transfer on."""
    return list(HELDOUT_DATASETS)


# ---------------------------------------------------------------------------
# Embedding cache (per process — avoids recomputing when user switches task)
# ---------------------------------------------------------------------------

_EMB_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _cache_key(model: ModelEntry, dataset: str) -> tuple[str, str]:
    return (model.key, dataset)


def compute_embeddings(model: ModelEntry, graph, device: torch.device,
                       progress=lambda _: None) -> np.ndarray:
    key = _cache_key(model, graph.name)
    if key in _EMB_CACHE:
        return _EMB_CACHE[key]

    progress(f"embedding with {model.label}")
    if model.method == "random":
        emb = baselines.embed_random(graph.num_nodes)
    elif model.method == "structural":
        emb = baselines.embed_structural(graph)
    elif model.method == "svd":
        emb = baselines.embed_svd(graph)
    elif model.method == "netfm":
        emb = baselines.embed_netfm(graph, model.checkpoint, device)
    elif model.method == "node2vec":
        emb = baselines.embed_node2vec(graph, device, epochs=3)
    else:
        raise ValueError(f"unknown method: {model.method}")
    _EMB_CACHE[key] = emb
    return emb


def clear_embedding_cache() -> None:
    _EMB_CACHE.clear()


# ---------------------------------------------------------------------------
# Test construction — Node Classification
# ---------------------------------------------------------------------------

@dataclass
class NodeClsFrame:
    """One step in the NC animation. Colors and notes change per frame."""
    title: str
    subtitle: str
    node_colors: np.ndarray   # (n, 4) rgba in [0,1]
    node_sizes: np.ndarray    # (n,) float
    stats: list[tuple[str, str]] = field(default_factory=list)  # shown on the side


@dataclass
class NodeClsTest:
    dataset: str
    model_label: str
    sample_idx: np.ndarray        # nodes in the visualized subsample (original-graph ids)
    test_mask_in_sample: np.ndarray  # (|sample|,) bool — true for test nodes
    labels_in_sample: np.ndarray   # true class per sampled node
    preds_in_sample: np.ndarray    # predicted class per sampled node
    result: NodeClsResult
    frames: list[NodeClsFrame]


def _palette_rgb(k: int) -> np.ndarray:
    """k×3 palette in [0,1] — borrows from matplotlib tab20."""
    base = np.array([
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
        [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
        [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
    ], dtype=np.float32) / 255.0
    out = np.empty((k, 3), dtype=np.float32)
    for i in range(k):
        out[i] = base[i % len(base)]
    return out


def build_nc_test(
    model: ModelEntry,
    dataset: str,
    sample_idx: np.ndarray,
    device: torch.device,
    train_frac: float = 0.6,
    seed: int = 0,
    progress=lambda _: None,
) -> Optional[NodeClsTest]:
    """Run node classification on the full graph, return the sub-sample frames.

    `sample_idx` are indices into the full graph — typically from the viewer's
    current GraphBundle. The animation visualises these exact nodes.
    """
    progress("loading dataset")
    graph = load_dataset(dataset)
    y = graph.node_labels
    if y is None:
        return None
    y = y.cpu().numpy()
    if y.ndim != 1:
        return None  # multi-label (e.g. ogbn_proteins) — not supported here

    emb = compute_embeddings(model, graph, device, progress)
    if emb.std(axis=0).max() < 1e-10:
        return None  # degenerate — e.g. SVD on lastfm_asia

    progress("running classification")
    n = graph.num_nodes
    train_idx, _, test_idx = make_node_split(n, train_frac=train_frac, seed=seed)
    result = run_node_classification(
        emb, y, train_idx=train_idx, test_idx=test_idx, seed=seed
    )

    # Restrict to the sampled subset shown in the viewer
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    test_set = set(test_idx.tolist())
    is_test = np.array([int(i) in test_set for i in sample_idx], dtype=bool)
    labels_sub = y[sample_idx]

    # Build a prediction per sampled node: real prediction for test nodes,
    # ground truth for train nodes (they were supervised anyway).
    preds_sub = labels_sub.copy()
    test_to_pred = dict(zip(test_idx.tolist(), result.y_pred.tolist()))
    for i, gid in enumerate(sample_idx):
        if is_test[i]:
            preds_sub[i] = int(test_to_pred.get(int(gid), labels_sub[i]))

    # Colors
    num_classes = int(max(labels_sub.max(), preds_sub.max())) + 1
    palette = _palette_rgb(num_classes)
    alpha = np.ones((len(sample_idx), 1), dtype=np.float32)

    def rgba_from_classes(cls: np.ndarray, alpha_arr: np.ndarray) -> np.ndarray:
        c = np.concatenate([palette[cls], alpha_arr], axis=1)
        return c.astype(np.float32)

    sizes = np.full(len(sample_idx), 14.0, dtype=np.float32)
    sizes[is_test] = 20.0  # highlight test nodes

    # --- Frame 1: original (true labels) ---
    f1 = NodeClsFrame(
        title="1 · Ground truth",
        subtitle=f"{int(is_test.sum())} test nodes highlighted",
        node_colors=rgba_from_classes(labels_sub, alpha),
        node_sizes=sizes,
        stats=[
            ("dataset", dataset),
            ("total sampled", f"{len(sample_idx):,}"),
            ("train nodes", f"{int((~is_test).sum()):,}"),
            ("test nodes",  f"{int(is_test.sum()):,}"),
            ("classes",     f"{num_classes}"),
        ],
    )

    # --- Frame 2: test-node labels hidden (gray) ---
    colors2 = rgba_from_classes(labels_sub, alpha).copy()
    colors2[is_test] = np.array([0.45, 0.45, 0.45, 1.0], dtype=np.float32)
    sizes2 = sizes.copy()
    sizes2[is_test] = 14.0
    f2 = NodeClsFrame(
        title="2 · Test labels hidden",
        subtitle="Model sees only training nodes + graph structure",
        node_colors=colors2,
        node_sizes=sizes2,
        stats=[
            ("train size (full graph)", f"{result.train_size:,}"),
            ("test size  (full graph)", f"{result.test_size:,}"),
            ("encoder", model.label),
        ],
    )

    # --- Frame 3: predictions, train = true, test = predicted w/ ring ---
    colors3 = rgba_from_classes(preds_sub, alpha).copy()
    # correctness on test nodes
    correct = (preds_sub == labels_sub) & is_test
    wrong   = (preds_sub != labels_sub) & is_test
    sizes3 = sizes.copy()
    sizes3[correct] = 22.0
    sizes3[wrong]   = 22.0
    # Dim wrong predictions so they stand out with red hue via alpha
    alpha3 = np.ones((len(sample_idx), 1), dtype=np.float32)
    alpha3[wrong] = 1.0
    colors3[:, 3:4] = alpha3

    acc_sample = float(correct.sum()) / max(int(is_test.sum()), 1)
    f3 = NodeClsFrame(
        title="3 · Predictions",
        subtitle=f"test accuracy on this sample: {acc_sample:.3f}",
        node_colors=colors3,
        node_sizes=sizes3,
        stats=[
            ("accuracy",     f"{result.accuracy:.4f}"),
            ("top-5 acc",    f"{result.top5_accuracy:.4f}"),
            ("macro-F1",     f"{result.macro_f1:.4f}"),
            ("weighted-F1",  f"{result.weighted_f1:.4f}"),
            ("sample acc",   f"{acc_sample:.4f}"),
            ("sample ✓ / ✗", f"{int(correct.sum())} / {int(wrong.sum())}"),
        ],
    )

    return NodeClsTest(
        dataset=dataset,
        model_label=model.label,
        sample_idx=sample_idx,
        test_mask_in_sample=is_test,
        labels_in_sample=labels_sub,
        preds_in_sample=preds_sub,
        result=result,
        frames=[f1, f2, f3],
    )


# ---------------------------------------------------------------------------
# Test construction — Link Prediction
# ---------------------------------------------------------------------------

@dataclass
class LinkPredFrame:
    title: str
    subtitle: str
    # edges to draw in this frame, in SAMPLE coordinates (remapped local ids)
    edges_normal: np.ndarray    # (2, E) regular edges
    edges_hidden: np.ndarray    # (2, E) held-out ground-truth positives
    edges_pred_pos: np.ndarray  # (2, E) correctly retrieved
    edges_pred_neg: np.ndarray  # (2, E) false positives (high-scoring negatives)
    node_sizes: np.ndarray      # (n,) float
    stats: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class LinkPredTest:
    dataset: str
    model_label: str
    sample_idx: np.ndarray
    sample_remap: dict           # original_id -> local_id
    edges_sample_local: np.ndarray  # (2, E) full edges inside the sample (local ids)
    held_pos_local: np.ndarray      # (2, Eh) held-out pos edges inside the sample
    result: LinkPredResult
    frames: list[LinkPredFrame]


def _remap_edges(edges: np.ndarray, remap: dict) -> np.ndarray:
    """Keep only edges whose endpoints are both in the sample; return local ids."""
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    mask = np.array([(int(s) in remap) and (int(t) in remap)
                     for s, t in zip(edges[0], edges[1])], dtype=bool)
    if not mask.any():
        return np.zeros((2, 0), dtype=np.int64)
    sub = edges[:, mask]
    out = np.stack([np.array([remap[int(s)] for s in sub[0]], dtype=np.int64),
                    np.array([remap[int(t)] for t in sub[1]], dtype=np.int64)])
    return out


def build_lp_test(
    model: ModelEntry,
    dataset: str,
    sample_idx: np.ndarray,
    device: torch.device,
    held_frac: float = 0.1,
    seed: int = 0,
    top_k_visualize: int = 50,
    progress=lambda _: None,
) -> Optional[LinkPredTest]:
    progress("loading dataset")
    graph = load_dataset(dataset)

    emb = compute_embeddings(model, graph, device, progress)
    if emb.std(axis=0).max() < 1e-10:
        return None

    progress("holding out edges + scoring")
    pos_src, pos_dst, neg = make_edge_split(graph.edge_index, graph.num_nodes,
                                            held_frac=held_frac, seed=seed)
    result = run_link_prediction(
        emb, graph.edge_index, graph.num_nodes,
        pos_src_override=pos_src, pos_dst_override=pos_dst, neg_override=neg,
    )

    # Restrict to sample
    sample_idx = np.asarray(sample_idx, dtype=np.int64)
    remap = {int(o): i for i, o in enumerate(sample_idx)}
    n = len(sample_idx)

    full_ei = graph.edge_index.cpu().numpy()
    held_ei = np.stack([pos_src, pos_dst])
    # "normal" edges = full_ei with held-out pairs removed (undirected match)
    held_set = set()
    for s, t in zip(pos_src, pos_dst):
        held_set.add((int(s), int(t)))
        held_set.add((int(t), int(s)))
    normal_mask = np.array([(int(s), int(t)) not in held_set
                            for s, t in zip(full_ei[0], full_ei[1])], dtype=bool)
    normal_ei = full_ei[:, normal_mask]

    edges_normal_local = _remap_edges(normal_ei, remap)
    edges_held_local   = _remap_edges(held_ei, remap)

    # For the prediction frame: rank all candidate pairs restricted to sample
    # that were held-out positives or negatives. Show the top K by score with
    # color coding.
    neg_ei = neg.T  # (2, Eneg)
    # Score the held and negative pairs the model saw
    pos_scores = result.pos_scores
    neg_scores = result.neg_scores

    # Rank candidates inside the sample
    def _pairs_in_sample(ei: np.ndarray, scores: np.ndarray):
        mask = np.array([(int(s) in remap) and (int(t) in remap)
                         for s, t in zip(ei[0], ei[1])], dtype=bool)
        return ei[:, mask], scores[mask]

    pos_in, pos_sc = _pairs_in_sample(held_ei, pos_scores)
    neg_in, neg_sc = _pairs_in_sample(neg_ei,  neg_scores)

    # Merge into single scored candidate pool and pick top-K
    all_pairs = np.concatenate([pos_in, neg_in], axis=1) if pos_in.size + neg_in.size else np.zeros((2, 0), dtype=np.int64)
    all_scores = np.concatenate([pos_sc, neg_sc])
    is_pos = np.concatenate([np.ones(pos_in.shape[1], dtype=bool),
                             np.zeros(neg_in.shape[1], dtype=bool)])
    order = np.argsort(-all_scores)[:top_k_visualize]
    top_pairs = all_pairs[:, order] if all_pairs.size else all_pairs
    top_is_pos = is_pos[order] if is_pos.size else is_pos

    pred_pos_local = _remap_edges(top_pairs[:, top_is_pos] if top_pairs.size else top_pairs, remap)
    pred_neg_local = _remap_edges(top_pairs[:, ~top_is_pos] if top_pairs.size else top_pairs, remap)

    sizes = np.full(n, 14.0, dtype=np.float32)

    f1 = LinkPredFrame(
        title="1 · Full graph",
        subtitle=f"{edges_normal_local.shape[1] // 2 + edges_held_local.shape[1] // 2} edges in sample",
        edges_normal=edges_normal_local,
        edges_hidden=edges_held_local,
        edges_pred_pos=np.zeros((2, 0), dtype=np.int64),
        edges_pred_neg=np.zeros((2, 0), dtype=np.int64),
        node_sizes=sizes,
        stats=[
            ("dataset", dataset),
            ("sample nodes", f"{n:,}"),
            ("sample edges", f"{(edges_normal_local.shape[1] + edges_held_local.shape[1]) // 2:,}"),
            ("held-out (sample)", f"{edges_held_local.shape[1] // 2:,}"),
            ("held frac (full)", f"{held_frac:.0%}"),
        ],
    )
    f2 = LinkPredFrame(
        title="2 · Held-out edges hidden",
        subtitle="Model sees only remaining edges",
        edges_normal=edges_normal_local,
        edges_hidden=np.zeros((2, 0), dtype=np.int64),  # hidden
        edges_pred_pos=np.zeros((2, 0), dtype=np.int64),
        edges_pred_neg=np.zeros((2, 0), dtype=np.int64),
        node_sizes=sizes,
        stats=[
            ("encoder", model.label),
            ("held-out total (full)", f"{result.num_pos:,}"),
            ("negatives (full)", f"{result.num_neg:,}"),
        ],
    )
    f3 = LinkPredFrame(
        title=f"3 · Top-{top_k_visualize} predictions",
        subtitle=f"green = true edge, red = spurious",
        edges_normal=edges_normal_local,
        edges_hidden=np.zeros((2, 0), dtype=np.int64),
        edges_pred_pos=pred_pos_local,
        edges_pred_neg=pred_neg_local,
        node_sizes=sizes,
        stats=[
            ("AUC",        f"{result.auc:.4f}"),
            ("AP",         f"{result.ap:.4f}"),
            ("Hits@50",    f"{result.hits_at_50:.4f}"),
            ("Hits@100",   f"{result.hits_at_100:.4f}"),
            ("MRR",        f"{result.mrr:.4f}"),
            ("sample TP",  f"{pred_pos_local.shape[1]}"),
            ("sample FP",  f"{pred_neg_local.shape[1]}"),
        ],
    )

    return LinkPredTest(
        dataset=dataset,
        model_label=model.label,
        sample_idx=sample_idx,
        sample_remap=remap,
        edges_sample_local=edges_normal_local,
        held_pos_local=edges_held_local,
        result=result,
        frames=[f1, f2, f3],
    )
