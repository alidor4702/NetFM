"""
NetFM fine-tuning for few-shot downstream tasks.

Zero-shot evaluation freezes the encoder and trains only a linear probe.
Few-shot evaluation unfreezes the encoder and trains encoder + task head
jointly on a small label budget — this is the setting where a foundation
model is supposed to shine relative to a from-scratch GNN.

Each entry point loads a NetFM checkpoint, attaches a task-specific head,
runs a short joint optimization, then returns the same result dataclasses
the rest of the pipeline consumes (NodeClsResult / LinkPredResult).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from src.data import NetFMGraph
from src.features import DEFAULT_SVD_DIM, load_features
from src.model import NetFMModel
from src.tasks import LinkPredResult, NodeClsResult, _hits_at_k, _mrr


def _load_netfm(checkpoint_path: Path | str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    hidden = int(cfg.get("hidden_dim", 256))
    svd_dim = int(cfg.get("svd_dim", DEFAULT_SVD_DIM))
    num_layers = int(cfg.get("num_layers", 3))
    dropout = float(cfg.get("dropout", 0.1))
    model = NetFMModel(
        struct_dim=6,
        svd_dim=svd_dim,
        hidden_dim=hidden,
        num_layers=num_layers,
        dropout=dropout,
        mask_rate=cfg.get("mask_rate", 0.15),
        edge_drop_rate=cfg.get("edge_drop_rate", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["full_model"])
    return model, hidden, svd_dim


def _graph_summary(graph: NetFMGraph, device: torch.device) -> torch.Tensor:
    N = graph.num_nodes
    E_dir = int(graph.edge_index.size(1))
    avg_deg = E_dir / max(N, 1)
    return torch.tensor(
        [math.log(max(N, 1.0)),
         math.log(max(E_dir / 2.0, 1.0)),
         math.log(max(avg_deg, 1e-6))],
        dtype=torch.float32, device=device,
    )


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def finetune_netfm_node_classification(
    graph: NetFMGraph,
    checkpoint_path: Path | str,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    head_hidden: int = 0,
    verbose: bool = False,
) -> tuple[NodeClsResult, np.ndarray]:
    """Fine-tune encoder + linear head on labeled train nodes.

    Returns (result, final_embeddings). The head is discarded after training,
    so the returned embeddings can still be used for t-SNE.
    """
    model, hidden, svd_dim = _load_netfm(checkpoint_path, device)

    struct_np, svd_np = load_features(graph.name, d=svd_dim)
    struct = torch.from_numpy(struct_np).to(device)
    svd = torch.from_numpy(svd_np).to(device)
    ei = graph.edge_index.to(device)
    summary = _graph_summary(graph, device)
    y = torch.from_numpy(labels).long().to(device)
    num_classes = int(labels.max() + 1)

    if head_hidden > 0:
        head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, num_classes),
        ).to(device)
    else:
        head = nn.Linear(hidden, num_classes).to(device)

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    train_t = torch.from_numpy(train_idx).long().to(device)

    model.train()
    head.train()
    for epoch in range(epochs):
        opt.zero_grad()
        emb = model.encoder(struct, svd, ei, summary)
        logits = head(emb)
        loss = F.cross_entropy(logits[train_t], y[train_t])
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            with torch.no_grad():
                acc = (logits[train_t].argmax(-1) == y[train_t]).float().mean()
            print(f"    netfm-ft ncls epoch {epoch + 1}/{epochs}  "
                  f"loss={float(loss):.4f}  train_acc={float(acc):.3f}")

    model.eval()
    head.eval()
    with torch.no_grad():
        emb = model.encode_clean(struct, svd, ei, summary)
        logits = head(emb)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    emb_np = emb.cpu().numpy().astype(np.float32)

    y_true = labels[test_idx]
    y_pred = probs[test_idx].argmax(axis=1)
    eff_num_classes = int(max(num_classes, int(y_pred.max()) + 1))
    try:
        top5 = top_k_accuracy_score(y_true, probs[test_idx], k=5,
                                     labels=np.arange(eff_num_classes))
    except Exception:
        top5 = float("nan")
    result = NodeClsResult(
        accuracy=accuracy_score(y_true, y_pred),
        top5_accuracy=float(top5),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        per_class_f1=f1_score(y_true, y_pred, average=None, zero_division=0,
                              labels=np.arange(eff_num_classes)),
        confusion=confusion_matrix(y_true, y_pred, labels=np.arange(eff_num_classes)),
        num_classes=eff_num_classes,
        y_true=y_true,
        y_pred=y_pred,
        train_size=len(train_idx),
        test_size=len(test_idx),
    )
    return result, emb_np


# ---------------------------------------------------------------------------
# Link prediction
# ---------------------------------------------------------------------------

def finetune_netfm_link_prediction(
    graph: NetFMGraph,
    checkpoint_path: Path | str,
    pos_src: np.ndarray,
    pos_dst: np.ndarray,
    neg: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    neg_per_pos: int = 1,
    verbose: bool = False,
) -> tuple[LinkPredResult, np.ndarray]:
    """Fine-tune encoder via link-prediction BCE on held-out positives.

    IMPORTANT: this is the few-shot fine-tune setting — the positives given
    here are the training pairs, not the evaluation pairs. The test split is
    scored separately by the caller after this returns the new embeddings.
    """
    model, hidden, svd_dim = _load_netfm(checkpoint_path, device)

    struct_np, svd_np = load_features(graph.name, d=svd_dim)
    struct = torch.from_numpy(struct_np).to(device)
    svd = torch.from_numpy(svd_np).to(device)
    ei = graph.edge_index.to(device)
    summary = _graph_summary(graph, device)

    pos_s = torch.from_numpy(pos_src).long().to(device)
    pos_d = torch.from_numpy(pos_dst).long().to(device)
    num_nodes = graph.num_nodes

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        emb = model.encoder(struct, svd, ei, summary)
        n_neg = pos_s.size(0) * neg_per_pos
        neg_s = torch.randint(0, num_nodes, (n_neg,), device=device)
        neg_d = torch.randint(0, num_nodes, (n_neg,), device=device)
        pos_score = (emb[pos_s] * emb[pos_d]).sum(-1)
        neg_score = (emb[neg_s] * emb[neg_d]).sum(-1)
        scores = torch.cat([pos_score, neg_score])
        lbls = torch.cat([torch.ones_like(pos_score),
                          torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, lbls)
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    netfm-ft lp epoch {epoch + 1}/{epochs}  loss={float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        emb = model.encode_clean(struct, svd, ei, summary)
    emb_np = emb.cpu().numpy().astype(np.float32)

    # score the actual (held-out) positives vs negatives the caller passed
    pos_scores = (emb_np[pos_src] * emb_np[pos_dst]).sum(axis=1)
    neg_scores = (emb_np[neg[:, 0]] * emb_np[neg[:, 1]]).sum(axis=1)
    scores = np.concatenate([pos_scores, neg_scores])
    lbls = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    result = LinkPredResult(
        auc=roc_auc_score(lbls, scores),
        ap=average_precision_score(lbls, scores),
        hits_at_50=_hits_at_k(pos_scores, neg_scores, 50),
        hits_at_100=_hits_at_k(pos_scores, neg_scores, 100),
        mrr=_mrr(pos_scores, neg_scores),
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        num_pos=len(pos_scores),
        num_neg=len(neg_scores),
    )
    return result, emb_np


# ---------------------------------------------------------------------------
# Few-shot label subsampling
# ---------------------------------------------------------------------------

def few_shot_subsample(
    train_idx: np.ndarray,
    labels: np.ndarray,
    k_per_class: int,
    seed: int = 0,
) -> np.ndarray:
    """Pick at most `k_per_class` train examples from each label class."""
    rng = np.random.default_rng(seed)
    y_train = labels[train_idx]
    out: list[int] = []
    for c in np.unique(y_train):
        pool = train_idx[y_train == c]
        if len(pool) <= k_per_class:
            out.extend(pool.tolist())
        else:
            pick = rng.choice(pool, size=k_per_class, replace=False)
            out.extend(pick.tolist())
    return np.array(sorted(out), dtype=np.int64)
