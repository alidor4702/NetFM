"""
NetFM fine-tuning for few-shot downstream tasks.

Zero-shot evaluation freezes the encoder and trains only a linear probe.
Few-shot evaluation unfreezes the encoder and trains encoder + task head
jointly on a small label budget — this is the setting where a foundation
model is supposed to shine relative to a from-scratch GNN.

For large graphs (ogbn_mag, ogbn_proteins) the backward pass through the
full graph won't fit in a 24GB GPU. In that case we switch to PyG's
NeighborLoader / LinkNeighborLoader to train on sampled subgraphs.

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


# Graphs bigger than these thresholds go through the batched code path.
BATCH_EDGE_THRESHOLD = 3_000_000
BATCH_NODE_THRESHOLD = 300_000


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
    return model, hidden, svd_dim, num_layers


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


def _needs_batching(graph: NetFMGraph) -> bool:
    return (graph.num_nodes > BATCH_NODE_THRESHOLD
            or int(graph.edge_index.size(1)) > BATCH_EDGE_THRESHOLD)


# ---------------------------------------------------------------------------
# Batched helpers (NeighborLoader / LinkNeighborLoader)
# ---------------------------------------------------------------------------

def _encode_batched_inference(
    model: NetFMModel,
    struct: torch.Tensor,
    svd: torch.Tensor,
    edge_index: torch.Tensor,
    summary: torch.Tensor,
    num_nodes: int,
    num_layers: int,
    hidden: int,
    device: torch.device,
    batch_size: int = 4096,
    neighbors_per_hop: int = 10,
) -> torch.Tensor:
    """Produce full-graph embeddings via NeighborLoader, no gradients."""
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    sd = struct.size(1)
    data = Data(
        x=torch.cat([struct.cpu(), svd.cpu()], dim=-1),
        edge_index=edge_index.cpu(),
    )
    loader = NeighborLoader(
        data,
        num_neighbors=[neighbors_per_hop] * num_layers,
        batch_size=batch_size,
        input_nodes=torch.arange(num_nodes),
        shuffle=False,
    )
    out = torch.empty(num_nodes, hidden, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            bs = batch.batch_size
            b_struct = batch.x[:, :sd]
            b_svd = batch.x[:, sd:]
            emb = model.encoder(b_struct, b_svd, batch.edge_index, summary)
            out[batch.input_id.cpu()] = emb[:bs].detach().cpu()
    return out


def _train_nc_batched(
    model: NetFMModel,
    head: nn.Module,
    struct: torch.Tensor,
    svd: torch.Tensor,
    edge_index: torch.Tensor,
    summary: torch.Tensor,
    labels_t: torch.Tensor,
    train_idx: np.ndarray,
    num_layers: int,
    opt: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    batch_size: int = 512,
    neighbors_per_hop: int = 10,
    verbose: bool = False,
) -> None:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    sd = struct.size(1)
    data = Data(
        x=torch.cat([struct.cpu(), svd.cpu()], dim=-1),
        edge_index=edge_index.cpu(),
        y=labels_t.cpu(),
    )
    loader = NeighborLoader(
        data,
        num_neighbors=[neighbors_per_hop] * num_layers,
        batch_size=batch_size,
        input_nodes=torch.from_numpy(train_idx).long(),
        shuffle=True,
    )
    model.train()
    head.train()
    for epoch in range(epochs):
        ep_loss, ep_n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            b_struct = batch.x[:, :sd]
            b_svd = batch.x[:, sd:]
            emb = model.encoder(b_struct, b_svd, batch.edge_index, summary)
            bs = batch.batch_size
            logits = head(emb[:bs])
            loss = F.cross_entropy(logits, batch.y[:bs])
            loss.backward()
            opt.step()
            ep_loss += float(loss) * bs
            ep_n += bs
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    netfm-ft(batched) ncls epoch {epoch + 1}/{epochs}  "
                  f"loss={ep_loss / max(ep_n, 1):.4f}")


def _train_lp_batched(
    model: NetFMModel,
    struct: torch.Tensor,
    svd: torch.Tensor,
    edge_index: torch.Tensor,
    summary: torch.Tensor,
    pos_src: np.ndarray,
    pos_dst: np.ndarray,
    num_layers: int,
    opt: torch.optim.Optimizer,
    epochs: int,
    neg_per_pos: int,
    device: torch.device,
    batch_size: int = 4096,
    neighbors_per_hop: int = 10,
    verbose: bool = False,
) -> None:
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    sd = struct.size(1)
    data = Data(
        x=torch.cat([struct.cpu(), svd.cpu()], dim=-1),
        edge_index=edge_index.cpu(),
    )
    pos_edge = torch.stack([
        torch.from_numpy(pos_src).long(),
        torch.from_numpy(pos_dst).long(),
    ])
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[neighbors_per_hop] * num_layers,
        batch_size=batch_size,
        edge_label_index=pos_edge,
        edge_label=torch.ones(pos_edge.size(1)),
        neg_sampling_ratio=float(neg_per_pos),
        shuffle=True,
    )
    model.train()
    for epoch in range(epochs):
        ep_loss, ep_n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            b_struct = batch.x[:, :sd]
            b_svd = batch.x[:, sd:]
            emb = model.encoder(b_struct, b_svd, batch.edge_index, summary)
            s = batch.edge_label_index[0]
            d = batch.edge_label_index[1]
            scores = (emb[s] * emb[d]).sum(-1)
            loss = F.binary_cross_entropy_with_logits(
                scores, batch.edge_label.float(),
            )
            loss.backward()
            opt.step()
            ep_loss += float(loss) * s.size(0)
            ep_n += s.size(0)
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    netfm-ft(batched) lp epoch {epoch + 1}/{epochs}  "
                  f"loss={ep_loss / max(ep_n, 1):.4f}")


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

    Returns (result, final_embeddings). Auto-switches to NeighborLoader-
    batched training for graphs exceeding BATCH_*_THRESHOLD.
    """
    model, hidden, svd_dim, num_layers = _load_netfm(checkpoint_path, device)

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

    batched = _needs_batching(graph)
    if batched:
        print(f"    [batched] graph too large, using NeighborLoader "
              f"(N={graph.num_nodes}, E={int(ei.size(1))})")
        _train_nc_batched(
            model, head, struct, svd, ei, summary, y, train_idx,
            num_layers=num_layers, opt=opt, epochs=epochs, device=device,
            verbose=verbose,
        )
    else:
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

    # ---- inference ----
    if batched:
        emb = _encode_batched_inference(
            model, struct, svd, ei, summary, graph.num_nodes,
            num_layers=num_layers, hidden=hidden, device=device,
        ).to(device)
    else:
        model.eval()
        with torch.no_grad():
            emb = model.encode_clean(struct, svd, ei, summary)
    head.eval()
    with torch.no_grad():
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

    Auto-switches to LinkNeighborLoader-batched training for graphs
    exceeding BATCH_*_THRESHOLD.
    """
    model, hidden, svd_dim, num_layers = _load_netfm(checkpoint_path, device)

    struct_np, svd_np = load_features(graph.name, d=svd_dim)
    struct = torch.from_numpy(struct_np).to(device)
    svd = torch.from_numpy(svd_np).to(device)
    ei = graph.edge_index.to(device)
    summary = _graph_summary(graph, device)

    num_nodes = graph.num_nodes
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    batched = _needs_batching(graph)
    if batched:
        print(f"    [batched] graph too large, using LinkNeighborLoader "
              f"(N={num_nodes}, E={int(ei.size(1))})")
        _train_lp_batched(
            model, struct, svd, ei, summary, pos_src, pos_dst,
            num_layers=num_layers, opt=opt, epochs=epochs,
            neg_per_pos=neg_per_pos, device=device, verbose=verbose,
        )
    else:
        pos_s = torch.from_numpy(pos_src).long().to(device)
        pos_d = torch.from_numpy(pos_dst).long().to(device)
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
                print(f"    netfm-ft lp epoch {epoch + 1}/{epochs}  "
                      f"loss={float(loss):.4f}")

    # ---- inference ----
    if batched:
        emb = _encode_batched_inference(
            model, struct, svd, ei, summary, num_nodes,
            num_layers=num_layers, hidden=hidden, device=device,
        )
    else:
        model.eval()
        with torch.no_grad():
            emb = model.encode_clean(struct, svd, ei, summary)
    emb_np = emb.cpu().numpy().astype(np.float32)

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
