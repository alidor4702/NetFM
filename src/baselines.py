"""
Baseline embedders and supervised baselines for downstream task evaluation.

Every embedder returns a dense float32 matrix [N, d]. Tasks consume these
uniformly — switching methods is just switching which function produces the
embedding matrix.

Baselines:
  random        — random Gaussian vectors           (no learning, no data)
  structural    — the cached 6 structural features  (topology only)
  svd           — the cached SVD features           (attribute only)
  netfm         — a trained NetFM encoder           (our model)
  node2vec      — DeepWalk-style random-walk SSL    (classic graph SSL)
  supervised_*  — GCN trained end-to-end on the task (ceiling; no transfer)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Node2Vec

from src.data import NetFMGraph
from src.features import DEFAULT_SVD_DIM, load_features
from src.model import NetFMModel


# ---------------------------------------------------------------------------
# Embedders (output: [N, d] float32)
# ---------------------------------------------------------------------------

def embed_random(num_nodes: int, d: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_nodes, d), dtype=np.float32)


def embed_structural(graph: NetFMGraph, d: int = DEFAULT_SVD_DIM) -> np.ndarray:
    struct, _ = load_features(graph.name, d=d)
    return struct.astype(np.float32)


def embed_svd(graph: NetFMGraph, d: int = DEFAULT_SVD_DIM) -> np.ndarray:
    _, svd = load_features(graph.name, d=d)
    return svd.astype(np.float32)


def embed_netfm(
    graph: NetFMGraph,
    checkpoint_path: Path | str,
    device: torch.device,
    d: int = DEFAULT_SVD_DIM,
) -> np.ndarray:
    """Load a trained NetFM encoder and produce embeddings for the whole graph."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})
    hidden = int(cfg.get("hidden_dim", 256))
    svd_dim = int(cfg.get("svd_dim", d))
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
    model.eval()

    struct_np, svd_np = load_features(graph.name, d=svd_dim)
    struct = torch.from_numpy(struct_np).to(device)
    svd = torch.from_numpy(svd_np).to(device)
    ei = graph.edge_index.to(device)

    import math
    N = graph.num_nodes
    E_dir = int(graph.edge_index.size(1))
    avg_deg = E_dir / max(N, 1)
    summary = torch.tensor(
        [math.log(max(N, 1.0)),
         math.log(max(E_dir / 2.0, 1.0)),
         math.log(max(avg_deg, 1e-6))],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        emb = model.encode_clean(struct, svd, ei, summary)
    return emb.cpu().numpy().astype(np.float32)


def embed_node2vec(
    graph: NetFMGraph,
    device: torch.device,
    d: int = 128,
    walk_length: int = 40,
    context_size: int = 10,
    walks_per_node: int = 10,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.01,
) -> np.ndarray:
    """Train Node2Vec (DeepWalk with p=q=1 by default) on the graph's edges."""
    ei = graph.edge_index.to(device)
    model = Node2Vec(
        ei,
        embedding_dim=d,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=1.0, q=1.0,
        sparse=True,
    ).to(device)
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        n_batch = 0
        for pos_rw, neg_rw in loader:
            opt.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            opt.step()
            total += float(loss)
            n_batch += 1
        print(f"    node2vec epoch {epoch + 1}/{epochs}  loss={total / max(n_batch, 1):.4f}")

    model.eval()
    with torch.no_grad():
        emb = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    return emb


# ---------------------------------------------------------------------------
# Supervised baseline (no embed step — trains end-to-end on the task)
# ---------------------------------------------------------------------------

class _SupervisedGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def _concat_features(graph: NetFMGraph, d: int) -> np.ndarray:
    """Concat raw structural + SVD features as the input to supervised GCN."""
    struct, svd = load_features(graph.name, d=d)
    return np.concatenate([struct, svd], axis=1).astype(np.float32)


# Above these sizes, supervised training switches to NeighborLoader batching
# so the backward pass fits on a 24 GB GPU.
_SUP_BATCH_EDGE_THRESHOLD = 3_000_000
_SUP_BATCH_NODE_THRESHOLD = 300_000


def _sup_needs_batching(graph: NetFMGraph) -> bool:
    return (graph.num_nodes > _SUP_BATCH_NODE_THRESHOLD
            or int(graph.edge_index.size(1)) > _SUP_BATCH_EDGE_THRESHOLD)


def _sup_inference_batched(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_layers: int,
    out_dim: int,
    device: torch.device,
    batch_size: int = 4096,
    neighbors_per_hop: int = 10,
) -> torch.Tensor:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    data = Data(x=x.cpu(), edge_index=edge_index.cpu().contiguous())
    loader = NeighborLoader(
        data, num_neighbors=[neighbors_per_hop] * num_layers,
        batch_size=batch_size,
        input_nodes=torch.arange(num_nodes),
        shuffle=False,
    )
    out = torch.empty(num_nodes, out_dim)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch.x, batch.edge_index)
            bs = batch.batch_size
            out[batch.input_id.cpu()] = emb[:bs].detach().cpu()
    return out


def supervised_node_classification(
    graph: NetFMGraph,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    d: int = DEFAULT_SVD_DIM,
    hidden: int = 128,
    num_layers: int = 3,
    epochs: int = 200,
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a small GCN supervised on labels. Returns (preds, probs) for every node."""
    x = torch.from_numpy(_concat_features(graph, d)).to(device)
    ei = graph.edge_index.to(device)
    y = torch.from_numpy(labels).long().to(device)
    num_classes = int(labels.max() + 1)

    model = _SupervisedGCN(x.size(1), hidden, num_classes, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if _sup_needs_batching(graph):
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        print(f"    [batched] sup-gcn ncls using NeighborLoader "
              f"(N={graph.num_nodes}, E={int(ei.size(1))})")
        data = Data(x=x.cpu(), edge_index=ei.cpu().contiguous(), y=y.cpu())
        loader = NeighborLoader(
            data, num_neighbors=[10] * num_layers, batch_size=512,
            input_nodes=torch.from_numpy(train_idx).long(), shuffle=True,
        )
        for epoch in range(epochs):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                logits = model(batch.x, batch.edge_index)
                bs = batch.batch_size
                loss = F.cross_entropy(logits[:bs], batch.y[:bs])
                loss.backward()
                opt.step()
                ep_loss += float(loss) * bs
                ep_n += bs
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"    sup-gcn(batched) epoch {epoch + 1}/{epochs}  "
                      f"loss={ep_loss / max(ep_n, 1):.4f}")
        logits = _sup_inference_batched(
            model, x, ei, graph.num_nodes, num_layers, num_classes, device,
        ).to(device)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        return preds, probs

    train_t = torch.from_numpy(train_idx).long().to(device)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, ei)
        loss = F.cross_entropy(logits[train_t], y[train_t])
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (logits[train_t].argmax(-1) == y[train_t]).float().mean()
            print(f"    sup-gcn epoch {epoch + 1}/{epochs}  loss={float(loss):.4f}  train_acc={float(acc):.3f}")

    model.eval()
    with torch.no_grad():
        logits = model(x, ei)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=-1)
    return preds, probs


def supervised_link_prediction(
    graph: NetFMGraph,
    pos_train: np.ndarray,
    pos_test: np.ndarray,
    neg_test: np.ndarray,
    device: torch.device,
    d: int = DEFAULT_SVD_DIM,
    hidden: int = 128,
    num_layers: int = 3,
    epochs: int = 100,
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    verbose: bool = False,
) -> np.ndarray:
    """
    Train a GCN encoder end-to-end on link prediction. Training edges are
    the positives kept in the graph (everything not in the held-out set);
    scoring is dot product of the learned embeddings.
    Returns [N, hidden] embeddings.
    """
    x = torch.from_numpy(_concat_features(graph, d)).to(device)
    train_ei = graph.edge_index.to(device)
    num_nodes = graph.num_nodes

    model = _SupervisedGCN(x.size(1), hidden, hidden, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if _sup_needs_batching(graph):
        from torch_geometric.data import Data
        from torch_geometric.loader import LinkNeighborLoader
        print(f"    [batched] sup-gcn lp using LinkNeighborLoader "
              f"(N={num_nodes}, E={int(train_ei.size(1))})")
        data = Data(x=x.cpu(), edge_index=train_ei.cpu().contiguous())
        pos_edge = torch.from_numpy(np.stack([pos_train[0], pos_train[1]])).long()
        loader = LinkNeighborLoader(
            data, num_neighbors=[10] * num_layers, batch_size=4096,
            edge_label_index=pos_edge,
            edge_label=torch.ones(pos_edge.size(1)),
            neg_sampling_ratio=1.0, shuffle=True,
        )
        for epoch in range(epochs):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                emb = model(batch.x, batch.edge_index)
                s, dst = batch.edge_label_index[0], batch.edge_label_index[1]
                scores = (emb[s] * emb[dst]).sum(-1)
                loss = F.binary_cross_entropy_with_logits(
                    scores, batch.edge_label.float(),
                )
                loss.backward()
                opt.step()
                ep_loss += float(loss) * s.size(0)
                ep_n += s.size(0)
            if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
                print(f"    sup-gcn-link(batched) epoch {epoch + 1}/{epochs}  "
                      f"loss={ep_loss / max(ep_n, 1):.4f}")
        emb = _sup_inference_batched(
            model, x, train_ei, num_nodes, num_layers, hidden, device,
        )
        return emb.numpy().astype(np.float32)

    pos_t = torch.from_numpy(np.stack([pos_train[0], pos_train[1]])).long().to(device)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        emb = model(x, train_ei)
        # Sample fresh negatives each epoch
        neg_src = torch.randint(0, num_nodes, (pos_t.size(1),), device=device)
        neg_dst = torch.randint(0, num_nodes, (pos_t.size(1),), device=device)
        pos_score = (emb[pos_t[0]] * emb[pos_t[1]]).sum(-1)
        neg_score = (emb[neg_src] * emb[neg_dst]).sum(-1)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 25 == 0:
            print(f"    sup-gcn-link epoch {epoch + 1}/{epochs}  loss={float(loss):.4f}")

    model.eval()
    with torch.no_grad():
        return model(x, train_ei).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EmbedderName = Literal["random", "structural", "svd", "netfm", "node2vec"]

ALL_EMBEDDERS: list[str] = ["random", "structural", "svd", "netfm", "node2vec"]
