"""
NetFM Model — Graph Foundation Model

Architecture:
  NetFMEncoder
    Dual-channel input:
      channel 1: structural [N, 6]  → Linear(6→d) → LayerNorm → struct_norm
      channel 2: svd        [N, d]                → LayerNorm → svd_norm
      graph context: Linear(6→d)(mean_pool(structural))  (inductive, no dataset-id leak)
      combine: node_input = α·struct_norm + (1−α)·svd_norm + graph_ctx
               with α = σ(mix) a single learnable scalar
    BatchNorm + ReLU → 3× SAGEConv (mean aggr) + BN + ReLU + Dropout
    (last SAGEConv layer has BN only, no ReLU/Dropout)
    Output: [N, d] node embeddings

  Three SSL heads (architecture.md §4.2):
    MaskedFeatureHead   — 15% of node inputs replaced with [MASK] token;
                          MLP(d → d → 6+d) reconstructs struct+svd; MSE
    LinkPredHead        — hide 10% of edges, sample equal # of negatives;
                          dot-product scoring; BCE
    SubgraphPropHead    — MLP(d → d → 2) predicts [clustering, triangles]; MSE

  NetFMModel wrapper runs one corrupted forward + all 3 heads,
  combines losses with uncertainty weighting (Kendall et al. 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mann-Whitney U form of ROC-AUC — no sklearn, no CPU hop."""
    n = scores.numel()
    n_pos = labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5, device=scores.device)
    order = torch.argsort(scores, descending=False)
    ranks = torch.empty(n, device=scores.device, dtype=scores.dtype)
    ranks[order] = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype)
    sum_pos = ranks[labels.bool()].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class NetFMEncoder(nn.Module):
    """GraphSAGE encoder with LayerNormed dual-channel input + mix gate + graph ctx."""

    def __init__(
        self,
        struct_dim: int = 6,
        svd_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert svd_dim == hidden_dim, (
            f"SVD dim {svd_dim} must match hidden dim {hidden_dim} "
            f"(LayerNorm on svd expects hidden_dim)"
        )
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Channel projections
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        # graph_ctx input = [log N, log E, log avg_deg] — 3 scale/density summaries
        # computed per graph from edge_index alone (inductive, works on unseen graphs).
        self.graph_proj = nn.Linear(3, hidden_dim)

        # Channel LayerNorms
        self.ln_struct = nn.LayerNorm(hidden_dim)
        self.ln_svd = nn.LayerNorm(svd_dim)

        # Learnable mix gate: α = σ(mix); initialised so α = 0.5
        self.mix = nn.Parameter(torch.zeros(1))

        # Pre-GNN norm
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        # GraphSAGE layers
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

    def combine_channels(
        self,
        struct: torch.Tensor,
        svd: torch.Tensor,
        graph_summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine structural and SVD channels into a single node input tensor.

        struct:        [N, 6]   structural features (this batch / subgraph)
        svd:           [N, d]   svd features
        graph_summary: [3]      [log N, log E, log avg_deg] for the FULL graph.
                       Caller is responsible for precomputing this from the
                       full-graph edge_index (not the subgraph), so subsampled
                       batches still see the true graph-level context.
        returns:       [N, d]
        """
        struct_proj = self.struct_proj(struct)             # [N, d]
        struct_norm = self.ln_struct(struct_proj)          # [N, d]
        svd_norm = self.ln_svd(svd)                        # [N, d]

        alpha = torch.sigmoid(self.mix)                    # scalar in (0, 1)

        graph_ctx = self.graph_proj(graph_summary.unsqueeze(0))  # [1, d]

        return alpha * struct_norm + (1.0 - alpha) * svd_norm + graph_ctx

    def encode(self, node_input: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Run the GNN stack on already-combined node inputs.
        node_input: [N, d]   (may contain [MASK] tokens for Head 1)
        edge_index: [2, E]   (may be edge-dropped for Head 2)
        returns:    [N, d] node embeddings
        """
        x = self.input_bn(node_input)
        x = F.relu(x)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        struct: torch.Tensor,
        svd: torch.Tensor,
        edge_index: torch.Tensor,
        graph_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Clean path (no masking, no edge drop) — used at inference."""
        node_input = self.combine_channels(struct, svd, graph_summary)
        return self.encode(node_input, edge_index)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class MaskedFeatureHead(nn.Module):
    """
    BERT-style masking over node inputs.

    apply_mask:  replaces a random `mask_rate` fraction of rows with a learnable
                 [MASK] token BEFORE the encoder runs.
    loss:        MLP(d → d → struct_dim+svd_dim) predicts (struct, svd)
                 at masked positions; MSE vs originals.
    """

    def __init__(
        self,
        hidden_dim: int,
        struct_dim: int = 6,
        svd_dim: int = 256,
        mask_rate: float = 0.15,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.mask_rate = mask_rate
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, struct_dim + svd_dim),
        )
        self.struct_dim = struct_dim
        self.svd_dim = svd_dim

    def apply_mask(self, node_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          masked_input: [N, d] with ~mask_rate fraction of rows replaced
          mask_idx:     [M] indices of masked nodes
        """
        N = node_input.size(0)
        num_mask = max(1, int(round(self.mask_rate * N)))
        mask_idx = torch.randperm(N, device=node_input.device)[:num_mask]
        masked = node_input.clone()
        masked[mask_idx] = self.mask_token
        return masked, mask_idx

    def loss(
        self,
        embeddings: torch.Tensor,
        mask_idx: torch.Tensor,
        struct_target: torch.Tensor,
        svd_target: torch.Tensor,
    ) -> torch.Tensor:
        if mask_idx.numel() == 0:
            return torch.zeros((), device=embeddings.device)
        pred = self.mlp(embeddings[mask_idx])
        target = torch.cat(
            [struct_target[mask_idx], svd_target[mask_idx]], dim=1
        )
        return F.mse_loss(pred, target)


class LinkPredHead(nn.Module):
    """
    Hide `edge_drop_rate` fraction of edges as positives; sample uniform
    random pairs as negatives. Score each pair via dot product of node
    embeddings; BCE-with-logits loss.
    """

    def __init__(self, edge_drop_rate: float = 0.1):
        super().__init__()
        self.edge_drop_rate = edge_drop_rate

    def corrupt_edges(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (kept_edge_index, held_out_edges, neg_edges).
        kept goes into the encoder; held_out + neg go into the loss.
        """
        E = edge_index.size(1)
        device = edge_index.device
        if E == 0:
            empty = torch.zeros(2, 0, dtype=torch.long, device=device)
            return edge_index, empty, empty
        num_drop = max(1, int(round(self.edge_drop_rate * E)))
        perm = torch.randperm(E, device=device)
        drop_ids = perm[:num_drop]
        keep_mask = torch.ones(E, dtype=torch.bool, device=device)
        keep_mask[drop_ids] = False
        kept = edge_index[:, keep_mask]
        held_out = edge_index[:, drop_ids]
        # Approximate negative sampling — false-negative rate ≈ density; negligible
        # for the graphs in our corpus (avg density < 1%).
        neg_src = torch.randint(0, num_nodes, (num_drop,), device=device)
        neg_dst = torch.randint(0, num_nodes, (num_drop,), device=device)
        neg = torch.stack([neg_src, neg_dst], dim=0)
        return kept, held_out, neg

    def compute(
        self,
        embeddings: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (bce_loss, auc)."""
        if pos_edges.size(1) == 0:
            z = torch.zeros((), device=embeddings.device)
            return z, z + 0.5
        pos_score = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=-1)
        neg_score = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(dim=-1)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
        )
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        with torch.no_grad():
            auc = _rank_auc(scores, labels)
        return loss, auc


class SubgraphPropHead(nn.Module):
    """MLP(d → d → 2) predicts [clustering, normalized triangles]; MSE."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def compute(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mse_loss, r2[2])  where r2 = [clustering_r2, triangles_r2]."""
        pred = self.mlp(embeddings)
        loss = F.mse_loss(pred, targets)
        with torch.no_grad():
            res = (targets - pred).pow(2).sum(dim=0)
            tot = (targets - targets.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
            r2 = 1.0 - res / tot.clamp(min=1e-8)
        return loss, r2


# ---------------------------------------------------------------------------
# Full model with uncertainty-weighted multi-task loss
# ---------------------------------------------------------------------------

@dataclass
class NetFMOutput:
    loss: torch.Tensor
    loss_mask: torch.Tensor
    loss_link: torch.Tensor
    loss_sub: torch.Tensor
    sigma_mask: torch.Tensor
    sigma_link: torch.Tensor
    sigma_sub: torch.Tensor
    alpha: torch.Tensor
    embeddings: torch.Tensor
    link_auc: torch.Tensor
    sub_r2_clust: torch.Tensor
    sub_r2_tri: torch.Tensor


class NetFMModel(nn.Module):
    """
    One corrupted forward through the encoder; all three heads read the
    same embeddings. Loss combined via uncertainty weighting:

        L = Σ_i [ ½ · exp(−2·s_i) · L_i  +  s_i ]

    where s_i is a learnable log-variance parameter per head. Heads whose
    losses are intrinsically noisier get downweighted automatically.
    """

    def __init__(
        self,
        struct_dim: int = 6,
        svd_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        mask_rate: float = 0.15,
        edge_drop_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = NetFMEncoder(
            struct_dim=struct_dim,
            svd_dim=svd_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mask_head = MaskedFeatureHead(
            hidden_dim=hidden_dim,
            struct_dim=struct_dim,
            svd_dim=svd_dim,
            mask_rate=mask_rate,
        )
        self.link_head = LinkPredHead(edge_drop_rate=edge_drop_rate)
        self.sub_head = SubgraphPropHead(hidden_dim=hidden_dim)

        # Uncertainty weights (Kendall et al. 2018)
        self.log_sigma_mask = nn.Parameter(torch.zeros(1))
        self.log_sigma_link = nn.Parameter(torch.zeros(1))
        self.log_sigma_sub = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        structural: torch.Tensor,      # [N, 6]
        svd: torch.Tensor,             # [N, d]
        edge_index: torch.Tensor,      # [2, E]
        subgraph_targets: torch.Tensor,  # [N, 2] = [clustering, triangles]
        graph_summary: torch.Tensor,   # [3] = [log N, log E, log avg_deg]
    ) -> NetFMOutput:
        num_nodes = structural.size(0)

        # 1. Combine channels (clean)
        node_input = self.encoder.combine_channels(structural, svd, graph_summary)

        # 2. Mask 15% of node inputs
        masked_input, mask_idx = self.mask_head.apply_mask(node_input)

        # 3. Corrupt 10% of edges
        kept_ei, held_ei, neg_ei = self.link_head.corrupt_edges(edge_index, num_nodes)

        # 4. Single encoder forward on corrupted inputs + corrupted graph
        embeddings = self.encoder.encode(masked_input, kept_ei)

        # 5. Per-head losses + diagnostic metrics
        L_mask = self.mask_head.loss(embeddings, mask_idx, structural, svd)
        L_link, link_auc = self.link_head.compute(embeddings, held_ei, neg_ei)
        L_sub, sub_r2 = self.sub_head.compute(embeddings, subgraph_targets)

        # 6. Uncertainty-weighted combined loss
        L = (
            0.5 * torch.exp(-2.0 * self.log_sigma_mask) * L_mask + self.log_sigma_mask
            + 0.5 * torch.exp(-2.0 * self.log_sigma_link) * L_link + self.log_sigma_link
            + 0.5 * torch.exp(-2.0 * self.log_sigma_sub) * L_sub + self.log_sigma_sub
        ).squeeze()

        return NetFMOutput(
            loss=L,
            loss_mask=L_mask.detach(),
            loss_link=L_link.detach(),
            loss_sub=L_sub.detach(),
            sigma_mask=self.log_sigma_mask.detach().exp(),
            sigma_link=self.log_sigma_link.detach().exp(),
            sigma_sub=self.log_sigma_sub.detach().exp(),
            alpha=torch.sigmoid(self.encoder.mix).detach(),
            embeddings=embeddings.detach(),
            link_auc=link_auc.detach(),
            sub_r2_clust=sub_r2[0].detach(),
            sub_r2_tri=sub_r2[1].detach(),
        )

    @torch.no_grad()
    def encode_clean(
        self,
        structural: torch.Tensor,
        svd: torch.Tensor,
        edge_index: torch.Tensor,
        graph_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Inference path: no masking, no edge drop. Returns [N, d]."""
        self.eval()
        return self.encoder(structural, svd, edge_index, graph_summary)
