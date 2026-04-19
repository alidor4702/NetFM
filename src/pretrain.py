"""
NetFM Pre-training — multi-graph, multi-objective SSL.

Each step:
  1. Pick one pre-training graph via sqrt(N)-weighted sampling.
  2. For small graphs: full-batch forward.
     For larger graphs: sample a subgraph via NeighborLoader with per-graph fanout.
  3. Run NetFMModel (mask + link + subgraph heads) with uncertainty-weighted loss.
  4. Backprop + AdamW + CosineAnnealingLR.

Per-graph batching strategy (see architecture.md §4.3):
  • ≤ 20K nodes:                         full-batch
  • 20K–200K nodes, avg deg ≤ 100:       NeighborLoader, fanout [15,10,5]
  • > 200K nodes or avg deg > 100:       NeighborLoader, fanout [5, 3, 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.data import PRETRAIN_DATASETS, load_dataset
from src.features import DEFAULT_SVD_DIM, load_features
from src.model import NetFMModel
from src.plot_metrics import plot_training_metrics


# ---------------------------------------------------------------------------
# Per-graph context
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    strategy: str          # "full" or "neighbor"
    fanout: list[int] | None
    batch_size: int | None


class GraphContext:
    """Wrap a pre-training graph with its cached features and a NeighborLoader."""

    def __init__(self, name: str, device: torch.device, d: int = DEFAULT_SVD_DIM):
        g = load_dataset(name)
        struct_np, svd_np = load_features(name, d=d)

        self.name = name
        self.num_nodes = g.num_nodes
        self.num_edges = int(g.edge_index.size(1))
        self.device = device

        # Tensors held on CPU in a Data object so NeighborLoader can slice.
        # For small graphs we move once to GPU in `batch()` below.
        self.data = Data(
            edge_index=g.edge_index,
            x=torch.from_numpy(struct_np),              # [N, 6]
            svd=torch.from_numpy(svd_np),               # [N, d]
            subgraph_y=torch.from_numpy(struct_np[:, [1, 3]].copy()),  # [N, 2] = [clust, tri]
            num_nodes=g.num_nodes,
        )
        # Precomputed full-graph summary [log N, log E, log avg_deg].
        # Computed from the true full-graph edge_index so subsampled batches
        # still see accurate graph-level context (not the subgraph's stats).
        # Undirected edge count = directed / 2; avg_deg = 2E / N.
        N = float(self.num_nodes)
        E_dir = float(self.num_edges)
        E_undir = E_dir / 2.0
        avg_deg = E_dir / max(N, 1.0)
        self.graph_summary = torch.tensor(
            [math.log(max(N, 1.0)),
             math.log(max(E_undir, 1.0)),
             math.log(max(avg_deg, 1e-6))],
            dtype=torch.float32,
            device=device,
        )

        self.config = self._pick_config()
        self.loader = None
        self._loader_iter = None

    # -- configuration ------------------------------------------------------

    def _pick_config(self) -> BatchConfig:
        n = self.num_nodes
        avg_deg = self.num_edges / max(n, 1)
        if n <= 20_000:
            return BatchConfig(strategy="full", fanout=None, batch_size=None)
        if avg_deg > 100 or n > 200_000:
            return BatchConfig(strategy="neighbor", fanout=[5, 3, 2], batch_size=256)
        return BatchConfig(strategy="neighbor", fanout=[15, 10, 5], batch_size=256)

    # -- batching -----------------------------------------------------------

    def _ensure_loader(self):
        if self.loader is not None:
            return
        cfg = self.config
        self.loader = NeighborLoader(
            data=self.data,
            num_neighbors=cfg.fanout,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        self._loader_iter = iter(self.loader)

    def _next_neighbor_batch(self) -> Data:
        self._ensure_loader()
        try:
            return next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self.loader)
            return next(self._loader_iter)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (structural, svd, edge_index, subgraph_targets), all on device."""
        if self.config.strategy == "full":
            d = self.data
        else:
            d = self._next_neighbor_batch()
        return (
            d.x.to(self.device, non_blocking=True),
            d.svd.to(self.device, non_blocking=True),
            d.edge_index.to(self.device, non_blocking=True),
            d.subgraph_y.to(self.device, non_blocking=True),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    "epoch", "global_step", "elapsed_s",
    "loss_total", "loss_mask", "loss_link", "loss_sub",
    "sigma_mask", "sigma_link", "sigma_sub",
    "alpha", "lr", "grad_norm",
    "link_auc", "sub_r2_clust", "sub_r2_tri",
    "eff_rank", "visits",
]


def _effective_rank(embeddings: torch.Tensor) -> float:
    """exp(entropy of normalized covariance eigenvalues). Detects rank collapse."""
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / max(x.size(0) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    p = eig / eig.sum()
    H = -(p * p.log()).sum()
    return float(H.exp().item())


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Run directory: outputs/<timestamp>[_name]/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{run_id}_{args.run_name}"
    run_dir = Path(args.outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"
    png_path = run_dir / "training_metrics.png"
    ckpt_path = run_dir / "encoder.pt"
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run_id: {run_id}")
    print(f"run dir: {run_dir}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    print(f"loading {len(PRETRAIN_DATASETS)} pre-training graphs + cached features...")
    contexts: list[GraphContext] = []
    for name in PRETRAIN_DATASETS:
        ctx = GraphContext(name, device, d=args.svd_dim)
        contexts.append(ctx)
        print(
            f"  [{ctx.name:15s}] n={ctx.num_nodes:>7d}  e={ctx.num_edges:>9d}  "
            f"strategy={ctx.config.strategy:8s}  "
            + (f"fanout={ctx.config.fanout}" if ctx.config.fanout else "")
        )

    # sqrt(N)-weighted sampling across graphs
    weights = np.array([np.sqrt(ctx.num_nodes) for ctx in contexts], dtype=np.float64)
    weights /= weights.sum()
    print("sampling weights (sqrt N):")
    for ctx, w in zip(contexts, weights):
        print(f"  {ctx.name:15s}  p={w:.3f}")

    model = NetFMModel(
        struct_dim=6,
        svd_dim=args.svd_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mask_rate=args.mask_rate,
        edge_drop_rate=args.edge_drop_rate,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {num_params:,} parameters")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # Initialise CSV with header
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)

    global_step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        ep_L = ep_mask = ep_link = ep_sub = 0.0
        ep_grad = 0.0
        ep_auc = 0.0
        ep_r2_c = 0.0
        ep_r2_t = 0.0
        per_graph_counts = {c.name: 0 for c in contexts}
        last_embeddings = None

        for _ in range(args.steps_per_epoch):
            idx = int(rng.choice(len(contexts), p=weights))
            ctx = contexts[idx]
            per_graph_counts[ctx.name] += 1

            struct, svd, ei, sub_targets = ctx.batch()

            model.train()
            opt.zero_grad(set_to_none=True)
            out = model(struct, svd, ei, sub_targets, graph_summary=ctx.graph_summary)
            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            opt.step()
            sched.step()

            ep_L += out.loss.item()
            ep_mask += out.loss_mask.item()
            ep_link += out.loss_link.item()
            ep_sub += out.loss_sub.item()
            ep_grad += float(grad_norm)
            ep_auc += out.link_auc.item()
            ep_r2_c += out.sub_r2_clust.item()
            ep_r2_t += out.sub_r2_tri.item()
            last_embeddings = out.embeddings
            global_step += 1

            if global_step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                print(
                    f"  step {global_step:6d}  graph={ctx.name:15s}  "
                    f"L={out.loss.item():6.3f}  "
                    f"m={out.loss_mask.item():5.3f}  "
                    f"l={out.loss_link.item():5.3f}  "
                    f"s={out.loss_sub.item():5.3f}  "
                    f"auc={out.link_auc.item():.2f}  "
                    f"α={out.alpha.item():.3f}  "
                    f"σ=[{out.sigma_mask.item():.2f},"
                    f"{out.sigma_link.item():.2f},"
                    f"{out.sigma_sub.item():.2f}]  "
                    f"lr={lr:.2e}"
                )

        n = args.steps_per_epoch
        elapsed = time.time() - t_start
        lr_end = sched.get_last_lr()[0]
        eff_rank = _effective_rank(last_embeddings) if last_embeddings is not None else 0.0

        print(
            f"\n[epoch {epoch + 1:3d}/{args.epochs}]  "
            f"avg L={ep_L / n:6.3f}  mask={ep_mask / n:5.3f}  "
            f"link={ep_link / n:5.3f}  sub={ep_sub / n:5.3f}  "
            f"auc={ep_auc / n:.3f}  r2=({ep_r2_c / n:+.2f},{ep_r2_t / n:+.2f})  "
            f"eff_rank={eff_rank:.1f}  ({elapsed:.0f}s)"
        )
        visited = ", ".join(f"{k}={v}" for k, v in per_graph_counts.items() if v > 0)
        print(f"  visits: {visited}\n")

        # Append per-epoch row to CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, global_step, f"{elapsed:.1f}",
                f"{ep_L / n:.6f}", f"{ep_mask / n:.6f}",
                f"{ep_link / n:.6f}", f"{ep_sub / n:.6f}",
                f"{out.sigma_mask.item():.6f}",
                f"{out.sigma_link.item():.6f}",
                f"{out.sigma_sub.item():.6f}",
                f"{out.alpha.item():.6f}",
                f"{lr_end:.8f}",
                f"{ep_grad / n:.6f}",
                f"{ep_auc / n:.6f}",
                f"{ep_r2_c / n:.6f}",
                f"{ep_r2_t / n:.6f}",
                f"{eff_rank:.4f}",
                json.dumps(per_graph_counts),
            ])

        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "full_model": model.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
                "run_id": run_id,
            },
            ckpt_path,
        )

        # Refresh training_metrics.png every `plot_every` epochs (and at end)
        if (epoch + 1) % args.plot_every == 0 or (epoch + 1) == args.epochs:
            try:
                plot_training_metrics(csv_path, png_path, run_id)
                print(f"  [plot] {png_path}")
            except Exception as e:
                print(f"  [plot] skipped: {e}")

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    print(f"Run directory: {run_dir}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  metrics:    {csv_path}")
    print(f"  plot:       {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train NetFM")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--svd-dim", type=int, default=DEFAULT_SVD_DIM)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask-rate", type=float, default=0.15)
    p.add_argument("--edge-drop-rate", type=float, default=0.1)
    p.add_argument("--outputs-root", type=str, default="outputs/training",
                   help="Root directory for per-run output folders")
    p.add_argument("--run-name", type=str, default="",
                   help="Optional suffix appended to timestamp run id")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print a training step line every N global steps")
    p.add_argument("--plot-every", type=int, default=10,
                   help="Re-render training_metrics.png every N epochs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
