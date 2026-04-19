# NetFM — Architecture & Implementation Specification

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│  15 graphs (SNAP/OGB/PyG) → unified format                         │
│  Split: pre-training corpus (10) │ held-out evaluation (5)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                              │
│  Structural features (6-dim, topology) — universal for all graphs   │
│  SVD-compressed original features (d-dim) — domain info preserved   │
│  Combined: structural + SVD → input to GNN                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PRE-TRAINING                                    │
│  GraphSAGE encoder + 3 self-supervised heads                        │
│  L = λ₁·L_mask + λ₂·L_link + λ₃·L_subgraph                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DOWNSTREAM EVALUATION                             │
│  4 tasks × 5 held-out datasets × 3 settings × 5+ baselines         │
│  → 300+ data points → answer RQ1, RQ2, RQ3                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DEMO & LIBRARY                                 │
│  pip install netfm │ Streamlit app │ upload → instant analysis      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data

### 2.1 Dataset Inventory

**Pre-training corpus** (10 datasets — model trains on these, never evaluated on them):

| Domain         | Dataset              | Source | Loader                              | Nodes     | Edges      | Directed | Original Features          | Has Labels     |
|----------------|----------------------|--------|--------------------------------------|-----------|------------|----------|----------------------------|----------------|
| Social         | Facebook ego-nets    | PyG    | `SNAPDataset(name='ego-facebook')`   | ~4K       | ~88K       | No       | Ego features (variable)    | No             |
| Social         | Twitch EN            | PyG    | `Twitch(name='EN')`                 | ~7.1K     | ~35.3K     | No       | 7-dim user attributes      | Yes (binary)   |
| Citation       | Cora                 | PyG    | `Planetoid(name='Cora')`            | 2,708     | 10,556     | No       | 1,433-dim word embeddings  | Yes (7 classes) |
| Citation       | CiteSeer             | PyG    | `Planetoid(name='CiteSeer')`        | 3,327     | 9,104      | No       | 3,703-dim word embeddings  | Yes (6 classes) |
| Citation       | PubMed               | PyG    | `Planetoid(name='PubMed')`          | 19,717    | 88,648     | No       | 500-dim TF-IDF             | Yes (3 classes) |
| Biological     | PPI                  | PyG    | `PPI(split='train')`                | ~56.9K    | ~818K      | No       | 50-dim gene features       | Yes (121 multilabel) |
| Infrastructure | US Power Grid        | Manual | KONECT download (TSV edge list)      | 4,941     | 6,594      | No       | None                       | No             |
| Infrastructure | AS-733               | Manual | SNAP download (edge list snapshots)  | ~7.7K     | ~26K       | No       | None                       | No             |
| Collaboration  | DBLP Coauthor CS     | PyG    | `Coauthor(name='CS')`               | 18,333    | 163,788    | No       | 6,805-dim keyword vectors  | Yes (15 classes) |
| Collaboration  | DBLP (SNAP)          | Manual | SNAP download (edge list + communities) | ~317K  | ~1.05M     | No       | None                       | Yes (communities) |

**Held-out evaluation** (5 datasets — one per domain, never seen during pre-training):

| Domain         | Dataset              | Source | Loader                                   | Nodes     | Edges      | Directed | Original Features         | Has Labels       |
|----------------|----------------------|--------|------------------------------------------|-----------|------------|----------|---------------------------|------------------|
| Social         | LastFM Asia          | PyG    | `LastFMAsia()`                           | ~7.6K     | ~27.8K     | No       | 128-dim                   | Yes (18 classes) |
| Citation       | ogbn-arxiv           | OGB    | `PygNodePropPredDataset('ogbn-arxiv')`   | 169,343   | 1,166,243  | Yes      | 128-dim paper embeddings  | Yes (40 classes) |
| Biological     | ogbn-proteins        | OGB    | `PygNodePropPredDataset('ogbn-proteins')`| 132,534   | 39,561,252 | No       | None (edge features only) | Yes (112 multilabel) |
| Infrastructure | Euro Road            | Manual | graph-tool/NetworkRepository download     | 1,174     | 1,417      | No       | None                      | No               |
| Collaboration  | ogbn-mag (papers)    | OGB    | `PygNodePropPredDataset('ogbn-mag')`     | 736,389   | ~5.4M      | Yes      | 128-dim paper embeddings  | Yes (349 classes) |

**Notes:**
- ogbn-mag is heterogeneous (4 node types). We extract only the **paper-cites-paper** homogeneous subgraph (~736K paper nodes).
- ogbn-proteins has no node features — only edge features. SVD channel will be empty; the model relies on structural features alone for this graph.
- AS-733 contains 733 daily snapshots. We use the **last snapshot** (largest, most developed topology).
- All directed graphs are converted to undirected for pre-training (edges duplicated both ways).

### 2.2 Unified Graph Format

Every dataset gets normalized to a single internal representation:

```python
@dataclass
class NetFMGraph:
    name: str                              # e.g. "cora", "facebook_ego"
    domain: str                            # "social" | "citation" | "biological" | "infrastructure" | "collaboration"
    edge_index: Tensor                     # [2, num_edges] — COO format, undirected
    num_nodes: int
    node_labels: Optional[Tensor]          # None if no ground truth
    num_classes: Optional[int]
    original_features: Optional[Tensor]    # raw domain features, None if absent
    structural_features: Optional[Tensor]  # [num_nodes, 6] — computed in feature step
    svd_features: Optional[Tensor]         # [num_nodes, d] — SVD-compressed original features
    split: str                             # "pretrain" | "held_out"
```

### 2.3 Data Loading Strategy

- **Small/medium graphs** (< 100K nodes): loaded fully into GPU memory as single `Data` objects.
- **Large graphs** (ogbn-arxiv, ogbn-proteins, DBLP SNAP, ogbn-mag): use PyG's `NeighborLoader` for mini-batch sampling during training. Batch size per graph scaled proportionally so each epoch sees roughly the same fraction of nodes.
- All datasets cached locally under `data/raw/` (auto-downloaded on first run) and processed versions saved to `data/processed/`.

### 2.4 Pre-training Corpus Construction

During pre-training, we sample batches across all 10 pre-training graphs:

1. Each epoch iterates over all pre-training graphs.
2. For small graphs: full-batch forward pass.
3. For large graphs: `NeighborLoader` with `num_neighbors=[15, 10, 5]` (one per GraphSAGE layer), batch size 256.
4. Graphs are **interleaved** (not sequential) to prevent the model from overfitting to one domain before seeing others.

---

## 3. Feature Engineering — Dual-Channel Input

The key design decision: NetFM uses **two channels** of node features, combined before the GNN.

```
Channel 1 — Structural (universal, always available):
  6 topology features → Linear(6 → d) → structural_embedding [N, d]

Channel 2 — SVD-compressed domain features (when original features exist):
  Original features [N, F] → SVD truncation → svd_features [N, d]
  (or zeros [N, d] if no original features)

Combined input:
  node_input = structural_embedding + svd_features  →  [N, d]  →  GNN
```

### 3.1 Channel 1: Structural Features

Every node in every graph gets a 6-dimensional feature vector computed purely from topology.

| # | Feature                    | Definition                                                        | Library    |
|---|----------------------------|-------------------------------------------------------------------|------------|
| 1 | **Degree**                 | Number of edges incident to the node.                             | NetworkX   |
| 2 | **Local Clustering Coeff** | Fraction of pairs of neighbors that are connected to each other.  | NetworkX   |
| 3 | **PageRank**               | Stationary distribution of a random walk with damping 0.85.       | NetworkX   |
| 4 | **Triangle Count**         | Number of triangles the node participates in.                     | NetworkX   |
| 5 | **K-core Number**          | Maximum k such that the node belongs to the k-core subgraph.     | NetworkX   |
| 6 | **Eigenvector Centrality** | Leading eigenvector of the adjacency matrix (power iteration).    | NetworkX   |

**Normalization:** Per-graph z-score (zero mean, unit variance). Features with zero variance set to 0.

**Computation concerns:**
- Eigenvector centrality: can fail on disconnected graphs → compute per connected component, assign 0 to isolated nodes.
- Triangle count: O(m^{3/2}) → for graphs > 500K edges, precompute offline.
- All structural features are precomputed once and cached to disk.

**These features are then projected** through a learned `Linear(6 → d)` layer to match the GNN's hidden dimension.

### 3.2 Channel 2: SVD-Compressed Domain Features

For datasets that have original node features (word embeddings, gene expressions, etc.), we preserve that information via SVD compression to a fixed dimension d.

**How SVD feature compression works:**

Given a feature matrix X of shape [N, F] where F is the original feature dimension:

1. Compute SVD: `U, σ, Vᵀ = SVD(X)` — this decomposes X into components ranked by importance.
2. Truncate to top d components: `X_compressed = U[:, :d] × diag(σ[:d])` → shape [N, d].
3. If F < d: keep all F components, pad remaining columns with zeros.

This is a **deterministic, parameter-free** operation. No training required. It works on any feature matrix of any dimension and always produces a [N, d] output.

**Inspired by [OpenGraph (EMNLP 2024)](https://arxiv.org/abs/2403.01121)**, which demonstrated that SVD-based feature alignment outperforms one-hot encoding, degree embeddings, and random projections for cross-domain graph transfer. Their zero-shot model beat 5-shot trained baselines on Cora (75.0% vs 58.5%) and CiteSeer (61.0% vs 55.7%).

**Per-dataset SVD results:**

| Dataset           | Original dim | SVD output | Notes                              |
|-------------------|-------------|------------|------------------------------------|
| Cora              | 1,433       | [N, 256]   | Compressed, top 256 components     |
| CiteSeer          | 3,703       | [N, 256]   | Compressed                         |
| PubMed            | 500         | [N, 256]   | Compressed                         |
| PPI               | 50          | [N, 256]   | 50 real + 206 zero-padded          |
| Twitch EN         | 7           | [N, 256]   | 7 real + 249 zero-padded           |
| DBLP Coauthor CS  | 6,805       | [N, 256]   | Compressed                         |
| LastFM Asia       | 128         | [N, 256]   | 128 real + 128 zero-padded         |
| ogbn-arxiv        | 128         | [N, 256]   | 128 real + 128 zero-padded         |
| ogbn-mag (papers) | 128         | [N, 256]   | 128 real + 128 zero-padded         |
| Facebook ego      | variable    | [N, 256]   | SVD on whatever features exist     |
| Power Grid        | 0           | [N, 256]   | All zeros                          |
| AS-733            | 0           | [N, 256]   | All zeros                          |
| Euro Road         | 0           | [N, 256]   | All zeros                          |
| ogbn-proteins     | 0           | [N, 256]   | All zeros (edge features only)     |
| DBLP (SNAP)       | 0           | [N, 256]   | All zeros                          |

### 3.3 Why Two Channels?

- **Structural-only** would lose domain semantics (word embeddings, gene data) → poor node classification.
- **SVD-only** would fail on graphs without features (power grid, road networks) → limited universality.
- **Combined** gives both: structural channel enables cross-domain transfer, SVD channel preserves per-domain richness.

This also enables a clean **ablation study**: structural-only vs SVD-only vs combined. Useful for RQ2 (which tasks benefit from which channel).

---

## 4. Model Architecture

### 4.1 GraphSAGE Encoder

```
Structural features [N, 6]          SVD features [N, d]
        │                                   │
        ▼                                   │
 Linear(6 → d) → struct_emb [N, d]   svd_emb [N, d]
        │                                   │
        ▼                                   ▼
    LayerNorm                           LayerNorm             ← per-channel norm
        │                                   │
        │       α = σ(mix)   (1−α)          │
        └──── × ────┐                 ┌──── × ────┘
                    │                 │
                    ▼                 ▼
                    ( + ) ──── + ──── graph_ctx = Linear(3 → d)([log N, log E, log avg_deg])
                      │
                      ▼
               node_input [N, d]
                      │
                      ▼
               BatchNorm + ReLU
                      │
              ┌───────▼───────┐
              │  SAGEConv L1  │  d → d, aggr="mean"
              │  + BatchNorm  │
              │  + ReLU       │
              │  + Dropout    │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  SAGEConv L2  │  d → d (same as L1)
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  SAGEConv L3  │  d → d, BatchNorm only (no ReLU, no Dropout)
              └───────┬───────┘
                      │
                      ▼
        Output: [N, d] node embeddings
```

**Dual-channel combination — design refinements** (not naive sum):

1. **LayerNorm per channel.** Structural projection and SVD features have very different magnitudes (z-scored vs raw singular values). LayerNorm puts both on unit-variance per-node before combining so one channel doesn't drown out the other.

2. **Learnable mix gate α.** A single scalar `mix` parameter, passed through sigmoid:
   `α = σ(mix) ∈ (0, 1)`, initialized at 0.5. The encoder learns how much to weight each channel via gradients rather than hand-tuning:
   `combined = α · struct_norm + (1 − α) · svd_norm`
   Shared globally (not per-graph) to stay inductive.

3. **Graph context `graph_ctx`.** A 256-dim vector summarizing the graph's scale and density, added to every node's input:
   `graph_ctx = Linear(3 → d)([log N, log E, log avg_deg])`
   where `N`, `E`, `avg_deg` come from the full-graph `edge_index` (so NeighborLoader subgraphs still see the true graph-level context). The three inputs are log-scaled so the Linear layer sees inputs on a similar order of magnitude across graphs spanning four orders of magnitude in size.
   Derived from the graph's own topology, so it works on unseen graphs at inference (unlike a graph-id embedding, which would leak dataset identity). Gives the encoder a coarse "what kind of graph is this" signal — a degree-10 node means something different in a road network (log avg_deg ≈ 1) than in a social graph (log avg_deg ≈ 4) — without breaking zero-shot transfer.
   *Note*: we tried `Linear(6 → d)(mean_pool(structural))`, but because `features.py` z-scores structural features per-graph at cache time, the mean is always 0 and `graph_ctx` degenerates to a single learned bias. The 3-vector formulation avoids this degeneracy.

**Default hyperparameters:**
- `d` (hidden_dim): 256 (treat as hyperparameter — ablate with 128, 512)
- `num_layers`: 3
- `dropout`: 0.1
- `aggregator`: mean
- `mix` init: 0.0 (so α = σ(0) = 0.5 at start)

**Why GraphSAGE**: It's inductive — it learns aggregation functions over neighborhoods rather than fixed per-node embeddings. This means it can produce embeddings for nodes and graphs it has never seen, which is essential for a foundation model.

### 4.2 Pre-training Heads

Three heads sit on top of the shared encoder. All three are applied simultaneously during pre-training.

#### Head 1: Masked Feature Reconstruction

```
Goal: Predict masked input features from graph context.
Analogy: BERT's masked language model.

Procedure:
1. Randomly select 15% of nodes.
2. Replace their combined input (struct_emb + svd_emb) with a learnable [MASK] token.
3. Forward pass through encoder.
4. MLP head: Linear(d → d) → ReLU → Linear(d → 6 + d)
   Predicts both structural features (6) and SVD features (d).
5. Loss: MSE between predicted and original features (only on masked nodes).
```

#### Head 2: Link Prediction

```
Goal: Predict whether an edge exists between two nodes.

Procedure:
1. Randomly remove 10% of edges (positive samples).
2. Sample an equal number of non-edges (negative samples).
3. Forward pass through encoder (on the corrupted graph).
4. Score: dot product of node embeddings → sigmoid.
5. Loss: Binary cross-entropy.
```

#### Head 3: Subgraph Property Prediction

```
Goal: Predict local structural properties from learned embeddings.
Purpose: Forces the encoder to explicitly encode structural information.

Procedure:
1. For each node, precompute targets: [clustering_coeff, normalized_triangle_count].
2. MLP head: Linear(d → d) → ReLU → Linear(d → 2)
3. Loss: MSE between predicted and actual values.

Note: These targets overlap with structural input features, but the model must
reconstruct them from the GNN embedding (which aggregates neighborhood info),
not just memorize the input — especially since 15% of inputs are masked.
```

#### Combined Loss — Uncertainty Weighting (Kendall et al. 2018)

Hand-tuning `λ`'s is fragile: the mask head outputs 262 dims with MSE while the link head outputs 1 scalar with BCE — their raw magnitudes can differ by 10–100×. Rather than guessing weights, we learn them via uncertainty weighting:

```
Each head i gets a learnable log-variance parameter s_i.
Effective loss per head:   (1 / (2 · exp(2·s_i))) · L_i   +   s_i
                            ↑                               ↑
                    downweight noisy heads          regularizer on s_i

Total:
  L = (0.5 · exp(−2·s_mask)) · L_mask + s_mask
    + (0.5 · exp(−2·s_link)) · L_link + s_link
    + (0.5 · exp(−2·s_sub))  · L_sub  + s_sub
```

The model learns per-head `s_i` during pre-training. Heads with intrinsically noisier losses get downweighted automatically; heads with crisp gradients get more influence. +3 learnable scalars, no manual tuning.

**Validation split:** 10% of pre-training graphs held aside for loss monitoring (not weight updates).

### 4.3 Training Configuration

| Parameter       | Value     |
|-----------------|-----------|
| Optimizer       | AdamW     |
| Learning rate   | 1e-3      |
| Weight decay    | 1e-4      |
| Scheduler       | CosineAnnealingLR (T_max = epochs) |
| Epochs          | 100       |
| Batch size      | 256 (for NeighborLoader on large graphs) |
| GPU             | RTX 3090 (24GB VRAM) |

#### Graph Sampling Across the Pre-training Corpus

The 10 pre-training graphs span 4 orders of magnitude in node count (power_grid 4.9K → dblp_snap 317K) and ~4 in edge count. Naive uniform sampling over epochs would have dblp_snap dominate and power_grid get overfit within a few epochs.

**Fix — `sqrt(N)`-weighted graph sampling:**

At each training step, pick which graph to sample from with probability proportional to `sqrt(num_nodes)`. Square-root smoothing keeps the largest graph from dominating (proteins is 26× larger than cora, but only `sqrt(26) ≈ 5×` more likely to be sampled) while still giving big graphs meaningfully more compute than small ones.

```python
weights = np.array([sqrt(g.num_nodes) for g in pretrain_graphs])
weights /= weights.sum()
# each step:
graph_idx = np.random.choice(len(pretrain_graphs), p=weights)
```

**Per-graph batching strategy:**

| Graph size | Strategy | Fanout |
|---|---|---|
| ≤ 20K nodes | full-batch forward | — |
| 20K – 200K nodes | `NeighborLoader`, 256 seed nodes | `[15, 10, 5]` |
| > 200K nodes OR avg degree > 100 | `NeighborLoader`, 256 seeds, reduced fanout | `[5, 3, 2]` or smaller |

Effective subgraph size per step is ~3K–10K nodes regardless of full-graph size. This bounds GPU memory and ensures every step is a balanced mixture of contexts.

**Interleaving:** graphs are sampled **per step**, not per epoch. One gradient update can be on Cora, the next on ogbn-proteins. This prevents the model from overfitting to one domain before seeing others.

---

## 5. Downstream Evaluation

### 5.1 Tasks

#### Task 1: Node Classification

- **What**: Predict categorical node labels.
- **How**: Freeze encoder, train a linear probe `Linear(d → num_classes)` on top of embeddings.
- **Metrics**: Accuracy, Macro-F1.
- **Applicable held-out datasets**: LastFM (18 countries), ogbn-arxiv (40 categories), ogbn-proteins (112 functions), ogbn-mag (349 venues).

#### Task 2: Link Prediction

- **What**: Predict missing edges.
- **How**: Freeze encoder, compute embeddings, score candidate pairs via dot product.
- **Metrics**: AUC-ROC, Average Precision (AP).
- **Evaluation**: Hide 20% of edges as test positives, sample equal negatives. Rank all candidates.
- **Applicable held-out datasets**: All 5.

#### Task 3: Community Detection

- **What**: Discover community structure.
- **How**: Freeze encoder, run K-Means on node embeddings (k = number of ground-truth communities).
- **Metrics**: Normalized Mutual Information (NMI) against ground-truth communities.
- **Applicable held-out datasets**: LastFM (country clusters), ogbn-mag (venue clusters).

#### Task 4: Centrality Estimation

- **What**: Predict node centrality rankings without computing them explicitly.
- **How**: Freeze encoder, train a linear regressor `Linear(d → 1)` to predict betweenness centrality and PageRank.
- **Metrics**: Spearman rank correlation with exact values.
- **Applicable held-out datasets**: All 5.

### 5.2 Evaluation Settings

| Setting           | Description                                                  |
|-------------------|--------------------------------------------------------------|
| **Zero-shot**     | Use pre-trained encoder directly. No fine-tuning at all. Only the task head is trained (linear probe). |
| **10-shot**       | Fine-tune encoder + task head on 10 labeled examples per class. |
| **Full supervision** | Train a fresh GNN (same architecture) from scratch on the full labeled dataset. This is the upper-bound reference. |

### 5.3 Baselines

| Baseline                | Type       | Tasks                                |
|-------------------------|------------|--------------------------------------|
| Common Neighbors        | Heuristic  | Link prediction                      |
| Jaccard Coefficient     | Heuristic  | Link prediction                      |
| Adamic-Adar Index       | Heuristic  | Link prediction                      |
| Preferential Attachment | Heuristic  | Link prediction                      |
| Louvain                 | Heuristic  | Community detection                  |
| PageRank (exact)        | Heuristic  | Centrality estimation (oracle)       |
| node2vec + classifier   | Embedding  | All tasks                            |
| GCN (from scratch)      | GNN        | All tasks                            |
| GAT (from scratch)      | GNN        | All tasks                            |

---

## 6. Research Questions & Analysis

### RQ1: Does pre-training help?

Compare NetFM (zero-shot, 10-shot) vs. GCN/GAT from scratch across all tasks and datasets. The key signal: does NetFM with 10 labels beat a GNN trained on hundreds?

### RQ2: Which tasks and channels benefit most?

- For each of the 4 tasks, compare: structural-only vs. SVD-only vs. combined.
- Which tasks are driven by topology (structural channel dominant)?
- Which tasks need domain features (SVD channel dominant)?
- This is the ablation that justifies the dual-channel design.

### RQ3: What predicts transfer success?

Compute structural statistics for each dataset:
- Degree distribution (power-law exponent)
- Average clustering coefficient
- Diameter / average path length
- Homophily ratio (if labels exist)
- Density

Correlate these with transfer performance (Pearson/Spearman) to find which graph properties predict successful transfer from pre-training.

---

## 7. Streamlit Demo

### Interface

```
┌─────────────────────────────────────────────┐
│  NetFM — Graph Foundation Model Demo        │
├─────────────────────────────────────────────┤
│                                             │
│  [Upload Graph] (edge list / adj matrix)    │
│                                             │
│  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Graph Stats  │  │ Embedding Viz (t-SNE)│  │
│  │ - Nodes: ... │  │                     │   │
│  │ - Edges: ... │  │    (scatter plot)    │   │
│  │ - Density ..│  │                     │   │
│  └─────────────┘  └─────────────────────┘   │
│                                             │
│  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Communities  │  │ Centrality Ranking  │   │
│  │ (colored    │  │ - Top-10 nodes      │   │
│  │  graph viz) │  │ - PageRank / Betw.  │   │
│  └─────────────┘  └─────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ Link Predictions                      │   │
│  │ - Top-K most likely missing edges     │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Workflow

1. User uploads edge list (CSV/TXT) or adjacency matrix, optionally with node features.
2. Backend computes structural features (always) and SVD-compresses original features (if provided).
3. Pre-trained NetFM encoder produces embeddings.
4. Display: graph stats, t-SNE embedding visualization, community coloring, centrality ranking, top-K predicted links.

---

## 8. Code Structure

```
NetFM/
├── src/
│   ├── data.py              # Dataset downloading, loading, unified NetFMGraph format
│   ├── features.py          # Structural features + SVD compression + normalization
│   ├── model.py             # GraphSAGE encoder + dual-channel input + pre-training heads
│   ├── pretrain.py          # Pre-training loop (multi-graph, multi-objective)
│   ├── tasks.py             # Downstream task evaluation (4 tasks, 3 settings)
│   ├── baselines.py         # Classical + GNN baselines
│   ├── utils.py             # Shared utilities (metrics, logging, config loading)
│   └── demo.py              # Streamlit application
├── scripts/
│   ├── run_pretrain.py      # Entry point: pre-train NetFM
│   └── run_eval.py          # Entry point: evaluate on downstream tasks
├── configs/
│   └── default.yaml         # All hyperparameters
├── docs/
│   ├── proposal.pdf
│   ├── OpenGraph_EMNLP2024.pdf
│   └── architecture.md      # This file
├── data/                    # Auto-created, gitignored
│   ├── raw/
│   └── processed/
├── .gitignore
├── README.md
└── pyproject.toml
```

---

## 9. Implementation Order

| Phase | What                         | Files                        | Milestone                              |
|-------|------------------------------|------------------------------|----------------------------------------|
| 1     | Data pipeline                | `src/data.py`                | All 15 graphs load into NetFMGraph     |
| 2     | Feature engineering          | `src/features.py`            | Structural (6-dim) + SVD for all graphs|
| 3     | Model + pre-training         | `src/model.py`, `src/pretrain.py` | Loss decreasing on training corpus |
| 4     | Downstream tasks + baselines | `src/tasks.py`, `src/baselines.py` | Full results matrix generated    |
| 5     | Analysis                     | Notebooks / scripts          | RQ1-3 answered with plots              |
| 6     | Demo                         | `src/demo.py`                | Streamlit app running                  |
