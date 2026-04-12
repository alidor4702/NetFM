# NetFM — Architecture & Implementation Specification

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│  15+ graphs (SNAP/OGB) → unified format → structural features       │
│  Split: pre-training corpus (10+) │ held-out evaluation (5)         │
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

| Domain         | Dataset              | Source   | Nodes     | Edges      | Directed | Has Labels | Role       |
|----------------|----------------------|----------|-----------|------------|----------|------------|------------|
| **Social**     | Facebook ego-nets    | SNAP     | ~4K       | ~88K       | No       | Yes (circles) | Pre-train  |
| **Social**     | Twitch EN            | SNAP     | ~7.1K     | ~35.3K     | No       | Yes (binary) | Pre-train  |
| **Social**     | LastFM Asia          | SNAP     | ~7.6K     | ~27.8K     | No       | Yes (country) | **Held-out** |
| **Citation**   | Cora                 | PyG      | 2,708     | 10,556     | No       | Yes (7 classes) | Pre-train  |
| **Citation**   | CiteSeer             | PyG      | 3,327     | 9,104      | No       | Yes (6 classes) | Pre-train  |
| **Citation**   | PubMed               | PyG      | 19,717    | 88,648     | No       | Yes (3 classes) | Pre-train  |
| **Citation**   | ogbn-arxiv           | OGB      | 169,343   | 1,166,243  | Yes      | Yes (40 classes) | **Held-out** |
| **Biological** | PPI                  | PyG      | ~56.9K    | ~818K      | No       | Yes (multilabel) | Pre-train  |
| **Biological** | ogbn-proteins        | OGB      | 132,534   | 39,561,252 | No       | Yes (multilabel) | **Held-out** |
| **Infra**      | US Power Grid        | SNAP     | 4,941     | 6,594      | No       | No         | Pre-train  |
| **Infra**      | AS-733               | SNAP     | ~7.7K     | ~26K       | No       | No         | Pre-train  |
| **Infra**      | Euro Road            | SNAP     | 1,174     | 1,417      | No       | No         | **Held-out** |
| **Collab**     | DBLP                 | SNAP     | ~317K     | ~1.05M     | No       | Yes (communities) | Pre-train  |
| **Collab**     | ogbn-mag             | OGB      | 1,939,743 | 21,111,007 | Yes      | Yes (349 classes) | **Held-out** |

> **Held-out rule**: one dataset per domain is never seen during pre-training and is used exclusively for downstream evaluation.

### 2.2 Unified Graph Format

Every dataset gets normalized to a single internal representation:

```python
@dataclass
class NetFMGraph:
    name: str                          # e.g. "cora", "facebook_ego"
    domain: str                        # "social" | "citation" | "biological" | "infrastructure" | "collaboration"
    edge_index: Tensor                 # [2, num_edges] — COO format
    num_nodes: int
    is_directed: bool
    node_labels: Optional[Tensor]      # None if no ground truth
    num_classes: Optional[int]
    structural_features: Optional[Tensor]  # [num_nodes, 6] — computed in feature step
    split: str                         # "pretrain" | "held_out"
```

All graphs are converted to undirected for pre-training (directed edges are duplicated). The original direction is preserved as metadata for tasks that need it.

### 2.3 Data Loading Strategy

- **Small/medium graphs** (< 100K nodes): loaded fully into GPU memory as single `Data` objects.
- **Large graphs** (ogbn-arxiv, ogbn-proteins, DBLP, ogbn-mag): use PyG's `NeighborLoader` for mini-batch sampling during training. Batch size per graph scaled proportionally so each epoch sees roughly the same fraction of nodes.
- All datasets cached locally under `data/raw/` (auto-downloaded on first run) and processed versions saved to `data/processed/`.

### 2.4 Pre-training Corpus Construction

During pre-training, we sample batches across all pre-training graphs:

1. Each epoch iterates over all pre-training graphs.
2. For small graphs: full-batch forward pass.
3. For large graphs: `NeighborLoader` with `num_neighbors=[15, 10, 5]` (one per GraphSAGE layer), batch size 256.
4. Graphs are interleaved (not sequential) to prevent the model from overfitting to one domain before seeing others.

---

## 3. Structural Node Features

### 3.1 Feature Definitions

Every node in every graph gets a 6-dimensional feature vector. These are domain-agnostic — they encode structural role, not domain semantics.

| # | Feature                    | Definition                                                        | Library           |
|---|----------------------------|-------------------------------------------------------------------|--------------------|
| 1 | **Degree**                 | Number of edges incident to the node. For directed graphs: in-degree + out-degree. | NetworkX / PyG    |
| 2 | **Local Clustering Coeff** | Fraction of pairs of neighbors that are connected to each other.  | NetworkX          |
| 3 | **PageRank**               | Stationary distribution of a random walk with damping factor 0.85.| NetworkX          |
| 4 | **Triangle Count**         | Number of triangles the node participates in.                     | NetworkX          |
| 5 | **K-core Number**          | Maximum k such that the node belongs to the k-core subgraph.     | NetworkX          |
| 6 | **Eigenvector Centrality** | Leading eigenvector of the adjacency matrix (power iteration).    | NetworkX          |

### 3.2 Normalization

Each feature is normalized **per-graph** to zero mean and unit variance. This prevents a single large graph from dominating the feature scale. Features with zero variance (e.g., clustering coefficient in a tree) are set to 0.

### 3.3 Computation Concerns

- **Eigenvector centrality** can fail to converge on disconnected graphs. Fallback: compute per connected component, assign 0 to isolated nodes.
- **Triangle count** is O(m^{3/2}) — expensive on large graphs. For graphs with > 500K edges, use approximate triangle counting or precompute offline.
- **PageRank** is fast (power iteration converges in ~50 iterations typically).
- Features for large OGB graphs are precomputed once and cached.

---

## 4. Model Architecture

### 4.1 GraphSAGE Encoder

```
Input: [num_nodes, 6] structural features
    │
    ▼
┌─────────────────────────┐
│  Linear(6 → hidden_dim) │   ← input projection
│  + BatchNorm + ReLU     │
└────────────┬────────────┘
             │
     ┌───────▼───────┐
     │  SAGEConv L1  │   hidden_dim → hidden_dim, aggr="mean"
     │  + BatchNorm  │
     │  + ReLU       │
     │  + Dropout    │
     └───────┬───────┘
             │
     ┌───────▼───────┐
     │  SAGEConv L2  │   hidden_dim → hidden_dim
     │  + BatchNorm  │
     │  + ReLU       │
     │  + Dropout    │
     └───────┬───────┘
             │
     ┌───────▼───────┐
     │  SAGEConv L3  │   hidden_dim → hidden_dim
     │  + BatchNorm  │
     └───────┬───────┘
             │
             ▼
Output: [num_nodes, hidden_dim] node embeddings
```

**Default hyperparameters:**
- `hidden_dim`: 256
- `num_layers`: 3
- `dropout`: 0.1
- `aggregator`: mean

**Why GraphSAGE**: It's inductive — it learns aggregation functions over neighborhoods rather than fixed per-node embeddings. This means it can produce embeddings for nodes and graphs it has never seen, which is essential for a foundation model.

### 4.2 Pre-training Heads

Three heads sit on top of the shared encoder. All three are applied simultaneously during pre-training.

#### Head 1: Masked Feature Reconstruction

```
Goal: Predict masked structural features from graph context.
Analogy: BERT's masked language model.

Procedure:
1. Randomly select 15% of nodes.
2. Replace their 6-dim feature vector with a learnable [MASK] token.
3. Forward pass through encoder.
4. MLP head: Linear(hidden_dim → hidden_dim) → ReLU → Linear(hidden_dim → 6)
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
2. MLP head: Linear(hidden_dim → hidden_dim) → ReLU → Linear(hidden_dim → 2)
3. Loss: MSE between predicted and actual values.

Note: These targets overlap with input features, but the model must reconstruct
them from the GNN embedding (which aggregates neighborhood info), not just
memorize the input — especially since 15% of inputs are masked.
```

#### Combined Loss

```
L = λ₁ · L_mask + λ₂ · L_link + λ₃ · L_subgraph

Default: λ₁ = λ₂ = λ₃ = 1.0
Tuned on validation split (random 10% of pre-training graphs held aside for loss monitoring).
```

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

---

## 5. Downstream Evaluation

### 5.1 Tasks

#### Task 1: Node Classification

- **What**: Predict categorical node labels.
- **How**: Freeze encoder, train a linear probe `Linear(hidden_dim → num_classes)` on top of embeddings.
- **Metrics**: Accuracy, Macro-F1.
- **Applicable held-out datasets**: LastFM (country), ogbn-arxiv (paper category), ogbn-proteins (protein function), ogbn-mag (paper venue).

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
- **Applicable held-out datasets**: LastFM (country clusters), DBLP communities, ogbn-mag (venue clusters).

#### Task 4: Centrality Estimation

- **What**: Predict node centrality rankings without computing them explicitly.
- **How**: Freeze encoder, train a linear regressor `Linear(hidden_dim → 1)` to predict betweenness centrality and PageRank.
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

### RQ2: Which tasks transfer best?

For each of the 4 tasks, compute the average gap between NetFM zero-shot and from-scratch. Tasks where zero-shot is already competitive = highly transferable. Tasks where it fails = domain-specific.

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

1. User uploads edge list (CSV/TXT) or adjacency matrix.
2. Backend computes structural features.
3. Pre-trained NetFM encoder produces embeddings.
4. Display: graph stats, t-SNE embedding visualization, community coloring, centrality ranking, top-K predicted links.

---

## 8. Code Structure (Final)

```
NetFM/
├── src/
│   ├── data.py              # Dataset downloading, loading, unified NetFMGraph format
│   ├── features.py          # Structural feature computation + normalization
│   ├── model.py             # GraphSAGE encoder + pre-training heads
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
│   └── architecture.md      # This file
├── tests/
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
| 2     | Structural features          | `src/features.py`            | 6 features computed for all graphs     |
| 3     | Model + pre-training         | `src/model.py`, `src/pretrain.py` | Loss decreasing on training corpus |
| 4     | Downstream tasks + baselines | `src/tasks.py`, `src/baselines.py` | Full results matrix generated    |
| 5     | Analysis                     | Notebooks / scripts          | RQ1-3 answered with plots              |
| 6     | Demo                         | `src/demo.py`                | Streamlit app running                  |
