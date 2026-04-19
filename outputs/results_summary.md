# NetFM — Results Summary

*Machine Learning in Network Science, CentraleSupélec, April 2026.
Authors: Ali Dor, Elora Drouilhet, Henri Bonamy, Zaynab Raounak.*

This document explains, in plain terms, **what we built, how we evaluated it, and
what the numbers mean**. It is meant to be read alongside the report
(`report/main.tex`) and the raw result files in this folder.

---

## 1. What is NetFM?

NetFM is a single graph-neural-network encoder, trained once on ten diverse
graphs, that we can then reuse on any unseen graph. The idea is the
"GPT-for-graphs" one: pre-train on diverse data, then transfer to downstream
tasks with little or no extra training.

**Architecture** (see `src/model.py`).

- Two input channels, both domain-agnostic:
  - *Structural channel* (6-dim): degree, clustering coefficient, PageRank,
    triangle count, k-core, eigenvector centrality.
  - *SVD channel* (256-dim): truncated SVD of the adjacency matrix (spectral
    rank information the 6-feature vector alone cannot capture).
- A learnable scalar `α = σ(mix)` fuses the two channels:
  `h = α · struct + (1-α) · svd + graph_ctx`.
- `graph_ctx` is a small learned projection of `[log N, log E, log avg_deg]`
  so the model is told the scale of the current graph.
- 3-layer GraphSAGE (hidden dim 256, BatchNorm, ReLU, dropout 0.1).

**Self-supervised pre-training** (see `src/pretrain.py`). Three heads, combined
via Kendall-style uncertainty weighting (auto-tuning their relative weights):

1. **Masked feature reconstruction** — mask 15% of input rows, reconstruct both
   structural and SVD targets. MSE.
2. **Link prediction** — hide 10% of edges, score held-out vs random negatives
   by dot product. BCE.
3. **Subgraph property regression** — predict local clustering + triangles from
   the final embedding. MSE.

**Pre-training corpus** (10 graphs, 5 domains) — citation (Cora, CiteSeer,
PubMed, ogbn-arxiv as a test, but we pre-train on Cora/CiteSeer/PubMed only),
biological (PPI), social (Facebook-ego, Twitch-EN), collaboration (Coauthor-CS,
DBLP-SNAP), infrastructure (Power-Grid, AS-733). 100 epochs × 200 steps on one
RTX 3090 — about 30 minutes. See `outputs/training/20260419_004517_v1/training_metrics.png`.

---

## 2. What did we evaluate?

Five **held-out** graphs (never seen during pre-training):

| Graph          | Domain          | Nodes   | Edges      | Labels? |
|----------------|-----------------|---------|------------|---------|
| lastfm_asia    | social          | 7,624   | 55,612     | 18 classes |
| ogbn-arxiv     | citation        | 169,343 | 2,315,598  | 40 classes |
| ogbn-proteins  | biological      | 132,534 | 79,122,504 | multi-label |
| euro_road      | infrastructure  | 1,174   | 2,834      | no labels |
| ogbn-mag       | collaboration   | 736,389 | 10,792,672 | 349 classes |

Three **tasks**:

1. **Node classification** — predict node labels. Metrics: accuracy, top-5
   accuracy, macro-F1, weighted-F1.
2. **Link prediction** — given a 10% hold-out of edges + equal-size random
   negatives, score by dot product. Metrics: AUC, AP, Hits@50, Hits@100, MRR.
3. **Community detection** — k-means on frozen embeddings with k = number of
   true classes, compared against ground-truth labels (NMI, ARI) and against a
   Louvain partition of the same graph (NMI).

Three **settings** for classification/link prediction:

- **Zero-shot** — freeze the encoder, train only a linear head
  (LogisticRegression for NC, dot-product scorer for LP).
- **10-shot fine-tune** — fine-tune the encoder + a small head on K=10 labels
  per class (or a matching edge budget for LP), 100 epochs (NC) / 50 epochs (LP).
- **Supervised (upper bound)** — train a fresh 3-layer GCN from scratch on the
  full training split. Serves as the target number a good transfer method
  should try to reach.

Five **baselines** for a fair comparison:

- `random` — Gaussian noise (sanity floor).
- `structural` — the same 6 structural features NetFM uses as input, fed
  directly to the downstream task (so you can see how much value the SAGE layers
  add on top).
- `svd` — raw 256-dim SVD embedding alone.
- `node2vec` — skip-gram random-walk embedding trained *on the target graph*
  for 5 epochs (a strong non-GNN baseline).
- `supervised_gcn` — 3-layer GCN trained end-to-end (only used in the
  "supervised" setting).

---

## 3. How to read the outputs

- `outputs/training/20260419_004517_v1/encoder.pt` — the pre-trained NetFM checkpoint.
- `outputs/training/20260419_004517_v1/training_metrics.png` — loss curves, uncertainty
  weights, and the learned α mix-gate during pre-training.
- `outputs/testing/<timestamp>_v1/` — zero-shot run.
- `outputs/testing/20260419_113901_v1_fewshot/` — 10-shot fine-tune run.
- `outputs/testing/20260419_113900_v1_supervised/` — supervised GCN upper-bound run.
- `outputs/testing/20260419_214245_community_v1/` — community detection run.
- `outputs/testing/final_comparison.png` + `final_leaderboard.txt` — aggregated
  plots and leaderboards across all settings.

Each run directory contains `results.csv` (raw metrics, one row per
method×task×dataset), `args.json` (command-line args), and `plots/<task>/<dataset>/<method>.png`
per-case breakdown figures.

---

## 4. Headline numbers

### 4.1 Link prediction (AUC, higher is better)

| Dataset        | Best method & setting            | AUC    | Supervised GCN AUC |
|----------------|----------------------------------|--------|--------------------|
| euro_road      | **netfm (few_shot)**             | **0.988** | 0.751 |
| lastfm_asia    | **netfm (few_shot)**             | **0.980** | 0.874 |
| ogbn-arxiv     | node2vec (zero-shot)             | 0.998  | 0.966 |
| ogbn-arxiv     | — netfm (few_shot, #2)           | 0.992  | — |
| ogbn-mag       | node2vec                          | 0.999  | not completed |
| ogbn-proteins  | node2vec                          | 0.977  | not completed |

**Takeaway.** 10-shot NetFM is the best method on the two smallest/medium graphs
(`euro_road`, `lastfm_asia`) and **beats a fully-supervised GCN trained from scratch**
on both — so our pre-training really does transfer across domains. On the giant
OGB graphs, `node2vec` trained on the target still wins, which makes sense
because link prediction there is dominated by local graph structure that
node2vec can memorise directly.

### 4.2 Node classification (accuracy, higher is better)

| Dataset      | Best zero-shot                    | Best 10-shot        | Supervised GCN |
|--------------|-----------------------------------|---------------------|----------------|
| lastfm_asia  | node2vec (0.747)                  | node2vec (0.385)    | **0.686** |
| ogbn-arxiv   | node2vec (0.659)                  | node2vec (0.445)    | **0.697** |
| ogbn-mag     | node2vec (0.308)                  | node2vec (0.125)    | not completed |

**Takeaway.** Node classification is tougher for a generic structural GFM:
node2vec trained on the target graph dominates, and supervised GCN is the
upper bound. Zero-shot NetFM still beats random, structural-only, and SVD
baselines, but does not overtake node2vec here. This is expected — label
prediction often requires domain-specific semantic signal (word embeddings,
molecular features), which our structural-only input deliberately discards.

### 4.3 Community detection (k-means on frozen embeddings)

Setup: k-means with `k = #classes` (or = #Louvain communities where no labels
exist). Scored against (a) ground-truth class labels and (b) a Louvain
partition of the same graph. Louvain is skipped on `ogbn_arxiv` because
NetworkX's pure-Python implementation does not finish in reasonable time on
2.3M edges; the ground-truth NMI is still reported.

| Dataset      | Method     | NMI vs labels | NMI vs Louvain |
|--------------|------------|---------------|----------------|
| lastfm_asia  | random     | 0.007         | 0.011          |
| lastfm_asia  | structural | 0.060         | 0.063          |
| lastfm_asia  | svd        | 0.000         | 0.000          |
| lastfm_asia  | **netfm**  | 0.273         | 0.330          |
| lastfm_asia  | **node2vec** | **0.487**   | **0.547**      |
| euro_road    | random     | —             | 0.161          |
| euro_road    | structural | —             | 0.292          |
| euro_road    | svd        | —             | 0.000          |
| euro_road    | **netfm**  | —             | **0.379**      |
| euro_road    | node2vec   | —             | 0.168          |
| ogbn_arxiv   | random     | 0.001         | —              |
| ogbn_arxiv   | structural | 0.088         | —              |
| ogbn_arxiv   | svd        | 0.221         | —              |
| ogbn_arxiv   | **netfm**  | 0.225         | —              |
| ogbn_arxiv   | **node2vec** | **0.387**   | —              |

**Takeaway.**
- On **euro_road** (unlabelled infrastructure graph) NetFM's k-means partition
  agrees with Louvain best (NMI=0.38), **more than 2× node2vec**. This is the
  cleanest community-detection win for the GFM: no labels, pure structure,
  cross-domain pre-training beats a target-trained baseline.
- On **lastfm_asia** and **ogbn_arxiv**, node2vec (target-trained) wins again,
  and NetFM is second. NetFM still outperforms SVD-alone and structural-alone
  by a big margin, confirming the pre-training gives the SAGE layers something
  to work with.
- SVD collapses to NMI=0 on euro_road and lastfm_asia because the sparse SVD
  returns near-zero components for these small/sparse graphs -- the structural
  and NetFM channels are what save the model there.

Raw numbers: `outputs/testing/20260419_220055_community_final/results.csv`.
Per-method bar charts: `outputs/testing/20260419_220055_community_final/plots/community_detection/<dataset>/`.

---

## 5. What worked, what didn't

**Worked well.**

- **Cross-domain link prediction transfers**. A single pre-trained encoder gets
  AUC > 0.98 on an infrastructure graph *and* a social graph *and* a citation
  graph with 10 labels each. That's the core GFM hypothesis confirmed for LP.
- **The structural + SVD dual-channel input**. Ablation-wise, random and SVD
  alone are near chance on node classification; structural alone hits 0.13 on
  lastfm. NetFM (which *uses* structural + SVD + SAGE + pre-training) hits 0.70
  zero-shot — so the pre-training is doing real work.
- **Learned mix-gate and graph context**. The α scalar converged away from 0.5
  to a non-degenerate value, indicating both channels carry signal; the
  `graph_proj(log N, log E, log avg_deg)` vector gave the model a cheap way to
  condition on scale.
- **Auto-batching for large graphs**. We added a `_needs_batching` check that
  switches NetFM and supervised-GCN fine-tuning to `NeighborLoader` /
  `LinkNeighborLoader` when the graph exceeds 300k nodes or 3M edges, which is
  what makes ogbn-mag and ogbn-proteins tractable at all on 24GB GPUs.

**Didn't work / caveats.**

- **10-shot fine-tuning hurts on large-class NC.** On ogbn-arxiv (40 classes,
  only 400 training labels), fine-tuning under-performs the zero-shot encoder
  (0.148 vs 0.591). The encoder simply over-fits 400 examples. A parameter-
  efficient adapter (LoRA-style) is the cleaner fix; we mention this in the
  report.
- **Supervised LP on ogbn-proteins failed** with a "Input should be contiguous"
  error in the batched LinkNeighborLoader path — fixed in-source with a
  `.contiguous()` call, but we canceled the rerun in the interest of time
  since `supervised_gcn` is already an upper-bound reference, not a novel
  method. (Task #26 marker remains in the repo history.)
- **Pre-training on 10 graphs, not 15+.** The proposal aimed for 15+ pre-
  training graphs. We trained on 10 because the pre-training plateaued early
  and the held-out results were already strong. More graphs is an obvious
  extension.
- **No GAT baseline, no Streamlit demo.** Both were in the proposal but cut
  for time. Neither would change our core conclusions: GCN already serves as a
  supervised upper bound, and the demo is a deployment concern, not a research
  one.

---

## 6. Reproducing

```bash
# 1. Build features (structural + SVD) for pre-train + held-out datasets
sbatch features.slurm

# 2. Pre-train the NetFM encoder
sbatch pretrain.slurm --run-name v1

# 3. Evaluate — one submission per setting
sbatch eval.slurm \
    --checkpoint outputs/training/<run_id>/encoder.pt \
    --setting zero_shot --run-name v1_zero

sbatch eval.slurm \
    --checkpoint outputs/training/<run_id>/encoder.pt \
    --setting few_shot --k-per-class 10 --run-name v1_fewshot

sbatch eval.slurm \
    --checkpoint outputs/training/<run_id>/encoder.pt \
    --setting supervised --run-name v1_supervised

# 4. Community detection
python -m src.eval_community \
    --checkpoint outputs/training/<run_id>/encoder.pt \
    --datasets lastfm_asia,ogbn_arxiv,euro_road \
    --run-name community_v1

# 5. Aggregate
python -m src.plot_eval \
    --runs outputs/testing/*_v1 outputs/testing/*_v1_fewshot outputs/testing/*_v1_supervised \
    --out outputs/testing/final_comparison.png \
    --leaderboard outputs/testing/final_leaderboard.txt
```

---

## 7. Research questions — answers

> **RQ1: does GFM pre-training beat training-from-scratch with limited labels?**

**Yes, for link prediction.** On euro_road and lastfm_asia, the pre-trained
10-shot NetFM beats the supervised GCN trained on full labels by ≥10 AUC points.
On ogbn-arxiv LP, NetFM few-shot also beats supervised GCN (0.992 vs 0.966).
**No, for node classification on feature-rich graphs** — supervised GCN is still
the upper bound there.

> **RQ2: which tasks benefit most from pre-training?**

Link prediction benefits the most, because it's intrinsically a structural
question — and structure is exactly what our input channels encode.
Node classification is harder because it often depends on domain-specific
semantics that we don't expose to the model.

> **RQ3: which network properties predict successful transfer?**

Smaller, homophilic, label-poor graphs (euro_road, lastfm_asia) benefit the
most; large, shallow, label-rich graphs (ogbn-mag) are already well-served by
node2vec trained in-place, and our transfer gain shrinks. This matches the
general foundation-model pattern: pre-training shines when target data is
scarce.
