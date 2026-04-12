# NetFM

A Graph Foundation Model pre-trained on diverse networks for universal network science tasks (link prediction, community detection, centrality estimation, node classification).

## Overview

Foundation models have transformed NLP and computer vision — NetFM asks: **can we build a "GPT for graphs"?** A single GNN pre-trained on many diverse networks that generalizes to any network science task on any unseen graph.

NetFM is pre-trained on 15+ network datasets spanning social, citation, biological, infrastructure, and collaboration domains using domain-agnostic structural features.

## Tasks

- **Node Classification** — predict node labels (accuracy, macro-F1)
- **Link Prediction** — predict missing edges (AUC, AP)
- **Community Detection** — cluster embeddings vs. Louvain (NMI)
- **Centrality Estimation** — predict betweenness/PageRank (Spearman correlation)

## Installation

```bash
pip install -e .
```

## Author

Ali Dor
