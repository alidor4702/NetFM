"""
NetFM Data Pipeline
Downloads and loads all 15 datasets into a unified format.
"""

import os
import glob
import gzip
import tarfile
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Fix PyTorch 2.7+ / OGB compatibility (weights_only=True by default)
import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except ImportError:
    pass


@dataclass
class NetFMGraph:
    """Unified graph representation for NetFM."""
    name: str
    domain: str  # social | citation | biological | infrastructure | collaboration
    edge_index: Tensor  # [2, num_edges], undirected
    num_nodes: int
    node_labels: Optional[Tensor] = None
    num_classes: Optional[int] = None
    original_features: Optional[Tensor] = None
    structural_features: Optional[Tensor] = None
    svd_features: Optional[Tensor] = None
    split: str = "pretrain"  # pretrain | held_out

    def __repr__(self):
        feat_str = f", orig_feat={list(self.original_features.shape)}" if self.original_features is not None else ""
        label_str = f", classes={self.num_classes}" if self.num_classes else ""
        return (
            f"NetFMGraph({self.name}, domain={self.domain}, split={self.split}, "
            f"nodes={self.num_nodes}, edges={self.edge_index.size(1)}{feat_str}{label_str})"
        )


# ---------------------------------------------------------------------------
# PyG dataset loaders
# ---------------------------------------------------------------------------

def load_cora() -> NetFMGraph:
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=os.path.join(DATA_ROOT, "Cora"), name="Cora")
    data = dataset[0]
    return NetFMGraph(
        name="cora", domain="citation", split="pretrain",
        edge_index=to_undirected(data.edge_index),
        num_nodes=data.num_nodes,
        original_features=data.x.float(),
        node_labels=data.y,
        num_classes=dataset.num_classes,
    )


def load_citeseer() -> NetFMGraph:
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=os.path.join(DATA_ROOT, "CiteSeer"), name="CiteSeer")
    data = dataset[0]
    return NetFMGraph(
        name="citeseer", domain="citation", split="pretrain",
        edge_index=to_undirected(data.edge_index),
        num_nodes=data.num_nodes,
        original_features=data.x.float(),
        node_labels=data.y,
        num_classes=dataset.num_classes,
    )


def load_pubmed() -> NetFMGraph:
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=os.path.join(DATA_ROOT, "PubMed"), name="PubMed")
    data = dataset[0]
    return NetFMGraph(
        name="pubmed", domain="citation", split="pretrain",
        edge_index=to_undirected(data.edge_index),
        num_nodes=data.num_nodes,
        original_features=data.x.float(),
        node_labels=data.y,
        num_classes=dataset.num_classes,
    )


def load_ppi() -> NetFMGraph:
    """Load PPI as a single merged graph (all train/val/test splits combined)."""
    from torch_geometric.datasets import PPI
    all_edges = []
    all_x = []
    all_y = []
    offset = 0
    for split in ["train", "val", "test"]:
        dataset = PPI(root=os.path.join(DATA_ROOT, "PPI"), split=split)
        for data in dataset:
            all_edges.append(data.edge_index + offset)
            all_x.append(data.x)
            all_y.append(data.y)
            offset += data.num_nodes
    return NetFMGraph(
        name="ppi", domain="biological", split="pretrain",
        edge_index=to_undirected(torch.cat(all_edges, dim=1)),
        num_nodes=offset,
        original_features=torch.cat(all_x, dim=0).float(),
        node_labels=torch.cat(all_y, dim=0),  # multilabel
        num_classes=121,
    )


def load_facebook_ego() -> NetFMGraph:
    """Load Facebook ego-networks merged into a single graph."""
    from torch_geometric.datasets import SNAPDataset
    dataset = SNAPDataset(root=os.path.join(DATA_ROOT, "SNAPDataset"), name="ego-facebook")
    all_edges = []
    offset = 0
    total_nodes = 0
    for data in dataset:
        all_edges.append(data.edge_index + offset)
        offset += data.num_nodes
        total_nodes += data.num_nodes
    edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.zeros(2, 0, dtype=torch.long)
    return NetFMGraph(
        name="facebook_ego", domain="social", split="pretrain",
        edge_index=to_undirected(edge_index),
        num_nodes=total_nodes,
    )


def load_twitch() -> NetFMGraph:
    """Load Twitch EN gamers network from SNAP (PyG mirror is down)."""
    import json
    import zipfile

    raw_dir = os.path.join(DATA_ROOT, "Twitch")
    zip_path = os.path.join(raw_dir, "twitch_gamers.zip")
    _download_if_needed("https://snap.stanford.edu/data/twitch_gamers.zip", zip_path)

    extract_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    # Find edges file
    edges_files = glob.glob(os.path.join(extract_dir, "**", "*edges*"), recursive=True)
    if not edges_files:
        raise FileNotFoundError(f"No edges file found in {extract_dir}")

    # Parse edges CSV (src,dst format)
    edges = []
    with open(edges_files[0]) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))

    # Find features/target if available
    target_files = glob.glob(os.path.join(extract_dir, "**", "*target*"), recursive=True)
    node_labels = None
    original_features = None
    num_nodes = max(max(s, d) for s, d in edges) + 1

    if target_files:
        import csv
        with open(target_files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows and "mature" in rows[0]:
            labels = torch.zeros(num_nodes, dtype=torch.long)
            for row in rows:
                nid = int(row.get("numeric_id", row.get("id", -1)))
                if 0 <= nid < num_nodes:
                    labels[nid] = int(row["mature"])
            node_labels = labels

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return NetFMGraph(
        name="twitch_en", domain="social", split="pretrain",
        edge_index=edge_index,
        num_nodes=num_nodes,
        original_features=original_features,
        node_labels=node_labels,
        num_classes=2 if node_labels is not None else None,
    )


def load_lastfm() -> NetFMGraph:
    """Load LastFM Asia social network (PyG mirror is down, using GitHub CSV)."""
    import csv

    raw_dir = os.path.join(DATA_ROOT, "LastFMAsia")
    edges_path = os.path.join(raw_dir, "lastfm_asia_edges.csv")
    target_path = os.path.join(raw_dir, "lastfm_asia_target.csv")

    base_url = "https://raw.githubusercontent.com/sahityasetu/Social-media-network-analysis-in-python/main"
    _download_if_needed(f"{base_url}/lastfm_asia_edges.csv", edges_path)
    _download_if_needed(f"{base_url}/lastfm_asia_target.csv", target_path)

    # Parse edges
    edges = []
    with open(edges_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            edges.append((int(row[0]), int(row[1])))

    num_nodes = max(max(s, d) for s, d in edges) + 1

    # Parse labels
    labels = torch.zeros(num_nodes, dtype=torch.long)
    with open(target_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            labels[int(row[0])] = int(row[1])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    return NetFMGraph(
        name="lastfm_asia", domain="social", split="held_out",
        edge_index=edge_index,
        num_nodes=num_nodes,
        original_features=None,  # no features available from this source
        node_labels=labels,
        num_classes=18,
    )


def load_coauthor_cs() -> NetFMGraph:
    from torch_geometric.datasets import Coauthor
    dataset = Coauthor(root=os.path.join(DATA_ROOT, "Coauthor"), name="CS")
    data = dataset[0]
    return NetFMGraph(
        name="coauthor_cs", domain="collaboration", split="pretrain",
        edge_index=to_undirected(data.edge_index),
        num_nodes=data.num_nodes,
        original_features=data.x.float(),
        node_labels=data.y,
        num_classes=dataset.num_classes,
    )


# ---------------------------------------------------------------------------
# OGB dataset loaders
# ---------------------------------------------------------------------------

def load_ogbn_arxiv() -> NetFMGraph:
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(DATA_ROOT, "OGB"))
    data = dataset[0]
    return NetFMGraph(
        name="ogbn_arxiv", domain="citation", split="held_out",
        edge_index=to_undirected(data.edge_index),
        num_nodes=data.num_nodes,
        original_features=data.x.float(),
        node_labels=data.y.squeeze(),
        num_classes=40,
    )


def load_ogbn_proteins() -> NetFMGraph:
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name="ogbn-proteins", root=os.path.join(DATA_ROOT, "OGB"))
    data = dataset[0]
    return NetFMGraph(
        name="ogbn_proteins", domain="biological", split="held_out",
        edge_index=to_undirected(data.edge_index),
        num_nodes=data.num_nodes,
        original_features=None,  # no node features, only edge features
        node_labels=data.y.float(),  # multilabel
        num_classes=112,
    )


def load_ogbn_mag() -> NetFMGraph:
    """Load ogbn-mag and extract paper-cites-paper homogeneous subgraph."""
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(name="ogbn-mag", root=os.path.join(DATA_ROOT, "OGB"))
    data = dataset[0]

    # Extract paper-cites-paper subgraph
    # OGB returns a Data object with dict attributes (x_dict, y_dict, edge_index_dict)
    paper_edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]
    paper_x = data.x_dict['paper'].float()
    paper_y = data.y_dict['paper'].squeeze()
    num_papers = data.num_nodes_dict['paper']

    return NetFMGraph(
        name="ogbn_mag", domain="collaboration", split="held_out",
        edge_index=to_undirected(paper_edge_index),
        num_nodes=num_papers,
        original_features=paper_x,
        node_labels=paper_y,
        num_classes=349,
    )


# ---------------------------------------------------------------------------
# Manual dataset loaders (SNAP / KONECT / NetworkRepository)
# ---------------------------------------------------------------------------

def _download_if_needed(url: str, dest: str):
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)


def _parse_edge_list(filepath: str, comment_chars=("#", "%"), sep=None, one_indexed=True) -> list:
    """Parse a whitespace/tab-separated edge list file."""
    edges = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or any(line.startswith(c) for c in comment_chars):
                continue
            parts = line.split(sep)
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    if one_indexed:
                        src -= 1
                        dst -= 1
                    edges.append((src, dst))
                except ValueError:
                    continue
    return edges


def _edges_to_graph(edges: list, name: str, domain: str, split: str) -> NetFMGraph:
    """Convert edge list to NetFMGraph with contiguous node IDs."""
    if not edges:
        raise ValueError(f"No edges found for {name}")

    # Remap node IDs to contiguous range [0, num_nodes)
    nodes = set()
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
    node_map = {old: new for new, old in enumerate(sorted(nodes))}
    remapped = [(node_map[s], node_map[d]) for s, d in edges]

    edge_index = torch.tensor(remapped, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    num_nodes = len(nodes)

    return NetFMGraph(
        name=name, domain=domain, split=split,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


def load_power_grid() -> NetFMGraph:
    raw_dir = os.path.join(DATA_ROOT, "PowerGrid")
    archive_path = os.path.join(raw_dir, "opsahl-powergrid.tar.bz2")
    _download_if_needed(
        "http://konect.cc/files/download.tsv.opsahl-powergrid.tar.bz2",
        archive_path,
    )
    # Extract
    extracted_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir, exist_ok=True)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(extracted_dir)

    # Find the edge list file
    candidates = glob.glob(os.path.join(extracted_dir, "**", "out.*"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No edge list found in {extracted_dir}")

    edges = _parse_edge_list(candidates[0], comment_chars=("%",), one_indexed=True)
    return _edges_to_graph(edges, "power_grid", "infrastructure", "pretrain")


def load_as733() -> NetFMGraph:
    raw_dir = os.path.join(DATA_ROOT, "AS733")
    archive_path = os.path.join(raw_dir, "as-733.tar.gz")
    _download_if_needed(
        "https://snap.stanford.edu/data/as-733.tar.gz",
        archive_path,
    )
    extracted_dir = os.path.join(raw_dir, "extracted")
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extracted_dir)

    # Use the last (most developed) snapshot
    snapshots = sorted(glob.glob(os.path.join(extracted_dir, "**", "*.txt"), recursive=True))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {extracted_dir}")

    edges = _parse_edge_list(snapshots[-1], comment_chars=("#",), one_indexed=False)
    return _edges_to_graph(edges, "as733", "infrastructure", "pretrain")


def load_euro_road() -> NetFMGraph:
    raw_dir = os.path.join(DATA_ROOT, "EuroRoad")
    csv_path = os.path.join(raw_dir, "euroroad.csv")

    if not os.path.exists(csv_path):
        os.makedirs(raw_dir, exist_ok=True)
        zip_path = os.path.join(raw_dir, "euroroad.csv.zip")
        _download_if_needed(
            "https://networks.skewed.de/net/euroroad/files/euroroad.csv.zip",
            zip_path,
        )
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            # Extract the CSV file
            for name in z.namelist():
                if name.endswith(".csv"):
                    with z.open(name) as src, open(csv_path, "wb") as dst:
                        dst.write(src.read())
                    break

    edges = _parse_edge_list(csv_path, comment_chars=("#", "%", "s"), sep=",", one_indexed=False)
    return _edges_to_graph(edges, "euro_road", "infrastructure", "held_out")


def load_dblp_snap() -> NetFMGraph:
    raw_dir = os.path.join(DATA_ROOT, "DBLP_SNAP")
    gz_path = os.path.join(raw_dir, "com-dblp.ungraph.txt.gz")
    txt_path = os.path.join(raw_dir, "com-dblp.ungraph.txt")

    _download_if_needed(
        "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        gz_path,
    )

    if not os.path.exists(txt_path):
        with gzip.open(gz_path, "rt") as f_in, open(txt_path, "w") as f_out:
            f_out.write(f_in.read())

    edges = _parse_edge_list(txt_path, comment_chars=("#",), one_indexed=False)
    return _edges_to_graph(edges, "dblp_snap", "collaboration", "pretrain")


# ---------------------------------------------------------------------------
# Main registry and loading
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    # Pre-training (10 datasets)
    "cora": load_cora,
    "citeseer": load_citeseer,
    "pubmed": load_pubmed,
    "ppi": load_ppi,
    "facebook_ego": load_facebook_ego,
    "twitch_en": load_twitch,
    "coauthor_cs": load_coauthor_cs,
    "dblp_snap": load_dblp_snap,
    "power_grid": load_power_grid,
    "as733": load_as733,
    # Held-out evaluation (5 datasets)
    "lastfm_asia": load_lastfm,
    "ogbn_arxiv": load_ogbn_arxiv,
    "ogbn_proteins": load_ogbn_proteins,
    "euro_road": load_euro_road,
    "ogbn_mag": load_ogbn_mag,
}

PRETRAIN_DATASETS = [
    "cora", "citeseer", "pubmed", "ppi", "facebook_ego",
    "twitch_en", "coauthor_cs", "dblp_snap", "power_grid", "as733",
]

HELDOUT_DATASETS = [
    "lastfm_asia", "ogbn_arxiv", "ogbn_proteins", "euro_road", "ogbn_mag",
]


def load_dataset(name: str) -> NetFMGraph:
    """Load a single dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    print(f"Loading {name}...")
    graph = DATASET_REGISTRY[name]()
    print(f"  {graph}")
    return graph


def load_all_pretrain() -> dict[str, NetFMGraph]:
    """Load all pre-training datasets."""
    graphs = {}
    for name in PRETRAIN_DATASETS:
        graphs[name] = load_dataset(name)
    return graphs


def load_all_heldout() -> dict[str, NetFMGraph]:
    """Load all held-out evaluation datasets."""
    graphs = {}
    for name in HELDOUT_DATASETS:
        graphs[name] = load_dataset(name)
    return graphs


def load_all() -> dict[str, NetFMGraph]:
    """Load all 15 datasets."""
    graphs = {}
    for name in DATASET_REGISTRY:
        graphs[name] = load_dataset(name)
    return graphs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Load specific datasets
        for name in sys.argv[1:]:
            load_dataset(name)
    else:
        # Load all
        graphs = load_all()
        print(f"\n{'='*60}")
        print(f"Loaded {len(graphs)} datasets:")
        for name, g in graphs.items():
            print(f"  {g}")
