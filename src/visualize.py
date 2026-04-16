"""
NetFM Graph Explorer

A native Qt dashboard (PySide6 + pyqtgraph) for visualizing NetFM datasets
or any user-provided graph file in 2D / 3D.

Usage:
    pip install -e .[viz]
    python -m src.visualize
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_CACHE = PROJECT_ROOT / "figures" / "layouts"

DEFAULT_SAMPLE = 2000
SMALL_LIMIT = 3000
SEED = 0

# 20-entry qualitative palette (bright on dark background)
PALETTE = [
    "#5bc0de", "#f0ad4e", "#d9534f", "#5cb85c", "#9b59b6",
    "#e67e22", "#1abc9c", "#ecf0f1", "#f39c12", "#e74c3c",
    "#2ecc71", "#3498db", "#8e44ad", "#16a085", "#27ae60",
    "#d35400", "#2980b9", "#7f8c8d", "#bdc3c7", "#f8c471",
]

BG_DARK = "#0f1419"
PANEL_BG = "#161c24"
PANEL_BORDER = "#2a3440"
TEXT = "#e6ecef"
MUTED = "#8894a0"
ACCENT = "#5bc0de"

EDGE_INTRA = "#3a7bd5"
EDGE_CROSS = "#d9534f"
EDGE_HIGHLIGHT = "#f0ad4e"
NODE_HIGHLIGHT = "#f0ad4e"


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class GraphBundle:
    """Everything the viewer needs for one dataset render."""
    name: str
    domain: str
    split: str
    full_nodes: int
    keep_ids: np.ndarray          # original indices in source graph
    edge_index: np.ndarray        # remapped to [0, n)
    labels: np.ndarray            # per-node integer label
    label_kind: str               # "class" | "community" | "none"
    coords: np.ndarray            # (n, dim)
    dim: int
    colors: np.ndarray            # (n, 4) rgba in [0,1]
    legend: list[tuple[str, str]] # list of (label_str, hex_color)
    communities: np.ndarray       # Louvain assignments (for community layout)
    sample_method: str
    layout_name: str


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(h: str, alpha: int = 255):
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def palette_for(labels: np.ndarray):
    uniq = sorted(np.unique(labels).tolist())
    colors = np.zeros((len(labels), 4), dtype=np.float32)
    legend: list[tuple[str, str]] = []
    for i, u in enumerate(uniq):
        hx = PALETTE[i % len(PALETTE)]
        r, g, b, _ = hex_to_rgba(hx)
        colors[labels == u] = [r / 255, g / 255, b / 255, 1.0]
        legend.append((str(int(u)), hx))
    return colors, legend


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _csr(edge_index: np.ndarray, n: int):
    row = edge_index[0]
    order = np.argsort(row, kind="stable")
    row_s = row[order]
    col_s = edge_index[1][order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, row_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, col_s


def sample_ego(ei, n, target, rng):
    deg = np.bincount(ei[0], minlength=n)
    top = np.argsort(deg)[::-1][:max(10, n // 1000)]
    seed = int(rng.choice(top))
    indptr, indices = _csr(ei, n)
    visited = {seed}
    frontier = [seed]
    while len(visited) < target and frontier:
        nxt = []
        for u in frontier:
            for v in indices[indptr[u]:indptr[u + 1]]:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
                    if len(visited) >= target:
                        break
            if len(visited) >= target:
                break
        frontier = nxt
    return np.array(sorted(visited), dtype=np.int64)


def sample_random_walk(ei, n, target, rng, walk_len=20):
    indptr, indices = _csr(ei, n)
    deg = np.diff(indptr)
    nz = np.where(deg > 0)[0]
    if len(nz) == 0:
        return rng.choice(n, size=min(target, n), replace=False)
    seen: set[int] = set()
    attempts = 0
    while len(seen) < target and attempts < target * 20:
        u = int(rng.choice(nz))
        for _ in range(walk_len):
            seen.add(u)
            if len(seen) >= target:
                break
            s, e = indptr[u], indptr[u + 1]
            if e == s:
                break
            u = int(indices[rng.integers(s, e)])
        attempts += 1
    return np.array(sorted(list(seen))[:target], dtype=np.int64)


def sample_community(ei, n, target, rng):
    try:
        import community as louvain
        import networkx as nx
    except ImportError:
        return sample_random_walk(ei, n, target, rng)
    if n > 100_000:
        return sample_random_walk(ei, n, target, rng)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(ei.T.tolist())
    part = louvain.best_partition(G, random_state=SEED)
    by: dict[int, list[int]] = {}
    for u, c in part.items():
        by.setdefault(c, []).append(u)
    picked: list[int] = []
    for comm in sorted(by.values(), key=len, reverse=True):
        picked.extend(comm)
        if len(picked) >= target:
            break
    return np.array(sorted(picked[:target]), dtype=np.int64)


SAMPLERS: dict[str, Callable] = {
    "ego": sample_ego,
    "community": sample_community,
    "random_walk": sample_random_walk,
}


def induce_subgraph(ei: np.ndarray, keep: np.ndarray):
    max_id = int(ei.max()) + 2
    remap = -np.ones(max_id, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    mask = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    sub = np.stack([remap[ei[0][mask]], remap[ei[1][mask]]], axis=0)
    return sub


# ---------------------------------------------------------------------------
# Community detection helper
# ---------------------------------------------------------------------------

def detect_communities(ei: np.ndarray, n: int) -> np.ndarray:
    try:
        import community as louvain
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(ei.T.tolist())
        part = louvain.best_partition(G, random_state=SEED)
        return np.array([part.get(i, 0) for i in range(n)], dtype=np.int64)
    except Exception:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(ei.T.tolist())
        comp = {u: i for i, cc in enumerate(nx.connected_components(G)) for u in cc}
        return np.array([comp.get(i, 0) for i in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Layout algorithms
# ---------------------------------------------------------------------------

def _nx_graph(ei: np.ndarray, n: int):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(ei.T.tolist())
    return G


def layout_spring(ei, n, dim, **_):
    import networkx as nx
    G = _nx_graph(ei, n)
    pos = nx.spring_layout(G, dim=dim, seed=SEED, iterations=100)
    return np.array([pos[i] for i in range(n)], dtype=np.float32)


def layout_kamada_kawai(ei, n, dim, **_):
    import networkx as nx
    G = _nx_graph(ei, n)
    try:
        pos = nx.kamada_kawai_layout(G, dim=dim)
    except TypeError:
        if dim == 2:
            pos = nx.kamada_kawai_layout(G)
        else:
            # networkx pre-3.0 doesn't accept dim > 2 here
            p2 = nx.kamada_kawai_layout(G)
            pos = {k: np.array([v[0], v[1], 0.0]) for k, v in p2.items()}
    return np.array([pos[i] for i in range(n)], dtype=np.float32)


def layout_spectral(ei, n, dim, **_):
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import eigsh
    row, col = ei[0], ei[1]
    A = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    A = (A + A.T)
    A.data = np.ones_like(A.data)
    d = np.asarray(A.sum(axis=1)).flatten()
    d[d == 0] = 1.0
    d_inv = 1.0 / np.sqrt(d)
    L = eye(n) - diags(d_inv) @ A @ diags(d_inv)
    try:
        vals, vecs = eigsh(L, k=min(dim + 1, n - 1), which="SM")
    except Exception:
        rng = np.random.default_rng(SEED)
        return rng.normal(size=(n, dim)).astype(np.float32)
    coords = vecs[:, 1:dim + 1]
    if coords.shape[1] < dim:
        coords = np.hstack([coords, np.zeros((n, dim - coords.shape[1]))])
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-9)
    return coords.astype(np.float32)


def layout_circular(ei, n, dim, **_):
    if dim == 2:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    # 3D → Fibonacci sphere
    i = np.arange(n, dtype=np.float64)
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5))
    y = 1 - (i / (n - 1)) * 2
    r = np.sqrt(np.maximum(0.0, 1 - y * y))
    theta = phi * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1).astype(np.float32)


def layout_community(ei, n, dim, communities=None, **_):
    import networkx as nx
    if communities is None:
        communities = detect_communities(ei, n)
    n_comms = int(communities.max()) + 1
    centers = layout_circular(None, n_comms, dim) * 5.0
    coords = np.zeros((n, dim), dtype=np.float32)
    for c in range(n_comms):
        nodes_c = np.where(communities == c)[0]
        if len(nodes_c) == 0:
            continue
        mask = np.isin(ei[0], nodes_c) & np.isin(ei[1], nodes_c)
        local_ei = ei[:, mask]
        if len(nodes_c) == 1 or local_ei.shape[1] == 0:
            local_pos = np.zeros((len(nodes_c), dim), dtype=np.float32)
        else:
            remap = -np.ones(n, dtype=np.int64)
            remap[nodes_c] = np.arange(len(nodes_c))
            local_ei_remapped = np.stack(
                [remap[local_ei[0]], remap[local_ei[1]]], axis=0
            )
            G = nx.Graph()
            G.add_nodes_from(range(len(nodes_c)))
            G.add_edges_from(local_ei_remapped.T.tolist())
            pos = nx.spring_layout(G, dim=dim, seed=SEED, iterations=40, scale=1.0)
            local_pos = np.array([pos[i] for i in range(len(nodes_c))],
                                  dtype=np.float32)
        coords[nodes_c] = centers[c] + local_pos
    return coords


LAYOUTS: dict[str, Callable] = {
    "spring": layout_spring,
    "kamada_kawai": layout_kamada_kawai,
    "spectral": layout_spectral,
    "circular": layout_circular,
    "community": layout_community,
}


# ---------------------------------------------------------------------------
# Dataset / custom-file loading
# ---------------------------------------------------------------------------

def load_registry_dataset(name: str):
    from src.data import load_dataset
    g = load_dataset(name)
    return {
        "name": g.name,
        "domain": g.domain,
        "split": g.split,
        "num_nodes": g.num_nodes,
        "edge_index": g.edge_index.numpy(),
        "labels": g.node_labels.numpy() if g.node_labels is not None else None,
    }


def load_custom_file(path: str):
    """Accept edge list (.txt/.csv), GraphML, GEXF."""
    import networkx as nx
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".graphml", ".xml"):
        G = nx.read_graphml(p)
    elif ext == ".gexf":
        G = nx.read_gexf(p)
    elif ext in (".gpickle", ".pkl"):
        G = nx.read_gpickle(p)
    else:
        # fallback: edge list
        delim = "," if ext == ".csv" else None
        G = nx.read_edgelist(p, delimiter=delim, nodetype=str)
    G = nx.convert_node_labels_to_integers(G, label_attribute="orig")
    n = G.number_of_nodes()
    ei = np.array(list(G.edges()), dtype=np.int64).T if G.number_of_edges() else np.zeros((2, 0), dtype=np.int64)
    if ei.size:
        # ensure undirected both directions
        ei = np.concatenate([ei, ei[::-1]], axis=1)
    # pull labels if stored on nodes under common keys
    labels = None
    for key in ("label", "class", "y", "community"):
        vals = nx.get_node_attributes(G, key)
        if vals and len(vals) == n:
            try:
                labels = np.array([int(vals[i]) for i in range(n)])
                break
            except (TypeError, ValueError):
                pass
    return {
        "name": p.stem,
        "domain": "custom",
        "split": "custom",
        "num_nodes": n,
        "edge_index": ei,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _cache_path(name: str, layout: str, method: str, size: int, dim: int) -> Path:
    return LAYOUT_CACHE / f"{name}_{layout}_{method}_n{size}_d{dim}.npz"


def build_bundle(source: dict, dim: int, sample_method: str, sample_size: int,
                 layout_name: str, color_by: str,
                 progress: Callable[[str], None] = lambda _s: None) -> GraphBundle:
    progress("sampling")
    rng = np.random.default_rng(SEED)
    ei = source["edge_index"]
    full_n = source["num_nodes"]

    if full_n > sample_size:
        keep = SAMPLERS[sample_method](ei, full_n, sample_size, rng)
    else:
        keep = np.arange(full_n, dtype=np.int64)
        sample_method = "full"

    sub = induce_subgraph(ei, keep)
    n = len(keep)

    progress("community detection")
    communities = detect_communities(sub, n)

    cache = _cache_path(source["name"], layout_name, sample_method, n, dim)
    if cache.exists():
        progress("loading cached layout")
        coords = np.load(cache)["coords"]
    else:
        progress(f"computing {layout_name} layout")
        coords = LAYOUTS[layout_name](sub, n, dim, communities=communities)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, coords=coords)

    # node labels
    src_labels = source.get("labels")
    if color_by == "class" and src_labels is not None and src_labels.ndim == 1:
        labels = src_labels[keep].astype(np.int64)
        label_kind = "class"
    elif color_by == "degree":
        deg = np.bincount(sub[0], minlength=n)
        bins = np.linspace(0, deg.max() + 1, 8)
        labels = np.digitize(deg, bins).astype(np.int64)
        label_kind = "degree"
    else:
        labels = communities
        label_kind = "community"

    colors, legend = palette_for(labels)

    return GraphBundle(
        name=source["name"], domain=source["domain"], split=source["split"],
        full_nodes=full_n, keep_ids=keep, edge_index=sub, labels=labels,
        label_kind=label_kind, coords=coords, dim=dim, colors=colors,
        legend=legend, communities=communities, sample_method=sample_method,
        layout_name=layout_name,
    )


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT};
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}}
QFrame#sidebar, QFrame#legendPanel {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 8px;
}}
QLabel {{
    color: {TEXT};
}}
QLabel[role="section"] {{
    color: {MUTED};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding-top: 4px;
}}
QLabel[role="muted"] {{
    color: {MUTED};
    font-size: 11px;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {BG_DARK};
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    min-height: 22px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_BG};
    color: {TEXT};
    selection-background-color: {ACCENT};
    border: 1px solid {PANEL_BORDER};
}}
QPushButton {{
    background-color: {ACCENT};
    color: #0c1016;
    font-weight: 600;
    border: none;
    border-radius: 5px;
    padding: 7px 12px;
}}
QPushButton:hover {{ background-color: #6fcce3; }}
QPushButton:disabled {{ background-color: #2a3440; color: {MUTED}; }}
QPushButton[role="secondary"] {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
    font-weight: 500;
}}
QPushButton[role="secondary"]:hover {{ background-color: #20272f; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {PANEL_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
"""


# Qt widget classes are created lazily inside _make_qt_classes() so that
# `import visualize` works without PySide6 installed (e.g. on a headless
# server — for unit testing the data pipeline).

def _make_qt_classes():
    """Returns a namespace of Qt-dependent classes after importing the libs."""
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    pg.setConfigOption("background", BG_DARK)
    pg.setConfigOption("foreground", TEXT)
    pg.setConfigOption("antialias", True)

    # ------------------------------------------------------------------
    # Legend with actual color swatches
    # ------------------------------------------------------------------
    class LegendPanel(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("legendPanel")
            self.setMinimumWidth(220)
            self._layout = QtWidgets.QVBoxLayout(self)
            self._layout.setContentsMargins(14, 14, 14, 14)
            self._layout.setSpacing(6)
            self._header = QtWidgets.QLabel("LEGEND")
            self._header.setProperty("role", "section")
            self._layout.addWidget(self._header)
            self._content_box = QtWidgets.QVBoxLayout()
            self._content_box.setSpacing(4)
            container = QtWidgets.QWidget()
            container.setLayout(self._content_box)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(container)
            self._layout.addWidget(scroll, 1)

        def _clear(self):
            while self._content_box.count():
                it = self._content_box.takeAt(0)
                w = it.widget()
                if w:
                    w.deleteLater()

        def _section(self, title: str):
            lab = QtWidgets.QLabel(title)
            lab.setProperty("role", "section")
            self._content_box.addWidget(lab)

        def _row(self, color_hex: str, text: str, note: str = ""):
            row = QtWidgets.QWidget()
            row.setStyleSheet("background: transparent;")
            hl = QtWidgets.QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(8)
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(
                f"background-color: {color_hex}; border: 1px solid {PANEL_BORDER};"
                " border-radius: 3px;"
            )
            hl.addWidget(swatch)
            label = QtWidgets.QLabel(text)
            hl.addWidget(label, 1)
            if note:
                side = QtWidgets.QLabel(note)
                side.setProperty("role", "muted")
                hl.addWidget(side)
            self._content_box.addWidget(row)

        def _info(self, rows: list[tuple[str, str]]):
            for k, v in rows:
                line = QtWidgets.QLabel(f"<span style='color:{MUTED}'>"
                                        f"{k}</span>  {v}")
                line.setTextFormat(QtCore.Qt.RichText)
                self._content_box.addWidget(line)

        def set_bundle(self, b: GraphBundle, label_name_fn):
            self._clear()
            self._section("DATASET")
            self._info([
                ("name", b.name),
                ("domain", b.domain),
                ("split", b.split),
            ])
            self._section("SAMPLING")
            self._info([
                ("nodes", f"{len(b.keep_ids):,} / {b.full_nodes:,}"),
                ("edges", f"{b.edge_index.shape[1] // 2:,}"),
                ("method", b.sample_method),
                ("layout", b.layout_name),
                ("dim", f"{b.dim}D"),
            ])
            self._section(f"NODES BY {b.label_kind.upper()}")
            for label_id, hx in b.legend[:30]:
                txt = label_name_fn(int(label_id), b.label_kind)
                self._row(hx, txt)
            if len(b.legend) > 30:
                more = QtWidgets.QLabel(f"+ {len(b.legend) - 30} more")
                more.setProperty("role", "muted")
                self._content_box.addWidget(more)
            self._section("EDGES")
            self._row(EDGE_INTRA, f"intra-{b.label_kind}")
            self._row(EDGE_CROSS, f"cross-{b.label_kind}")
            self._row(EDGE_HIGHLIGHT, "highlighted (hover)")
            self._content_box.addStretch(1)

    # ------------------------------------------------------------------
    # 2D view
    # ------------------------------------------------------------------
    class GraphView2D(pg.PlotWidget):
        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setBackground(BG_DARK)
            self.getPlotItem().hideAxis("bottom")
            self.getPlotItem().hideAxis("left")
            self.getPlotItem().getViewBox().setAspectLocked(True)
            self.getPlotItem().setMenuEnabled(False)
            self.getPlotItem().setMouseEnabled(True, True)

            self._bundle: Optional[GraphBundle] = None
            self._scatter: Optional[pg.ScatterPlotItem] = None
            self._edge_intra = None
            self._edge_cross = None
            self._edge_highlight = None
            self._tooltip = pg.TextItem("", color=TEXT, anchor=(0, 1),
                                         fill=pg.mkBrush(22, 28, 36, 230),
                                         border=pg.mkPen(PANEL_BORDER))
            self._tooltip.setZValue(500)
            self.addItem(self._tooltip)
            self._tooltip.hide()

            self._static_labels: list[pg.TextItem] = []

            self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        def set_bundle(self, b: GraphBundle, node_size: float, edge_width: float,
                       edge_alpha: float, show_labels: bool):
            self.getPlotItem().clear()
            self.addItem(self._tooltip)
            self._bundle = b
            coords = b.coords
            edges = b.edge_index
            labels = b.labels

            # edges split intra/cross
            intra_mask = labels[edges[0]] == labels[edges[1]]
            self._edge_intra = self._add_edges(coords, edges[:, intra_mask],
                                               EDGE_INTRA, edge_width, edge_alpha)
            self._edge_cross = self._add_edges(coords, edges[:, ~intra_mask],
                                               EDGE_CROSS, edge_width, edge_alpha)

            # highlight layer on top (empty at first)
            self._edge_highlight = pg.PlotDataItem(
                [], [], connect="pairs",
                pen=pg.mkPen(color=(*hex_to_rgba(EDGE_HIGHLIGHT)[:3], 255),
                             width=edge_width + 1.5))
            self._edge_highlight.setZValue(200)
            self.addItem(self._edge_highlight)

            # nodes
            brushes = [pg.mkBrush(c[0] * 255, c[1] * 255, c[2] * 255, 245)
                       for c in b.colors]
            deg = np.bincount(edges[0], minlength=len(coords)).astype(np.float32)
            sizes = node_size + 2.0 * np.log1p(deg)
            self._scatter = pg.ScatterPlotItem(
                pos=coords, size=sizes, brush=brushes,
                pen=pg.mkPen(color="#0b1014", width=0.8), hoverable=False,
            )
            self._scatter.setZValue(300)
            self.addItem(self._scatter)

            # static labels if small
            self._static_labels = []
            if show_labels and len(coords) <= 200:
                for i, kid in enumerate(b.keep_ids):
                    t = pg.TextItem(text=str(int(kid)), color=MUTED,
                                    anchor=(0.5, -0.5))
                    t.setPos(coords[i, 0], coords[i, 1])
                    self.addItem(t)
                    self._static_labels.append(t)

            self.autoRange(padding=0.05)

        def _add_edges(self, coords, edges, color_hex, width, alpha):
            if edges.size == 0:
                return None
            x = np.empty(2 * edges.shape[1])
            y = np.empty(2 * edges.shape[1])
            x[0::2] = coords[edges[0], 0]; x[1::2] = coords[edges[1], 0]
            y[0::2] = coords[edges[0], 1]; y[1::2] = coords[edges[1], 1]
            r, g, b, _ = hex_to_rgba(color_hex)
            item = pg.PlotDataItem(
                x, y, connect="pairs",
                pen=pg.mkPen(color=(r, g, b, int(alpha)), width=width))
            item.setZValue(100)
            self.addItem(item)
            return item

        # -------- hover --------
        def _on_mouse_moved(self, scene_pos):
            if self._bundle is None or self._scatter is None:
                return
            vb = self.getPlotItem().getViewBox()
            if not self.sceneBoundingRect().contains(scene_pos):
                self._tooltip.hide()
                self._edge_highlight.setData([], [])
                return
            mouse_point = vb.mapSceneToView(scene_pos)
            mx, my = mouse_point.x(), mouse_point.y()

            coords = self._bundle.coords
            diffs = coords - np.array([mx, my])
            dists = np.linalg.norm(diffs, axis=1)
            # pixel threshold in data coords
            pix = vb.viewPixelSize()
            px_thresh = 12 * max(pix[0], pix[1])
            i = int(np.argmin(dists))
            if dists[i] < px_thresh:
                self._show_node_tooltip(i, mx, my)
                return

            # no node → try edges
            edge_idx = self._nearest_edge(mx, my, px_thresh * 0.8)
            if edge_idx is not None:
                self._show_edge_tooltip(edge_idx, mx, my)
            else:
                self._tooltip.hide()
                self._edge_highlight.setData([], [])

        def _show_node_tooltip(self, i, mx, my):
            b = self._bundle
            edges = b.edge_index
            deg = int(((edges[0] == i) | (edges[1] == i)).sum() / 2 * 2)
            deg = int((edges[0] == i).sum())
            lines = [
                f"<b>node {int(b.keep_ids[i])}</b>",
                f"{b.label_kind}: {int(b.labels[i])}",
                f"degree: {deg}",
                f"community: {int(b.communities[i])}",
            ]
            self._tooltip.setHtml("<br>".join(lines))
            self._tooltip.setPos(mx, my)
            self._tooltip.show()
            # highlight its edges
            mask = (edges[0] == i) | (edges[1] == i)
            sub = edges[:, mask]
            if sub.size:
                coords = b.coords
                x = np.empty(2 * sub.shape[1])
                y = np.empty(2 * sub.shape[1])
                x[0::2] = coords[sub[0], 0]; x[1::2] = coords[sub[1], 0]
                y[0::2] = coords[sub[0], 1]; y[1::2] = coords[sub[1], 1]
                self._edge_highlight.setData(x, y)
            else:
                self._edge_highlight.setData([], [])

        def _nearest_edge(self, mx, my, thresh):
            b = self._bundle
            edges = b.edge_index
            if edges.shape[1] == 0:
                return None
            p = b.coords[edges[0]]
            q = b.coords[edges[1]]
            r = np.array([mx, my])
            ab = q - p
            t = np.einsum("ij,ij->i", r - p, ab) / (
                np.einsum("ij,ij->i", ab, ab) + 1e-12)
            t = np.clip(t, 0.0, 1.0)
            proj = p + (ab.T * t).T
            d = np.linalg.norm(proj - r, axis=1)
            i = int(np.argmin(d))
            if d[i] < thresh:
                return i
            return None

        def _show_edge_tooltip(self, edge_i, mx, my):
            b = self._bundle
            u, v = int(b.edge_index[0, edge_i]), int(b.edge_index[1, edge_i])
            lu, lv = int(b.labels[u]), int(b.labels[v])
            kind = "intra" if lu == lv else "cross"
            lines = [
                f"<b>edge</b>",
                f"{int(b.keep_ids[u])} → {int(b.keep_ids[v])}",
                f"{kind}-{b.label_kind}  ({lu} ↔ {lv})",
            ]
            self._tooltip.setHtml("<br>".join(lines))
            self._tooltip.setPos(mx, my)
            self._tooltip.show()
            # highlight this edge
            x = [b.coords[u, 0], b.coords[v, 0]]
            y = [b.coords[u, 1], b.coords[v, 1]]
            self._edge_highlight.setData(x, y)

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------
    class GraphView3D(gl.GLViewWidget):
        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setBackgroundColor(pg.mkColor(BG_DARK))
            self.setMouseTracking(True)
            self._bundle: Optional[GraphBundle] = None
            self._scatter = None
            self._edge_intra = None
            self._edge_cross = None
            self._edge_highlight = None
            self._label_items: list = []

            self._tooltip = QtWidgets.QLabel("", self)
            self._tooltip.setStyleSheet(
                f"color: {TEXT};"
                f"background-color: rgba(22, 28, 36, 230);"
                f"border: 1px solid {PANEL_BORDER};"
                f"padding: 6px 8px; border-radius: 4px;"
                f"font-family: monospace; font-size: 11px;"
            )
            self._tooltip.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self._tooltip.hide()

        def set_bundle(self, b: GraphBundle, node_size: float, edge_width: float,
                       edge_alpha: float, show_labels: bool):
            # clear previous items
            for it in list(self.items):
                self.removeItem(it)
            self._label_items = []
            self._bundle = b

            coords = b.coords.astype(np.float32)
            edges = b.edge_index
            labels = b.labels
            intra_mask = labels[edges[0]] == labels[edges[1]]

            # scale coords into a reasonable viewing cube
            span = coords.max(axis=0) - coords.min(axis=0)
            scale = 10.0 / max(float(span.max()), 1e-6)
            coords = (coords - coords.mean(axis=0)) * scale
            b.coords = coords  # keep in sync for hover projection

            self._add_edges(coords, edges[:, intra_mask], EDGE_INTRA,
                            edge_width, edge_alpha, "intra")
            self._add_edges(coords, edges[:, ~intra_mask], EDGE_CROSS,
                            edge_width, edge_alpha, "cross")

            deg = np.bincount(edges[0], minlength=len(coords)).astype(np.float32)
            sizes = node_size + 1.5 * np.log1p(deg)
            self._scatter = gl.GLScatterPlotItem(
                pos=coords, size=sizes, color=b.colors, pxMode=True,
            )
            self._scatter.setGLOptions("translucent")
            self.addItem(self._scatter)

            if show_labels and len(coords) <= 150 and hasattr(gl, "GLTextItem"):
                for i, kid in enumerate(b.keep_ids):
                    try:
                        t = gl.GLTextItem(pos=coords[i], text=str(int(kid)),
                                           color=(200, 210, 220, 200))
                        self.addItem(t)
                        self._label_items.append(t)
                    except Exception:
                        break

            radius = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
            self.opts["distance"] = max(radius * 1.5, 4.0)
            self.update()

        def _add_edges(self, coords, edges, color_hex, width, alpha, kind):
            if edges.size == 0:
                setattr(self, f"_edge_{kind}", None)
                return
            verts = np.empty((2 * edges.shape[1], 3), dtype=np.float32)
            verts[0::2] = coords[edges[0]]
            verts[1::2] = coords[edges[1]]
            r, g, b, _ = hex_to_rgba(color_hex)
            item = gl.GLLinePlotItem(
                pos=verts, mode="lines", antialias=True,
                color=(r / 255, g / 255, b / 255, alpha / 255),
                width=float(width),
            )
            item.setGLOptions("translucent")
            self.addItem(item)
            setattr(self, f"_edge_{kind}", item)

        # -------- 3D hover via screen-space projection --------
        @staticmethod
        def _qmat_to_np(m) -> np.ndarray:
            # QMatrix4x4 indexing is m[row, col]; we build a numpy row-major array.
            return np.array([[m[r, c] for c in range(4)] for r in range(4)],
                            dtype=np.float64)

        def _project(self, pts: np.ndarray) -> np.ndarray:
            try:
                V = self._qmat_to_np(self.viewMatrix())
                P = self._qmat_to_np(self.projectionMatrix())
            except Exception:
                return np.full((len(pts), 3), np.inf)
            homog = np.hstack([pts, np.ones((len(pts), 1))])
            eye = homog @ V.T
            clip = eye @ P.T
            w = clip[:, 3:4]
            w = np.where(np.abs(w) < 1e-9, 1.0, w)
            ndc = clip[:, :3] / w
            sw, sh = self.width(), self.height()
            screen = np.empty_like(ndc)
            screen[:, 0] = (ndc[:, 0] + 1) * 0.5 * sw
            screen[:, 1] = (1 - (ndc[:, 1] + 1) * 0.5) * sh
            # points behind the camera (OpenGL convention: eye-space z > 0 is behind)
            behind = eye[:, 2] > 0
            screen[behind, 0] = np.inf
            return screen

        def mouseMoveEvent(self, ev):
            super().mouseMoveEvent(ev)
            if self._bundle is None or self._scatter is None:
                return
            pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            cx, cy = float(pos.x()), float(pos.y())
            screen = self._project(self._bundle.coords)
            dists = np.hypot(screen[:, 0] - cx, screen[:, 1] - cy)
            i = int(np.argmin(dists))
            if dists[i] < 14.0:
                b = self._bundle
                edges = b.edge_index
                deg = int((edges[0] == i).sum())
                neighbors = set(edges[1, edges[0] == i].tolist())
                self._tooltip.setText(
                    f"id {int(b.keep_ids[i])}\n"
                    f"{b.label_kind} {int(b.labels[i])}\n"
                    f"community {int(b.communities[i])}\n"
                    f"degree {deg}\n"
                    f"neighbors {min(len(neighbors), 6)}"
                    + (" …" if len(neighbors) > 6 else "")
                )
                self._tooltip.adjustSize()
                self._tooltip.move(int(cx + 14), int(cy + 14))
                self._tooltip.raise_()
                self._tooltip.show()
            else:
                self._tooltip.hide()

        def leaveEvent(self, ev):
            super().leaveEvent(ev)
            self._tooltip.hide()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    class Sidebar(QtWidgets.QFrame):
        render_requested = QtCore.Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("sidebar")
            self.setMinimumWidth(310)
            self.setMaximumWidth(360)

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            title = QtWidgets.QLabel("NetFM Graph Explorer")
            title.setStyleSheet("font-size: 16px; font-weight: 600;")
            sub = QtWidgets.QLabel("Interactive 2D/3D graph viewer")
            sub.setProperty("role", "muted")
            root.addWidget(title)
            root.addWidget(sub)

            # dataset
            root.addWidget(self._section("DATASET"))
            self.dataset_combo = QtWidgets.QComboBox()
            self._populate_datasets()
            self.dataset_combo.currentTextChanged.connect(self._on_dataset_change)
            root.addWidget(self.dataset_combo)

            self.custom_btn = QtWidgets.QPushButton("Load custom file…")
            self.custom_btn.setProperty("role", "secondary")
            self.custom_btn.clicked.connect(self._on_custom_file)
            root.addWidget(self.custom_btn)
            self.custom_path_lbl = QtWidgets.QLabel("")
            self.custom_path_lbl.setProperty("role", "muted")
            root.addWidget(self.custom_path_lbl)
            self.custom_file: Optional[str] = None

            # dim
            root.addWidget(self._section("VIEW"))
            dim_row = QtWidgets.QHBoxLayout()
            self.dim_2d = QtWidgets.QPushButton("2D")
            self.dim_3d = QtWidgets.QPushButton("3D")
            for b in (self.dim_2d, self.dim_3d):
                b.setCheckable(True)
                b.setProperty("role", "secondary")
            self.dim_3d.setChecked(True)
            self.dim_2d.clicked.connect(lambda: self._set_dim(2))
            self.dim_3d.clicked.connect(lambda: self._set_dim(3))
            dim_row.addWidget(self.dim_2d)
            dim_row.addWidget(self.dim_3d)
            w = QtWidgets.QWidget(); w.setLayout(dim_row)
            root.addWidget(w)

            # layout
            self.layout_combo = QtWidgets.QComboBox()
            self.layout_combo.addItems(list(LAYOUTS.keys()))
            self.layout_combo.setCurrentText("spring")
            root.addWidget(QtWidgets.QLabel("Layout algorithm"))
            root.addWidget(self.layout_combo)

            # color by
            self.color_combo = QtWidgets.QComboBox()
            self.color_combo.addItems(["class", "community", "degree"])
            root.addWidget(QtWidgets.QLabel("Color by"))
            root.addWidget(self.color_combo)

            # sampling
            root.addWidget(self._section("SAMPLING"))
            self.sample_combo = QtWidgets.QComboBox()
            self.sample_combo.addItems(list(SAMPLERS.keys()))
            root.addWidget(QtWidgets.QLabel("Method"))
            root.addWidget(self.sample_combo)

            self.sample_spin = QtWidgets.QSpinBox()
            self.sample_spin.setRange(50, 20000)
            self.sample_spin.setSingleStep(100)
            self.sample_spin.setValue(DEFAULT_SAMPLE)
            root.addWidget(QtWidgets.QLabel("Sample size"))
            root.addWidget(self.sample_spin)

            # appearance
            root.addWidget(self._section("APPEARANCE"))
            self.node_slider = self._make_slider(4, 40, 14, "Node size")
            self.edge_slider = self._make_slider(1, 6, 2, "Edge width")
            self.alpha_slider = self._make_slider(20, 255, 110, "Edge opacity")
            for w in self.node_slider[0], self.edge_slider[0], self.alpha_slider[0]:
                root.addWidget(w)

            self.labels_chk = QtWidgets.QCheckBox("Show node labels (small graphs)")
            self.labels_chk.setChecked(True)
            root.addWidget(self.labels_chk)

            # render
            root.addStretch(1)
            self.render_btn = QtWidgets.QPushButton("Render graph")
            self.render_btn.clicked.connect(lambda: self.render_requested.emit())
            root.addWidget(self.render_btn)

            self.status = QtWidgets.QLabel("")
            self.status.setProperty("role", "muted")
            self.status.setWordWrap(True)
            root.addWidget(self.status)

            self._dim = 3
            self._on_dataset_change(self.dataset_combo.currentText())

        def _section(self, title):
            lab = QtWidgets.QLabel(title)
            lab.setProperty("role", "section")
            return lab

        def _make_slider(self, mn, mx, val, label):
            wrapper = QtWidgets.QWidget()
            vl = QtWidgets.QVBoxLayout(wrapper)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(3)
            hdr = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel(label)
            val_lbl = QtWidgets.QLabel(str(val))
            val_lbl.setProperty("role", "muted")
            hdr.addWidget(name); hdr.addStretch(); hdr.addWidget(val_lbl)
            w = QtWidgets.QWidget(); w.setLayout(hdr)
            vl.addWidget(w)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(mn, mx); sl.setValue(val)
            sl.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
            vl.addWidget(sl)
            return wrapper, sl

        def _populate_datasets(self):
            try:
                from src.data import PRETRAIN_DATASETS, HELDOUT_DATASETS
                names = PRETRAIN_DATASETS + HELDOUT_DATASETS
            except Exception:
                names = []
            self.dataset_combo.clear()
            self.dataset_combo.addItems(names)
            self.dataset_combo.addItem("— custom file —")

        def _on_dataset_change(self, name):
            is_custom = name == "— custom file —"
            self.custom_btn.setVisible(is_custom)
            self.custom_path_lbl.setVisible(is_custom and bool(self.custom_file))
            if not is_custom:
                self.status.setText(f"Selected: {name}")

        def _on_custom_file(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open graph file", "",
                "Graphs (*.txt *.csv *.graphml *.gexf *.gpickle *.pkl);;All files (*)",
            )
            if path:
                self.custom_file = path
                self.custom_path_lbl.setText(Path(path).name)
                self.custom_path_lbl.setVisible(True)
                self.status.setText(f"Custom file: {Path(path).name}")

        def _set_dim(self, d):
            self._dim = d
            self.dim_2d.setChecked(d == 2)
            self.dim_3d.setChecked(d == 3)

        def current_options(self):
            name = self.dataset_combo.currentText()
            source_kind = "custom" if name == "— custom file —" else "registry"
            return {
                "dataset": name,
                "source_kind": source_kind,
                "custom_path": self.custom_file,
                "dim": self._dim,
                "layout": self.layout_combo.currentText(),
                "color_by": self.color_combo.currentText(),
                "sample_method": self.sample_combo.currentText(),
                "sample_size": self.sample_spin.value(),
                "node_size": self.node_slider[1].value(),
                "edge_width": self.edge_slider[1].value(),
                "edge_alpha": self.alpha_slider[1].value(),
                "show_labels": self.labels_chk.isChecked(),
            }

        def set_busy(self, busy, text=""):
            self.render_btn.setDisabled(busy)
            self.render_btn.setText("Rendering…" if busy else "Render graph")
            if text:
                self.status.setText(text)

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("NetFM · Graph Explorer")
            self.resize(1500, 900)
            self.setStyleSheet(_STYLE)

            splitter = QtWidgets.QSplitter(self)
            splitter.setHandleWidth(6)
            self.setCentralWidget(splitter)

            self.sidebar = Sidebar()
            splitter.addWidget(self.sidebar)

            self.graph_container = QtWidgets.QWidget()
            self.graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
            self.graph_layout.setContentsMargins(6, 6, 6, 6)
            self.placeholder = QtWidgets.QLabel(
                "Pick a dataset on the left and click <b>Render graph</b>.\n"
                "Large graphs are auto-sampled.")
            self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
            self.placeholder.setStyleSheet(f"color: {MUTED}; font-size: 14px;")
            self.graph_layout.addWidget(self.placeholder)
            splitter.addWidget(self.graph_container)

            self.legend = LegendPanel()
            splitter.addWidget(self.legend)

            splitter.setSizes([340, 940, 240])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setStretchFactor(2, 0)

            self.current_view = None
            self.sidebar.render_requested.connect(self._render)

        def _swap_view(self, new_widget):
            while self.graph_layout.count():
                it = self.graph_layout.takeAt(0)
                w = it.widget()
                if w:
                    w.deleteLater()
            self.graph_layout.addWidget(new_widget, 1)
            self.current_view = new_widget

        def _render(self):
            opts = self.sidebar.current_options()
            try:
                if opts["source_kind"] == "custom":
                    if not opts["custom_path"]:
                        self.sidebar.set_busy(False, "⚠ no custom file chosen")
                        return
                    self.sidebar.set_busy(True, "loading custom file…")
                    QtWidgets.QApplication.processEvents()
                    source = load_custom_file(opts["custom_path"])
                else:
                    self.sidebar.set_busy(True, f"loading {opts['dataset']}…")
                    QtWidgets.QApplication.processEvents()
                    source = load_registry_dataset(opts["dataset"])

                def prog(msg):
                    self.sidebar.status.setText(msg)
                    QtWidgets.QApplication.processEvents()

                bundle = build_bundle(
                    source, opts["dim"], opts["sample_method"],
                    opts["sample_size"], opts["layout"], opts["color_by"],
                    progress=prog,
                )

                view = GraphView2D() if opts["dim"] == 2 else GraphView3D()
                view.set_bundle(
                    bundle,
                    node_size=opts["node_size"],
                    edge_width=opts["edge_width"],
                    edge_alpha=opts["edge_alpha"],
                    show_labels=opts["show_labels"],
                )
                self._swap_view(view)
                self.legend.set_bundle(bundle, self._label_name)

                self.sidebar.set_busy(
                    False, f"rendered {len(bundle.keep_ids):,} nodes · "
                           f"{bundle.edge_index.shape[1] // 2:,} edges")
            except Exception as ex:
                import traceback
                traceback.print_exc()
                self.sidebar.set_busy(False, f"⚠ {type(ex).__name__}: {ex}")

        def _label_name(self, label_id: int, kind: str) -> str:
            if kind == "degree":
                return f"bucket {label_id}"
            return f"{kind} {label_id}"

    return {
        "LegendPanel": LegendPanel,
        "GraphView2D": GraphView2D,
        "GraphView3D": GraphView3D,
        "Sidebar": Sidebar,
        "MainWindow": MainWindow,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NetFM Graph Explorer")
    parser.add_argument("--dataset", default=None,
                        help="Skip the dashboard and render this dataset immediately")
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
    parser.add_argument("--layout", default="spring", choices=list(LAYOUTS.keys()))
    parser.add_argument("--sample", default="ego", choices=list(SAMPLERS.keys()))
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    try:
        import pyqtgraph  # noqa: F401
        from pyqtgraph.Qt import QtWidgets  # noqa: F401
    except ImportError:
        print("Visualizer requires PySide6 + pyqtgraph."
              " Install with: pip install -e .[viz]")
        sys.exit(1)

    classes = _make_qt_classes()
    MainWindow = classes["MainWindow"]

    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    if args.dataset is not None:
        # pre-fill sidebar and kick off a render
        combo = win.sidebar.dataset_combo
        idx = combo.findText(args.dataset)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        win.sidebar._set_dim(args.dim)
        win.sidebar.layout_combo.setCurrentText(args.layout)
        win.sidebar.sample_combo.setCurrentText(args.sample)
        win.sidebar.sample_spin.setValue(args.size)
        win.sidebar.render_requested.emit()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
