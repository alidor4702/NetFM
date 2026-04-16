"""
NetFM Graph Visualizer

A native Qt (PySide6 + pyqtgraph) 2D/3D explorer for any dataset in the
NetFM registry. Launched from a Rich-styled terminal picker.

Usage:
    pip install -e .[viz]
    python -m src.visualize

Large graphs are automatically sub-sampled (ego / community / random walk)
so the viewer stays interactive. Layouts and community assignments are
cached under figures/layouts/ so repeated runs are instant.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_CACHE = PROJECT_ROOT / "figures" / "layouts"

DEFAULT_SAMPLE = 2000
SMALL_LIMIT = 3000
SEED = 0

# 20-color palette (colorblind-ish) cycled for >20 classes
PALETTE = [
    "#5bc0de", "#f0ad4e", "#d9534f", "#5cb85c", "#9b59b6",
    "#e67e22", "#1abc9c", "#34495e", "#f39c12", "#c0392b",
    "#2ecc71", "#3498db", "#8e44ad", "#16a085", "#27ae60",
    "#d35400", "#2980b9", "#7f8c8d", "#bdc3c7", "#95a5a6",
]

BG_DARK = "#0f1419"
EDGE_INTRA = "#3a7bd5"   # intra-class edge color (within same community/class)
EDGE_CROSS = "#d9534f"   # cross-class edge color
EDGE_ALPHA = 70          # 0-255


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(h: str, alpha: int = 255) -> tuple:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def palette_for(labels: np.ndarray):
    """Map labels to palette RGBA tuples. Returns (colors[N,4], legend)."""
    uniq = np.unique(labels)
    legend = {}
    colors = np.zeros((len(labels), 4), dtype=np.float32)
    for i, u in enumerate(uniq):
        hex_color = PALETTE[i % len(PALETTE)]
        r, g, b, a = hex_to_rgba(hex_color)
        colors[labels == u] = [r / 255, g / 255, b / 255, 1.0]
        legend[int(u)] = hex_color
    return colors, legend


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _csr_neighbors(edge_index: np.ndarray, num_nodes: int):
    """Return (indptr, indices) adjacency from an undirected edge_index."""
    row = edge_index[0]
    col = edge_index[1]
    order = np.argsort(row, kind="stable")
    row_s = row[order]
    col_s = col[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, row_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, col_s


def sample_ego(edge_index: np.ndarray, num_nodes: int, target: int,
               rng: np.random.Generator) -> np.ndarray:
    """BFS out from a high-degree seed until we have `target` nodes."""
    deg = np.bincount(edge_index[0], minlength=num_nodes)
    top = np.argsort(deg)[::-1][:max(10, num_nodes // 1000)]
    seed = int(rng.choice(top))

    indptr, indices = _csr_neighbors(edge_index, num_nodes)
    visited = {seed}
    frontier = [seed]
    while len(visited) < target and frontier:
        next_frontier = []
        for u in frontier:
            for v in indices[indptr[u]:indptr[u + 1]]:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    next_frontier.append(v)
                    if len(visited) >= target:
                        break
            if len(visited) >= target:
                break
        frontier = next_frontier
    return np.array(sorted(visited), dtype=np.int64)


def sample_random_walk(edge_index: np.ndarray, num_nodes: int, target: int,
                       rng: np.random.Generator, num_walks: int = 400,
                       walk_len: int = 20) -> np.ndarray:
    """Random-walk-induced subgraph."""
    indptr, indices = _csr_neighbors(edge_index, num_nodes)
    deg = np.diff(indptr)
    nonzero = np.where(deg > 0)[0]
    if len(nonzero) == 0:
        return rng.choice(num_nodes, size=min(target, num_nodes), replace=False)

    seen = set()
    while len(seen) < target:
        start = int(rng.choice(nonzero))
        u = start
        for _ in range(walk_len):
            seen.add(u)
            if len(seen) >= target:
                break
            s, e = indptr[u], indptr[u + 1]
            if e == s:
                break
            u = int(indices[rng.integers(s, e)])
        if len(seen) >= target * 3:
            break
    return np.array(sorted(seen)[:target], dtype=np.int64)


def sample_community(edge_index: np.ndarray, num_nodes: int, target: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Louvain on the full graph, keep largest communities up to `target`."""
    try:
        import community as louvain
        import networkx as nx
    except ImportError:
        return sample_random_walk(edge_index, num_nodes, target, rng)

    # For huge graphs this is still expensive — fall back if too big
    if num_nodes > 100_000:
        return sample_random_walk(edge_index, num_nodes, target, rng)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T.tolist())
    part = louvain.best_partition(G, random_state=SEED)

    by_comm: dict[int, list[int]] = {}
    for n, c in part.items():
        by_comm.setdefault(c, []).append(n)
    # Largest communities first
    ordered = sorted(by_comm.values(), key=len, reverse=True)
    picked: list[int] = []
    for comm in ordered:
        picked.extend(comm)
        if len(picked) >= target:
            break
    return np.array(sorted(picked[:target]), dtype=np.int64)


SAMPLERS = {
    "ego": sample_ego,
    "community": sample_community,
    "random_walk": sample_random_walk,
}


def induce_subgraph(edge_index: np.ndarray, keep_nodes: np.ndarray):
    """Return (remapped_edge_index, remap) where remap[old]=new or -1."""
    remap = -np.ones(int(edge_index.max()) + 2, dtype=np.int64)
    remap[keep_nodes] = np.arange(len(keep_nodes))
    mask = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)
    sub = edge_index[:, mask]
    sub = np.stack([remap[sub[0]], remap[sub[1]]], axis=0)
    return sub, remap


# ---------------------------------------------------------------------------
# Layout + community detection (with cache)
# ---------------------------------------------------------------------------

def _cache_path(dataset: str, method: str, size: int, dim: int, seed: int) -> Path:
    return LAYOUT_CACHE / f"{dataset}_{method}_n{size}_d{dim}_s{seed}.npz"


def compute_layout_and_community(sub_edge: np.ndarray, n: int, dim: int,
                                 cache_file: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache_file.exists():
        z = np.load(cache_file)
        return z["coords"], z["communities"]

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(sub_edge.T.tolist())

    print(f"  computing spring layout (dim={dim}, n={n})...")
    pos = nx.spring_layout(G, dim=dim, seed=SEED, iterations=80)
    coords = np.array([pos[i] for i in range(n)], dtype=np.float32)

    try:
        import community as louvain
        part = louvain.best_partition(G, random_state=SEED)
        communities = np.array([part.get(i, 0) for i in range(n)], dtype=np.int64)
    except ImportError:
        # fallback: connected components
        comp = {u: i for i, comp in enumerate(nx.connected_components(G)) for u in comp}
        communities = np.array([comp.get(i, 0) for i in range(n)], dtype=np.int64)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, coords=coords, communities=communities)
    return coords, communities


# ---------------------------------------------------------------------------
# Qt 2D / 3D views
# ---------------------------------------------------------------------------

def _make_app():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def show_2d(coords, edges, node_colors, node_sizes, node_text,
            intra_mask, legend_entries, title):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

    app = _make_app()
    pg.setConfigOption("background", BG_DARK)
    pg.setConfigOption("foreground", "#e6e6e6")

    win = pg.GraphicsLayoutWidget(show=True, size=(1200, 800), title=title)
    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.hideAxis("bottom")
    plot.hideAxis("left")
    plot.setTitle(title, color="#e6e6e6", size="11pt")

    # edges as two PlotDataItems (intra + cross)
    def edges_to_xy(mask):
        e = edges[:, mask]
        x = np.empty(2 * e.shape[1])
        y = np.empty(2 * e.shape[1])
        x[0::2] = coords[e[0], 0]; x[1::2] = coords[e[1], 0]
        y[0::2] = coords[e[0], 1]; y[1::2] = coords[e[1], 1]
        return x, y

    if intra_mask.any():
        x, y = edges_to_xy(intra_mask)
        plot.plot(x, y, connect="pairs",
                  pen=pg.mkPen(color=(*hex_to_rgba(EDGE_INTRA)[:3], EDGE_ALPHA), width=1))
    if (~intra_mask).any():
        x, y = edges_to_xy(~intra_mask)
        plot.plot(x, y, connect="pairs",
                  pen=pg.mkPen(color=(*hex_to_rgba(EDGE_CROSS)[:3], EDGE_ALPHA), width=1))

    # nodes
    brushes = [pg.mkBrush(c[0] * 255, c[1] * 255, c[2] * 255, 230) for c in node_colors]
    scatter = pg.ScatterPlotItem(pos=coords, size=node_sizes,
                                 brush=brushes, pen=pg.mkPen(color="#202830", width=0.5),
                                 hoverable=True, tip=None, data=node_text)
    plot.addItem(scatter)

    # hover label
    hover = pg.TextItem(text="", color="#e6e6e6", anchor=(0, 1),
                        fill=pg.mkBrush(20, 26, 34, 200),
                        border=pg.mkPen("#3a4550"))
    hover.setZValue(100)
    plot.addItem(hover)
    hover.hide()

    def on_hover(_scatter, points, _ev):
        if len(points) > 0:
            p = points[0]
            hover.setText(str(p.data()))
            hover.setPos(p.pos())
            hover.show()
        else:
            hover.hide()

    scatter.sigHovered.connect(on_hover)

    # static labels for small graphs
    if len(coords) <= 200:
        for i, name in enumerate(node_text):
            t = pg.TextItem(text=str(name), color="#9aa5b1", anchor=(0.5, -0.5))
            t.setPos(coords[i, 0], coords[i, 1])
            plot.addItem(t)

    # legend text
    legend = pg.TextItem(text=legend_entries, color="#e6e6e6", anchor=(0, 0),
                         fill=pg.mkBrush(20, 26, 34, 220),
                         border=pg.mkPen("#3a4550"))
    legend.setParentItem(plot.getViewBox())
    legend.setPos(10, 10)

    app.exec()


def show_3d(coords, edges, node_colors, node_sizes, node_text,
            intra_mask, legend_entries, title):
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

    app = _make_app()
    view = gl.GLViewWidget()
    view.setWindowTitle(title)
    view.resize(1200, 800)
    view.setBackgroundColor(pg.mkColor(BG_DARK))
    radius = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    view.opts["distance"] = max(radius * 1.6, 2.0)

    def edge_vertices(mask):
        e = edges[:, mask]
        verts = np.empty((2 * e.shape[1], 3), dtype=np.float32)
        verts[0::2] = coords[e[0]]
        verts[1::2] = coords[e[1]]
        return verts

    if intra_mask.any():
        r, g, b, _ = hex_to_rgba(EDGE_INTRA)
        view.addItem(gl.GLLinePlotItem(
            pos=edge_vertices(intra_mask), mode="lines", antialias=True,
            color=(r / 255, g / 255, b / 255, EDGE_ALPHA / 255), width=1.0,
        ))
    if (~intra_mask).any():
        r, g, b, _ = hex_to_rgba(EDGE_CROSS)
        view.addItem(gl.GLLinePlotItem(
            pos=edge_vertices(~intra_mask), mode="lines", antialias=True,
            color=(r / 255, g / 255, b / 255, EDGE_ALPHA / 255), width=1.0,
        ))

    # nodes
    scatter = gl.GLScatterPlotItem(
        pos=coords, size=node_sizes, color=node_colors, pxMode=True,
    )
    scatter.setGLOptions("translucent")
    view.addItem(scatter)

    # labels for small graphs (GLTextItem available in recent pyqtgraph)
    try:
        if len(coords) <= 150 and hasattr(gl, "GLTextItem"):
            for i, name in enumerate(node_text):
                t = gl.GLTextItem(pos=coords[i], text=str(name), color=(200, 210, 220, 200))
                view.addItem(t)
    except Exception:
        pass

    # HUD overlay with legend (floating QLabel)
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(view)

    hud = QtWidgets.QLabel(legend_entries, container)
    hud.setStyleSheet(
        "color: #e6e6e6;"
        "background-color: rgba(20,26,34,220);"
        "border: 1px solid #3a4550;"
        "padding: 8px; font-family: monospace;"
    )
    hud.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
    hud.adjustSize()
    hud.move(10, 10)
    hud.raise_()

    container.resize(1200, 800)
    container.setWindowTitle(title)
    container.show()
    app.exec()


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------

def run_viewer(dataset: str, dim: int, sample_method: str, sample_size: int):
    from src.data import load_dataset

    g = load_dataset(dataset)
    edge_index = g.edge_index.numpy()

    rng = np.random.default_rng(SEED)
    needs_sampling = g.num_nodes > sample_size
    if needs_sampling:
        print(f"  sampling {sample_size}/{g.num_nodes} nodes via {sample_method}...")
        keep = SAMPLERS[sample_method](edge_index, g.num_nodes, sample_size, rng)
    else:
        keep = np.arange(g.num_nodes, dtype=np.int64)
        sample_method = "full"

    sub_edge, _ = induce_subgraph(edge_index, keep)
    n = len(keep)

    cache = _cache_path(dataset, sample_method, n, dim, SEED)
    coords, communities = compute_layout_and_community(sub_edge, n, dim, cache)

    # labels: use ground truth if single-label, else community
    labels_are = "community"
    if g.node_labels is not None and g.node_labels.ndim == 1:
        labels = g.node_labels.numpy()[keep]
        labels_are = "class"
    else:
        labels = communities

    node_colors, legend_map = palette_for(labels)

    # node sizes by log-degree
    deg = np.bincount(sub_edge[0], minlength=n).astype(np.float32)
    node_sizes = 4.0 + 2.0 * np.log1p(deg)
    node_sizes = np.clip(node_sizes, 4.0, 18.0)

    # edge types: intra- vs inter-label
    intra_mask = labels[sub_edge[0]] == labels[sub_edge[1]]

    # node text
    node_text = np.array([
        f"id={int(keep[i])} {labels_are}={int(labels[i])} deg={int(deg[i])}"
        for i in range(n)
    ])

    # legend string
    legend_lines = [
        f"dataset : {g.name}",
        f"domain  : {g.domain}",
        f"split   : {g.split}",
        f"nodes   : {n:,} (of {g.num_nodes:,})",
        f"edges   : {sub_edge.shape[1] // 2:,}",
        f"sample  : {sample_method}",
        f"coloring: {labels_are} ({len(np.unique(labels))} groups)",
        "",
        f"edges   : intra-{labels_are}  ·  cross-{labels_are}",
        f"          {EDGE_INTRA}            {EDGE_CROSS}",
    ]
    legend_text = "\n".join(legend_lines)

    title = f"NetFM · {g.name} · {dim}D · {sample_method}"

    if dim == 2:
        show_2d(coords, sub_edge, node_colors, node_sizes, node_text,
                intra_mask, legend_text, title)
    else:
        show_3d(coords, sub_edge, node_colors, node_sizes, node_text,
                intra_mask, legend_text, title)


# ---------------------------------------------------------------------------
# Rich terminal launcher
# ---------------------------------------------------------------------------

def _read_summary_row(dataset: str) -> Optional[dict]:
    stats_path = PROJECT_ROOT / "figures" / "data" / dataset / "stats.json"
    if not stats_path.exists():
        return None
    import json
    return json.loads(stats_path.read_text())


def launch_terminal():
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.prompt import Prompt, IntPrompt
    except ImportError:
        print("Missing rich. Install viz extras: pip install -e .[viz]")
        sys.exit(1)

    from src.data import DATASET_REGISTRY, PRETRAIN_DATASETS, HELDOUT_DATASETS

    console = Console()
    console.print("\n[bold cyan]NetFM graph viewer[/bold cyan]\n")

    names = PRETRAIN_DATASETS + HELDOUT_DATASETS

    table = Table(show_header=True, header_style="bold magenta",
                  border_style="#3a4550")
    table.add_column("#", style="dim", width=3)
    table.add_column("dataset", style="cyan")
    table.add_column("domain", style="green")
    table.add_column("split")
    table.add_column("nodes", justify="right")
    table.add_column("edges", justify="right")
    table.add_column("labels")

    for i, name in enumerate(names):
        stats = _read_summary_row(name)
        if stats is None:
            table.add_row(str(i), name, "?", "?", "?", "?", "?")
            continue
        lb = "-"
        if stats.get("labels"):
            if stats["labels"]["type"] == "single":
                lb = f"{stats['labels']['observed_classes']} cls"
            else:
                lb = f"{stats['labels']['num_labels']} multi"
        table.add_row(
            str(i), stats["name"], stats["domain"], stats["split"],
            f"{stats['num_nodes']:,}", f"{stats['num_edges_undirected']:,}", lb,
        )
    console.print(table)
    console.print("")

    choice = Prompt.ask("[bold]dataset[/bold] (name or #)", default="cora")
    if choice.isdigit():
        idx = int(choice)
        if not (0 <= idx < len(names)):
            console.print("[red]invalid index[/red]"); sys.exit(1)
        dataset = names[idx]
    else:
        if choice not in DATASET_REGISTRY:
            console.print(f"[red]unknown dataset: {choice}[/red]"); sys.exit(1)
        dataset = choice

    dim = IntPrompt.ask("[bold]dimensions[/bold] (2 or 3)", default=3,
                        choices=["2", "3"])

    stats = _read_summary_row(dataset)
    num_nodes = stats["num_nodes"] if stats else 0
    if num_nodes > SMALL_LIMIT:
        method = Prompt.ask(
            "[bold]sample method[/bold]", default="ego",
            choices=["ego", "community", "random_walk"],
        )
        size = IntPrompt.ask("[bold]sample size[/bold]", default=DEFAULT_SAMPLE)
    else:
        method, size = "ego", num_nodes or DEFAULT_SAMPLE

    console.print(f"\n[dim]launching viewer for [cyan]{dataset}[/cyan] "
                  f"({dim}D, {method}, n={size})[/dim]\n")
    run_viewer(dataset, dim, method, size)


def main():
    parser = argparse.ArgumentParser(description="NetFM graph visualizer")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
    parser.add_argument("--sample", default="ego",
                        choices=["ego", "community", "random_walk"])
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    if args.dataset is None:
        launch_terminal()
    else:
        run_viewer(args.dataset, args.dim, args.sample, args.size)


if __name__ == "__main__":
    main()
