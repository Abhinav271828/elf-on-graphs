"""Render a graph with the model's predicted path highlighted, for wandb
logging. Uses `networkx` purely for layout (`spring_layout`) and drawing
helpers; all graph math (validity, shortest-path, diameter) lives in
graphgen.py / metrics.py -- this module only visualizes.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .graphgen import Graph
from .metrics import EvalExample

PATH_COLOR = "#d62728"    # highlighted path edges/interior nodes
START_COLOR = "#2ca02c"   # predicted path's first node
END_COLOR = "#9467bd"     # predicted path's last node
BASE_NODE_COLOR = "#a9c6e8"
BASE_EDGE_COLOR = "#c8c8c8"


def _to_networkx(graph: Graph) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(graph.nodes)
    g.add_edges_from(graph.edges)
    return g


def render_graph_path(graph: Graph, predicted_path: Optional[List[int]], diameter: int,
                       metrics: Optional[dict] = None, title: Optional[str] = None) -> "plt.Figure":
    g = _to_networkx(graph)
    pos = nx.spring_layout(g, seed=0)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=140)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=BASE_EDGE_COLOR, width=1.2)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=BASE_NODE_COLOR, node_size=380, edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9)

    path_edges = []
    if predicted_path and len(predicted_path) >= 2:
        for u, v in zip(predicted_path, predicted_path[1:]):
            if g.has_edge(u, v):
                path_edges.append((u, v))
        if path_edges:
            nx.draw_networkx_edges(g, pos, ax=ax, edgelist=path_edges, edge_color=PATH_COLOR, width=3.0)

        path_nodes_in_graph = [n for n in predicted_path if n in g.nodes]
        interior = [n for n in path_nodes_in_graph[1:-1]]
        if interior:
            nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=interior, node_color=PATH_COLOR, node_size=420)
        if predicted_path[0] in g.nodes:
            nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=[predicted_path[0]], node_color=START_COLOR, node_size=460)
        if predicted_path[-1] in g.nodes:
            nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=[predicted_path[-1]], node_color=END_COLOR, node_size=460)
        nx.draw_networkx_labels(g, pos, ax=ax, labels={n: n for n in path_nodes_in_graph}, font_size=9, font_color="white")

    if title is None:
        pred_len = len(predicted_path) - 1 if predicted_path else None
        m = metrics or {}
        title = (f"diameter={diameter}  pred_len={pred_len}\n"
                 f"valid={m.get('valid_path')}  shortest={m.get('shortest_path')}  "
                 f"correct_len={m.get('correct_length')}  optimal={m.get('optimal_path')}")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_eval_example(example: EvalExample) -> "plt.Figure":
    return render_graph_path(example.graph, example.predicted_path, example.diameter, metrics=example.metrics)


def wandb_images_for_examples(examples: List[EvalExample]):
    """Build `wandb.Image` objects, captioned with the model's raw text
    output, for a list of EvalExample. Imports wandb lazily so this module
    (and its figures) can be unit-tested without wandb installed/configured."""
    import wandb
    images = []
    for ex in examples:
        fig = render_eval_example(ex)
        images.append(wandb.Image(fig, caption=ex.predicted_text or "(empty output)"))
        plt.close(fig)
    return images
