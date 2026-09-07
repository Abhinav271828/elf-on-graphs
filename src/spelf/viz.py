"""Human-readable graph+path visualization: ground-truth vs. generated diametric path,
for wandb image logging. Layout is seeded from each graph's own content (see
_content_seed), not its position in the logged set, so a given graph always lays out
the same way -- but *which* graphs get logged each eval round is randomized
(metrics.sample_examples) so a few early-solved examples don't dominate every image
forever, hiding whether the model is still improving elsewhere in the val set."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from . import tokenizer as tok


def _draw_panel(ax, G: nx.Graph, pos: dict, path: list[int] | None,
                 path_color: str = "tab:blue", title: str = "", title_color: str = "black") -> None:
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightgray", node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    if path:
        # A generated path can reference a node-id token that's valid in the vocab but
        # isn't actually one of *this* graph's nodes (node ids are drawn from a pool
        # wider than any single graph's own node count -- see graphgen.sample_graph's
        # label_pool_size). metrics.evaluate_generation already excludes such paths
        # from `valid`, but the raw (possibly-invalid) path is still what gets drawn
        # here -- so only draw the nodes/edges that actually exist in G, rather than
        # indexing `pos` (which only has entries for G's real nodes) for all of them.
        real_nodes = [p for p in path if p in G]
        real_edges = [(a, b) for a, b in zip(path, path[1:]) if a in G and b in G and G.has_edge(a, b)]
        if real_edges:
            nx.draw_networkx_edges(G, pos, edgelist=real_edges, ax=ax, edge_color=path_color, width=3)
        if real_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=real_nodes, ax=ax, node_color=path_color, node_size=300)
            # Highlight the path's own two endpoints -- there's no fixed query
            # start/end for this task, so "start"/"end" here just mean the drawn
            # path's own ends (among its real, in-graph nodes).
            nx.draw_networkx_nodes(G, pos, nodelist=[real_nodes[0]], ax=ax, node_color="tab:green", node_size=400)
            if len(real_nodes) > 1:
                nx.draw_networkx_nodes(G, pos, nodelist=[real_nodes[-1]], ax=ax, node_color="tab:orange", node_size=400)
    ax.set_title(title, color=title_color)
    ax.axis("off")


def _content_seed(decoded_input: dict) -> int:
    """Deterministic spring_layout seed derived from the graph itself (not from its
    position in whatever list it's being plotted alongside), so the same graph always
    lays out the same way -- but which graph gets logged at wandb-image slot 0 can
    freely change from one eval round to the next (see log_examples_to_wandb) without
    that graph's layout jumping around if it happens to reappear. Python's hash() of a
    tuple of ints is unaffected by PYTHONHASHSEED (that randomization only applies to
    str/bytes), so this is stable across separate runs too, not just within one."""
    canon_edges = tuple(sorted(tuple(sorted(e)) for e in decoded_input["edges"]))
    key = (tuple(sorted(decoded_input["node_list"])), canon_edges)
    return hash(key) % (2**31)


def plot_example(example: dict, seed: int | None = None):
    """example: one dict as returned by metrics.evaluate_generation, extended with
    input_ids/gen_ids/target_ids (as metrics.run_eval produces). Returns a matplotlib
    Figure -- caller is responsible for plt.close(fig) after use (e.g. after handing it
    to wandb.Image). `seed` defaults to a hash of the graph's own content (see
    _content_seed); pass an explicit int to override."""
    decoded_input = example["decoded_input"]
    if seed is None:
        seed = _content_seed(decoded_input)
    G = nx.Graph()
    # Node labels are drawn from a pool wider than [0, n) (see graphgen.sample_graph),
    # so the real node set is node_list, not range(n) -- using range(n) here would add
    # spurious zero-edge placeholder nodes alongside the real ones, showing up as
    # phantom "unconnected" dots in the plot even though the actual graph is connected.
    G.add_nodes_from(decoded_input["node_list"])
    G.add_edges_from(decoded_input["edges"])
    pos = nx.spring_layout(G, seed=seed)

    gt_path = tok.decode_target(example["target_ids"])
    gen_path = example["gen_path"]
    valid = example["valid"]
    diameter = example["diameter"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw_panel(axes[0], G, pos, gt_path, path_color="tab:blue", title="Ground truth")

    if gen_path:
        color = "tab:blue" if valid else "tab:red"
        title = "Generated" if valid else "Generated -- INVALID PATH"
        _draw_panel(axes[1], G, pos, gen_path, path_color=color,
                    title=title, title_color=("black" if valid else "red"))
    else:
        _draw_panel(axes[1], G, pos, None, title="Generated -- UNPARSEABLE", title_color="red")
        raw = " ".join(tok.token_name(t) for t in example["gen_ids"])
        axes[1].text(0.5, -0.08, raw, transform=axes[1].transAxes, ha="center",
                     fontsize=6, color="red", wrap=True)

    fig.suptitle(
        f"diameter={diameter}   exact_match={example['exact_match']}   "
        f"valid={example['valid']}   shortest={example['shortest']}   "
        f"correct_length={example['correct_length']}   optimal={example['optimal']}"
    )
    fig.tight_layout()
    return fig


def log_examples_to_wandb(run, examples: list[dict], split_name: str, step: int) -> None:
    import wandb
    images = []
    for i, ex in enumerate(examples):
        fig = plot_example(ex)
        images.append(wandb.Image(fig, caption=f"{split_name}_{i}"))
        plt.close(fig)
    run.log({f"eval/{split_name}/examples": images}, step=step)
