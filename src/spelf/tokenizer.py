"""Tokenizer for the graph-description / diametric-path-generation task.

Vocab layout (21 tokens):
  0 <PAD>  1 <G>  2 <E>  3 <N>  4 <P>  5 <EOS>  6 <MASK>
  7 .. 7+MAX_NODES-1   node ids 0 .. MAX_NODES-1

Input sequence:  <G> u1 v1 <E> u2 v2 <E> ... <N> n0 n1 ... n(k-1)
Target sequence: <P> p0 p1 ... pk <EOS> <PAD> ... <PAD>   (padded to TARGET_LENGTH)

The input names only the graph -- there is no query. The task is to find *some* path
whose length (in edges) equals the graph's diameter (see graphgen.diametric_pairs); the
path's own endpoints are whatever the model chooses them to be.

`decode_input`/`decode_target` are the single source of truth for parsing token ids
back into graph/path objects -- used identically for ground truth and for (possibly
malformed) model generations.
"""
from __future__ import annotations

from typing import Optional, Sequence

MAX_NODES = 14

PAD, G, E, N, P, EOS, MASK = range(7)
NODE_OFFSET = 7
VOCAB_SIZE = NODE_OFFSET + MAX_NODES

SPECIAL_NAMES = {
    PAD: "<PAD>",
    G: "<G>",
    E: "<E>",
    N: "<N>",
    P: "<P>",
    EOS: "<EOS>",
    MASK: "<MASK>",
}

# Global fixed target length (L_TGT), used uniformly across ID (6-10 node) and
# OOD (11-14 node) graphs so DLM tensors have one constant shape. A diametric path can
# have at most MAX_NODES nodes (it's a simple path), so this still fits every case.
TARGET_LENGTH = MAX_NODES + 2


def node_tok(node_id: int) -> int:
    if not (0 <= node_id < MAX_NODES):
        raise ValueError(f"node id {node_id} out of range [0, {MAX_NODES})")
    return NODE_OFFSET + node_id


def is_node_tok(tok: int) -> bool:
    return NODE_OFFSET <= tok < NODE_OFFSET + MAX_NODES


def tok_to_node(tok: int) -> int:
    assert is_node_tok(tok)
    return tok - NODE_OFFSET


def token_name(tok: int) -> str:
    if tok in SPECIAL_NAMES:
        return SPECIAL_NAMES[tok]
    if is_node_tok(tok):
        return str(tok_to_node(tok))
    return f"<UNK:{tok}>"


def input_length(n: int, m: int) -> int:
    """Exact unpadded length of an encoded input sequence for n nodes, m edges.
    L_in(n, m) = 1 (<G>) + 3m (u,v,<E> per edge) + 1 (<N>) + n (node list)
    """
    return n + 3 * m + 2


def encode_input(
    n: int,
    edges: Sequence[tuple[int, int]],
    node_list: Optional[Sequence[int]] = None,
) -> list[int]:
    """Encode <G> edges... <N> node_list. Returns an unpadded list of ints. `edges`
    order is used as given -- callers wanting order-invariance robustness should
    shuffle before calling this."""
    if node_list is None:
        node_list = list(range(n))
    ids = [G]
    for u, v in edges:
        ids += [node_tok(u), node_tok(v), E]
    ids += [N]
    ids += [node_tok(x) for x in node_list]
    assert len(ids) == input_length(n, len(edges))
    return ids


def pad_input(ids: Sequence[int], length: int) -> tuple[list[int], list[bool]]:
    assert len(ids) <= length, f"input length {len(ids)} exceeds pad target {length}"
    mask = [True] * len(ids) + [False] * (length - len(ids))
    padded = list(ids) + [PAD] * (length - len(ids))
    return padded, mask


def decode_input(ids: Sequence[int]) -> Optional[dict]:
    """Parse an (possibly padded) encoded input sequence back to a graph.
    Returns None if malformed."""
    ids = list(ids)
    if not ids or ids[0] != G:
        return None
    i = 1
    edges: list[tuple[int, int]] = []
    while i < len(ids) and ids[i] != N:
        if i + 2 >= len(ids) or not is_node_tok(ids[i]) or not is_node_tok(ids[i + 1]) or ids[i + 2] != E:
            return None
        edges.append((tok_to_node(ids[i]), tok_to_node(ids[i + 1])))
        i += 3
    if i >= len(ids) or ids[i] != N:
        return None
    i += 1
    node_list: list[int] = []
    while i < len(ids) and is_node_tok(ids[i]):
        node_list.append(tok_to_node(ids[i]))
        i += 1
    if any(t != PAD for t in ids[i:]):
        return None
    return {"n": len(node_list), "edges": edges, "node_list": node_list}


def encode_target(path: Sequence[int], length: int = TARGET_LENGTH) -> list[int]:
    """<P> path... <EOS> <PAD>...<PAD>, padded to `length`."""
    ids = [P] + [node_tok(x) for x in path] + [EOS]
    if len(ids) > length:
        raise ValueError(f"path of length {len(path)} does not fit in target length {length}")
    ids += [PAD] * (length - len(ids))
    return ids


def decode_target(ids: Sequence[int]) -> Optional[list[int]]:
    """Parse a (possibly model-generated) target sequence back to a node-id path.
    Must be <P>, zero-or-more node tokens, then <EOS> -- content after <EOS> is
    ignored, not validated. Missing/duplicated <P>/<EOS>, or non-node tokens in the
    path, still yield None. This is the single source of truth for both ground truth
    and raw model output.

    Trailing content is deliberately unchecked (not required to be <PAD>): dlm.loss
    excludes pad_id positions from both its denoising and decode losses (matching the
    canonical ELF implementation's own masking, verified directly against its source),
    so nothing trains the DLM on what belongs after <EOS> -- a generation is meant to
    be read by finding the first <EOS>, not by validating its tail. Requiring literal
    <PAD> there would make every DLM generation with an untrained (and therefore
    effectively arbitrary) tail spuriously invalid, regardless of whether its real
    content was correct."""
    ids = list(ids)
    if not ids or ids[0] != P:
        return None
    i = 1
    path: list[int] = []
    while i < len(ids) and is_node_tok(ids[i]):
        path.append(tok_to_node(ids[i]))
        i += 1
    if i >= len(ids) or ids[i] != EOS:
        return None
    return path
