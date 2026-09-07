"""The task's serialization grammar, and the (real, pretrained) T5 tokenizer.

Earlier this module defined a small closed-vocabulary whitespace tokenizer,
used together with a T5 encoder trained from scratch on this project's own
data. Both are gone: the encoder is now a real pretrained T5
(`transformers.T5EncoderModel.from_pretrained`, see `t5_encoder.py`), so
tokenization must use *its* vocabulary -- `transformers.T5TokenizerFast` --
not a bespoke one. This module now just wraps that loader and documents the
plain-text grammar every serialized example follows.

Grammar (lowercase, so common words tokenize as single pieces under T5's
vocabulary -- e.g. "▁edges", "▁find" -- rather than splitting; verified with
`scripts/inspect_t5_tokenization.py`, see `data_sample.txt`):

    condition:  nodes: <n0> <n1> ... edges: <u0> - <v0> <u1> - <v1> ... find diametric path
    target:     <p0> <p1> ... <pk>

Every node id is single-space-delimited from its neighbors on both sides
(as its own list entry in the node list, and as the two space-separated
operands of `-` in each edge) so that under T5's SentencePiece tokenizer --
which treats leading whitespace as part of a token (`"▁12"` for " 12") --
no two node ids can run together into one token or bleed into an adjacent
one: `tok("11 12")` tokenizes to `["▁11", "▁12"]`, two clean tokens, never
`["▁1112"]` or similar. This is what "keeps node IDs separate" means
concretely, and it's why the edge separator is written `u - v` (spaced) and
not `u-v`.
"""

from __future__ import annotations

NODES_HDR = "nodes:"
EDGES_HDR = "edges:"
EDGE_SEP = "-"
PROMPT = "find diametric path"


def load_tokenizer(model_name: str):
    """Load the pretrained tokenizer that matches the frozen T5 encoder."""
    from transformers import T5TokenizerFast
    return T5TokenizerFast.from_pretrained(model_name)


def get_pad_id(tokenizer, pad_token: str = "pad") -> int:
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id
