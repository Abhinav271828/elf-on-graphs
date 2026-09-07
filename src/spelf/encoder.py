"""Bidirectional graph encoder + its self-supervised (masked node-token) pretraining
objective. No path-answer labels are ever used here -- only the <G>...<N>
input sequence."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as tok
from .modules import EncoderLayer, LearnedPosEnc, SharedEmbedding


class GraphEncoder(nn.Module):
    def __init__(self, l_in: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 8,
                 d_mlp: int = 512, dropout: float = 0.1, embedding: SharedEmbedding | None = None):
        super().__init__()
        self.l_in = l_in
        self.d_model = d_model
        self.embedding = embedding if embedding is not None else SharedEmbedding(d_model=d_model)
        self.pos_enc = LearnedPosEnc(l_in, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_mlp, dropout) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, input_mask)
        return self.out_ln(x)

    def freeze(self) -> None:
        """Freeze every parameter except the embedding table's <P>/<EOS> rows (which
        this encoder's own input never contains -- see SharedEmbedding docstring).
        embedding.weight itself keeps requires_grad=True (needed so <P>/<EOS> can still
        train when this same embedding module is shared into a DLM/ARLM decoder), with
        a gradient hook zeroing every other row -- see SharedEmbedding.freeze_pretrained_rows.
        Callers should still run this encoder's own forward pass under torch.no_grad()
        during downstream training, since none of its parameters should accumulate
        gradient through that path."""
        for name, p in self.named_parameters():
            if name.startswith("embedding."):
                continue
            p.requires_grad = False
        self.embedding.freeze_pretrained_rows()
        self.eval()


# Structural marker tokens are never masked -- the useful self-supervised signal here
# is relational/structural understanding of node identity, not of the fixed grammar.
MASKABLE_TOKENS_ONLY_NODES = True


def mlm_forward(model: GraphEncoder, input_ids: torch.Tensor, input_mask: torch.Tensor,
                 mlm_prob: float = 0.15) -> dict:
    """One masked-language-model training step's worth of loss/accuracy. Masks a random
    ~mlm_prob fraction of node-id token positions (edge endpoints, <N> list), replaces
    them with <MASK>, and predicts the original id via the tied unembedding at those
    positions only."""
    is_node = (input_ids >= tok.NODE_OFFSET) & (input_ids < tok.NODE_OFFSET + tok.MAX_NODES)
    eligible = is_node & input_mask
    mask_positions = eligible & (torch.rand_like(input_ids, dtype=torch.float) < mlm_prob)

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_positions] = tok.MASK

    hidden = model(masked_input_ids, input_mask)
    logits = model.embedding.unembed(hidden)  # [B, L, vocab]

    labels = torch.full_like(input_ids, fill_value=-100)
    labels[mask_positions] = input_ids[mask_positions]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )
    with torch.no_grad():
        n_masked = mask_positions.sum().clamp(min=1)
        preds = logits.argmax(dim=-1)
        correct = ((preds == input_ids) & mask_positions).sum()
        accuracy = (correct.float() / n_masked.float()).item()
    return {"loss": loss, "accuracy": accuracy, "n_masked": int(mask_positions.sum().item())}


def load_frozen_encoder(checkpoint_path: str, device) -> GraphEncoder:
    """Load a GraphEncoder checkpoint (as saved by scripts/pretrain_encoder.py) and
    freeze it, ready to condition a DLM/ARLM decoder. Shared by train_dlm.py,
    train_arlm.py, and eval_only.py so there's one place that knows the checkpoint
    schema."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    encoder = GraphEncoder(
        l_in=cfg["l_in"], d_model=cfg.get("d_model", 128), n_layers=cfg.get("n_layers", 2),
        n_heads=cfg.get("n_heads", 8), d_mlp=cfg.get("d_mlp", 512), dropout=cfg.get("dropout", 0.1),
    ).to(device)
    encoder.load_state_dict(state["model_state"])
    encoder.freeze()
    return encoder
