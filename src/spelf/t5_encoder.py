"""Optional pretrained-T5 conditioning encoder, selected via --encoder_kind t5 in
train_dlm.py / train_arlm.py / eval_only.py. This is closer to the ELF paper's actual
setup (frozen pretrained T5 text encoder) than the project's default `GraphEncoder`
(which is trained from scratch on this task's own custom node-id vocab via MLM, see
encoder.py) -- it lets us ask whether a strong pretrained text encoder is a better (or
worse) conditioning source than a small from-scratch graph encoder, holding the DLM/ARLM
decoder architecture fixed.

Unlike GraphEncoder, whose output sequence is length-aligned with its own input_ids
(so callers can reuse input_mask as context_mask), this encoder serializes the graph to
a natural-language description and retokenizes it with T5's own subword tokenizer, so
its output sequence length/mask are unrelated to input_mask. See
common.encode_context for the uniform (context, context_mask) dispatch every call site
uses instead of calling either encoder directly.

T5's own embedding space is pretrained on natural-language subwords and has nothing to
do with this project's 22-token graph vocab, so -- unlike GraphEncoder, whose embedding
table *is* the pretrained conditioning signal and is shared verbatim into the decoder --
this encoder owns a fresh SharedEmbedding over the graph vocab that trains from scratch
alongside the decoder, plus a trainable linear projection from T5's hidden size down to
this project's d_model. Only those two pieces are trainable; the T5 stack itself stays
frozen throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import tokenizer as tok
from .modules import SharedEmbedding


def graph_to_text(decoded: dict) -> str:
    """Deterministic text serialization of a decoded input (see tokenizer.decode_input)
    for T5 to encode. Node labels are written as literal integers -- T5's subword
    tokenizer splits multi-digit numbers into digit/subword pieces on its own, which is
    fine since we never re-parse this text, only encode it."""
    edges = ", ".join(f"{u}-{v}" for u, v in decoded["edges"])
    nodes = " ".join(str(x) for x in decoded["node_list"])
    return (
        f"Graph with nodes: {nodes}. Edges: {edges}. "
        f"Find a path whose length equals the graph's diameter "
        f"(the longest shortest path between any two nodes)."
    )


class T5GraphEncoder(nn.Module):
    context_requires_grad = True  # see common.encode_context

    def __init__(self, model_name: str = "t5-small", d_model: int = 128, max_text_len: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.model_name = model_name
        self.max_text_len = max_text_len
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5 = T5EncoderModel.from_pretrained(model_name)

        self.d_model = d_model
        self.proj = nn.Linear(self.t5.config.d_model, d_model)
        self.embedding = SharedEmbedding(vocab_size=tok.VOCAB_SIZE, d_model=d_model)

        self.freeze()

    def freeze(self) -> None:
        """Freeze the pretrained T5 stack only; `proj` and `embedding` stay trainable
        (see module docstring). Kept as an explicit method, mirroring
        GraphEncoder.freeze(), so load_t5_encoder reads the same as load_frozen_encoder."""
        for p in self.t5.parameters():
            p.requires_grad = False
        self.t5.eval()

    def train(self, mode: bool = True):
        """Override so an accidental `t5_graph_encoder.train()` (e.g. via a parent
        module's recursive .train()) can never put the frozen T5 stack into train mode
        (dropout etc.) -- proj/embedding still switch normally since they're plain
        submodules without their own train()/eval() semantics beyond dropout, which
        neither has."""
        super().train(mode)
        self.t5.eval()
        return self

    def trainable_state_dict(self) -> dict:
        """The only weights this encoder needs checkpointed -- the frozen T5 stack is
        reproducible from `model_name` alone, so re-saving its ~tens-of-millions of
        frozen params on every checkpoint would be pure waste."""
        return {"proj": self.proj.state_dict(), "embedding": self.embedding.state_dict()}

    def load_trainable_state_dict(self, state: dict) -> None:
        self.proj.load_state_dict(state["proj"])
        self.embedding.load_state_dict(state["embedding"])

    def _texts_from_batch(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> list[str]:
        texts = []
        for row, m in zip(input_ids.cpu().tolist(), input_mask.cpu().tolist()):
            real = [t for t, keep in zip(row, m) if keep]
            decoded = tok.decode_input(real)
            assert decoded is not None, "T5GraphEncoder requires well-formed (unmasked) inputs"
            texts.append(graph_to_text(decoded))
        return texts

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        texts = self._texts_from_batch(input_ids, input_mask)
        enc = self.t5_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_len,
        ).to(input_ids.device)
        with torch.no_grad():
            hidden = self.t5(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        context = self.proj(hidden)  # trainable projection -> gradient flows here even though `hidden` came from no_grad
        context_mask = enc["attention_mask"].bool()
        return context, context_mask


def load_t5_encoder(model_name: str, d_model: int, device) -> T5GraphEncoder:
    encoder = T5GraphEncoder(model_name=model_name, d_model=d_model).to(device)
    encoder.eval()
    return encoder
