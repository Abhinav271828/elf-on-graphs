"""Graph <-> text serialization, the training/eval Dataset, and batch collation.

Serialization grammar (see tokenizer.py for the full rationale):

    condition:  nodes: <n0> <n1> ... edges: <u0> - <v0> <u1> - <v1> ... find diametric path
    target:     <p0> <p1> ... <pk>                        (+ EOS, added at encode time)

Both are encoded with the real pretrained T5 tokenizer (`add_special_tokens=
False`, matching ELF's own minimal data-prep recipe). Batching follows
ELF's `data_utils.py` pattern exactly: condition and target ids are
concatenated into one sequence, and three masks describe it:

  - `cond_seq_mask`:          1 at condition positions, 0 at target positions.
  - `attention_mask`:         1 at any valid (non-pad) position -- this is
                               the loss mask, and also what the ELF backbone
                               attends over.
  - `encoder_attention_mask`: the *self*-attention mask fed to the frozen T5
                               encoder over the whole (condition + target)
                               sequence: condition tokens attend only to
                               condition tokens, target tokens attend to
                               everything valid.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import Config
from .graphgen import DiametricExample, Graph
from .tokenizer import EDGE_SEP, EDGES_HDR, NODES_HDR, PROMPT, get_pad_id


def serialize_condition_text(graph: Graph) -> str:
    parts = [NODES_HDR, *[str(n) for n in graph.nodes], EDGES_HDR]
    for u, v in graph.edges:
        parts.extend([str(u), EDGE_SEP, str(v)])
    parts.append(PROMPT)
    return " ".join(parts)


def serialize_target_text(path: Sequence[int]) -> str:
    return " ".join(str(n) for n in path)


def parse_path_text(text: str) -> Optional[List[int]]:
    """Parse decoded target text back into a node-id path.

    Returns None if any whitespace-separated piece is not a bare node-id
    (a stray word or punctuation in the decoded body is a formatting error,
    not something to silently filter out -- callers treat None as
    "invalid"). Empty text also parses to None (no path at all).
    """
    pieces = text.split()
    if not pieces:
        return None
    path: List[int] = []
    for p in pieces:
        if not p.isdigit():
            return None
        path.append(int(p))
    return path


def build_example(example: DiametricExample, tokenizer) -> Dict:
    cond_text = serialize_condition_text(example.graph)
    target_text = serialize_target_text(example.path)
    condition_ids = tokenizer(cond_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {
        "condition_input_ids": condition_ids,
        "input_ids": target_ids,
        "input": cond_text,
        "target": target_text,
        "diameter": example.diameter,
        "nodes": list(example.graph.nodes),
        "edges": [list(e) for e in example.graph.edges],
        "source": example.source,
        "path_target": example.target,
    }


class PathDataset(Dataset):
    """Wraps a list of raw JSON example dicts (from data_cache.py) and
    tokenizes on the fly with the real T5 tokenizer.
    """

    def __init__(self, raw_examples: List[Dict], tokenizer):
        self.raw_examples = raw_examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.raw_examples)

    def __getitem__(self, idx: int) -> Dict:
        raw = self.raw_examples[idx]
        graph = Graph(nodes=tuple(raw["nodes"]), edges=tuple(tuple(e) for e in raw["edges"]))
        example = DiametricExample(
            graph=graph, source=raw["source"], target=raw["target"],
            path=tuple(raw["path"]), diameter=raw["diameter"],
        )
        item = build_example(example, self.tokenizer)
        item["index"] = idx
        return item


def load_examples_jsonl(path: str) -> List[Dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_self_attn_cond_masks(is_cond: np.ndarray, is_valid: np.ndarray):
    """Port of ELF's `encoder_utils.build_self_attn_cond_masks`."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(np.float32)
    attention_mask = is_valid.astype(np.float32)
    cond_seq_mask = is_cond.astype(np.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask


def _pad_and_truncate(ids_list, target_len: int, pad_id: int):
    padded, lengths = [], []
    for ids in ids_list:
        ids = np.asarray(ids, dtype=np.int64)
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_id, dtype=np.int64)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def make_collate_fn(config: Config, pad_id: int):
    max_length = config.max_length
    max_input_length = config.max_input_length

    def collate_fn(batch: List[Dict]) -> Dict:
        seq_list, cond_lens = [], []
        for item in batch:
            cond = np.asarray(item["condition_input_ids"], dtype=np.int64)[:max_input_length]
            tgt = np.asarray(item["input_ids"], dtype=np.int64)
            seq_list.append(np.concatenate([cond, tgt]))
            cond_lens.append(len(cond))
        cond_lens = np.array(cond_lens)

        ids, total_lens = _pad_and_truncate(seq_list, max_length, pad_id)
        pos = np.arange(max_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, cond_mask = build_self_attn_cond_masks(is_cond, is_valid)

        result = {
            "input_ids": torch.from_numpy(ids),
            "encoder_attention_mask": torch.from_numpy(encoder_attn),
            "attention_mask": torch.from_numpy(attn),
            "cond_seq_mask": torch.from_numpy(cond_mask),
            "cond_len": torch.from_numpy(cond_lens),
        }
        for key in ("index", "input", "target", "diameter"):
            if key in batch[0]:
                result[key] = [item[key] for item in batch]
        return result

    return collate_fn


def get_dataloader(dataset: Dataset, config: Config, tokenizer,
                    batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    pad_id = get_pad_id(tokenizer, config.pad_token)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=config.num_workers, collate_fn=make_collate_fn(config, pad_id),
    )
