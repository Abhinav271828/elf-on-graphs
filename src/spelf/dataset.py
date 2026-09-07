"""Thin dataset wrappers around the tensors cached by data_cache.generate_all."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, default_collate


def load_meta(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json") as f:
        return json.load(f)


class PathDataset(Dataset):
    """Labeled path-generation examples (train.pt / val_id.pt / val_ood.pt)."""

    def __init__(self, path: str | Path):
        d = torch.load(path)
        self.input_ids = d["input_ids"]
        self.input_mask = d["input_mask"]
        self.target_ids = d["target_ids"]
        self.n_nodes = d["n_nodes"]
        self.graph_id = d["graph_id"]

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "input_mask": self.input_mask[idx],
            "target_ids": self.target_ids[idx],
            "n_nodes": self.n_nodes[idx],
            "graph_id": self.graph_id[idx],
        }


class EncoderPretrainDataset(Dataset):
    """Input-only (no path) examples for graph-encoder MLM pretraining: the training
    split's 6-10 node inputs concatenated with encoder_pretrain_extra.pt's 11-14 node
    inputs, so the encoder is exposed to both node-count ranges without train.pt
    duplicating any data on disk."""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        train = torch.load(data_dir / "train.pt")
        extra = torch.load(data_dir / "encoder_pretrain_extra.pt")
        self.input_ids = torch.cat([train["input_ids"], extra["input_ids"]], dim=0)
        self.input_mask = torch.cat([train["input_mask"], extra["input_mask"]], dim=0)
        self.n_nodes = torch.cat([train["n_nodes"], extra["n_nodes"]], dim=0)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "input_mask": self.input_mask[idx],
            "n_nodes": self.n_nodes[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Examples are already fixed-shape and pre-padded, so this is just the default
    per-key stacking collate -- named explicitly for clarity at DataLoader call sites."""
    return default_collate(batch)
