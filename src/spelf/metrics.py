"""The four per-example eval metrics, plus the generation + scoring loop that
runs the trained ELF model over a data split.

Metrics (each a boolean per example; the task asks for all four, logged as
their mean rate over the eval split):

  1. valid_path      -- the decoded output is an actual path in the graph:
                         every node exists, consecutive nodes are connected
                         by an edge, and no node repeats.
  2. shortest_path    -- the output is *a* shortest path between its own two
                         endpoints (requires valid_path; length equals the
                         BFS distance between predicted_path[0] and [-1]).
  3. correct_length   -- the output's length (edge count) equals the graph's
                         diameter. This is a pure length check, independent
                         of validity -- a wrong-but-right-length output still
                         scores here, which is deliberately informative
                         (distinguishes "got the length right" from "got a
                         real path at all").
  4. optimal_path     -- valid_path AND shortest_path AND correct_length:
                         the output is a genuine diametric path of the graph
                         (any one of the graph's possibly-several diametric
                         paths, not necessarily the specific one stored as
                         ground truth).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .common import Config
from .dataset import PathDataset, get_dataloader, parse_path_text
from .graphgen import Graph, bfs_distances
from .sampling import decode_batch, generate_samples_single_batch, get_sampling_steps, mask_after_eos, shift_left
from .t5_encoder import encode_text
from .tokenizer import get_pad_id
from .train_state import unwrap_model


def _is_valid_path(graph: Graph, path: Optional[List[int]]) -> bool:
    if not path or len(path) < 2:
        return False
    node_set = set(graph.nodes)
    if any(n not in node_set for n in path):
        return False
    if len(set(path)) != len(path):
        return False
    adj = graph.adjacency()
    return all(v in adj[u] for u, v in zip(path, path[1:]))


def evaluate_path(graph: Graph, predicted_path: Optional[List[int]], diameter: int) -> Dict[str, bool]:
    valid = _is_valid_path(graph, predicted_path)
    path_len = (len(predicted_path) - 1) if predicted_path else None
    correct_length = path_len is not None and path_len == diameter

    is_shortest = False
    if valid:
        s, t = predicted_path[0], predicted_path[-1]
        dist = bfs_distances(graph.adjacency(), s).get(t)
        is_shortest = dist is not None and path_len == dist

    return {
        "valid_path": valid,
        "shortest_path": is_shortest,
        "correct_length": correct_length,
        "optimal_path": valid and is_shortest and correct_length,
    }


@dataclass
class EvalExample:
    input_text: str
    target_text: str
    predicted_text: str
    graph: Graph
    diameter: int
    predicted_path: Optional[List[int]]
    metrics: Dict[str, bool]


@torch.no_grad()
def run_eval(model: torch.nn.Module, encoder: torch.nn.Module, tokenizer,
             raw_examples: List[dict], config: Config, device: torch.device,
             generator: torch.Generator, num_examples: int, batch_size: int) -> List[EvalExample]:
    """Generate + score up to `num_examples` examples from `raw_examples`.

    A model in eval mode is expected (callers typically pass an EMA-weight
    copy; see checkpoint.py / scripts/eval.py).
    """
    if num_examples > 0:
        raw_examples = raw_examples[:num_examples]
    dataset = PathDataset(raw_examples, tokenizer)
    pad_id = get_pad_id(tokenizer, config.pad_token)
    dataloader = get_dataloader(dataset, config, tokenizer, batch_size=batch_size, shuffle=False, drop_last=False)

    d_model = unwrap_model(model).text_encoder_dim
    dtype = next(model.parameters()).dtype
    eos_id = tokenizer.eos_token_id

    results: List[EvalExample] = []
    for batch in dataloader:
        bsz = batch["input_ids"].shape[0]
        input_ids = batch["input_ids"].to(device).long()
        encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32)
        cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32)
        cond_len = batch["cond_len"].to(device)

        cond_seq = encode_text(input_ids=input_ids, attention_mask=encoder_attention_mask,
                                encoder=encoder, latent_mean=config.latent_mean, latent_std=config.latent_std).to(dtype)

        t_steps = get_sampling_steps(config.num_sampling_steps, config.sampling_time_schedule,
                                      config.denoiser_p_mean, config.denoiser_p_std, device=device, dtype=dtype)
        z = torch.randn((bsz, config.max_length, d_model), generator=generator, dtype=dtype).to(device) * config.denoiser_noise_scale

        latent = generate_samples_single_batch(
            model=model, generator=generator, z=z, t_steps=t_steps,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask, config=config,
            cfg_scale=config.cfg_scale, self_cond_cfg_scale=config.self_cond_cfg_scale,
            sampling_method=config.sampling_method, sde_gamma=config.sde_gamma,
        )
        predicted_ids = decode_batch(latent, model, t_steps[-1].item(), config, config.self_cond_cfg_scale)
        predicted_ids = shift_left(predicted_ids, cond_len, pad_value=pad_id)
        gen_length = config.max_length - config.max_input_length
        predicted_ids = predicted_ids[:, :gen_length]
        predicted_ids = mask_after_eos(predicted_ids, eos_id=eos_id, pad_id=pad_id)

        for i in range(bsz):
            idx = batch["index"][i]
            raw = raw_examples[idx]
            graph = Graph(nodes=tuple(raw["nodes"]), edges=tuple(tuple(e) for e in raw["edges"]))
            pred_text = tokenizer.decode(predicted_ids[i].detach().cpu().tolist(), skip_special_tokens=True)
            predicted_path = parse_path_text(pred_text)
            metrics = evaluate_path(graph, predicted_path, raw["diameter"])
            results.append(EvalExample(
                input_text=batch["input"][i], target_text=batch["target"][i],
                predicted_text=pred_text, graph=graph, diameter=raw["diameter"],
                predicted_path=predicted_path, metrics=metrics,
            ))
    return results


def aggregate_metrics(results: List[EvalExample]) -> Dict[str, float]:
    if not results:
        return {"valid_path_rate": 0.0, "shortest_path_rate": 0.0, "correct_length_rate": 0.0,
                "optimal_path_rate": 0.0, "num_examples": 0}
    n = len(results)
    return {
        "valid_path_rate": sum(r.metrics["valid_path"] for r in results) / n,
        "shortest_path_rate": sum(r.metrics["shortest_path"] for r in results) / n,
        "correct_length_rate": sum(r.metrics["correct_length"] for r in results) / n,
        "optimal_path_rate": sum(r.metrics["optimal_path"] for r in results) / n,
        "num_examples": n,
    }


def pick_viz_examples(results: List[EvalExample], k: int, seed: int = 0) -> List[EvalExample]:
    if len(results) <= k:
        return list(results)
    rng = random.Random(seed)
    return rng.sample(results, k)
