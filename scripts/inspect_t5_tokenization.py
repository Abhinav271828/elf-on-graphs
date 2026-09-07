#!/usr/bin/env python3
"""Show exactly how T5's real tokenizer segments a batch of real graph/path examples
(see t5_encoder.graph_to_text / path_to_text) -- the text a --encoder_kind=t5 run
actually feeds T5, and (for the path) the text whose T5 tokenization the DLM literally
diffuses into (see t5_encoder.T5DiffusionEncoder). Token boundaries are marked with `|`
so a human can see the segmentation directly, rather than trusting the module
docstrings' claim that "every node-id number gets its own token span, never merged with
a different node id" on faith.

That claim is also checked automatically for every sample (not just eyeballed): each
node-id number's character span in the source text is compared against every token's
own character span (via the tokenizer's offset mapping); a single token whose span
covers characters from two different numbers is a hard failure. The script exits
non-zero if any sample fails this check.

Usage:
  python scripts/inspect_t5_tokenization.py --data_dir data/ --n_samples 20
"""
import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from spelf import t5_encoder, tokenizer as tok


def _number_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\d+", text)]


def _tokens_fuse_two_numbers(offsets: list[tuple[int, int]], number_spans: list[tuple[int, int]]) -> bool:
    """True if any single token's character span overlaps two DIFFERENT number spans
    -- i.e. tokenization merged two distinct node-id numbers into one token, exactly
    the failure mode graph_to_text/path_to_text's spacing is meant to prevent."""
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            continue  # zero-width special tokens (e.g. </s>) can't overlap anything
        overlapping = [i for i, (s, e) in enumerate(number_spans) if tok_start < e and tok_end > s]
        if len(overlapping) > 1:
            return True
    return False


def _render(pieces: list[str]) -> str:
    return " | ".join(p.replace("\n", "\\n") for p in pieces)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--split", type=str, default="val_id", choices=["train", "val_id", "val_ood"])
    p.add_argument("--t5_model_name", type=str, default="t5-small")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="defaults to {data_dir}/t5_tokenization_samples.txt")
    args = p.parse_args()

    from transformers import AutoTokenizer
    t5_tokenizer = AutoTokenizer.from_pretrained(args.t5_model_name)
    assert t5_tokenizer.is_fast, (
        f"'{args.t5_model_name}' did not load a fast tokenizer -- this script needs "
        f"return_offsets_mapping support for its automated boundary check"
    )

    data_dir = Path(args.data_dir)
    d = torch.load(data_dir / f"{args.split}.pt")
    n = d["input_ids"].shape[0]
    idx = random.Random(args.seed).sample(range(n), min(args.n_samples, n))

    lines = [
        f"T5 ({args.t5_model_name}) tokenization of {len(idx)} sampled examples from "
        f"{args.split}.pt (seed={args.seed})",
        "Token boundaries marked with '|'. Each of the two texts per example is also "
        "checked automatically: no single token's character span may cover two "
        "different node-id numbers (see t5_encoder.graph_to_text/path_to_text's "
        "spacing discipline).",
        "",
    ]
    n_checked, n_failed = 0, 0

    for i in idx:
        input_row = [t for t, keep in zip(d["input_ids"][i].tolist(), d["input_mask"][i].tolist()) if keep]
        decoded_input = tok.decode_input(input_row)
        assert decoded_input is not None, "ground-truth input should always be well-formed"
        graph_text = t5_encoder.graph_to_text(decoded_input)

        path = tok.decode_target(d["target_ids"][i].tolist())
        assert path is not None, "ground-truth target should always be well-formed"
        path_text = t5_encoder.path_to_text(path)

        lines.append(f"=== example {i} (n_nodes={decoded_input['n']}, path_len={len(path)}) ===")
        for label, text in [("graph (input) ", graph_text), ("path (target)", path_text)]:
            enc = t5_tokenizer(text, return_offsets_mapping=True)
            pieces = t5_tokenizer.convert_ids_to_tokens(enc["input_ids"])
            fused = _tokens_fuse_two_numbers(enc["offset_mapping"], _number_spans(text))
            n_checked += 1
            n_failed += int(fused)
            status = "FAIL -- a token spans two different node-id numbers" if fused else "PASS"
            lines.append(f"  [{label}] {status}  ({len(pieces)} tokens)")
            lines.append(f"    text:   {text}")
            lines.append(f"    tokens: {_render(pieces)}")
        lines.append("")

    lines.append(
        f"== summary: {n_checked - n_failed}/{n_checked} checks passed "
        f"(no node-id number ever shares a token with a different one) =="
    )

    out_path = Path(args.out) if args.out else data_dir / "t5_tokenization_samples.txt"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print(f"{n_checked - n_failed}/{n_checked} checks passed")
    if n_failed:
        raise SystemExit(f"{n_failed} tokenization check(s) failed -- see {out_path}")


if __name__ == "__main__":
    main()
