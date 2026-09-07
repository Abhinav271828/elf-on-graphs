import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest

from spelf.common import Config
from spelf.t5_encoder import build_pretrained_encoder
from spelf.tokenizer import load_tokenizer


@pytest.fixture
def tiny_config() -> Config:
    """A Config with small dims everywhere (except the encoder, which is
    real pretrained t5-small -- see `tokenizer`/`encoder` fixtures below),
    for fast unit/integration tests."""
    return Config(
        id_min_nodes=4, id_max_nodes=6, ood_min_nodes=7, ood_max_nodes=8,
        max_input_length=96, max_length=112,
        bottleneck_dim=8, num_time_tokens=2, num_self_cond_cfg_tokens=2, num_model_mode_tokens=2,
        batch_size=4, eval_num_examples=8, eval_num_viz=2,
        use_wandb=False,
    )


@pytest.fixture(scope="session")
def tokenizer():
    """The real T5 tokenizer -- session-scoped since it's the same object
    for every test (downloaded/cached once by `transformers`)."""
    return load_tokenizer("t5-small")


@pytest.fixture(scope="session")
def encoder():
    """The real, frozen, pretrained T5 encoder (t5-small, d_model=512) --
    session-scoped so its weights are loaded only once for the whole suite.
    """
    return build_pretrained_encoder("t5-small")
