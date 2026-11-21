import numpy as np
import pytest

from app.model import BLOCK_SIZE, KV, Model, ModelConfig, empty_kv


@pytest.fixture(scope="module")
def model():
    return Model(ModelConfig(vocab_size=32, d_model=16, n_heads=2, seed=3))


def test_block_size_is_four():
    assert BLOCK_SIZE == 4


def test_full_forward_logits_shape(model):
    tokens = np.arange(7, dtype=np.int64)
    logits, kv = model.forward(tokens)
    assert logits.shape == (7, model.cfg.vocab_size)
    assert len(kv) == 7
    assert list(kv.positions) == list(range(7))


def test_chunked_forward_matches_full_forward(model):
    tokens = np.array([1, 4, 2, 8, 5, 3, 9, 6, 7, 0], dtype=np.int64)
    full, _ = model.forward(tokens)

    cache = None
    collected = []
    for chunk in (tokens[0:1], tokens[1:4], tokens[4:5], tokens[5:8], tokens[8:]):
        logits, cache = model.forward(chunk, cache)
        collected.append(logits)
    chunked = np.concatenate(collected, axis=0)

    assert np.max(np.abs(chunked - full)) <= 1e-5


def test_returned_kv_is_whole_cache(model):
    tokens = np.array([3, 1, 4, 1, 5, 9], dtype=np.int64)
    _, first = model.forward(tokens[:2])
    _, second = model.forward(tokens[2:], first)
    assert len(first) == 2
    assert len(second) == len(tokens)
    assert list(second.positions) == list(range(len(tokens)))
    _, direct = model.forward(tokens)
    assert np.max(np.abs(second.k - direct.k)) <= 1e-12


def test_logits_depend_only_on_prefix(model):
    """A cached continuation at position 4 sees exactly positions 0..4."""
    a = np.array([2, 7, 1, 9, 4], dtype=np.int64)
    b = np.array([2, 7, 1, 9, 6], dtype=np.int64)
    full_a, _ = model.forward(a)
    full_b, _ = model.forward(b)
    assert np.max(np.abs(full_a[:4] - full_b[:4])) <= 1e-12
    assert np.max(np.abs(full_a[4] - full_b[4])) > 1e-3


def test_rejects_out_of_range_tokens(model):
    with pytest.raises(ValueError):
        model.forward(np.array([model.cfg.vocab_size], dtype=np.int64))
