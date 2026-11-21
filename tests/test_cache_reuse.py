"""Tests that the cache really is reused: only new tokens get fresh K/V."""

import numpy as np
import pytest

from app.model import BLOCK_SIZE, Model, ModelConfig
from app.store import BlockStore


class CountingModel(Model):
    """Model that records how many tokens it computes K/V for."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.kv_tokens = 0
        self.forward_calls = 0

    def _qkv(self, tokens, positions):
        self.kv_tokens += int(tokens.shape[0])
        return super()._qkv(tokens, positions)

    def forward(self, tokens, memory=None):
        self.forward_calls += 1
        return super().forward(tokens, memory)


@pytest.fixture
def model():
    return CountingModel(ModelConfig(vocab_size=32, d_model=16, n_heads=2, seed=7))


def test_only_new_tokens_compute_kv(model):
    store = BlockStore(model, max_blocks=64, block_size=BLOCK_SIZE)
    sid = store.create_session([1, 2, 3])[0]["session_id"]
    assert model.kv_tokens == 3

    model.kv_tokens = 0
    store.append(sid, [4, 5, 6, 7, 8])
    # Exactly the five appended tokens, not the eight-position prefix.
    assert model.kv_tokens == 5

    model.kv_tokens = 0
    store.append(sid, [9])
    assert model.kv_tokens == 1


def test_fork_then_append_computes_only_new_tokens(model):
    store = BlockStore(model, max_blocks=64, block_size=BLOCK_SIZE)
    sid = store.create_session(list(range(10)))[0]["session_id"]
    child = store.fork(sid, 6).session_id
    model.kv_tokens = 0
    store.append(child, [11, 12])
    assert model.kv_tokens == 2


def test_blocks_store_correct_absolute_positions(model):
    store = BlockStore(model, max_blocks=64, block_size=BLOCK_SIZE)
    sid = store.create_session([1, 2, 3, 4, 5, 6, 7])[0]["session_id"]
    store.append(sid, [8, 9, 10])
    store.fork(sid, 5)
    with store._lock:
        for sess in store._sessions.values():
            expected_next = 0
            for ref in sess.blocks:
                block = store._blocks[ref.block_id]
                assert list(block.positions[: ref.used]) == list(
                    range(expected_next, expected_next + ref.used)
                )
                expected_next += ref.used
            assert expected_next == sess.length


def test_kv_rows_of_a_block_are_self_contained(model):
    """A fork's cached rows equal the rows a full forward produces there."""
    store = BlockStore(model, max_blocks=64, block_size=BLOCK_SIZE)
    tokens = [4, 2, 9, 1, 6, 3, 8, 5, 7, 0]
    sid = store.create_session(tokens)[0]["session_id"]
    child = store.fork(sid, 5).session_id
    _, full = model.forward(np.array(tokens))
    for sess_id in (sid, child):
        with store._lock:
            sess = store._sessions[sess_id]
            ks = []
            for ref in sess.blocks:
                block = store._blocks[ref.block_id]
                ks.append(block.k[: ref.used])
            kv = np.concatenate(ks)
        assert np.max(np.abs(kv - full.k[: sess.length])) <= 1e-12
