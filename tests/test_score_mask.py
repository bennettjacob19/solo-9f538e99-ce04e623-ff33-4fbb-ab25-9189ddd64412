"""Tests for masked-context scoring (POST /sessions/{id}/score-mask).

The endpoint reports, for each target token, its natural log-probability
under the full prefix and under the prefix with one interval hidden from
attention.  These tests pin down the scoring semantics (teacher forcing from
the last prefix position, RoPE positions preserved under masking), the
read-only/cache-only nature of scoring, and its snapshot behaviour under
concurrent appends and deletes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model import BLOCK_SIZE, KV, Model, ModelConfig
from app.store import (
    BlockStore,
    InvalidRange,
    InvalidRequest,
    SessionNotFound,
)


@pytest.fixture
def model():
    return Model(ModelConfig(vocab_size=32, d_model=16, n_heads=2, seed=9))


@pytest.fixture
def store(model):
    return BlockStore(model, max_blocks=16, block_size=BLOCK_SIZE)


def log_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


def reference_logprobs(model, prefix, targets, hidden=()):
    """Independent masked-scoring reference assembled from model primitives.

    ``hidden`` prefix positions are dropped from the attention context while
    every remaining row keeps its absolute position.
    """
    L, m = len(prefix), len(targets)
    _, kv = model.forward(np.asarray(prefix, dtype=np.int64))
    keep = [i for i in range(L) if i not in hidden]
    k_mem, v_mem, p_mem = kv.k[keep], kv.v[keep], kv.positions[keep]

    q_tokens = np.asarray([prefix[-1]] + list(targets[:-1]), dtype=np.int64)
    q_pos = np.asarray([L - 1] + list(range(L, L + m - 1)), dtype=np.int64)
    q, k_new, v_new = model._qkv(q_tokens, q_pos)
    # The last prefix token's K/V row already lives in the cache; only the
    # targets that precede another target contribute fresh rows.
    k_all = np.concatenate([k_mem, k_new[1:]], axis=0)
    v_all = np.concatenate([v_mem, v_new[1:]], axis=0)
    p_all = np.concatenate([p_mem, q_pos[1:]], axis=0)

    scale = 1.0 / np.sqrt(model.cfg.d_head)
    scores = np.einsum("qhd,khd->hqk", q, k_all) * scale
    causal = p_all[None, :] <= q_pos[:, None]
    scores = np.where(causal[None, :, :], scores, -np.inf)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    attn = np.einsum("hqk,khd->qhd", probs, v_all).reshape(m, model.cfg.d_model)
    logits = model._head(q_tokens, attn)
    return log_softmax(logits)[np.arange(m), np.asarray(targets, dtype=np.int64)]


# --------------------------------------------------------------------- model


def test_score_continuation_matches_full_forward(model):
    prefix = [3, 1, 4, 1, 5]
    targets = [9, 2, 6]
    _, kv = model.forward(np.asarray(prefix, dtype=np.int64))
    got = model.score_continuation(kv, prefix[-1], np.asarray(targets))

    full, _ = model.forward(np.asarray(prefix + targets, dtype=np.int64))
    rows = log_softmax(full)[len(prefix) - 1 : len(prefix) - 1 + len(targets)]
    expected = rows[np.arange(len(targets)), targets]
    assert np.max(np.abs(got - expected)) <= 1e-9


def test_score_continuation_single_target_uses_last_position_only(model):
    prefix = [2, 7, 1, 8]
    _, kv = model.forward(np.asarray(prefix, dtype=np.int64))
    got = model.score_continuation(kv, prefix[-1], np.asarray([5]))
    full, _ = model.forward(np.asarray(prefix, dtype=np.int64))
    expected = log_softmax(full)[-1, 5]
    assert np.max(np.abs(got - [expected])) <= 1e-9


def test_score_continuation_mask_matches_reference(model):
    prefix = [3, 1, 4, 1, 5, 9, 2]
    targets = [6, 5, 3]
    _, kv = model.forward(np.asarray(prefix, dtype=np.int64))
    visible = np.ones(len(prefix), dtype=bool)
    visible[2:5] = False
    got = model.score_continuation(kv, prefix[-1], np.asarray(targets), visible=visible)
    expected = reference_logprobs(model, prefix, targets, hidden=range(2, 5))
    assert np.max(np.abs(got - expected)) <= 1e-9


def test_score_contexts_matches_single_calls(model):
    prefix = [3, 1, 4, 1, 5, 9, 2]
    targets = np.asarray([6, 5, 3])
    _, kv = model.forward(np.asarray(prefix, dtype=np.int64))
    visible = np.ones(len(prefix), dtype=bool)
    visible[1:4] = False
    both = model.score_contexts(kv, prefix[-1], targets, [None, visible])
    assert both.shape == (2, 3)
    assert np.allclose(
        both[0], model.score_continuation(kv, prefix[-1], targets), atol=1e-12
    )
    assert np.allclose(
        both[1],
        model.score_continuation(kv, prefix[-1], targets, visible=visible),
        atol=1e-12,
    )


def test_score_continuation_validation(model):
    _, kv = model.forward(np.asarray([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError):
        model.score_continuation(kv, 3, np.asarray([], dtype=np.int64))
    with pytest.raises(ValueError):
        model.score_continuation(kv, 3, np.asarray([model.cfg.vocab_size]))
    empty = KV(
        k=np.zeros((0, model.cfg.n_heads, model.cfg.d_head)),
        v=np.zeros((0, model.cfg.n_heads, model.cfg.d_head)),
        positions=np.zeros((0,), dtype=np.int64),
    )
    with pytest.raises(ValueError):
        model.score_continuation(empty, 3, np.asarray([1]))
    with pytest.raises(ValueError):
        model.score_continuation(kv, 3, np.asarray([1]), visible=[True, False])


# --------------------------------------------------------------------- store


def test_store_scores_match_reference(model, store):
    prefix = [3, 1, 4, 1, 5, 9, 2, 6]
    sid = store.create_session(prefix)[0]["session_id"]
    targets = [5, 3, 8]
    res = store.score_masked(sid, 2, 5, targets)
    assert res.prefix_length == len(prefix)
    assert res.mask_start == 2 and res.mask_end == 5
    assert res.targets == targets
    assert np.allclose(
        res.logprobs_full, reference_logprobs(model, prefix, targets), atol=1e-9
    )
    assert np.allclose(
        res.logprobs_masked,
        reference_logprobs(model, prefix, targets, hidden=range(2, 5)),
        atol=1e-9,
    )
    # Masking actually changes the scores for this interval.
    assert np.max(np.abs(res.logprobs_full - res.logprobs_masked)) > 1e-6


def test_mask_preserves_rope_positions(model, store):
    """Masking must not re-index: it differs from scoring a shorter prefix."""
    prefix = [1, 2, 3, 4, 5, 6, 7]
    targets = [8, 9]
    sid = store.create_session(prefix)[0]["session_id"]
    masked = store.score_masked(sid, 2, 4, targets)

    shortened = prefix[:2] + prefix[4:]  # interval deleted, positions shifted
    other = store.create_session(shortened)[0]["session_id"]
    shifted = store.score_masked(other, 0, 1, targets)

    assert np.max(np.abs(masked.logprobs_masked - shifted.logprobs_full)) > 1e-6


def test_scoring_does_not_touch_session_or_blocks(model, store):
    sid = store.create_session([1, 2, 3, 4, 5, 6])[0]["session_id"]
    store.append(sid, [7, 8])
    before_session = store.get_session(sid)
    before_stats = store.stats()
    store.score_masked(sid, 1, 4, [9, 10, 11])
    assert store.get_session(sid) == before_session
    assert store.stats() == before_stats


def test_scoring_works_when_cache_is_full():
    model = Model(ModelConfig(vocab_size=16, d_model=8, n_heads=2, seed=6))
    store = BlockStore(model, max_blocks=2, block_size=BLOCK_SIZE)
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]  # fills both blocks exactly
    sid = store.create_session(prefix)[0]["session_id"]
    assert store.stats()["free_blocks"] == 0
    res = store.score_masked(sid, 0, 7, [9, 10])
    assert res.prefix_length == 8
    assert np.allclose(
        res.logprobs_full, reference_logprobs(model, prefix, [9, 10]), atol=1e-9
    )
    assert store.stats()["free_blocks"] == 0


def test_scoring_computes_kv_only_for_uncached_targets():
    """Scoring reads cached prefix rows; it never recomputes them."""

    class CountingModel(Model):
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

    model = CountingModel(ModelConfig(vocab_size=32, d_model=16, n_heads=2, seed=13))
    store = BlockStore(model, max_blocks=16, block_size=BLOCK_SIZE)
    sid = store.create_session([1, 2, 3, 4, 5, 6])[0]["session_id"]

    model.kv_tokens = 0
    model.forward_calls = 0
    store.score_masked(sid, 1, 3, [7, 8, 9, 10])
    # Only the targets a later query conditions on (all but the last) get
    # fresh K/V; the cached prefix rows are read, not recomputed.
    assert model.kv_tokens == 3
    assert model.forward_calls == 0

    model.kv_tokens = 0
    store.score_masked(sid, 1, 3, [7])
    assert model.kv_tokens == 0


def test_score_on_forked_session(model, store):
    parent = store.create_session([1, 2, 3, 4, 5, 6, 7])[0]["session_id"]
    child = store.fork(parent, 5).session_id
    store.append(child, [8, 9])
    tokens = [1, 2, 3, 4, 5, 8, 9]
    res = store.score_masked(child, 1, 3, [2, 4])
    assert res.prefix_length == 7
    assert np.allclose(
        res.logprobs_full, reference_logprobs(model, tokens, [2, 4]), atol=1e-9
    )
    assert np.allclose(
        res.logprobs_masked,
        reference_logprobs(model, tokens, [2, 4], hidden=range(1, 3)),
        atol=1e-9,
    )


def test_score_after_parent_deleted(model, store):
    parent = store.create_session([1, 2, 3, 4, 5, 6])[0]["session_id"]
    child = store.fork(parent, 6).session_id
    store.delete(parent)
    res = store.score_masked(child, 0, 2, [1])
    assert np.allclose(
        res.logprobs_full,
        reference_logprobs(model, [1, 2, 3, 4, 5, 6], [1]),
        atol=1e-9,
    )


def test_score_masked_errors(model, store):
    sid = store.create_session([1, 2, 3, 4])[0]["session_id"]
    with pytest.raises(SessionNotFound):
        store.score_masked("nope", 0, 1, [1])
    # Interval must contain a token and keep the last prefix position.
    for start, end in [(-1, 2), (2, 2), (3, 2), (0, 4), (2, 4), (0, 5), (4, 4)]:
        with pytest.raises(InvalidRange):
            store.score_masked(sid, start, end, [1])
    with pytest.raises(InvalidRequest):
        store.score_masked(sid, 0, 1, [])
    with pytest.raises(InvalidRequest):
        store.score_masked(sid, 0, 1, [1] * 65)
    with pytest.raises(InvalidRequest):
        store.score_masked(sid, 0, 1, [32])  # vocab_size == 32
    with pytest.raises(InvalidRequest):
        store.score_masked(sid, 0, 1, [True])
    # An empty session has no valid interval.
    empty = store.create_session([])[0]["session_id"]
    with pytest.raises(InvalidRange):
        store.score_masked(empty, 0, 1, [1])
    # Errors leave the session untouched.
    assert store.get_session(sid)["tokens"] == [1, 2, 3, 4]


def test_score_masked_boundary_intervals(model, store):
    prefix = [1, 2, 3, 4, 5]
    sid = store.create_session(prefix)[0]["session_id"]
    # Everything but the last position hidden.
    res = store.score_masked(sid, 0, 4, [6])
    assert np.allclose(
        res.logprobs_masked,
        reference_logprobs(model, prefix, [6], hidden=range(0, 4)),
        atol=1e-9,
    )
    # Single-token interval ending right before the last position.
    res = store.score_masked(sid, 3, 4, [6])
    assert np.allclose(
        res.logprobs_masked,
        reference_logprobs(model, prefix, [6], hidden=range(3, 4)),
        atol=1e-9,
    )


# --------------------------------------------------------------- concurrency


def test_scores_pinned_to_snapshot_during_appends(model):
    """Scores reflect the prefix fixed at snapshot time, whatever lands later."""
    store = BlockStore(model, max_blocks=64, block_size=BLOCK_SIZE)
    prefix = [3, 1, 4, 1, 5, 9]
    sid = store.create_session(prefix)[0]["session_id"]
    extra = [2, 6, 5, 3]
    targets = [7, 8]
    start, end = 1, 3

    def append_all():
        for t in extra:
            store.append(sid, [t])

    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(store.score_masked, sid, start, end, targets) for _ in range(16)]
        append_future = pool.submit(append_all)
        results = [f.result() for f in futures]
        append_future.result()

    assert store.get_session(sid)["length"] == len(prefix) + len(extra)
    for res in results:
        L = res.prefix_length
        assert len(prefix) <= L <= len(prefix) + len(extra)
        tokens = prefix + extra[: L - len(prefix)]
        assert np.allclose(
            res.logprobs_full, reference_logprobs(model, tokens, targets), atol=1e-9
        )
        assert np.allclose(
            res.logprobs_masked,
            reference_logprobs(model, tokens, targets, hidden=range(start, end)),
            atol=1e-9,
        )


def test_scores_pinned_to_snapshot_during_delete(model):
    """A delete racing scoring: each score is 404-equivalent or exact."""
    store = BlockStore(model, max_blocks=64, block_size=BLOCK_SIZE)
    prefix = [3, 1, 4, 1, 5, 9]
    targets = [7, 8]
    start, end = 1, 3
    expected_full = reference_logprobs(model, prefix, targets)
    expected_masked = reference_logprobs(model, prefix, targets, hidden=range(start, end))

    for _ in range(20):
        sid = store.create_session(prefix)[0]["session_id"]

        def score_one():
            try:
                return store.score_masked(sid, start, end, targets)
            except SessionNotFound:
                return None

        with ThreadPoolExecutor(max_workers=9) as pool:
            futures = [pool.submit(score_one) for _ in range(16)]
            delete_future = pool.submit(store.delete, sid)
            results = [f.result() for f in futures]
            delete_future.result()

        for res in results:
            if res is None:
                continue
            assert res.prefix_length == len(prefix)
            assert np.allclose(res.logprobs_full, expected_full, atol=1e-9)
            assert np.allclose(res.logprobs_masked, expected_masked, atol=1e-9)
        with pytest.raises(SessionNotFound):
            store.get_session(sid)
    assert store.stats()["used_blocks"] == 0


# ----------------------------------------------------------------------- api


@pytest.fixture
def client():
    return TestClient(
        create_app(
            max_blocks=8,
            model_config=ModelConfig(vocab_size=16, d_model=8, n_heads=2, seed=4),
        )
    )


def test_score_mask_endpoint_happy_path(client):
    prefix = [1, 2, 3, 4, 5, 6]
    sid = client.post("/sessions", json={"tokens": prefix}).json()["session_id"]
    r = client.post(
        f"/sessions/{sid}/score-mask", json={"start": 1, "end": 3, "targets": [7, 8, 9]}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == sid
    assert body["prefix_length"] == 6
    assert body["mask_start"] == 1 and body["mask_end"] == 3
    assert [s["position"] for s in body["scores"]] == [6, 7, 8]
    assert [s["token"] for s in body["scores"]] == [7, 8, 9]
    for s in body["scores"]:
        assert s["logprob_full"] <= 0.0
        assert s["logprob_masked"] <= 0.0

    model = client.app.state.store._model
    assert np.allclose(
        [s["logprob_full"] for s in body["scores"]],
        reference_logprobs(model, prefix, [7, 8, 9]),
        atol=1e-9,
    )
    assert np.allclose(
        [s["logprob_masked"] for s in body["scores"]],
        reference_logprobs(model, prefix, [7, 8, 9], hidden=range(1, 3)),
        atol=1e-9,
    )


def test_score_mask_unknown_session_404(client):
    r = client.post("/sessions/nope/score-mask", json={"start": 0, "end": 1, "targets": [1]})
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "session_not_found"


def test_score_mask_invalid_range_422(client):
    sid = client.post("/sessions", json={"tokens": [1, 2, 3, 4]}).json()["session_id"]
    for start, end in [(-1, 2), (2, 2), (3, 2), (0, 4), (2, 4), (0, 5)]:
        r = client.post(
            f"/sessions/{sid}/score-mask",
            json={"start": start, "end": end, "targets": [1]},
        )
        assert r.status_code == 422, (start, end, r.text)
        assert r.json()["detail"]["error"] == "invalid_range"
    empty = client.post("/sessions", json={"tokens": []}).json()["session_id"]
    r = client.post(
        f"/sessions/{empty}/score-mask", json={"start": 0, "end": 1, "targets": [1]}
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_range"


def test_score_mask_invalid_targets_422(client):
    sid = client.post("/sessions", json={"tokens": [1, 2, 3, 4]}).json()["session_id"]
    r = client.post(
        f"/sessions/{sid}/score-mask", json={"start": 0, "end": 1, "targets": [16]}
    )
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_token"
    # 0 or 65 targets are rejected; 1 and 64 are accepted.
    assert client.post(
        f"/sessions/{sid}/score-mask", json={"start": 0, "end": 1, "targets": []}
    ).status_code == 422
    assert client.post(
        f"/sessions/{sid}/score-mask",
        json={"start": 0, "end": 1, "targets": [1] * 65},
    ).status_code == 422
    assert client.post(
        f"/sessions/{sid}/score-mask", json={"start": 0, "end": 1, "targets": [1]}
    ).status_code == 200
    assert client.post(
        f"/sessions/{sid}/score-mask",
        json={"start": 0, "end": 1, "targets": [1] * 64},
    ).status_code == 200


def test_score_mask_needs_no_free_blocks_and_changes_nothing(client):
    # max_blocks=8 with 4-position blocks: 32 tokens fill the cache exactly.
    sid = client.post("/sessions", json={"tokens": [1] * 32}).json()["session_id"]
    assert client.get("/stats").json()["free_blocks"] == 0
    before_stats = client.get("/stats").json()
    before_session = client.get(f"/sessions/{sid}").json()
    r = client.post(
        f"/sessions/{sid}/score-mask", json={"start": 0, "end": 31, "targets": [2, 3]}
    )
    assert r.status_code == 200, r.text
    assert client.get("/stats").json() == before_stats
    assert client.get(f"/sessions/{sid}").json() == before_session
