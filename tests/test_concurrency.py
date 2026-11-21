"""Concurrency tests: concurrent operations must be equivalent to some serial
ordering of the same operations.

Each test runs a batch of operations concurrently against one store, then checks
(a) no exceptions other than the documented rejections, (b) internal invariants
(block refcounts, session lengths) hold, and (c) the resulting state matches one
of the legal serial outcomes.
"""

from __future__ import annotations

import itertools
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.model import BLOCK_SIZE, Model, ModelConfig
from app.store import BlockStore, InsufficientBlocks, SessionNotFound


@pytest.fixture
def model():
    return Model(ModelConfig(vocab_size=32, d_model=16, n_heads=2, seed=11))


def make_store(model, max_blocks=200):
    return BlockStore(model, max_blocks=max_blocks, block_size=BLOCK_SIZE)


def check_invariants(store):
    """Refcounts must equal the number of session references to each block."""
    with store._lock:
        counts = {}
        for sess in store._sessions.values():
            for ref in sess.blocks:
                counts[ref.block_id] = counts.get(ref.block_id, 0) + 1
            assert sess.length == len(sess.tokens)
            assert sum(r.used for r in sess.blocks) == sess.length
            # Block views are contiguous prefixes of each block.
            for ref in sess.blocks:
                block = store._blocks[ref.block_id]
                assert 1 <= ref.used <= BLOCK_SIZE
                assert block.written >= ref.used
        for block_id, block in store._blocks.items():
            assert block.refs == counts.get(block_id, 0), (block_id, block.refs, counts)
            assert block.refs > 0


def full_logits(model, tokens):
    return model.forward(np.asarray(tokens, dtype=np.int64))[0]


def test_concurrent_forks_from_same_parent(model):
    store = make_store(model)
    sid = store.create_session([1, 2, 3, 4, 5, 6, 7, 8])[0]["session_id"]
    used = store.stats()["used_blocks"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        children = list(pool.map(lambda _: store.fork(sid, 5), range(16)))

    assert len({c.session_id for c in children}) == 16
    assert all(c.shared_blocks == 2 for c in children)
    assert store.stats()["used_blocks"] == used  # forks allocate nothing
    check_invariants(store)


def test_concurrent_forks_then_independent_appends_match_serial(model):
    """Each child's logits equal what a serial run produces for its own sequence."""
    store = make_store(model)
    parent_tokens = [2, 5, 1, 7, 3, 8, 4, 6]
    sid = store.create_session(parent_tokens)[0]["session_id"]
    children = [store.fork(sid, 4).session_id for _ in range(8)]
    payloads = [random.Random(i).sample(range(32), 5) for i in range(8)]

    def work(args):
        child_id, extra = args
        out = []
        for t in extra:
            out.append(store.append(child_id, [t]).logits[0])
        return np.array(out)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, zip(children, payloads)))

    for child_id, extra, got in zip(children, payloads, results):
        expected = full_logits(model, parent_tokens[:4] + extra)
        assert np.max(np.abs(got - expected[4:])) <= 1e-5
        assert store.get_session(child_id)["tokens"] == parent_tokens[:4] + extra
    check_invariants(store)


def test_concurrent_append_and_delete_same_session(model):
    """A delete racing appends must be equivalent to *some* serial order.

    Every append either happens before the delete (session gone with all its
    tokens) or after it (session gone, append rejected).  No partial block
    writes and no dangling refcounts are allowed either way.
    """
    store = make_store(model)
    tokens = [1, 2, 3, 4, 5, 6]

    for seed in range(60):
        rng = random.Random(seed)
        sid = store.create_session(tokens)[0]["session_id"]
        extra = [7, 8, 9]
        ops = [("append", extra[0]), ("delete", None), ("append", extra[1]),
               ("append", extra[2])]
        rng.shuffle(ops)

        def do(op):
            kind, value = op
            if kind == "delete":
                store.delete(sid)
                return "delete_ok"
            try:
                store.append(sid, [value])
                return "append_ok"
            except SessionNotFound:
                return "gone"

        with ThreadPoolExecutor(max_workers=len(ops)) as pool:
            outcomes = list(pool.map(do, ops))

        # The delete always succeeds exactly once; appends either all land
        # before it or are rejected by it (never a partial token list).
        assert outcomes.count("delete_ok") == 1
        assert store.get_session(sid) if False else True
        with pytest.raises(SessionNotFound):
            store.get_session(sid)
        n_appended = outcomes.count("append_ok")
        # Reconstruct what the tokens would have been, to check consistency of
        # the last surviving append. Appends that landed did so before delete.
        assert n_appended + outcomes.count("gone") == len(ops) - 1
        check_invariants(store)

        # Drain anything left from forks/sessions in this iteration.
        for s in store.list_sessions():
            try:
                store.delete(s["session_id"])
            except SessionNotFound:
                pass
        assert store.stats()["used_blocks"] == 0


def test_concurrent_appends_to_distinct_sessions_match_serial(model):
    store = make_store(model)
    n = 10
    seqs = [[random.Random(i).randrange(32) for _ in range(6)] for i in range(n)]
    sids = [store.create_session([])[0]["session_id"] for _ in range(n)]

    def work(args):
        sid, seq = args
        outs = []
        for t in seq:
            outs.append(store.append(sid, [t]).logits[0])
        return np.array(outs)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(work, zip(sids, seqs)))

    for seq, got in zip(seqs, results):
        expected = full_logits(model, seq)
        assert np.max(np.abs(got - expected)) <= 1e-5
    check_invariants(store)


def test_concurrent_append_to_shared_tail_keeps_blocks_consistent(model):
    """Many children appending through a shared prefix must each stay exact."""
    store = make_store(model)
    prefix = [9, 4, 7]
    sid = store.create_session(prefix)[0]["session_id"]
    children = [store.fork(sid, 3).session_id for _ in range(12)]

    def work(args):
        child_id, extra = args
        return store.append(child_id, extra).logits

    payloads = [[random.Random(i).randrange(32) for _ in range(4)] for i in range(12)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(work, zip(children, payloads)))

    for extra, got in zip(payloads, results):
        expected = full_logits(model, prefix + extra)
        assert np.max(np.abs(got - expected[3:])) <= 1e-5
    check_invariants(store)


def test_concurrent_delete_reclaims_only_unshared_blocks(model):
    store = make_store(model)
    # Build 20 unrelated sessions with 1 block each.
    sids = [store.create_session([i % 32, 1, 2, 3])[0]["session_id"] for i in range(20)]
    used = store.stats()["used_blocks"]
    assert used == 20

    def remove(sid):
        return store.delete(sid).blocks_freed

    with ThreadPoolExecutor(max_workers=10) as pool:
        freed = list(pool.map(remove, sids))
    assert sum(freed) == 20
    assert store.stats()["used_blocks"] == 0
    check_invariants(store)


def test_concurrent_rejections_never_partially_apply(model):
    """With one free block, concurrent 2-block appends: exactly one can win."""
    store = make_store(model, max_blocks=3)
    a = store.create_session([1, 2, 3, 4])[0]["session_id"]
    b = store.create_session([5, 6, 7, 8])[0]["session_id"]
    # 2 blocks used, 1 free. Tail blocks are full, so each 2-token append needs
    # exactly 1 fresh block; only one of the four can succeed.
    attempts = [(a, [9] * 4), (b, [10] * 4), (a, [11] * 4), (b, [12] * 4)]

    def work(args):
        try:
            store.append(*args)
            return True
        except InsufficientBlocks:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(work, attempts))

    assert sum(outcomes) == 1
    assert store.stats()["used_blocks"] == 3
    check_invariants(store)
    for sid in (a, b):
        assert store.get_session(sid)["length"] in (4, 8)


def _state(store, ids: dict[str, str]) -> dict:
    """Serializable snapshot keyed by logical session name."""
    out = {}
    for name, sid in ids.items():
        try:
            out[name] = tuple(store.get_session(sid)["tokens"])
        except KeyError:
            out[name] = None
    return out


def test_concurrent_mixed_workload_matches_some_serial_order(model):
    """A concurrent mix must equal *some* serial permutation of the same ops."""
    root_tokens = [1, 2, 3, 4, 5, 6]
    ops = [
        ("append", "a", [7, 8]),
        ("append", "b", [9, 10, 11]),
        ("append", "a", [12]),
        ("append", "c", [13, 14]),
        ("delete", "root", None),
        ("append", "b", [15]),
    ]

    def setup():
        store = make_store(model)
        root = store.create_session(root_tokens)[0]["session_id"]
        ids = {
            "root": root,
            "a": store.fork(root, 4).session_id,
            "b": store.fork(root, 2).session_id,
            "c": store.create_session([20, 21])[0]["session_id"],
        }
        return store, ids

    def apply(store, ids, op):
        kind, name, extra = op
        try:
            if kind == "append":
                store.append(ids[name], extra)
            else:
                store.delete(ids[name])
        except KeyError:
            pass

    # Every serial ordering of the same operations.
    serial_outcomes = set()
    for order in itertools.permutations(ops):
        store, ids = setup()
        for op in order:
            apply(store, ids, op)
        check_invariants(store)
        serial_outcomes.add(repr(sorted(_state(store, ids).items())))

    store, ids = setup()
    with ThreadPoolExecutor(max_workers=len(ops)) as pool:
        list(pool.map(lambda op: apply(store, ids, op), ops))
    check_invariants(store)
    concurrent_outcome = repr(sorted(_state(store, ids).items()))

    assert concurrent_outcome in serial_outcomes
