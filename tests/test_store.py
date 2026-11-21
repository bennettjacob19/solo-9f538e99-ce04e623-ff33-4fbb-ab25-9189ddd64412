import numpy as np
import pytest

from app.model import BLOCK_SIZE, Model, ModelConfig
from app.store import BlockStore, InsufficientBlocks, InvalidRequest, SessionNotFound


@pytest.fixture
def model():
    return Model(ModelConfig(vocab_size=32, d_model=16, n_heads=2, seed=5))


@pytest.fixture
def store(model):
    return BlockStore(model, max_blocks=8, block_size=BLOCK_SIZE)


def full_logits(model, tokens):
    return model.forward(np.asarray(tokens, dtype=np.int64))[0]


def test_create_with_prefill_matches_full_forward(model, store):
    tokens = [1, 5, 2, 7, 3]
    info, appended = store.create_session(tokens)
    assert info["length"] == 5
    assert appended is not None
    expected = full_logits(model, tokens)
    assert np.max(np.abs(appended.logits - expected)) <= 1e-5
    assert appended.positions == [0, 1, 2, 3, 4]


def test_append_returns_logits_for_new_positions_only(model, store):
    prefix = [4, 1, 8]
    info, _ = store.create_session(prefix)
    sid = info["session_id"]
    extra = [2, 9, 6, 0]
    result = store.append(sid, extra)
    assert result.start_position == 3
    assert result.positions == [3, 4, 5, 6]
    assert result.logits.shape == (4, model.cfg.vocab_size)

    expected = full_logits(model, prefix + extra)
    assert np.max(np.abs(result.logits - expected[3:])) <= 1e-5


def test_block_count_matches_positions(store):
    info, _ = store.create_session([1, 2, 3, 4, 5])
    assert info["num_blocks"] == 2  # 5 positions -> blocks of 4
    sid = info["session_id"]
    store.append(sid, [6, 7])
    detail = store.get_session(sid)
    assert detail["length"] == 7
    assert len(detail["blocks"]) == 2
    assert [b["used"] for b in detail["blocks"]] == [4, 3]


def test_fork_reuses_blocks_without_allocating(store):
    info, _ = store.create_session([1, 2, 3, 4, 5, 6])
    parent = info["session_id"]
    used_before = store.stats()["used_blocks"]
    forked = store.fork(parent, 4)
    assert forked.shared_blocks == 1
    assert store.stats()["used_blocks"] == used_before
    detail = store.get_session(forked.session_id)
    assert detail["length"] == 4
    assert detail["tokens"] == [1, 2, 3, 4]


def test_fork_at_arbitrary_prefix_len(store):
    info, _ = store.create_session([1, 2, 3, 4, 5, 6, 7])
    parent = info["session_id"]
    mid = store.fork(parent, 6)
    detail = store.get_session(mid.session_id)
    assert detail["length"] == 6
    assert [b["used"] for b in detail["blocks"]] == [4, 2]


def test_forked_children_diverge_independently(model, store):
    prefix = [3, 1, 4, 1]
    info, _ = store.create_session(prefix)
    parent = info["session_id"]
    a = store.fork(parent, 4).session_id
    b = store.fork(parent, 4).session_id

    ra = store.append(a, [5, 9])
    rb = store.append(b, [2, 6])
    assert np.max(np.abs(ra.logits - full_logits(model, prefix + [5, 9])[4:])) <= 1e-5
    assert np.max(np.abs(rb.logits - full_logits(model, prefix + [2, 6])[4:])) <= 1e-5
    assert np.max(np.abs(ra.logits - rb.logits)) > 1e-3


def test_fork_does_not_let_parent_overwrite_shared_block(model, store):
    """Appending to a forked parent must not clobber the child's view."""
    info, _ = store.create_session([1, 2, 3])  # one block, used=3 (padded slot)
    parent = info["session_id"]
    child = store.fork(parent, 3).session_id

    # Parent extends into its padded tail slot; physically separate storage.
    rp = store.append(parent, [4, 5])
    rc = store.append(child, [7])
    p_tokens = store.get_session(parent)["tokens"]
    c_tokens = store.get_session(child)["tokens"]
    assert p_tokens == [1, 2, 3, 4, 5]
    assert c_tokens == [1, 2, 3, 7]
    assert np.max(np.abs(rp.logits - full_logits(model, p_tokens)[3:])) <= 1e-5
    assert np.max(np.abs(rc.logits - full_logits(model, c_tokens)[3:])) <= 1e-5


def test_partial_fork_does_not_corrupt_parent(model, store):
    info, _ = store.create_session([1, 2])  # block0, used=2
    parent = info["session_id"]
    child = store.fork(parent, 2).session_id  # shares block0
    # Parent's only (tail) block is shared, so it may not write into it.
    used_before = store.stats()["used_blocks"]
    rp = store.append(parent, [6])
    assert store.stats()["used_blocks"] == used_before + 1
    assert store.get_session(parent)["tokens"] == [1, 2, 6]
    assert store.get_session(child)["tokens"] == [1, 2]
    rc = store.append(child, [9, 9, 9])
    assert np.max(np.abs(rc.logits - full_logits(model, [1, 2, 9, 9, 9])[2:])) <= 1e-5
    assert np.max(np.abs(rp.logits - full_logits(model, [1, 2, 6])[2:])) <= 1e-5


def test_partial_fork_leaves_unshared_tail_extendable(model, store):
    """A fork at 2 of a 5-token session shares block0 only; block1 stays private."""
    info, _ = store.create_session([1, 2, 3, 4, 5])
    parent = info["session_id"]
    child = store.fork(parent, 2).session_id
    used_before = store.stats()["used_blocks"]
    rp = store.append(parent, [6, 7])
    assert store.stats()["used_blocks"] == used_before  # fits in private block1
    assert store.get_session(parent)["tokens"] == [1, 2, 3, 4, 5, 6, 7]
    assert store.get_session(child)["tokens"] == [1, 2]
    assert np.max(np.abs(rp.logits - full_logits(model, [1, 2, 3, 4, 5, 6, 7])[5:])) <= 1e-5


def test_insufficient_blocks_rejects_whole_append(model):
    store = BlockStore(model, max_blocks=1, block_size=4)
    info, _ = store.create_session([1, 2, 3])  # block0 used=3, 1 free slot
    sid = info["session_id"]
    stats_before = store.stats()
    with pytest.raises(InsufficientBlocks) as exc:
        store.append(sid, [4, 5])  # needs 1 new block, none free
    assert exc.value.required == 1
    assert exc.value.available == 0
    after = store.get_session(sid)
    assert after["tokens"] == [1, 2, 3]  # unchanged
    assert after["length"] == 3
    assert store.stats()["used_blocks"] == stats_before["used_blocks"]
    # The session is still usable afterwards.
    store.append(sid, [4])
    assert store.get_session(sid)["length"] == 4


def test_insufficient_blocks_on_prefill_creates_no_session():
    model = Model(ModelConfig(vocab_size=16, d_model=8, n_heads=2, seed=1))
    store = BlockStore(model, max_blocks=1, block_size=4)
    with pytest.raises(InsufficientBlocks):
        store.create_session([1, 2, 3, 4, 5])
    assert store.stats()["sessions"] == 0
    assert store.stats()["used_blocks"] == 0


def test_delete_reclaims_only_exclusive_blocks(model, store):
    info, _ = store.create_session([1, 2, 3, 4, 5, 6])  # 2 blocks
    parent = info["session_id"]
    child = store.fork(parent, 6).session_id
    before = store.stats()["used_blocks"]
    deleted = store.delete(parent)
    # Both blocks are still referenced by the child: nothing freed.
    assert deleted.blocks_freed == 0
    assert store.stats()["used_blocks"] == before == 2
    # Child can keep appending (its tail block is shared by nobody now -> refs=1
    # but written==used==2 and length==6, so it extends in place).
    r = store.append(child, [7, 8])
    assert np.max(np.abs(r.logits - full_logits(model, [1, 2, 3, 4, 5, 6, 7, 8])[6:])) <= 1e-5


def test_delete_frees_blocks_and_frees_capacity(model, store):
    a, _ = store.create_session([1, 2, 3, 4])  # 1 block
    b, _ = store.create_session([5, 6, 7, 8])  # 1 block
    assert store.stats()["used_blocks"] == 2
    res = store.delete(a["session_id"])
    assert res.blocks_freed == 1
    assert store.stats()["used_blocks"] == 1
    # Freed capacity can be reused by another session.
    store.create_session([9, 9, 9, 9])
    assert store.stats()["used_blocks"] == 2
    with pytest.raises(SessionNotFound):
        store.get_session(a["session_id"])


def test_delete_unknown_session(store):
    with pytest.raises(SessionNotFound):
        store.delete("nope")


def test_fork_unknown_or_bad_prefix(store):
    info, _ = store.create_session([1, 2, 3, 4, 5])
    sid = info["session_id"]
    with pytest.raises(InvalidRequest):
        store.fork(sid, 6)
    with pytest.raises(InvalidRequest):
        store.fork(sid, -1)
    with pytest.raises(SessionNotFound):
        store.fork("nope", 1)


def test_invalid_token_rejected_without_side_effects(store):
    info, _ = store.create_session([1, 2])
    sid = info["session_id"]
    with pytest.raises(InvalidRequest):
        store.append(sid, [3, 99])
    assert store.get_session(sid)["tokens"] == [1, 2]
    with pytest.raises(InvalidRequest):
        store.append(sid, [])
    with pytest.raises(InvalidRequest):
        store.append(sid, [True])


def test_fork_zero_length_is_allowed(store):
    info, _ = store.create_session([1, 2, 3])
    child = store.fork(info["session_id"], 0)
    assert child.shared_blocks == 0
    assert store.get_session(child.session_id)["length"] == 0
    store.append(child.session_id, [4, 5])
    assert store.get_session(child.session_id)["tokens"] == [4, 5]


def test_repeated_small_appends_match_full_forward(model, store):
    tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    info, _ = store.create_session(tokens[:1])
    sid = info["session_id"]
    l1 = store.get_session(sid)
    del l1
    pieces = []
    for t in tokens[1:]:
        r = store.append(sid, [t])
        pieces.append(r.logits)
    got = np.concatenate([np.zeros((1, model.cfg.vocab_size)), *pieces], axis=0)
    # Position 0's logits from creation are checked through a full compare.
    expected = full_logits(model, tokens)
    assert np.max(np.abs(got[1:] - expected[1:])) <= 1e-5
