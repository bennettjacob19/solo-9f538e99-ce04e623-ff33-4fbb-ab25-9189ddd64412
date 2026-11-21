"""Reference-counted KV block storage shared by sessions.

A *block* is the allocation unit: it holds the K/V rows of at most
``BLOCK_SIZE`` (4) absolute positions.  A *session* is an ordered list of
references into blocks together with how many positions of each block it uses.
Blocks are append-only and immutable once written, which is what lets a fork
point at its parent's blocks without copying any tensor data.

All mutating operations take a single re-entrant lock.  Every call is therefore
applied atomically, and concurrent fork/append/delete calls are equivalent to
some serial ordering of those same calls.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

import numpy as np

from .model import BLOCK_SIZE, KV, Model


class SessionNotFound(KeyError):
    """Raised when a session id is unknown."""


class InsufficientBlocks(Exception):
    """Raised when an append cannot be satisfied within the block budget."""

    def __init__(self, required: int, available: int, max_blocks: int) -> None:
        super().__init__(
            f"append needs {required} new block(s), only {available} of "
            f"{max_blocks} are free"
        )
        self.required = required
        self.available = available
        self.max_blocks = max_blocks


class InvalidRequest(ValueError):
    """Raised for requests that are well-formed but semantically invalid."""


class InvalidRange(InvalidRequest):
    """Raised when a mask interval is not a valid range of the prefix."""


@dataclass
class _Block:
    id: int
    k: np.ndarray
    v: np.ndarray
    positions: np.ndarray
    written: int = 0
    refs: int = 1


@dataclass(frozen=True)
class BlockRef:
    """One session's view of a block: how many leading positions it uses."""

    block_id: int
    used: int


@dataclass
class Session:
    id: str
    tokens: list[int]
    blocks: list[BlockRef] = field(default_factory=list)
    length: int = 0


@dataclass
class AppendResult:
    session_id: str
    start_position: int
    positions: list[int]
    logits: np.ndarray
    blocks_allocated: int
    length: int


@dataclass
class ForkResult:
    session_id: str
    parent_id: str
    length: int
    shared_blocks: int


@dataclass
class DeleteResult:
    session_id: str
    blocks_freed: int
    blocks_used: int


@dataclass
class ScoreMaskResult:
    session_id: str
    prefix_length: int
    mask_start: int
    mask_end: int
    targets: list[int]
    logprobs_full: np.ndarray
    logprobs_masked: np.ndarray


class BlockStore:
    """Owns all model K/V blocks and visible session metadata."""

    def __init__(
        self,
        model: Model,
        max_blocks: int,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        if max_blocks < 0:
            raise ValueError("max_blocks must be non-negative")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self._model = model
        self._max_blocks = max_blocks
        self._block_size = block_size
        self._lock = threading.RLock()
        self._blocks: dict[int, _Block] = {}
        self._sessions: dict[str, Session] = {}
        self._next_block_id = 0

    # ------------------------------------------------------------------ reads

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    def stats(self) -> dict:
        with self._lock:
            used = len(self._blocks)
            return {
                "block_size": self._block_size,
                "max_blocks": self._max_blocks,
                "used_blocks": used,
                "free_blocks": self._max_blocks - used,
                "cached_positions": sum(b.written for b in self._blocks.values()),
                "shared_blocks": sum(1 for b in self._blocks.values() if b.refs > 1),
                "sessions": len(self._sessions),
            }

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [self._session_summary_locked(s) for s in self._sessions.values()]

    def get_session(self, session_id: str, include_tokens: bool = True) -> dict:
        with self._lock:
            sess = self._get_locked(session_id)
            info = self._session_summary_locked(sess)
            info["blocks"] = [
                {
                    "block_id": ref.block_id,
                    "used": ref.used,
                    "shared_by": self._blocks[ref.block_id].refs,
                }
                for ref in sess.blocks
            ]
            if include_tokens:
                info["tokens"] = list(sess.tokens)
            return info

    # -------------------------------------------------------------- mutations

    def create_session(
        self, tokens: list[int] | None = None
    ) -> tuple[dict, AppendResult | None]:
        """Create a session, optionally prefilled with ``tokens``.

        Returns the session summary plus the prefill's :class:`AppendResult`
        (or ``None`` when no tokens were supplied).
        """
        tokens = list(tokens or [])
        self._validate_tokens(tokens)
        session_id = uuid.uuid4().hex
        sess = Session(id=session_id, tokens=[], length=0)
        with self._lock:
            # Build the prefill first; only publish the session once it has
            # succeeded, so a rejected prefill leaves neither a session nor
            # leaked blocks behind.
            appended = self._append_locked(sess, tokens) if tokens else None
            self._sessions[session_id] = sess
            return self._session_summary_locked(sess), appended

    def append(self, session_id: str, tokens: list[int]) -> AppendResult:
        tokens = list(tokens)
        if not tokens:
            raise InvalidRequest("tokens must contain at least one token id")
        self._validate_tokens(tokens)
        with self._lock:
            sess = self._get_locked(session_id)
            return self._append_locked(sess, tokens)

    def fork(self, session_id: str, prefix_length: int) -> ForkResult:
        with self._lock:
            parent = self._get_locked(session_id)
            if prefix_length < 0 or prefix_length > parent.length:
                raise InvalidRequest(
                    f"prefix_length must be between 0 and {parent.length}"
                )
            refs: list[BlockRef] = []
            remaining = prefix_length
            for ref in parent.blocks:
                if remaining <= 0:
                    break
                take = min(ref.used, remaining)
                refs.append(BlockRef(ref.block_id, take))
                remaining -= take

            # Referenced blocks are shared, never copied: no new block, and
            # the global occupancy is unchanged.
            for ref in refs:
                self._blocks[ref.block_id].refs += 1

            child = Session(
                id=uuid.uuid4().hex,
                tokens=list(parent.tokens[:prefix_length]),
                blocks=refs,
                length=prefix_length,
            )
            self._sessions[child.id] = child
            return ForkResult(
                session_id=child.id,
                parent_id=session_id,
                length=child.length,
                shared_blocks=len(refs),
            )

    def delete(self, session_id: str) -> DeleteResult:
        with self._lock:
            sess = self._get_locked(session_id)
            del self._sessions[session_id]
            freed = 0
            for ref in sess.blocks:
                block = self._blocks[ref.block_id]
                block.refs -= 1
                if block.refs == 0:
                    del self._blocks[block.id]
                    freed += 1
            return DeleteResult(
                session_id=session_id,
                blocks_freed=freed,
                blocks_used=len(self._blocks),
            )

    # --------------------------------------------------------------- scoring

    def score_masked(
        self, session_id: str, start: int, end: int, targets: list[int]
    ) -> ScoreMaskResult:
        """Score ``targets`` as a continuation of the session's prefix, with ``[start, end)`` hidden.

        The interval is 0-based, includes ``start``, excludes ``end``, must
        contain at least one token, and must keep the last prefix position
        visible.  Each target's natural log-probability is reported twice:
        with the full prefix visible and with the interval hidden from
        attention (RoPE positions are kept either way).

        The prefix is snapshotted under the lock; the read-only math runs
        after the lock is released, so concurrent appends and deletes against
        the same session proceed while scoring is in flight and never change
        the result.  Scoring allocates no blocks and does not modify the
        session, so it also works when the cache is full.
        """
        targets = list(targets)
        if not targets:
            raise InvalidRequest("targets must contain at least one token id")
        if len(targets) > 64:
            raise InvalidRequest("targets must contain at most 64 token ids")
        self._validate_tokens(targets)
        with self._lock:
            sess = self._get_locked(session_id)
            length = sess.length
            if not 0 <= start < end <= length - 1:
                raise InvalidRange(
                    f"mask range [{start}, {end}) must contain at least one "
                    f"token and keep the last prefix position visible: need "
                    f"0 <= start < end <= {length - 1} for length {length}"
                )
            # Snapshot: concatenating copies the rows out of the blocks, so
            # later appends or deletes cannot change what gets scored.
            kv = self._session_kv_locked(sess)
            last_token = sess.tokens[-1]

        t = np.asarray(targets, dtype=np.int64)
        visible = np.ones(length, dtype=bool)
        visible[start:end] = False
        full, masked = self._model.score_contexts(kv, last_token, t, [None, visible])
        return ScoreMaskResult(
            session_id=session_id,
            prefix_length=length,
            mask_start=start,
            mask_end=end,
            targets=targets,
            logprobs_full=full,
            logprobs_masked=masked,
        )

    # ---------------------------------------------------------------- private

    def _get_locked(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id) from None

    def _validate_tokens(self, tokens: list[int]) -> None:
        vocab = self._model.cfg.vocab_size
        for t in tokens:
            if not isinstance(t, (int, np.integer)) or isinstance(t, bool):
                raise InvalidRequest(f"token id {t!r} is not an integer")
            if t < 0 or t >= vocab:
                raise InvalidRequest(
                    f"token id {t} out of range [0, {vocab})"
                )

    def _new_block_locked(self) -> _Block:
        cfg = self._model.cfg
        nh, dh = cfg.n_heads, cfg.d_head
        block = _Block(
            id=self._next_block_id,
            k=np.zeros((self._block_size, nh, dh), dtype=np.float64),
            v=np.zeros((self._block_size, nh, dh), dtype=np.float64),
            positions=np.full((self._block_size,), -1, dtype=np.int64),
        )
        self._next_block_id += 1
        self._blocks[block.id] = block
        return block

    def _session_kv_locked(self, sess: Session) -> KV:
        cfg = self._model.cfg
        if not sess.blocks:
            return KV(
                k=np.zeros((0, cfg.n_heads, cfg.d_head)),
                v=np.zeros((0, cfg.n_heads, cfg.d_head)),
                positions=np.zeros((0,), dtype=np.int64),
            )
        ks, vs, ps = [], [], []
        for ref in sess.blocks:
            block = self._blocks[ref.block_id]
            ks.append(block.k[: ref.used])
            vs.append(block.v[: ref.used])
            ps.append(block.positions[: ref.used])
        return KV(
            k=np.concatenate(ks, axis=0),
            v=np.concatenate(vs, axis=0),
            positions=np.concatenate(ps, axis=0),
        )

    def _tail_free_locked(self, sess: Session) -> int:
        """Free slots in the last block that this session may safely extend.

        A block can be extended in place only when this session is its only
        owner and no rows exist past this session's view.  Otherwise a fork (or
        a deleted sibling) makes appending in place either corrupt shared state
        or overwrite rows that belong to a different continuation.
        """
        if not sess.blocks:
            return 0
        ref = sess.blocks[-1]
        block = self._blocks[ref.block_id]
        if block.refs != 1 or block.written != ref.used:
            return 0
        return self._block_size - ref.used

    def _append_locked(self, sess: Session, tokens: list[int]) -> AppendResult:
        n = len(tokens)
        start = sess.length

        tail_free = self._tail_free_locked(sess)
        overflow = max(0, n - tail_free)
        blocks_needed = (overflow + self._block_size - 1) // self._block_size
        available = self._max_blocks - len(self._blocks)
        if blocks_needed > available:
            # Rejection happens before any state is touched: the session and
            # the block table are exactly as they were.
            raise InsufficientBlocks(blocks_needed, available, self._max_blocks)

        memory = self._session_kv_locked(sess)
        logits, full_kv = self._model.forward(
            np.asarray(tokens, dtype=np.int64), memory
        )
        new_kv = full_kv.tail(n)

        cursor = 0
        take = min(n, tail_free)
        if take:
            ref = sess.blocks[-1]
            block = self._blocks[ref.block_id]
            end = ref.used + take
            block.k[ref.used:end] = new_kv.k[cursor : cursor + take]
            block.v[ref.used:end] = new_kv.v[cursor : cursor + take]
            block.positions[ref.used:end] = new_kv.positions[cursor : cursor + take]
            block.written = max(block.written, end)
            sess.blocks[-1] = BlockRef(block.id, end)
            cursor += take

        allocated = 0
        while cursor < n:
            chunk = min(self._block_size, n - cursor)
            block = self._new_block_locked()
            block.k[:chunk] = new_kv.k[cursor : cursor + chunk]
            block.v[:chunk] = new_kv.v[cursor : cursor + chunk]
            block.positions[:chunk] = new_kv.positions[cursor : cursor + chunk]
            block.written = chunk
            sess.blocks.append(BlockRef(block.id, chunk))
            cursor += chunk
            allocated += 1

        sess.tokens.extend(tokens)
        sess.length += n
        return AppendResult(
            session_id=sess.id,
            start_position=start,
            positions=list(range(start, start + n)),
            logits=logits,
            blocks_allocated=allocated,
            length=sess.length,
        )

    def _session_summary_locked(self, sess: Session) -> dict:
        return {
            "session_id": sess.id,
            "length": sess.length,
            "num_blocks": len(sess.blocks),
            "shared_blocks": sum(
                1
                for ref in sess.blocks
                if self._blocks[ref.block_id].refs > 1
            ),
        }
