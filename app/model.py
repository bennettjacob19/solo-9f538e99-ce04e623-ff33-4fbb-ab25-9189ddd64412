"""A tiny single-layer causal Transformer with rotary position encoding.

The model is deliberately small and pure NumPy so that everything the service
does is explicit and reproducible.  Its cacheable state is exactly the
per-position K/V rows of its single attention layer, which is what the block
store in :mod:`app.store` manages.

Because there is a single layer, the K/V row for a position depends only on the
token at that position and its absolute position -- never on other tokens.  That
is what makes sharing cached K/V blocks between sessions exact rather than
approximate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

#: Number of positions a KV block can hold.
BLOCK_SIZE = 4

#: Every angle in a head's rotary embedding divides `ROPE_ANGLE_BASE` by
#: ``base ** (2k/d)`` with the base independent of the head size, so a head can
#: be split into any prefix of its dimensions and the position encoding of that
#: prefix stays identical.
ROPE_ANGLE_BASE = 10_000.0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 64
    d_model: int = 16
    n_heads: int = 2
    seed: int = 0

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def rope_width(self) -> int:
        """Even prefix of each head that carries rotary position encoding."""
        return self.d_head - (self.d_head % 2)


@dataclass(frozen=True)
class KV:
    """Cached attention state, expressed in absolute positions.

    ``k``/``v`` are shaped ``(n, n_heads, d_head)`` and ``positions`` holds
    the absolute position of each row.  Each row is fully independent of its
    neighbours, so a contiguous run of rows can be moved, shared or dropped as
    a unit.
    """

    k: np.ndarray
    v: np.ndarray
    positions: np.ndarray

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def concat(self, other: "KV") -> "KV":
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        return KV(
            np.concatenate([self.k, other.k], axis=0),
            np.concatenate([self.v, other.v], axis=0),
            np.concatenate([self.positions, other.positions], axis=0),
        )

    def tail(self, n: int) -> "KV":
        """The last ``n`` rows."""
        if n > len(self):
            raise ValueError("tail is longer than the KV")
        if n == 0:
            return KV(
                k=self.k[:0].copy(), v=self.v[:0].copy(), positions=self.positions[:0].copy()
            )
        return KV(
            k=self.k[-n:].copy(), v=self.v[-n:].copy(), positions=self.positions[-n:].copy()
        )


def empty_kv(n_heads: int, d_head: int) -> KV:
    return KV(
        k=np.zeros((0, n_heads, d_head), dtype=np.float64),
        v=np.zeros((0, n_heads, d_head), dtype=np.float64),
        positions=np.zeros((0,), dtype=np.int64),
    )


def _rope_tables(positions: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    if width % 2:
        raise ValueError("rotary width must be even")
    idx = np.arange(0, width, 2, dtype=np.float64)
    inv_freq = ROPE_ANGLE_BASE ** (-idx / width)
    angles = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(angles), np.sin(angles)


def _rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate the leading even prefix of the last axis of ``x``.

    ``x`` has shape ``(n, n_heads, d_head)`` and ``cos``/``sin`` shape
    ``(n, rope_width/2)``.  Dimensions past ``rope_width`` pass through
    untouched.
    """
    width = cos.shape[-1] * 2
    out = x.copy()
    even = x[..., :width:2]
    odd = x[..., 1:width:2]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out[..., :width:2] = even * c - odd * s
    out[..., 1:width:2] = even * s + odd * c
    return out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


class Model:
    """One causal self-attention layer wrapped in LayerNorm + MLP blocks."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or ModelConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        d, v = self.cfg.d_model, self.cfg.vocab_size
        self.emb = self._init((v, d), 1.0 / np.sqrt(d))
        self.w_qkv = self._init((3 * d, d), 1.0 / np.sqrt(d))
        self.b_qkv = np.zeros(3 * d)
        self.w_o = self._init((d, d), 1.0 / np.sqrt(d))
        self.b_o = np.zeros(d)
        self.w_ff = self._init((4 * d, d), 1.0 / np.sqrt(d))
        self.b_ff = np.zeros(4 * d)
        self.w_proj = self._init((d, 4 * d), 1.0 / np.sqrt(4 * d))
        self.b_proj = np.zeros(d)

    def _init(self, shape: tuple[int, ...], scale: float) -> np.ndarray:
        return self._rng.normal(0.0, scale, size=shape).astype(np.float64)

    # ------------------------------------------------------------------ config

    @property
    def n_pos_per_block(self) -> int:
        return BLOCK_SIZE

    def config_summary(self) -> dict:
        c = self.cfg
        return {
            "vocab_size": c.vocab_size,
            "d_model": c.d_model,
            "n_heads": c.n_heads,
            "d_head": c.d_head,
            "d_head_rope": c.rope_width,
            "n_layers": 1,
            "position_encoding": "rope",
            "positions_per_block": BLOCK_SIZE,
        }

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for name in (
            "emb",
            "w_qkv",
            "b_qkv",
            "w_o",
            "b_o",
            "w_ff",
            "b_ff",
            "w_proj",
            "b_proj",
        ):
            arr = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            h.update(name.encode())
            h.update(arr.tobytes())
        return h.hexdigest()

    # -------------------------------------------------------------------- math

    def _ln(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5)

    def _qkv(
        self, tokens: np.ndarray, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Queries/keys/values for ``tokens`` at absolute ``positions``.

        Keys are rotated by their own absolute position, so a key row is valid
        in any sequence that places that token at the same position.  Queries
        are rotated identically, which makes attention scores depend only on
        the relative offset between query and key.
        """
        h = self._ln(self.emb[tokens]) @ self.w_qkv.T + self.b_qkv
        d = self.cfg.d_model
        q, k, v = h[:, :d], h[:, d : 2 * d], h[:, 2 * d :]
        nh, dh = self.cfg.n_heads, self.cfg.d_head
        q = q[:, : nh * dh].reshape(-1, nh, dh)
        k = k[:, : nh * dh].reshape(-1, nh, dh)
        v = v[:, : nh * dh].reshape(-1, nh, dh)
        cos, sin = _rope_tables(np.asarray(positions, dtype=np.int64), dh)
        q = _rotate(q, cos, sin)
        k = _rotate(k, cos, sin)
        return q, k, v

    def _queries(self, tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Rotary queries for tokens whose K/V rows already live in the cache.

        This is exactly the query third of :meth:`_qkv`, but it never touches
        the key/value projections, so scoring a cached position does not
        recompute its cache row.
        """
        d = self.cfg.d_model
        nh, dh = self.cfg.n_heads, self.cfg.d_head
        h = self._ln(self.emb[tokens]) @ self.w_qkv[:d].T + self.b_qkv[:d]
        q = h.reshape(-1, nh, dh)
        cos, sin = _rope_tables(np.asarray(positions, dtype=np.int64), dh)
        return _rotate(q, cos, sin)

    def _head(self, tokens: np.ndarray, attn: np.ndarray) -> np.ndarray:
        """Vocabulary logits for ``tokens`` given their attention output."""
        x = self._ln(self.emb[tokens])
        x = x + attn @ self.w_o.T + self.b_o
        x = x + np.maximum(self._ln(x) @ self.w_ff.T + self.b_ff, 0.0) @ self.w_proj.T + self.b_proj
        return self._ln(x) @ self.emb.T

    def forward(
        self, tokens: np.ndarray, memory: KV | None = None
    ) -> tuple[np.ndarray, KV]:
        """Run ``tokens`` after ``memory`` and return logits plus the full cache.

        ``memory`` holds cached rows for earlier absolute positions.  The
        returned logits are for the supplied tokens (the last row is the
        distribution for the token that would come next), and the returned
        :class:`KV` is ``memory`` followed by the rows of ``tokens``.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        n = tokens.shape[0]
        if n == 0:
            raise ValueError("forward needs at least one token")
        if np.any(tokens < 0) or np.any(tokens >= self.cfg.vocab_size):
            raise ValueError("token id out of range")

        nh, dh = self.cfg.n_heads, self.cfg.d_head
        mem = memory if memory is not None else empty_kv(nh, dh)
        n_mem = len(mem)
        if n_mem:
            start = int(mem.positions[-1]) + 1
        else:
            start = 0
        positions = np.arange(start, start + n, dtype=np.int64)

        q, k_new, v_new = self._qkv(tokens, positions)
        full = mem.concat(KV(k=k_new, v=v_new, positions=positions))
        k_all, v_all, all_pos = full.k, full.v, full.positions
        if np.any(all_pos[1:] <= all_pos[:-1]):
            raise ValueError("KV positions must be strictly increasing")

        # Attention scores depend only on relative offset thanks to RoPE.
        scale = 1.0 / np.sqrt(dh)
        scores = np.einsum("qhd,khd->hqk", q, k_all) * scale
        causal = all_pos[None, :] <= positions[:, None]
        scores = np.where(causal[None, :, :], scores, -np.inf)
        probs = _softmax(scores, axis=-1)
        attn = np.einsum("hqk,khd->qhd", probs, v_all)
        attn = attn.reshape(n, self.cfg.d_model)

        logits = self._head(tokens, attn)
        return logits, KV(k=k_all, v=v_all, positions=all_pos)

    def score_continuation(
        self,
        memory: KV,
        last_token: int,
        targets: np.ndarray,
        visible: np.ndarray | None = None,
    ) -> np.ndarray:
        """Natural log-probabilities of ``targets`` continuing ``memory``.

        Convenience wrapper around :meth:`score_contexts` for a single
        visibility mask (``None`` means the whole prefix is visible).
        """
        return self.score_contexts(memory, last_token, targets, [visible])[0]

    def score_contexts(
        self,
        memory: KV,
        last_token: int,
        targets: np.ndarray,
        visibles: list,
    ) -> np.ndarray:
        """Log-probabilities of ``targets`` under each visibility mask.

        ``memory`` holds the cached K/V rows of the prefix and ``last_token``
        is the token at its last position.  The first target is scored from
        that position's query; every later target is scored from the query of
        the target before it, placed at the next absolute position
        (teacher forcing).  Each entry of ``visibles`` is either ``None``
        (whole prefix visible) or a boolean mask over the prefix rows; hidden
        rows keep their RoPE positions and every other position is unchanged,
        so masking only changes attention visibility.

        The cached rows are read as-is and never recomputed: K/V is computed
        only for ``targets[:-1]`` (the targets a later query conditions on),
        shared across all contexts, and only the query is computed for the
        last prefix token.  Nothing is allocated or stored.  Returns an array
        of shape ``(len(visibles), len(targets))`` of natural
        log-probabilities.
        """
        targets = np.asarray(targets, dtype=np.int64)
        m = int(targets.shape[0])
        if m == 0:
            raise ValueError("score_contexts needs at least one target")
        if np.any(targets < 0) or np.any(targets >= self.cfg.vocab_size):
            raise ValueError("token id out of range")
        n_mem = len(memory)
        if n_mem == 0:
            raise ValueError("score_contexts needs a non-empty prefix")
        masks = []
        for visible in visibles:
            if visible is None:
                masks.append(np.ones(n_mem, dtype=bool))
                continue
            keep = np.asarray(visible, dtype=bool)
            if keep.shape != (n_mem,):
                raise ValueError("visible must have one entry per memory row")
            masks.append(keep)

        nh, dh = self.cfg.n_heads, self.cfg.d_head
        last_pos = int(memory.positions[-1])
        # Queries: the last prefix token (its K/V is already cached, so only
        # its query is computed) followed by every target but the last, which
        # also contribute the fresh K/V rows later queries condition on.
        q_last = self._queries(
            np.array([last_token], dtype=np.int64),
            np.array([last_pos], dtype=np.int64),
        )
        if m > 1:
            new_pos = np.arange(last_pos + 1, last_pos + m, dtype=np.int64)
            q_new, k_new, v_new = self._qkv(targets[:-1], new_pos)
        else:
            new_pos = np.zeros((0,), dtype=np.int64)
            q_new = np.zeros((0, nh, dh), dtype=np.float64)
            k_new = np.zeros((0, nh, dh), dtype=np.float64)
            v_new = np.zeros((0, nh, dh), dtype=np.float64)

        q = np.concatenate([q_last, q_new], axis=0)
        q_pos = np.concatenate([np.array([last_pos], dtype=np.int64), new_pos])
        q_tokens = np.concatenate([np.array([last_token], dtype=np.int64), targets[:-1]])
        scale = 1.0 / np.sqrt(dh)

        out = np.empty((len(masks), m), dtype=np.float64)
        for i, keep in enumerate(masks):
            k_all = np.concatenate([memory.k[keep], k_new], axis=0)
            v_all = np.concatenate([memory.v[keep], v_new], axis=0)
            pos_all = np.concatenate([memory.positions[keep], new_pos], axis=0)

            scores = np.einsum("qhd,khd->hqk", q, k_all) * scale
            causal = pos_all[None, :] <= q_pos[:, None]
            scores = np.where(causal[None, :, :], scores, -np.inf)
            probs = _softmax(scores, axis=-1)
            attn = np.einsum("hqk,khd->qhd", probs, v_all)
            attn = attn.reshape(m, self.cfg.d_model)

            logits = self._head(q_tokens, attn)
            logp = _log_softmax(logits)
            out[i] = logp[np.arange(m), targets]
        return out
