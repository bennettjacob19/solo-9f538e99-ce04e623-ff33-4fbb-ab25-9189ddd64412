# Block-Shared KV Cache Inference Service

A local inference service for a small, single-layer causal Transformer
implemented in pure NumPy, behind a FastAPI HTTP API. Multiple continuations
("sessions") can share the KV cache of a common prompt: forking a session
references the parent's KV blocks instead of copying them, and every append
computes K/V only for the newly added tokens.

## What it does

- **Model**: one causal self-attention layer with rotary position encoding,
  LayerNorm, and a small MLP head. Weights are deterministic (seeded), so the
  same sequence always yields the same logits.
- **Blocks**: the KV cache is allocated in blocks of **4 positions**
  (`BLOCK_SIZE`). The total number of blocks is capped by `MAX_BLOCKS`.
- **Sessions**: an ordered list of references into blocks. Appending computes
  K/V for the new tokens only and returns vocab logits for those positions.
- **Forking**: `POST /sessions/{id}/fork` with a `prefix_length` creates a child
  that shares the parent's blocks. Sharing allocates **no new blocks**; the
  global block usage is unchanged.
- **Rejection**: if an append would exceed the global block budget, the *entire*
  append is rejected (HTTP 409). The session's sequence and the block table are
  left exactly as they were.
- **Deletion**: deleting a session frees only the blocks no other session
  references; other sessions keep working.
- **Concurrency**: all mutating operations are serialized by a single lock, so
  concurrent fork/append/delete calls are equivalent to some serial ordering of
  the same calls.

## Correctness guarantee

For any session, the logits returned by an append match a single full-sequence
forward pass over the same tokens to within **1e-5** absolute error. This holds
for prefill, incremental appends, forked prefixes at any length, and after
neighbouring sessions are deleted. It is verified by the test suite.

The reason it is exact: with a single layer, each position's K/V row depends
only on that position's token and its absolute position (RoPE is applied to the
keys at write time), never on other tokens. A cached row is therefore valid in
any sequence that places the token at the same absolute position, which is
exactly what a fork preserves.

## API

Interactive docs are served at **`/docs`** (OpenAPI at `/openapi.json`).

| Method   | Path                        | Purpose |
|----------|-----------------------------|---------|
| `GET`    | `/health`                   | Liveness probe |
| `GET`    | `/model`                    | Model and cache geometry |
| `GET`    | `/stats`                    | Global block occupancy (used / free / max) |
| `POST`   | `/sessions`                 | Create a session; optional `tokens` prefill |
| `GET`    | `/sessions`                 | List sessions |
| `GET`    | `/sessions/{id}`            | Session tokens and block map |
| `POST`   | `/sessions/{id}/tokens`     | Append token ids; returns logits for the new positions |
| `POST`   | `/sessions/{id}/fork`       | Branch from `prefix_length`; shares blocks |
| `DELETE` | `/sessions/{id}`            | Delete; frees only exclusive blocks |

### Typical flow

```bash
# Create a session with a 5-token prompt (occupies 2 blocks).
curl -s -X POST localhost:8000/sessions \
  -H 'content-type: application/json' \
  -d '{"tokens": [1, 2, 3, 4, 5]}'
# -> {"session_id": "...", "length": 5, "num_blocks": 2, "logits": [...]}

# Append tokens; logits are for positions 5.. only.
curl -s -X POST localhost:8000/sessions/$SID/tokens \
  -H 'content-type: application/json' \
  -d '{"tokens": [6, 7]}'
# -> {"start_position": 5, "positions": [5, 6], "logits": [[...], [...]]}

# Fork at prefix 3: no new blocks, /stats used_blocks is unchanged.
curl -s -X POST localhost:8000/sessions/$SID/fork \
  -H 'content-type: application/json' -d '{"prefix_length": 3}'

# Occupancy.
curl -s localhost:8000/stats
```

### Error responses

Errors use a consistent JSON body under `detail`:

```json
{"detail": {"error": "insufficient_blocks",
            "message": "append needs 2 new block(s), only 1 of 8 are free",
            "required_blocks": 2, "available_blocks": 1, "max_blocks": 8}}
```

| Status | `error` | Meaning |
|--------|---------|---------|
| 404 | `session_not_found` | Unknown session id |
| 409 | `insufficient_blocks` | Append exceeds the global block budget |
| 422 | `invalid_token` | Token id not an integer in `[0, vocab_size)` |
| 422 | `invalid_prefix` | `prefix_length` outside `[0, session length]` |

## Running with Docker Compose

```bash
docker compose up --build
# service on http://localhost:8000, docs at http://localhost:8000/docs
```

Configuration via environment:

| Variable     | Default | Meaning |
|--------------|---------|---------|
| `MAX_BLOCKS` | `64`    | Global cap on 4-position KV blocks |
| `PORT`       | `8000`  | Host port published by compose |

```bash
MAX_BLOCKS=256 PORT=9000 docker compose up --build
```

## Running locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

## Tests

```bash
pip install pytest httpx
pytest -q
```

The suite covers:

- `tests/test_model.py` — full vs. chunked forward equivalence, cache shape.
- `tests/test_store.py` — block accounting, fork sharing, rejection atomicity,
  delete reclamation, logits vs. full forward.
- `tests/test_cache_reuse.py` — proves only new tokens compute K/V, and that
  block rows carry the correct absolute positions.
- `tests/test_concurrency.py` — concurrent fork/append/delete are equivalent to
  a serial order; refcount invariants hold; no partial application.
- `tests/test_api.py` — HTTP status codes, error bodies, and end-to-end logits.

## Layout

```
app/model.py   single-layer causal model + KV type
app/store.py   refcounted block store, sessions, fork/append/delete
app/main.py    FastAPI app and schemas
tests/         pytest suite
Dockerfile, docker-compose.yml
```
