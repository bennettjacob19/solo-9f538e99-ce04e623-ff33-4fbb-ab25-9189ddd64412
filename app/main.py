"""HTTP API for the block-shared KV cache.

Endpoint map (see ``/docs`` for the interactive version):

============================  ====================================================
``GET  /health``              liveness probe
``GET  /model``               model and cache geometry
``GET  /stats``               global block occupancy
``POST /sessions``            create a session (optionally prefilled)
``GET  /sessions``            list sessions
``GET  /sessions/{id}``       session details, block map and tokens
``POST /sessions/{id}/tokens`` append token ids, returns logits for the new ones
``POST /sessions/{id}/fork``  branch from a prefix of an existing session
``DELETE /sessions/{id}``     drop a session and reclaim exclusive blocks
============================  ====================================================
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from .model import Model, ModelConfig
from .store import (
    BlockStore,
    InsufficientBlocks,
    InvalidRequest,
    SessionNotFound,
)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None


class CreateSessionRequest(BaseModel):
    tokens: list[int] = Field(
        default_factory=list,
        description="Optional prompt token ids to prefill and cache.",
        examples=[[1, 2, 3]],
    )


class AppendRequest(BaseModel):
    tokens: list[int] = Field(
        ...,
        min_length=1,
        description="Token ids to append. Must all be valid before any is stored.",
        examples=[[7, 8, 9]],
    )


class ForkRequest(BaseModel):
    prefix_length: int = Field(
        ...,
        ge=0,
        description="Number of leading positions to share with the child session.",
        examples=[2],
    )


class SessionCreated(BaseModel):
    session_id: str
    length: int
    num_blocks: int
    shared_blocks: int
    blocks_allocated: int
    start_position: int | None = None
    positions: list[int] = Field(default_factory=list)
    logits: list[list[float]] | None = None


class AppendResponse(BaseModel):
    session_id: str
    start_position: int
    positions: list[int]
    length: int
    blocks_allocated: int
    logits: list[list[float]] = Field(
        description="Vocabulary logits for each appended position, in order."
    )


class ForkResponse(BaseModel):
    session_id: str
    parent_id: str
    length: int
    shared_blocks: int
    blocks_allocated: int = 0


class DeleteResponse(BaseModel):
    session_id: str
    blocks_freed: int
    blocks_used_global: int


class StatsResponse(BaseModel):
    block_size: int
    max_blocks: int
    used_blocks: int
    free_blocks: int
    cached_positions: int
    shared_blocks: int
    sessions: int


def _error(status: int, code: str, message: str, **extra) -> HTTPException:
    detail = {"error": code, "message": message}
    detail.update(extra)
    return HTTPException(status_code=status, detail=detail)


def create_app(
    *,
    max_blocks: int | None = None,
    model_config: ModelConfig | None = None,
) -> FastAPI:
    cfg = model_config or ModelConfig()
    model = Model(cfg)
    store = BlockStore(
        model,
        max_blocks=_int_env("MAX_BLOCKS", 64) if max_blocks is None else max_blocks,
        block_size=model.n_pos_per_block,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model
        app.state.store = store
        yield

    app = FastAPI(
        title="Block-Shared KV Cache Inference Service",
        version="1.0.0",
        description=(
            "Local inference over a single-layer causal Transformer with rotary "
            "position encoding. Sessions own an ordered list of KV blocks. A fork "
            "references the parent's blocks instead of copying them, so shared "
            "prefixes cost no new blocks. Appends compute attention only for the "
            "new positions, and their logits match a full-sequence forward pass."
        ),
        lifespan=lifespan,
    )
    app.state.model = model
    app.state.store = store

    # ------------------------------------------------------------------ infra

    @app.get("/health", tags=["infra"], summary="Liveness probe")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model", tags=["infra"], summary="Model and cache geometry")
    def model_info() -> dict:
        info = model.config_summary()
        info["fingerprint"] = model.fingerprint()
        info["block_size"] = store.block_size
        info["max_blocks"] = store.max_blocks
        return info

    @app.get(
        "/stats",
        response_model=StatsResponse,
        tags=["infra"],
        summary="Global block occupancy",
    )
    def stats() -> dict:
        return store.stats()

    # --------------------------------------------------------------- sessions

    @app.post(
        "/sessions",
        response_model=SessionCreated,
        status_code=201,
        tags=["sessions"],
        summary="Create a session, optionally prefilled with tokens",
    )
    def create_session(body: CreateSessionRequest) -> dict:
        try:
            info, appended = store.create_session(body.tokens)
        except InvalidRequest as exc:
            raise _error(422, "invalid_token", str(exc)) from None
        except InsufficientBlocks as exc:
            raise _error(
                409,
                "insufficient_blocks",
                str(exc),
                required_blocks=exc.required,
                available_blocks=exc.available,
                max_blocks=exc.max_blocks,
            ) from None

        info["blocks_allocated"] = info["num_blocks"]
        if appended is not None:
            info["start_position"] = appended.start_position
            info["positions"] = appended.positions
            info["logits"] = appended.logits.tolist()
        return info

    @app.get("/sessions", tags=["sessions"], summary="List sessions")
    def list_sessions() -> dict:
        sessions = store.list_sessions()
        return {"count": len(sessions), "sessions": sessions}

    @app.get("/sessions/{session_id}", tags=["sessions"], summary="Session details")
    def get_session(session_id: str = Path(...)) -> dict:
        try:
            return store.get_session(session_id)
        except SessionNotFound:
            raise _error(404, "session_not_found", f"unknown session {session_id}") from None

    @app.post(
        "/sessions/{session_id}/tokens",
        response_model=AppendResponse,
        tags=["sessions"],
        summary="Append token ids and get logits for the new positions",
    )
    def append_tokens(
        body: AppendRequest, session_id: str = Path(...)
    ) -> dict:
        try:
            result = store.append(session_id, body.tokens)
        except SessionNotFound:
            raise _error(404, "session_not_found", f"unknown session {session_id}") from None
        except InvalidRequest as exc:
            raise _error(422, "invalid_token", str(exc)) from None
        except InsufficientBlocks as exc:
            raise _error(
                409,
                "insufficient_blocks",
                str(exc),
                required_blocks=exc.required,
                available_blocks=exc.available,
                max_blocks=exc.max_blocks,
            ) from None
        return {
            "session_id": result.session_id,
            "start_position": result.start_position,
            "positions": result.positions,
            "length": result.length,
            "blocks_allocated": result.blocks_allocated,
            "logits": result.logits.tolist(),
        }

    @app.post(
        "/sessions/{session_id}/fork",
        response_model=ForkResponse,
        status_code=201,
        tags=["sessions"],
        summary="Branch a new session from a prefix, sharing blocks",
    )
    def fork_session(body: ForkRequest, session_id: str = Path(...)) -> dict:
        try:
            result = store.fork(session_id, body.prefix_length)
        except SessionNotFound:
            raise _error(404, "session_not_found", f"unknown session {session_id}") from None
        except InvalidRequest as exc:
            raise _error(422, "invalid_prefix", str(exc)) from None
        return {
            "session_id": result.session_id,
            "parent_id": result.parent_id,
            "length": result.length,
            "shared_blocks": result.shared_blocks,
            "blocks_allocated": 0,
        }

    @app.delete(
        "/sessions/{session_id}",
        response_model=DeleteResponse,
        tags=["sessions"],
        summary="Delete a session, reclaiming only its exclusive blocks",
    )
    def delete_session(session_id: str = Path(...)) -> dict:
        try:
            result = store.delete(session_id)
        except SessionNotFound:
            raise _error(404, "session_not_found", f"unknown session {session_id}") from None
        return {
            "session_id": result.session_id,
            "blocks_freed": result.blocks_freed,
            "blocks_used_global": result.blocks_used,
        }

    return app


app = create_app()
