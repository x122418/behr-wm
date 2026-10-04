"""HTTP service for frozen-actor TextWorld distribution consistency."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.reward.textworld_consistency_engine import TextWorldConsistencyEngine


logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: str


class ConsistencyRequest(BaseModel):
    history: list[Message]
    real_observation: str = Field(min_length=1)
    predicted_observation: str = Field(min_length=1)
    expert_action: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    reward_metric: Literal["union_topk_other_js", "full_vocab_js"]


class ConsistencyMicroBatcher:
    """Coalesce concurrent HTTP requests into bounded engine batches."""

    def __init__(
        self,
        engine: TextWorldConsistencyEngine,
        batch_wait_ms: float,
        max_batch_size: int,
    ) -> None:
        if batch_wait_ms <= 0:
            raise ValueError("batch_wait_ms must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.engine = engine
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        self.max_batch_size = max_batch_size
        self._pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._runner: asyncio.Task | None = None

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._lock:
            self._pending.append((request, future))
            if self._runner is None or self._runner.done():
                self._runner = asyncio.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        await asyncio.sleep(self.batch_wait_seconds)
        while True:
            async with self._lock:
                batch = self._pending[: self.max_batch_size]
                del self._pending[: self.max_batch_size]
                if not batch:
                    self._runner = None
                    return
            await self._score_batch(batch)
            async with self._lock:
                if not self._pending:
                    self._runner = None
                    return

    async def _score_batch(
        self, batch: list[tuple[dict[str, Any], asyncio.Future]]
    ) -> None:
        requests = [request for request, _ in batch]
        try:
            results = await asyncio.to_thread(self.engine.score_batch, requests)
            if len(results) != len(batch):
                raise RuntimeError("consistency engine returned the wrong batch size")
        except ValueError:
            await self._score_individually(batch)
            return
        except Exception as error:
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)
            return
        for result, (_, future) in zip(results, batch, strict=True):
            if not future.done():
                future.set_result(result)

    async def _score_individually(
        self, batch: list[tuple[dict[str, Any], asyncio.Future]]
    ) -> None:
        for request, future in batch:
            if future.done():
                continue
            try:
                result = await asyncio.to_thread(self.engine.score, **request)
            except Exception as error:
                future.set_exception(error)
            else:
                future.set_result(result)


def create_app(
    engine: TextWorldConsistencyEngine | None = None,
    batch_wait_ms: float = 0.0,
    max_batch_size: int = 32,
) -> FastAPI:
    """Create an app around an injected ready scorer engine."""
    if batch_wait_ms < 0:
        raise ValueError("batch_wait_ms must be non-negative")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    app = FastAPI(title="TextWorld Actor Consistency Scorer")
    app.state.engine = engine
    app.state.batch_wait_ms = batch_wait_ms
    app.state.max_batch_size = max_batch_size
    app.state.batcher = (
        ConsistencyMicroBatcher(engine, batch_wait_ms, max_batch_size)
        if engine is not None and batch_wait_ms > 0
        else None
    )

    def ready_engine() -> TextWorldConsistencyEngine:
        current = app.state.engine
        if current is None:
            raise HTTPException(status_code=503, detail="scorer is not ready")
        return current

    @app.get("/health")
    async def health() -> dict[str, Any]:
        current = ready_engine()
        return {
            "status": "ok",
            "model": current.model_name,
            "device": str(current.device),
            "dtype": current.dtype,
            "top_k": current.top_k,
            "batch_wait_ms": app.state.batch_wait_ms,
            "max_batch_size": app.state.max_batch_size,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        current = ready_engine()
        return {
            "object": "list",
            "data": [
                {
                    "id": current.model_name,
                    "object": "model",
                    "top_k": current.top_k,
                    "dtype": current.dtype,
                }
            ],
        }

    @app.post("/v1/behavior-consistency")
    async def behavior_consistency(request: ConsistencyRequest):
        current = ready_engine()
        if request.top_k != current.top_k:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"request top_k={request.top_k} does not match "
                    f"server top_k={current.top_k}"
                ),
            )
        request_kwargs = {
            "history": [message.model_dump() for message in request.history],
            "real_observation": request.real_observation,
            "predicted_observation": request.predicted_observation,
            "expert_action": request.expert_action,
            "reward_metric": request.reward_metric,
        }
        try:
            if app.state.batcher is not None:
                return await app.state.batcher.submit(request_kwargs)
            return await asyncio.to_thread(
                current.score,
                **request_kwargs,
            )
        except ValueError as error:
            logger.warning(
                "actor consistency request rejected: %s: %s",
                type(error).__name__,
                error,
            )
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception:
            request_id = uuid.uuid4().hex
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "actor consistency inference failed",
                    "request_id": request_id,
                },
            )

    return app


def load_engine(model_path: str, top_k: int) -> TextWorldConsistencyEngine:
    """Load one local frozen actor checkpoint onto the visible CUDA device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = str(Path(model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    return TextWorldConsistencyEngine(model, tokenizer, resolved, top_k=top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--max-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms must be non-negative")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be positive")
    if not Path(args.model).exists():
        parser.error(f"local model path not found: {args.model}")

    import uvicorn

    uvicorn.run(
        create_app(
            load_engine(args.model, args.top_k),
            batch_wait_ms=args.batch_wait_ms,
            max_batch_size=args.max_batch_size,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
