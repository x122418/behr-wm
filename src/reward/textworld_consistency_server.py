"""HTTP service for frozen-actor TextWorld distribution consistency."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.reward.textworld_consistency_engine import TextWorldConsistencyEngine


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


def create_app(engine: TextWorldConsistencyEngine | None = None) -> FastAPI:
    """Create an app around an injected ready scorer engine."""
    app = FastAPI(title="TextWorld Actor Consistency Scorer")
    app.state.engine = engine

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
        try:
            return await asyncio.to_thread(
                current.score,
                history=[message.model_dump() for message in request.history],
                real_observation=request.real_observation,
                predicted_observation=request.predicted_observation,
                expert_action=request.expert_action,
                reward_metric=request.reward_metric,
            )
        except ValueError as error:
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
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not Path(args.model).exists():
        parser.error(f"local model path not found: {args.model}")

    import uvicorn

    uvicorn.run(create_app(load_engine(args.model, args.top_k)), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
