"""FastAPI Service: Guarded local inference endpoint with defense-in-depth safety checks."""

from contextlib import asynccontextmanager
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Response

from sql_grpo.data.schema_serializer import SchemaSerializer
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.serving.generator import SQLGenerator
from sql_grpo.serving.inference import ModelUnavailableError
from sql_grpo.serving.schemas import (
    ConsensusStatus,
    ExecutionStatus,
    TextToSQLRequest,
    TextToSQLResponse,
    ValidationStatus,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

_MAX_CONCURRENT_INFERENCE = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCE", "2")))
_INFERENCE_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_INFERENCE)

registry = DatabaseRegistry()
sandbox = SQLiteSandbox()
serializer = SchemaSerializer()
generator = SQLGenerator(registry=registry, sandbox=sandbox)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("QueryAlign serving API starting (max_concurrent=%d).", _MAX_CONCURRENT_INFERENCE)
    yield
    logger.info("QueryAlign serving API shutting down cleanly.")


app = FastAPI(
    title="Autonomous Text-to-SQL Engine (RLVR + GRPO)",
    description="Guarded local-inference API service for Text-to-SQL generation and execution.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe: confirms the HTTP service process is alive without exposing internals."""
    return {"status": "ok", "service": "sql-grpo-serving"}


@app.get("/ready")
def ready(response: Response) -> Dict[str, Any]:
    """Readiness probe: distinguishes process alive from local model/database readiness."""
    dbs = registry.list_available_databases()
    model_ready = generator.inference_backend.is_ready()
    is_ready = bool(dbs) and model_ready
    if not is_ready:
        response.status_code = 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "model_ready": model_ready,
        "databases_count": len(dbs),
    }


@app.get("/databases")
def get_databases() -> Dict[str, Any]:
    return {"databases": registry.list_available_databases()}


@app.get("/databases/{db_id}/schema")
def get_schema(db_id: str) -> Dict[str, Any]:
    if not registry.is_registered(db_id):
        raise HTTPException(
            status_code=400,
            detail=f"Database '{db_id}' is not in approved registry.",
        )
    try:
        db_path = registry.get_path(db_id)
        info = serializer.serialize_database(db_path)
        text = serializer.to_compact_text(info)
        return {"database_id": db_id, "schema_text": text, "metadata": info}
    except Exception as exc:
        logger.exception("Schema serialization failed for database_id=%s: %s", db_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Failed to inspect schema metadata.")


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg or type(exc).__name__ == "OutOfMemoryError"


@app.post("/generate", response_model=TextToSQLResponse)
def generate(req: TextToSQLRequest) -> TextToSQLResponse:
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    if not registry.is_registered(req.database_id):
        raise HTTPException(
            status_code=400,
            detail=f"Database '{req.database_id}' is not in approved registry.",
        )

    acquired = _INFERENCE_SEMAPHORE.acquire(timeout=0.05)
    if not acquired:
        logger.warning("request_id=%s rejected due to inference queue capacity", req_id)
        raise HTTPException(
            status_code=503,
            detail="Inference queue is currently full; please retry shortly.",
        )

    try:
        mode = "Majority@8 (Self-Consistency)" if req.num_candidates > 1 else "Greedy (Pass@1)"
        result = generator.execute_and_evaluate(
            question=req.question,
            database_id=req.database_id,
            model_choice=req.model_choice,
            inference_mode=mode,
            execute=req.execute,
            num_candidates=req.num_candidates,
        )
    except ModelUnavailableError as exc:
        logger.error("request_id=%s model unavailable: %s", req_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Local Text-to-SQL model is currently unavailable.",
        )
    except Exception as exc:
        if _is_oom_error(exc):
            logger.error("request_id=%s inference OOM: %s", req_id, type(exc).__name__)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            raise HTTPException(
                status_code=503,
                detail="Inference device ran out of memory; reduce candidate count and retry.",
            )
        logger.exception("request_id=%s unexpected generation error: %s", req_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Internal generation error.")
    finally:
        _INFERENCE_SEMAPHORE.release()

    total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    latencies: Dict[str, float] = {
        "generation": result.generation_latency_ms,
        "validation": result.validation_latency_ms,
        "execution": result.execution_latency_ms,
        "total": total_ms,
    }

    cand_count = len(result.candidates) if result.candidates else 1
    safe_count = (
        sum(1 for c in result.candidates if c.is_ast_safe)
        if result.candidates
        else (1 if result.ast_safe else 0)
    )
    exec_count = (
        sum(1 for c in result.candidates if c.is_executed)
        if result.candidates
        else (1 if (result.execution_result and result.execution_result.success) else 0)
    )

    consensus_status = None
    if result.voting_outcome:
        vo = result.voting_outcome
        consensus_status = ConsensusStatus(
            enabled=True,
            cluster_size=vo.cluster_size,
            total_candidates=vo.total_candidates,
            confidence_score=round(vo.confidence_score, 3),
            abstention_reason=vo.abstention_reason,
            num_safe=vo.num_safe,
            num_executed=vo.num_executed,
        )

    if not result.format_valid:
        resp = TextToSQLResponse(
            request_id=req_id,
            status="format_error",
            sql=None,
            execution_verified=False,
            validation=ValidationStatus(
                format_valid=False,
                ast_safe=False,
                error_code=result.ast_error_code or "TAG_SYNTAX_ERROR",
            ),
            consensus=consensus_status,
            candidate_count=cand_count,
            safe_candidate_count=safe_count,
            executed_candidate_count=exec_count,
            adapter_sha256=result.adapter_sha256,
            latency_ms=latencies,
        )
        logger.info("request_id=%s db=%s status=%s total_ms=%.2f", req_id, req.database_id, resp.status, total_ms)
        return resp

    if not result.ast_safe:
        resp = TextToSQLResponse(
            request_id=req_id,
            status="ast_rejected",
            sql=None,
            execution_verified=False,
            validation=ValidationStatus(
                format_valid=True,
                ast_safe=False,
                error_code=result.ast_error_code,
                error_message=result.ast_error_message,
            ),
            consensus=consensus_status,
            candidate_count=cand_count,
            safe_candidate_count=safe_count,
            executed_candidate_count=exec_count,
            adapter_sha256=result.adapter_sha256,
            latency_ms=latencies,
        )
        logger.info("request_id=%s db=%s status=%s err=%s total_ms=%.2f", req_id, req.database_id, resp.status, result.ast_error_code, total_ms)
        return resp

    if result.voting_outcome and result.voting_outcome.is_abstained:
        resp = TextToSQLResponse(
            request_id=req_id,
            status="abstained",
            sql=None,
            execution_verified=False,
            verification_label=result.voting_outcome.abstention_reason or "ABSTAINED",
            validation=ValidationStatus(format_valid=True, ast_safe=True),
            consensus=consensus_status,
            candidate_count=cand_count,
            safe_candidate_count=safe_count,
            executed_candidate_count=exec_count,
            adapter_sha256=result.adapter_sha256,
            latency_ms=latencies,
        )
        logger.info(
            "request_id=%s db=%s status=abstained reason=%s total_ms=%.2f",
            req_id,
            req.database_id,
            result.voting_outcome.abstention_reason,
            total_ms,
        )
        return resp

    if not req.execute:
        resp = TextToSQLResponse(
            request_id=req_id,
            status="not_execution_verified",
            sql=result.selected_sql,
            execution_verified=False,
            verification_label="not execution-verified",
            validation=ValidationStatus(format_valid=True, ast_safe=True),
            execution=None,
            result_preview=None,
            columns=None,
            consensus=consensus_status,
            candidate_count=cand_count,
            safe_candidate_count=safe_count,
            executed_candidate_count=0,
            adapter_sha256=result.adapter_sha256,
            latency_ms=latencies,
        )
        logger.info("request_id=%s db=%s status=%s total_ms=%.2f", req_id, req.database_id, resp.status, total_ms)
        return resp

    exec_status = None
    result_preview = None
    columns = None
    resp_status = "success"
    exec_verified = False

    if result.execution_result:
        res = result.execution_result
        exec_status = ExecutionStatus(
            success=res.success,
            complete=res.complete,
            row_count=res.row_count_seen,
            truncated=res.truncated,
            error_code=res.error_code,
            error_message=res.error_message_safe,
        )
        if res.truncated or not res.complete:
            resp_status = "truncated"
        elif not res.success:
            resp_status = "execution_error"
        else:
            exec_verified = True
            if res.rows is not None:
                result_preview = [list(r) for r in res.rows[:10]]
                columns = res.columns

    resp = TextToSQLResponse(
        request_id=req_id,
        status=resp_status,
        sql=result.selected_sql,
        execution_verified=exec_verified,
        validation=ValidationStatus(format_valid=True, ast_safe=True),
        execution=exec_status,
        result_preview=result_preview,
        columns=columns,
        consensus=consensus_status,
        candidate_count=cand_count,
        safe_candidate_count=safe_count,
        executed_candidate_count=exec_count,
        adapter_sha256=result.adapter_sha256,
        latency_ms=latencies,
    )
    logger.info(
        "request_id=%s db=%s status=%s rows=%s total_ms=%.2f",
        req_id,
        req.database_id,
        resp.status,
        exec_status.row_count if exec_status else 0,
        total_ms,
    )
    return resp
