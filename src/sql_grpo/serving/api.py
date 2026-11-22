"""FastAPI Service: Production inference endpoint with defense-in-depth safety checks."""

import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sql_grpo.data.schema_serializer import SchemaSerializer
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser
from sql_grpo.serving.generator import SQLGenerator
from sql_grpo.serving.schemas import (
    ConsensusStatus,
    ExecutionStatus,
    TextToSQLRequest,
    TextToSQLResponse,
    ValidationStatus,
)

app = FastAPI(
    title="Autonomous Text-to-SQL Engine (RLVR + GRPO)",
    description="Guarded API service for Text-to-SQL generation and execution.",
    version="0.2.0",
)

registry = DatabaseRegistry()
sandbox = SQLiteSandbox()
serializer = SchemaSerializer()
generator = SQLGenerator(registry=registry, sandbox=sandbox)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sql-grpo-serving"}


@app.get("/databases")
def get_databases():
    return {"databases": registry.list_available_databases()}


@app.get("/databases/{db_id}/schema")
def get_schema(db_id: str):
    if not registry.is_registered(db_id):
        raise HTTPException(
            status_code=400,
            detail=f"Database '{db_id}' is not in approved registry.",
        )
    db_path = registry.get_path(db_id)
    info = serializer.serialize_database(db_path)
    text = serializer.to_compact_text(info)
    return {"database_id": db_id, "schema_text": text, "metadata": info}


@app.post("/generate", response_model=TextToSQLResponse)
def generate(req: TextToSQLRequest):
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    if not registry.is_registered(req.database_id):
        raise HTTPException(
            status_code=400,
            detail=f"Database '{req.database_id}' is not in approved registry.",
        )

    mode = "Majority@8 (Self-Consistency)" if req.num_candidates > 1 else "Greedy (Pass@1)"
    result = generator.execute_and_evaluate(
        question=req.question,
        database_id=req.database_id,
        inference_mode=mode,
    )

    if not result.format_valid:
        return TextToSQLResponse(
            request_id=req_id,
            status="format_error",
            sql=None,
            validation=ValidationStatus(
                format_valid=False,
                ast_safe=False,
                error_code="TAG_SYNTAX_ERROR",
            ),
            latency_ms={"total": round((time.perf_counter() - start_time) * 1000.0, 2)},
        )

    if not result.ast_safe:
        return TextToSQLResponse(
            request_id=req_id,
            status="ast_rejected",
            sql=result.selected_sql,
            validation=ValidationStatus(
                format_valid=True,
                ast_safe=False,
                error_code=result.ast_error_code,
                error_message=result.ast_error_message,
            ),
            latency_ms={"total": round((time.perf_counter() - start_time) * 1000.0, 2)},
        )

    exec_status = None
    result_preview = None
    columns = None
    if req.execute and result.execution_result:
        res = result.execution_result
        exec_status = ExecutionStatus(
            success=res.success,
            complete=res.complete,
            row_count=res.row_count_seen,
            truncated=res.truncated,
            error_code=res.error_code,
            error_message=res.error_message_safe,
        )
        if res.rows is not None:
            result_preview = [list(r) for r in res.rows[:10]]
            columns = res.columns

    consensus_status = None
    if result.voting_outcome:
        vo = result.voting_outcome
        consensus_status = ConsensusStatus(
            enabled=True,
            cluster_size=vo.cluster_size,
            total_candidates=vo.total_candidates,
            confidence_score=round(vo.confidence_score, 3),
            abstention_reason=vo.abstention_reason,
        )

    total_time = (time.perf_counter() - start_time) * 1000.0
    return TextToSQLResponse(
        request_id=req_id,
        status="success",
        sql=result.selected_sql,
        validation=ValidationStatus(format_valid=True, ast_safe=True),
        execution=exec_status,
        result_preview=result_preview,
        columns=columns,
        consensus=consensus_status,
        latency_ms={"total": round(total_time, 2), "sandbox": result.latency_ms},
    )
