"""Pydantic request and response schemas for FastAPI serving."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TextToSQLRequest(BaseModel):
    database_id: str = Field(..., description="Approved database ID")
    question: str = Field(..., max_length=500, description="Natural language question")
    num_candidates: int = Field(default=1, ge=1, le=8, description="Number of candidate queries (1 for greedy, 8 for Majority@8)")
    execute: bool = Field(default=True, description="Whether to execute the chosen query in the sandbox")


class ValidationStatus(BaseModel):
    format_valid: bool
    ast_safe: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ExecutionStatus(BaseModel):
    success: bool
    complete: bool
    row_count: int
    truncated: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ConsensusStatus(BaseModel):
    enabled: bool
    cluster_size: int
    total_candidates: int
    confidence_score: float
    abstention_reason: Optional[str] = None


class TextToSQLResponse(BaseModel):
    request_id: str
    status: str
    sql: Optional[str] = None
    validation: ValidationStatus
    execution: Optional[ExecutionStatus] = None
    result_preview: Optional[List[List[Any]]] = None
    columns: Optional[List[str]] = None
    consensus: Optional[ConsensusStatus] = None
    latency_ms: Dict[str, float] = Field(default_factory=dict)
