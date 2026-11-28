"""Pydantic request and response schemas for FastAPI serving."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TextToSQLRequest(BaseModel):
    database_id: str = Field(..., min_length=1, max_length=128, description="Approved database ID")
    question: str = Field(..., min_length=1, max_length=500, description="Natural language question")
    num_candidates: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of candidate queries (1 for greedy, up to 8 for Majority@8)",
    )
    execute: bool = Field(default=True, description="Whether to execute the chosen query in the sandbox")
    model_choice: str = Field(
        default="SFT + GRPO (RLVR)",
        max_length=80,
        description="Local model variant: 'SFT + GRPO (RLVR)', 'QLoRA SFT Baseline', or 'Qwen2.5-Coder-1.5B (Zero-Shot)'",
    )

    @field_validator("database_id", "question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be empty or whitespace-only.")
        return stripped


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
    num_safe: Optional[int] = None
    num_executed: Optional[int] = None


class TextToSQLResponse(BaseModel):
    request_id: str
    status: str
    sql: Optional[str] = None
    execution_verified: bool = True
    verification_label: Optional[str] = None
    validation: ValidationStatus
    execution: Optional[ExecutionStatus] = None
    result_preview: Optional[List[List[Any]]] = None
    columns: Optional[List[str]] = None
    consensus: Optional[ConsensusStatus] = None
    candidate_count: Optional[int] = None
    safe_candidate_count: Optional[int] = None
    executed_candidate_count: Optional[int] = None
    adapter_sha256: Optional[str] = None
    latency_ms: Dict[str, float] = Field(default_factory=dict)
