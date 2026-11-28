"""Text-to-SQL Generation Engine for Serving and Interactive Dashboard.

Executes strictly local open-weight model inference (Qwen2.5-Coder-1.5B-Instruct
with trained QLoRA SFT / GRPO LoRA adapters) coupled with SQLGlot AST allowlisting,
hardened read-only SQLite execution, and Majority@8 consensus clustering.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from sql_grpo.data.schema_serializer import SchemaSerializer, build_chat_prompt
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import ExecutionResult, SQLiteSandbox
from sql_grpo.rewards.composite import RewardBreakdown
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser
from sql_grpo.serving.inference import (
    LocalSQLInferenceBackend,
    SQLInferenceBackend,
    ServingModelConfig,
    load_serving_config,
)
from sql_grpo.serving.voting import MajorityVotingEngine, VotingOutcome


@dataclass
class CandidateResult:
    index: int
    raw_completion: str
    sql: Optional[str]
    is_format_valid: bool
    is_ast_safe: bool
    ast_error: Optional[str] = None
    is_executed: bool = False
    execution_error: Optional[str] = None
    row_count: int = 0
    cluster_id: Optional[int] = None


@dataclass
class GenerationResult:
    question: str
    database_id: str
    model_choice: str
    inference_mode: str
    selected_sql: Optional[str]
    raw_completion: str
    format_valid: bool
    ast_safe: bool
    ast_error_code: Optional[str] = None
    ast_error_message: Optional[str] = None
    execution_result: Optional[ExecutionResult] = None
    voting_outcome: Optional[VotingOutcome] = None
    candidates: List[CandidateResult] = field(default_factory=list)
    rewards: Optional[RewardBreakdown] = None
    latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    adapter_sha256: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    rows: List[Tuple[Any, ...]] = field(default_factory=list)


class SQLGenerator:
    """End-to-end local Text-to-SQL generator, AST validator, and sandbox executor."""

    def __init__(
        self,
        registry: Optional[DatabaseRegistry] = None,
        sandbox: Optional[SQLiteSandbox] = None,
        data_root: Optional[Path | str] = None,
        inference_backend: Optional[SQLInferenceBackend] = None,
        serving_config: Optional[ServingModelConfig] = None,
    ):
        self.config = serving_config or load_serving_config()
        self.registry = registry or DatabaseRegistry(data_root=data_root or "./data")
        self.sandbox = sandbox or SQLiteSandbox()
        self.serializer = SchemaSerializer()
        self.inference_backend: SQLInferenceBackend = inference_backend or LocalSQLInferenceBackend(
            config=self.config
        )
        self.voting_engine = MajorityVotingEngine(
            sandbox=self.sandbox,
            min_plurality_threshold=self.config.min_plurality_threshold,
        )

    def _validate_single_completion(
        self,
        raw_completion: str,
        valid_tables: Set[str],
        valid_columns: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[Optional[str], bool, bool, Optional[str], Optional[str]]:
        parsed = SQLCompletionParser.parse(raw_completion)
        if not parsed.is_valid_format or not parsed.sql:
            return None, False, False, parsed.error_code, "Completion missing valid <sql>...</sql> block"

        checker = ASTPolicyChecker(allowed_tables=valid_tables, allowed_columns=valid_columns)
        ast_res = checker.validate(parsed.sql)
        if not ast_res.is_safe:
            return parsed.sql, True, False, ast_res.error_code, ast_res.error_message

        return parsed.sql, True, True, None, None

    def execute_and_evaluate(
        self,
        question: str,
        database_id: str,
        model_choice: str = "SFT + GRPO (RLVR)",
        inference_mode: str = "Greedy (Pass@1)",
        execute: bool = True,
        num_candidates: Optional[int] = None,
    ) -> GenerationResult:
        """Generate SQL via local open-weight LLM, validate AST, and execute in sandbox."""
        total_start = time.perf_counter()

        db_path = self.registry.get_path(database_id)
        schema_info = self.serializer.serialize_database(db_path)
        valid_tables = set(schema_info["tables"].keys())
        valid_columns = {
            tbl: {col["name"] for col in meta.get("columns", [])}
            for tbl, meta in schema_info.get("tables", {}).items()
        }
        schema_text = self.serializer.to_compact_text(schema_info)
        prompt_messages = build_chat_prompt(question=question, serialized_schema=schema_text)

        resolved_candidates = num_candidates if num_candidates is not None else (
            self.config.num_candidates if "Majority@8" in inference_mode else 1
        )
        is_majority = resolved_candidates > 1

        inf_out = self.inference_backend.generate(
            prompt_messages=prompt_messages,
            num_candidates=resolved_candidates,
            model_choice=model_choice,
        )
        gen_ms = inf_out.generation_latency_ms
        completions = inf_out.completions

        voting_outcome: Optional[VotingOutcome] = None
        candidate_results: List[CandidateResult] = []
        exec_res: Optional[ExecutionResult] = None
        columns: List[str] = []
        rows: List[Tuple[Any, ...]] = []
        selected_sql: Optional[str] = None
        raw_completion = completions[0] if completions else ""
        format_valid = False
        ast_safe = False
        ast_error_code: Optional[str] = None
        ast_error_message: Optional[str] = None
        val_ms = 0.0
        exec_ms = 0.0

        if is_majority and execute:
            t_vote = time.perf_counter()
            voting_outcome = self.voting_engine.elect(
                candidate_completions=completions,
                db_path=db_path,
                database_id=database_id,
                valid_tables=valid_tables,
                valid_columns=valid_columns,
            )
            vote_total_ms = (time.perf_counter() - t_vote) * 1000.0
            exec_ms = sum(
                ev.execution_result.latency_ms
                for ev in voting_outcome.candidate_evaluations
                if ev.execution_result is not None
            )
            val_ms = max(0.0, vote_total_ms - exec_ms)

            for ev in voting_outcome.candidate_evaluations:
                candidate_results.append(
                    CandidateResult(
                        index=ev.index,
                        raw_completion=ev.raw_completion,
                        sql=ev.sql,
                        is_format_valid=ev.is_format_valid,
                        is_ast_safe=ev.is_ast_safe,
                        ast_error=ev.ast_error,
                        is_executed=ev.is_executed,
                        execution_error=ev.execution_error,
                        row_count=ev.row_count,
                        cluster_id=ev.cluster_id,
                    )
                )

            selected_sql = voting_outcome.selected_sql
            if selected_sql:
                raw_completion = f"<sql>{selected_sql}</sql>"
                format_valid = True
                ast_safe = True
                exec_res = voting_outcome.selected_execution_result
                if exec_res and exec_res.success and exec_res.rows is not None:
                    columns = exec_res.columns
                    rows = exec_res.rows
            else:
                format_valid = any(c.is_format_valid for c in candidate_results)
                ast_safe = any(c.is_ast_safe for c in candidate_results)
                if not format_valid:
                    ast_error_code = "TAG_SYNTAX_ERROR"
                elif not ast_safe:
                    first_ast_err = next((c.ast_error for c in candidate_results if c.ast_error), "AST_REJECTED")
                    ast_error_code = first_ast_err
                    ast_error_message = "All candidates failed AST safety validation"
        elif is_majority and not execute:
            # Preview-only mode: parse and AST-validate candidates without executing in SQLite
            t_val = time.perf_counter()
            for idx, comp in enumerate(completions, start=1):
                sql, f_ok, a_ok, err_code, _ = self._validate_single_completion(
                    comp, valid_tables, valid_columns=valid_columns
                )
                candidate_results.append(
                    CandidateResult(
                        index=idx,
                        raw_completion=comp,
                        sql=sql,
                        is_format_valid=f_ok,
                        is_ast_safe=a_ok,
                        ast_error=err_code,
                    )
                )
            val_ms = (time.perf_counter() - t_val) * 1000.0
            first_safe = next((c for c in candidate_results if c.is_ast_safe and c.sql), None)
            if first_safe:
                selected_sql = first_safe.sql
                raw_completion = first_safe.raw_completion
                format_valid = True
                ast_safe = True
            else:
                first_fmt = next((c for c in candidate_results if c.is_format_valid), None)
                if first_fmt:
                    selected_sql = first_fmt.sql
                    raw_completion = first_fmt.raw_completion
                    format_valid = True
                    ast_error_code = first_fmt.ast_error
                else:
                    ast_error_code = "TAG_SYNTAX_ERROR"
        else:
            t_val = time.perf_counter()
            selected_sql, format_valid, ast_safe, ast_error_code, ast_error_message = (
                self._validate_single_completion(
                    raw_completion, valid_tables, valid_columns=valid_columns
                )
            )
            val_ms = (time.perf_counter() - t_val) * 1000.0

            if execute and format_valid and ast_safe and selected_sql:
                exec_res = self.sandbox.execute(
                    db_path=db_path,
                    database_id=database_id,
                    sql=selected_sql,
                    allowed_tables=valid_tables,
                    allowed_columns=valid_columns,
                )
                exec_ms = exec_res.latency_ms
                if exec_res.success and exec_res.rows is not None:
                    columns = exec_res.columns
                    rows = exec_res.rows

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return GenerationResult(
            question=question,
            database_id=database_id,
            model_choice=model_choice,
            inference_mode=inference_mode,
            selected_sql=selected_sql,
            raw_completion=raw_completion,
            format_valid=format_valid,
            ast_safe=ast_safe,
            ast_error_code=ast_error_code,
            ast_error_message=ast_error_message,
            execution_result=exec_res,
            voting_outcome=voting_outcome,
            candidates=candidate_results,
            rewards=None,
            latency_ms=round(total_ms, 2),
            generation_latency_ms=round(gen_ms, 2),
            validation_latency_ms=round(val_ms, 2),
            execution_latency_ms=round(exec_ms, 2),
            adapter_sha256=inf_out.adapter_sha256,
            columns=columns,
            rows=rows,
        )
