"""Majority@8 Consensus Engine: Equivalence clustering and structured abstention."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sql_grpo.env.result_normalizer import _canonical_row
from sql_grpo.env.sqlite_sandbox import ExecutionResult, SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser


@dataclass
class CandidateEvaluation:
    """Single-pass parse, AST validation, sandbox execution, and cluster metadata."""

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
    execution_result: Optional[ExecutionResult] = None


@dataclass
class VotingOutcome:
    selected_sql: Optional[str]
    confidence_score: float
    cluster_size: int
    total_candidates: int
    num_safe: int
    num_executed: int
    is_abstained: bool
    abstention_reason: Optional[str] = None
    clusters_info: Optional[Dict[int, int]] = None
    selected_execution_result: Optional[ExecutionResult] = None
    candidate_evaluations: List[CandidateEvaluation] = field(default_factory=list)


class MajorityVotingEngine:
    """Clusters candidate outputs by result equivalence and selects consensus SQL."""

    def __init__(self, sandbox: SQLiteSandbox, min_plurality_threshold: int = 3):
        self.sandbox = sandbox
        self.min_plurality = min_plurality_threshold

    def elect(
        self,
        candidate_completions: List[str],
        db_path: Path,
        database_id: str,
        valid_tables: Optional[Set[str]] = None,
    ) -> VotingOutcome:
        total_candidates = len(candidate_completions)
        if total_candidates == 0:
            return VotingOutcome(
                selected_sql=None,
                confidence_score=0.0,
                cluster_size=0,
                total_candidates=0,
                num_safe=0,
                num_executed=0,
                is_abstained=True,
                abstention_reason="ALL_CANDIDATES_FAILED_OR_UNSAFE",
            )

        checker = ASTPolicyChecker(allowed_tables=valid_tables)
        evaluations: List[CandidateEvaluation] = []
        clusters: Dict[Tuple[Any, ...], List[Tuple[CandidateEvaluation, ExecutionResult]]] = defaultdict(list)
        safe_count = 0
        executed_count = 0

        for idx, comp in enumerate(candidate_completions, start=1):
            parsed = SQLCompletionParser.parse(comp)
            if not parsed.is_valid_format or not parsed.sql:
                evaluations.append(
                    CandidateEvaluation(
                        index=idx,
                        raw_completion=comp,
                        sql=None,
                        is_format_valid=False,
                        is_ast_safe=False,
                        ast_error=parsed.error_code,
                    )
                )
                continue

            ast_res = checker.validate(parsed.sql)
            if not ast_res.is_safe:
                evaluations.append(
                    CandidateEvaluation(
                        index=idx,
                        raw_completion=comp,
                        sql=parsed.sql,
                        is_format_valid=True,
                        is_ast_safe=False,
                        ast_error=ast_res.error_code,
                    )
                )
                continue

            safe_count += 1
            exec_res = self.sandbox.execute(
                db_path=db_path,
                database_id=database_id,
                sql=parsed.sql,
                allowed_tables=valid_tables,
            )

            exec_ok = exec_res.success and exec_res.complete and not exec_res.truncated
            ev = CandidateEvaluation(
                index=idx,
                raw_completion=comp,
                sql=parsed.sql,
                is_format_valid=True,
                is_ast_safe=True,
                is_executed=exec_ok,
                execution_error=exec_res.error_code or exec_res.error_message_safe,
                row_count=exec_res.row_count_seen,
                execution_result=exec_res,
            )
            evaluations.append(ev)

            if exec_ok:
                executed_count += 1
                rows = exec_res.rows or []
                sig = tuple(sorted((_canonical_row(r) for r in rows), key=repr))
                clusters[sig].append((ev, exec_res))

        if not clusters:
            return VotingOutcome(
                selected_sql=None,
                confidence_score=0.0,
                cluster_size=0,
                total_candidates=total_candidates,
                num_safe=safe_count,
                num_executed=0,
                is_abstained=True,
                abstention_reason="ALL_CANDIDATES_FAILED_OR_UNSAFE",
                candidate_evaluations=evaluations,
            )

        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        clusters_info: Dict[int, int] = {}
        for cluster_num, members in enumerate(sorted_clusters, start=1):
            clusters_info[cluster_num] = len(members)
            for ev, _ in members:
                ev.cluster_id = cluster_num

        largest_cluster = sorted_clusters[0]
        largest_size = len(largest_cluster)
        rep_eval, rep_exec = largest_cluster[0]

        if len(sorted_clusters) > 1 and len(sorted_clusters[1]) == largest_size:
            return VotingOutcome(
                selected_sql=rep_eval.sql,
                confidence_score=largest_size / total_candidates,
                cluster_size=largest_size,
                total_candidates=total_candidates,
                num_safe=safe_count,
                num_executed=executed_count,
                is_abstained=True,
                abstention_reason="NO_CONSENSUS_TIE",
                clusters_info=clusters_info,
                selected_execution_result=rep_exec,
                candidate_evaluations=evaluations,
            )

        if largest_size < self.min_plurality:
            return VotingOutcome(
                selected_sql=rep_eval.sql,
                confidence_score=largest_size / total_candidates,
                cluster_size=largest_size,
                total_candidates=total_candidates,
                num_safe=safe_count,
                num_executed=executed_count,
                is_abstained=True,
                abstention_reason="LOW_CONFIDENCE_THRESHOLD_UNMET",
                clusters_info=clusters_info,
                selected_execution_result=rep_exec,
                candidate_evaluations=evaluations,
            )

        return VotingOutcome(
            selected_sql=rep_eval.sql,
            confidence_score=largest_size / total_candidates,
            cluster_size=largest_size,
            total_candidates=total_candidates,
            num_safe=safe_count,
            num_executed=executed_count,
            is_abstained=False,
            clusters_info=clusters_info,
            selected_execution_result=rep_exec,
            candidate_evaluations=evaluations,
        )
