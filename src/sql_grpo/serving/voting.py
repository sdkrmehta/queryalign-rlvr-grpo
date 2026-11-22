"""Majority@8 Consensus Engine: Equivalence clustering and structured abstention."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sql_grpo.env.result_normalizer import _canonical_row
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser


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


class MajorityVotingEngine:
    """Clusters candidate outputs by result equivalence and selects consensus SQL."""

    def __init__(self, sandbox: SQLiteSandbox, min_plurality_threshold: int = 4):
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
        checker = ASTPolicyChecker(allowed_tables=valid_tables)

        valid_executions: List[Tuple[str, List[Tuple[Any, ...]]]] = []
        safe_count = 0

        for comp in candidate_completions:
            parsed = SQLCompletionParser.parse(comp)
            if not parsed.is_valid_format or not parsed.sql:
                continue

            ast_res = checker.validate(parsed.sql)
            if not ast_res.is_safe:
                continue
            safe_count += 1

            # Execute in sandbox
            exec_res = self.sandbox.execute(
                db_path=db_path,
                database_id=database_id,
                sql=parsed.sql,
                allowed_tables=valid_tables,
            )

            if exec_res.success and exec_res.complete and not exec_res.truncated:
                valid_executions.append((parsed.sql, exec_res.rows or []))

        if not valid_executions:
            return VotingOutcome(
                selected_sql=None,
                confidence_score=0.0,
                cluster_size=0,
                total_candidates=total_candidates,
                num_safe=safe_count,
                num_executed=0,
                is_abstained=True,
                abstention_reason="ALL_CANDIDATES_FAILED_OR_UNSAFE",
            )

        # Cluster by multiset signature of rows
        clusters: Dict[Tuple, List[str]] = defaultdict(list)
        for sql_str, rows in valid_executions:
            # Hashable representation of row multiset
            sorted_canonical_rows = tuple(sorted([_canonical_row(r) for r in rows], key=lambda x: str(x)))
            clusters[sorted_canonical_rows].append(sql_str)

        # Sort clusters by size
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
        largest_cluster = sorted_clusters[0]
        largest_size = len(largest_cluster)

        # Check for tie with second largest cluster
        if len(sorted_clusters) > 1 and len(sorted_clusters[1]) == largest_size:
            return VotingOutcome(
                selected_sql=largest_cluster[0],
                confidence_score=largest_size / total_candidates,
                cluster_size=largest_size,
                total_candidates=total_candidates,
                num_safe=safe_count,
                num_executed=len(valid_executions),
                is_abstained=True,
                abstention_reason="NO_CONSENSUS_TIE",
            )

        # Check plurality threshold
        if largest_size < self.min_plurality:
            return VotingOutcome(
                selected_sql=largest_cluster[0],
                confidence_score=largest_size / total_candidates,
                cluster_size=largest_size,
                total_candidates=total_candidates,
                num_safe=safe_count,
                num_executed=len(valid_executions),
                is_abstained=True,
                abstention_reason="LOW_CONFIDENCE_THRESHOLD_UNMET",
            )

        return VotingOutcome(
            selected_sql=largest_cluster[0],
            confidence_score=largest_size / total_candidates,
            cluster_size=largest_size,
            total_candidates=total_candidates,
            num_safe=safe_count,
            num_executed=len(valid_executions),
            is_abstained=False,
        )
