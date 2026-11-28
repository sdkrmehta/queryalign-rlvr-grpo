"""Standalone Benchmark Runner: Evaluates base, SFT, and GRPO variants identically."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

from sql_grpo.data.schema_serializer import SchemaSerializer, build_chat_prompt
from sql_grpo.data.validate_dataset import load_records
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.result_normalizer import compare_results
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.evaluation.metrics import compute_benchmark_metrics
from sql_grpo.safety.ast_policy import ASTPolicyChecker
from sql_grpo.safety.sql_parser import SQLCompletionParser
from sql_grpo.serving.inference import (
    LocalSQLInferenceBackend,
    SQLInferenceBackend,
    load_serving_config,
)
from sql_grpo.serving.voting import MajorityVotingEngine

MODEL_VARIANT_TO_CHOICE = {
    "base": "Qwen2.5-Coder-1.5B (Zero-Shot)",
    "sft": "SFT Baseline",
    "grpo": "SFT + GRPO (RLVR)",
}


def evaluate_examples(
    examples: List[Dict[str, Any]],
    inference_backend: SQLInferenceBackend,
    model_variant: str = "grpo",
    registry: Optional[DatabaseRegistry] = None,
    sandbox: Optional[SQLiteSandbox] = None,
    majority_k: int = 1,
    min_plurality_threshold: int = 3,
    run_id: Optional[str] = None,
    ordered: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates a list of manifest records through fixed safety & canonical equivalence checks."""
    reg = registry or DatabaseRegistry()
    sb = sandbox or SQLiteSandbox(row_limit=5000, timeout_seconds=2.0)
    serializer = SchemaSerializer()
    voting_engine = MajorityVotingEngine(sandbox=sb, min_plurality_threshold=min_plurality_threshold)
    model_choice = MODEL_VARIANT_TO_CHOICE.get(model_variant.lower(), model_variant)
    active_run_id = run_id or f"{model_variant}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    cfg_hash = hashlib.sha256(
        json.dumps({"model_variant": model_variant, "majority_k": majority_k, "ordered": ordered}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    records: List[Dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        ex_id = str(ex.get("example_id") or f"ex_{idx:06d}")
        db_id = str(ex["db_id"])
        db_path = reg.get_path(db_id)
        schema_info = serializer.serialize_database(db_path)
        valid_tables = set(ex.get("valid_tables") or schema_info["tables"].keys())
        valid_columns = ex.get("valid_columns") or {
            t: {c["name"] for c in meta["columns"]} for t, meta in schema_info["tables"].items()
        }
        valid_columns_sets = {t: set(cs) for t, cs in valid_columns.items()}

        prompt_messages = ex.get("prompt")
        if not prompt_messages:
            schema_txt = serializer.to_compact_text(schema_info, question=ex.get("question"))
            prompt_messages = build_chat_prompt(question=ex["question"], serialized_schema=schema_txt)

        gold_raw = json.loads(ex.get("gold_result_json") or "[]")
        gold_rows = [tuple(r) if isinstance(r, list) else (r,) for r in gold_raw]

        t_total_start = time.perf_counter()
        inf_out = inference_backend.generate(
            prompt_messages=prompt_messages,
            num_candidates=majority_k,
            model_choice=model_choice,
        )
        gen_ms = inf_out.generation_latency_ms
        raw_comp = inf_out.completions[0] if inf_out.completions else ""

        t_val_start = time.perf_counter()
        parsed = SQLCompletionParser.parse(raw_comp)
        fmt_ok = parsed.is_valid_format and bool(parsed.sql)
        ast_ok = False
        err_code = parsed.error_code if not fmt_ok else None
        if fmt_ok and parsed.sql:
            ast_res = ASTPolicyChecker(allowed_tables=valid_tables, allowed_columns=valid_columns_sets).validate(parsed.sql)
            ast_ok = ast_res.is_safe
            if not ast_ok:
                err_code = ast_res.error_code
        val_ms = (time.perf_counter() - t_val_start) * 1000.0

        exec_ok = False
        complete = False
        truncated = False
        match = False
        exec_ms = 0.0
        if fmt_ok and ast_ok and parsed.sql:
            exec_res = sb.execute(
                db_path=db_path,
                database_id=db_id,
                sql=parsed.sql,
                allowed_tables=valid_tables,
                allowed_columns=valid_columns_sets,
            )
            exec_ms = exec_res.latency_ms
            truncated = exec_res.truncated
            complete = exec_res.complete
            exec_ok = exec_res.success and exec_res.complete and not exec_res.truncated
            if not exec_ok:
                err_code = exec_res.error_code
            elif exec_res.rows is not None:
                match = compare_results(
                    candidate_rows=exec_res.rows,
                    gold_rows=gold_rows,
                    ordered=ordered,
                    candidate_truncated=exec_res.truncated,
                )

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        row_out: Dict[str, Any] = {
            "run_id": active_run_id,
            "model_variant": model_variant,
            "example_id": ex_id,
            "db_id": db_id,
            "generation_config_hash": cfg_hash,
            "adapter_sha256": inf_out.adapter_sha256,
            "raw_completion": raw_comp,
            "sql_extracted": parsed.sql if fmt_ok else None,
            "format_valid": fmt_ok,
            "ast_safe": ast_ok,
            "execution_success": exec_ok,
            "result_complete": complete,
            "truncated": truncated,
            "execution_match": match,
            "gold_is_empty": bool(ex.get("gold_is_empty", len(gold_rows) == 0)),
            "latency_generation_ms": round(gen_ms, 3),
            "latency_validation_ms": round(val_ms, 3),
            "latency_execution_ms": round(exec_ms, 3),
            "latency_total_ms": round(total_ms, 3),
            "error_code": err_code,
        }

        if majority_k > 1:
            t_maj_start = time.perf_counter()
            vote = voting_engine.elect(
                candidate_completions=inf_out.completions,
                db_path=db_path,
                database_id=db_id,
                valid_tables=valid_tables,
                valid_columns=valid_columns_sets,
            )
            maj_lat_ms = gen_ms + (time.perf_counter() - t_maj_start) * 1000.0
            maj_match = False
            maj_exec_ok = False
            maj_sql = None if vote.is_abstained else vote.selected_sql
            if not vote.is_abstained and vote.selected_execution_result and vote.selected_execution_result.rows is not None:
                maj_exec_ok = vote.selected_execution_result.success and vote.selected_execution_result.complete
                maj_match = compare_results(
                    candidate_rows=vote.selected_execution_result.rows,
                    gold_rows=gold_rows,
                    ordered=ordered,
                    candidate_truncated=vote.selected_execution_result.truncated,
                )
            status_label = (
                "CONSENSUS_ELECTED"
                if not vote.is_abstained
                else (
                    "LOW_CONFIDENCE"
                    if vote.abstention_reason == "LOW_CONFIDENCE_THRESHOLD_UNMET"
                    else ("NO_CONSENSUS_TIE" if vote.abstention_reason == "NO_CONSENSUS_TIE" else "ALL_FAILED")
                )
            )
            row_out.update(
                {
                    "majority8_status": status_label,
                    "majority8_sql": maj_sql,
                    "majority8_exec_ok": maj_exec_ok,
                    "majority8_correct": maj_match,
                    "majority8_cluster_size": vote.cluster_size,
                    "majority8_cluster_sizes": sorted((vote.clusters_info or {}).values(), reverse=True),
                    "majority8_n_valid_candidates": vote.num_executed,
                    "majority8_latency_ms": round(maj_lat_ms, 3),
                }
            )

        records.append(row_out)

    summary = {
        "run_id": active_run_id,
        "model_variant": model_variant,
        "majority_k": majority_k,
        "metrics": compute_benchmark_metrics(records),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QueryAlign model variants under fixed sandbox protocol.")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--model", choices=["base", "sft", "grpo"], default="grpo")
    parser.add_argument("--manifest", default="data/processed/dev_manifest.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--majority-k", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    eval_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir or eval_cfg.get("output", {}).get("reports_dir", "./reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(Path(args.manifest))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    backend = LocalSQLInferenceBackend(config=load_serving_config())
    records, summary = evaluate_examples(
        examples=rows,
        inference_backend=backend,
        model_variant=args.model,
        majority_k=args.majority_k,
    )

    preds_path = out_dir / f"{args.model}_eval_predictions.jsonl"
    preds_path.write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""), encoding="utf-8")
    summary_path = out_dir / f"{args.model}_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
