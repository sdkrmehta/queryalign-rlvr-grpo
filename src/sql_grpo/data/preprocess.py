"""Data Preprocessor: Validates raw Spider/BIRD items into verified manifests and exclusion logs."""

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from sql_grpo.data.schema_serializer import SchemaSerializer, build_chat_prompt
from sql_grpo.data.validate_dataset import DatasetValidator, load_records
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker


class DataPreprocessor:
    """Preprocesses Text-to-SQL datasets and computes verified gold query ground truth."""

    def __init__(
        self,
        data_root: str | Path = "./data",
        gold_timeout_seconds: float = 1.0,
        max_samples_per_col: int = 3,
        max_sample_chars: int = 50,
        max_prompt_tokens: Optional[int] = 1536,
    ):
        self.data_root = Path(data_root).resolve()
        self.registry = DatabaseRegistry(data_root=self.data_root)
        self.sandbox = SQLiteSandbox(timeout_seconds=gold_timeout_seconds)
        self.serializer = SchemaSerializer(
            max_samples_per_col=max_samples_per_col,
            max_sample_chars=max_sample_chars,
        )
        self.max_prompt_tokens = max_prompt_tokens
        self._schema_cache: Dict[Path, Dict[str, Any]] = {}

    def _get_schema_info(self, db_path: Path) -> Dict[str, Any]:
        if db_path not in self._schema_cache:
            self._schema_cache[db_path] = self.serializer.serialize_database(db_path)
        return self._schema_cache[db_path]

    def process_raw_item(
        self,
        example_id: str,
        dataset_name: str,
        db_id: str,
        question: str,
        gold_sql: str,
        split: str = "train",
        external_knowledge: Optional[str] = None,
        result_sidecar_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """Validates and processes a single question-SQL pair with full gold classification."""
        try:
            db_path = self.registry.get_path(db_id)
        except Exception:
            return None

        schema_dict = self._get_schema_info(db_path)
        valid_tables = list(schema_dict["tables"].keys())
        valid_columns = {
            tbl: [col["name"] for col in meta["columns"]]
            for tbl, meta in schema_dict["tables"].items()
        }

        ast_res = ASTPolicyChecker(
            allowed_tables=set(valid_tables),
            allowed_columns={t: set(cs) for t, cs in valid_columns.items()},
        ).validate(gold_sql)

        gold_error_code: Optional[str] = None
        if not ast_res.is_safe:
            gold_status = "ast_rejected"
            gold_error_code = ast_res.error_code
            gold_rows = None
            is_empty = False
            gold_latency_ms = 0.0
        else:
            gold_res = self.sandbox.execute(
                db_path=db_path,
                database_id=db_id,
                sql=gold_sql,
                allowed_tables=set(valid_tables),
                allowed_columns={t: set(cs) for t, cs in valid_columns.items()},
            )
            gold_latency_ms = round(gold_res.latency_ms, 2)
            if not gold_res.success:
                gold_error_code = gold_res.error_code
                gold_status = "timeout" if gold_res.error_code in ("TIMEOUT", "HARD_TIMEOUT_KILLED") else "error"
                gold_rows = None
                is_empty = False
            elif gold_res.truncated or not gold_res.complete:
                gold_error_code = gold_res.error_code or "RESULT_TRUNCATED"
                gold_status = "truncated"
                gold_rows = None
                is_empty = False
            else:
                gold_rows = gold_res.rows
                is_empty = (len(gold_rows) == 0) if gold_rows is not None else False
                gold_status = "empty" if is_empty else "success"

        serialized_schema_text = self.serializer.to_compact_text(
            schema_dict,
            external_knowledge=external_knowledge,
            max_tokens=self.max_prompt_tokens,
            question=question,
        )
        prompt_messages = build_chat_prompt(
            question=question,
            serialized_schema=serialized_schema_text,
            external_knowledge=external_knowledge,
        )

        gold_result_ref: Optional[str] = None
        if result_sidecar_dir is not None and gold_rows is not None:
            result_sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = result_sidecar_dir / f"{example_id}.json"
            sidecar_path.write_text(json.dumps(gold_rows), encoding="utf-8")
            gold_result_ref = str(sidecar_path.relative_to(self.data_root).as_posix()) if sidecar_path.is_relative_to(self.data_root) else str(sidecar_path)

        try:
            db_rel_path = str(db_path.relative_to(self.data_root).as_posix())
        except ValueError:
            db_rel_path = db_path.name

        return {
            "example_id": example_id,
            "dataset": dataset_name,
            "split": split,
            "db_id": db_id,
            "db_relative_path": f"data/{db_rel_path}" if not db_rel_path.startswith("data/") else db_rel_path,
            "dialect": "sqlite",
            "question": question,
            "gold_sql": gold_sql,
            "gold_status": gold_status,
            "gold_error_code": gold_error_code,
            "gold_is_empty": is_empty,
            "gold_row_count": len(gold_rows) if gold_rows is not None else 0,
            "gold_result_json": json.dumps(gold_rows) if gold_rows is not None else "[]",
            "gold_result_ref": gold_result_ref,
            "gold_latency_ms": gold_latency_ms,
            "schema_hash": schema_dict["schema_hash"],
            "valid_tables": valid_tables,
            "valid_columns": valid_columns,
            "prompt": prompt_messages,
            "completion": f"<sql>{gold_sql}</sql>",
        }

    def process_dataset(
        self,
        raw_items: List[Dict[str, Any]],
        dataset_name: str,
        split: str,
        exclude_non_success_for_train: bool = False,
        result_sidecar_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Processes a list of raw items, producing processed records, exclusion log, and summary stats."""
        processed: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        db_counts: Counter[str] = Counter()

        for idx, item in enumerate(raw_items):
            ex_id = str(item.get("example_id") or f"{split}_{idx:06d}")
            db_id = str(item.get("db_id") or "")
            question = str(item.get("question") or "").strip()
            gold_sql = str(item.get("query") or item.get("gold_sql") or "").strip()
            evidence = item.get("evidence") or item.get("external_knowledge")

            if not db_id or not question or not gold_sql:
                status_counts["invalid_input"] += 1
                excluded.append({"example_id": ex_id, "db_id": db_id, "reason": "INVALID_INPUT"})
                continue

            rec = self.process_raw_item(
                example_id=ex_id,
                dataset_name=dataset_name,
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                split=split,
                external_knowledge=evidence,
                result_sidecar_dir=result_sidecar_dir,
            )
            if rec is None:
                status_counts["missing_db"] += 1
                excluded.append({"example_id": ex_id, "db_id": db_id, "reason": "MISSING_OR_UNAPPROVED_DB"})
                continue

            status_counts[rec["gold_status"]] += 1
            db_counts[db_id] += 1

            if exclude_non_success_for_train and split == "train" and rec["gold_status"] in ("error", "timeout", "ast_rejected", "truncated"):
                excluded.append(
                    {
                        "example_id": ex_id,
                        "db_id": db_id,
                        "reason": f"GOLD_{rec['gold_status'].upper()}",
                        "error_code": rec.get("gold_error_code"),
                    }
                )
                continue

            processed.append(rec)

        summary = {
            "dataset": dataset_name,
            "split": split,
            "raw_count": len(raw_items),
            "retained_count": len(processed),
            "excluded_count": len(excluded),
            "distinct_dbs": len(db_counts),
            "gold_status_counts": dict(status_counts),
        }
        return processed, excluded, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess datasets or audit existing manifests.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing manifests without rewriting files.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(cfg.get("data_root", "./data"))
    processed_dir = Path(cfg.get("processed_dir", data_root / "processed"))

    if args.validate_only:
        validator = DatasetValidator(data_root=data_root)
        train_rows = load_records(processed_dir / "train_manifest.jsonl")
        dev_rows = load_records(processed_dir / "dev_manifest.jsonl")
        report = validator.validate_records(train_rows, eval_records=dev_rows)
        print(json.dumps(report.to_dict(), indent=2))
        if not report.passed:
            raise SystemExit(1)
        return

    pre_cfg = cfg.get("preprocessing", {})
    prep = DataPreprocessor(
        data_root=data_root,
        gold_timeout_seconds=float(pre_cfg.get("gold_timeout_seconds", 1.0)),
        max_samples_per_col=int(pre_cfg.get("max_sample_values_per_column", 3)),
        max_sample_chars=int(pre_cfg.get("max_sample_value_chars", 50)),
        max_prompt_tokens=pre_cfg.get("max_prompt_tokens"),
    )
    spider_cfg = cfg.get("datasets", {}).get("spider", {})
    spider_dir = Path(cfg.get("raw_dir", data_root / "raw")) / "spider"
    train_raw = json.loads((spider_dir / spider_cfg.get("train_file", "train_spider.json")).read_text(encoding="utf-8"))
    dev_raw = json.loads((spider_dir / spider_cfg.get("dev_file", "dev.json")).read_text(encoding="utf-8"))

    _, train_excl, train_sum = prep.process_dataset(train_raw, "spider", "train", exclude_non_success_for_train=True)
    _, dev_excl, dev_sum = prep.process_dataset(dev_raw, "spider", "dev", exclude_non_success_for_train=False)
    print(json.dumps({"train": train_sum, "dev": dev_sum, "train_excluded_sample": train_excl[:5], "dev_excluded_sample": dev_excl[:5]}, indent=2))


if __name__ == "__main__":
    main()
