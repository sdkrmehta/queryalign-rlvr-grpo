"""Dataset Validator: Verifies manifest schema integrity, DB mappings, SQL parse, and leakage."""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

from sql_grpo.data.schema_serializer import SchemaSerializer
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.safety.ast_policy import ASTPolicyChecker


@dataclass
class DatasetValidationReport:
    total_records: int
    valid_schema_records: int
    missing_fields_count: int
    missing_db_count: int
    unreadable_db_count: int
    duplicate_example_ids: List[str] = field(default_factory=list)
    duplicate_db_question_pairs: int = 0
    invalid_gold_sql_count: int = 0
    gold_status_counts: Dict[str, int] = field(default_factory=dict)
    db_overlap: List[str] = field(default_factory=list)
    cross_split_question_overlap: List[str] = field(default_factory=list)
    issue_samples: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_schema_records": self.valid_schema_records,
            "missing_fields_count": self.missing_fields_count,
            "missing_db_count": self.missing_db_count,
            "unreadable_db_count": self.unreadable_db_count,
            "duplicate_example_ids_count": len(self.duplicate_example_ids),
            "duplicate_example_ids_sample": self.duplicate_example_ids[:10],
            "duplicate_db_question_pairs": self.duplicate_db_question_pairs,
            "invalid_gold_sql_count": self.invalid_gold_sql_count,
            "gold_status_counts": self.gold_status_counts,
            "db_overlap": self.db_overlap,
            "cross_split_question_overlap_count": len(self.cross_split_question_overlap),
            "cross_split_question_overlap_sample": self.cross_split_question_overlap[:10],
            "issue_samples": self.issue_samples[:20],
            "passed": self.passed,
        }


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Loads either a JSON array file or JSONL manifest file."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        data = json.loads(text)
        return list(data) if isinstance(data, list) else []
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class DatasetValidator:
    """Validates raw or processed Text-to-SQL datasets and detects split leakage."""

    def __init__(self, data_root: str | Path = "./data"):
        self.data_root = Path(data_root).resolve()
        self.registry = DatabaseRegistry(data_root=self.data_root)
        self.serializer = SchemaSerializer()

    def validate_records(
        self,
        records: List[Dict[str, Any]],
        eval_records: Optional[List[Dict[str, Any]]] = None,
        check_gold_ast: bool = True,
    ) -> DatasetValidationReport:
        seen_ids: Set[str] = set()
        seen_pairs: Set[Tuple[str, str]] = set()
        dup_ids: List[str] = []
        dup_pairs = 0
        missing_fields = 0
        missing_db = 0
        unreadable_db = 0
        invalid_sql = 0
        valid_records = 0
        status_counts: Counter[str] = Counter()
        issues: List[Dict[str, Any]] = []

        db_schema_cache: Dict[str, Optional[Tuple[Set[str], Dict[str, Set[str]]]]] = {}
        db_readable_cache: Dict[str, bool] = {}

        def _check_single_split(split_rows: List[Dict[str, Any]], split_label: str) -> Tuple[Set[str], Set[str]]:
            nonlocal dup_pairs, missing_fields, missing_db, unreadable_db, invalid_sql, valid_records
            split_dbs: Set[str] = set()
            split_qs: Set[str] = set()

            for idx, rec in enumerate(split_rows):
                ex_id = str(rec.get("example_id") or f"{split_label}_{idx}")
                db_id = (rec.get("db_id") or "").strip()
                question = (rec.get("question") or "").strip()
                gold_sql = (rec.get("gold_sql") or rec.get("query") or "").strip()
                status_counts[str(rec.get("gold_status") or "unknown")] += 1

                if ex_id in seen_ids:
                    dup_ids.append(ex_id)
                    issues.append({"example_id": ex_id, "reason": "DUPLICATE_EXAMPLE_ID"})
                seen_ids.add(ex_id)

                if not db_id or not question or not gold_sql:
                    missing_fields += 1
                    issues.append({"example_id": ex_id, "reason": "MISSING_REQUIRED_FIELD"})
                    continue

                split_dbs.add(db_id)
                split_qs.add(question.lower())

                pair = (db_id.lower(), question.lower())
                if pair in seen_pairs:
                    dup_pairs += 1
                seen_pairs.add(pair)

                # Verify DB path exists and can open read-only
                if db_id not in db_readable_cache:
                    try:
                        db_path = self.registry.get_path(db_id)
                        conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=1.0)
                        conn.execute("PRAGMA query_only = ON")
                        conn.close()
                        db_readable_cache[db_id] = True
                        info = self.serializer.serialize_database(db_path)
                        tables = set(info["tables"].keys())
                        cols = {t: {c["name"] for c in m["columns"]} for t, m in info["tables"].items()}
                        db_schema_cache[db_id] = (tables, cols)
                    except KeyError:
                        db_readable_cache[db_id] = False
                        db_schema_cache[db_id] = None
                        missing_db += 1
                        issues.append({"example_id": ex_id, "db_id": db_id, "reason": "MISSING_DB"})
                        continue
                    except Exception:
                        db_readable_cache[db_id] = False
                        db_schema_cache[db_id] = None
                        unreadable_db += 1
                        issues.append({"example_id": ex_id, "db_id": db_id, "reason": "UNREADABLE_DB"})
                        continue
                elif not db_readable_cache[db_id]:
                    if db_id not in {i.get("db_id") for i in issues if i.get("reason") == "MISSING_DB"}:
                        unreadable_db += 1
                    else:
                        missing_db += 1
                    continue

                if check_gold_ast:
                    schema_entry = db_schema_cache.get(db_id)
                    checker = (
                        ASTPolicyChecker(allowed_tables=schema_entry[0], allowed_columns=schema_entry[1])
                        if schema_entry
                        else ASTPolicyChecker()
                    )
                    ast_res = checker.validate(gold_sql)
                    if not ast_res.is_safe:
                        invalid_sql += 1
                        issues.append(
                            {
                                "example_id": ex_id,
                                "db_id": db_id,
                                "reason": f"GOLD_AST_{ast_res.error_code}",
                            }
                        )
                        continue

                valid_records += 1

            return split_dbs, split_qs

        train_dbs, train_qs = _check_single_split(records, "primary")
        db_overlap: List[str] = []
        q_overlap: List[str] = []
        if eval_records is not None:
            eval_dbs, eval_qs = _check_single_split(eval_records, "eval")
            db_overlap = sorted(train_dbs & eval_dbs)
            q_overlap = sorted(train_qs & eval_qs)

        total = len(records) + (len(eval_records) if eval_records is not None else 0)
        passed = (
            missing_fields == 0
            and missing_db == 0
            and unreadable_db == 0
            and len(dup_ids) == 0
            and len(db_overlap) == 0
        )

        return DatasetValidationReport(
            total_records=total,
            valid_schema_records=valid_records,
            missing_fields_count=missing_fields,
            missing_db_count=missing_db,
            unreadable_db_count=unreadable_db,
            duplicate_example_ids=dup_ids,
            duplicate_db_question_pairs=dup_pairs,
            invalid_gold_sql_count=invalid_sql,
            gold_status_counts=dict(status_counts),
            db_overlap=db_overlap,
            cross_split_question_overlap=q_overlap,
            issue_samples=issues,
            passed=passed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset manifests and leakage controls.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--eval-manifest", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(cfg.get("data_root", "./data"))
    processed_dir = Path(cfg.get("processed_dir", data_root / "processed"))

    train_path = Path(args.train_manifest or processed_dir / "train_manifest.jsonl")
    eval_path = Path(args.eval_manifest or processed_dir / "dev_manifest.jsonl")

    validator = DatasetValidator(data_root=data_root)
    train_rows = load_records(train_path)
    eval_rows = load_records(eval_path) if eval_path.exists() else None
    report = validator.validate_records(train_rows, eval_records=eval_rows)
    print(json.dumps(report.to_dict(), indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
