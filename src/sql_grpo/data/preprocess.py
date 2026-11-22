"""Data Preprocessor: Preprocesses raw Spider and BIRD benchmarks into verified training manifests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sql_grpo.data.schema_serializer import SchemaSerializer, build_chat_prompt
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox


class DataPreprocessor:
    """Preprocesses Text-to-SQL datasets and computes gold query ground truth."""

    def __init__(
        self,
        data_root: str | Path = "./data",
        gold_timeout_seconds: float = 1.0,
    ):
        self.data_root = Path(data_root).resolve()
        self.registry = DatabaseRegistry(data_root=self.data_root)
        self.sandbox = SQLiteSandbox(timeout_seconds=gold_timeout_seconds)
        self.serializer = SchemaSerializer()

    def process_raw_item(
        self,
        example_id: str,
        dataset_name: str,
        db_id: str,
        question: str,
        gold_sql: str,
        external_knowledge: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Validates and processes a single question-SQL pair."""
        try:
            db_path = self.registry.get_path(db_id)
        except Exception:
            return None

        # Execute gold query
        gold_res = self.sandbox.execute(
            db_path=db_path,
            database_id=db_id,
            sql=gold_sql,
        )

        if not gold_res.success:
            gold_status = "error"
            gold_rows = None
            is_empty = False
        else:
            gold_rows = gold_res.rows
            is_empty = (len(gold_rows) == 0) if gold_rows is not None else False
            gold_status = "empty" if is_empty else "success"

        # Serialize schema
        schema_dict = self.serializer.serialize_database(db_path)
        serialized_schema_text = self.serializer.to_compact_text(schema_dict, external_knowledge=external_knowledge)

        # Build prompt
        prompt_messages = build_chat_prompt(
            question=question,
            serialized_schema=serialized_schema_text,
            external_knowledge=external_knowledge,
        )

        valid_tables = list(schema_dict["tables"].keys())

        return {
            "example_id": example_id,
            "dataset": dataset_name,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "gold_status": gold_status,
            "gold_is_empty": is_empty,
            "gold_row_count": len(gold_rows) if gold_rows is not None else 0,
            "gold_result_json": json.dumps(gold_rows) if gold_rows is not None else "[]",
            "gold_latency_ms": round(gold_res.latency_ms, 2),
            "valid_tables": valid_tables,
            "prompt": prompt_messages,
        }
