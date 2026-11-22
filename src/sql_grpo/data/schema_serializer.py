"""Schema Serializer: Generates compact, deterministic M-Schema representations for prompts."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class SchemaSerializer:
    """Serializes SQLite schemas into token-budgeted prompt context."""

    def __init__(
        self,
        max_samples_per_col: int = 3,
        max_sample_chars: int = 50,
    ):
        self.max_samples_per_col = max_samples_per_col
        self.max_sample_chars = max_sample_chars

    def serialize_database(self, db_path: Path, table_filter: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Extracts structural metadata and safe sample values directly from SQLite."""
        resolved = db_path.resolve(strict=True)
        uri = f"file:{resolved.as_posix()}?mode=ro"

        schema_info: Dict[str, Any] = {
            "tables": {},
            "foreign_keys": [],
        }

        with sqlite3.connect(uri, uri=True, timeout=2.0) as conn:
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()

            # 1. Fetch tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            all_tables = [row[0] for row in cursor.fetchall()]

            target_tables = [t for t in all_tables if (not table_filter or t.lower() in table_filter)]

            for table in target_tables:
                # 2. Columns & Primary Keys via PRAGMA table_info
                cursor.execute(f"PRAGMA table_info('{table}');")
                columns_info = []
                pks = []
                for col in cursor.fetchall():
                    cid, cname, ctype, notnull, dflt_value, pk = col
                    is_pk = bool(pk)
                    if is_pk:
                        pks.append(cname)

                    # Fetch safe sample values
                    samples: List[str] = []
                    try:
                        cursor.execute(
                            f"SELECT DISTINCT \"{cname}\" FROM \"{table}\" WHERE \"{cname}\" IS NOT NULL LIMIT {self.max_samples_per_col};"
                        )
                        for s_row in cursor.fetchall():
                            val = s_row[0]
                            val_str = str(val).strip()
                            if len(val_str) > self.max_sample_chars:
                                val_str = val_str[: self.max_sample_chars] + "..."
                            samples.append(val_str)
                    except Exception:
                        pass

                    columns_info.append({
                        "name": cname,
                        "type": ctype or "TEXT",
                        "is_pk": is_pk,
                        "samples": samples,
                    })

                # 3. Foreign Keys via PRAGMA foreign_key_list
                fks = []
                cursor.execute(f"PRAGMA foreign_key_list('{table}');")
                for fk in cursor.fetchall():
                    # (id, seq, table, from, to, on_update, on_delete, match)
                    fks.append({
                        "from_table": table,
                        "from_column": fk[3],
                        "to_table": fk[2],
                        "to_column": fk[4],
                    })

                schema_info["tables"][table] = {
                    "columns": columns_info,
                    "primary_keys": pks,
                }
                schema_info["foreign_keys"].extend(fks)

        return schema_info

    def to_compact_text(self, schema_info: Dict[str, Any], external_knowledge: Optional[str] = None) -> str:
        """Converts structured schema info into formatted prompt text."""
        lines = []
        lines.append("DATABASE SCHEMA:")

        for table_name, t_data in sorted(schema_info["tables"].items()):
            col_strs = []
            for c in t_data["columns"]:
                pk_tag = ", PK" if c["is_pk"] else ""
                sample_str = f" [examples: {', '.join(repr(s) for s in c['samples'])}]" if c["samples"] else ""
                col_strs.append(f"    - {c['name']} ({c['type']}{pk_tag}){sample_str}")

            lines.append(f"  Table: {table_name}")
            lines.extend(col_strs)

        if schema_info["foreign_keys"]:
            lines.append("  Foreign Keys:")
            for fk in schema_info["foreign_keys"]:
                lines.append(f"    - {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")

        if external_knowledge:
            lines.append(f"\nEXTERNAL KNOWLEDGE:\n  {external_knowledge.strip()}")

        return "\n".join(lines)


SYSTEM_PROMPT = """You generate SQLite SELECT queries for the supplied database.
Rules:
- Use only the supplied schema and task-provided domain evidence.
- Return exactly one read-only SELECT-family SQL statement.
- Do not use DDL, DML, PRAGMA, ATTACH, DETACH or multiple statements.
- Do not invent tables or columns.
- Use the correct join path and predicates for the question.
- If the question cannot be answered from the schema, return a structured abstention.
- Output only the required response structure.

RESPONSE FORMAT:
<sql>SELECT ...</sql>"""


def build_chat_prompt(
    question: str,
    serialized_schema: str,
    external_knowledge: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Builds chat messages conforming to Qwen chat template."""
    user_content = f"Database dialect: SQLite\n\n{serialized_schema}\n\nQuestion: {question}"
    if external_knowledge:
        user_content = f"Database dialect: SQLite\n\n{serialized_schema}\n\nExternal knowledge: {external_knowledge}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
