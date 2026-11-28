"""Schema Serializer: Generates compact, token-budgeted, deterministic M-Schema representations."""

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

_SENSITIVE_COL_RE = re.compile(r"(password|passwd|secret|token|api_key|private_key|ssn)", re.IGNORECASE)
_BLOB_TYPE_RE = re.compile(r"\b(blob|bytea|binary|varbinary)\b", re.IGNORECASE)


def _quote_sqlite_ident(identifier: str) -> str:
    """Safely quotes an arbitrary SQLite identifier using doubled double-quotes."""
    return '"' + identifier.replace('"', '""') + '"'


class SchemaSerializer:
    """Serializes SQLite schemas into deterministic, privacy-filtered, token-budgeted prompt context."""

    def __init__(
        self,
        max_samples_per_col: int = 3,
        max_sample_chars: int = 50,
        skip_blob_samples: bool = True,
        skip_sensitive_samples: bool = True,
    ):
        self.max_samples_per_col = max_samples_per_col
        self.max_sample_chars = max_sample_chars
        self.skip_blob_samples = skip_blob_samples
        self.skip_sensitive_samples = skip_sensitive_samples

    def serialize_database(
        self,
        db_path: Path,
        table_filter: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Extracts structural metadata and bounded safe sample values directly from read-only SQLite."""
        resolved = Path(db_path).resolve(strict=True)
        uri = resolved.as_uri() + "?mode=ro"

        schema_info: Dict[str, Any] = {
            "tables": {},
            "foreign_keys": [],
            "schema_hash": "",
        }

        norm_filter = {t.lower() for t in table_filter} if table_filter is not None else None
        conn: Optional[sqlite3.Connection] = None
        cursor: Optional[sqlite3.Cursor] = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
            conn.execute("PRAGMA query_only = ON")
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            )
            all_tables = [row[0] for row in cursor.fetchall()]

            target_tables = [
                t for t in all_tables if (norm_filter is None or t.lower() in norm_filter)
            ]

            for table in target_tables:
                q_table = _quote_sqlite_ident(table)
                cursor.execute(f"PRAGMA table_info({q_table});")
                columns_info: List[Dict[str, Any]] = []
                pks: List[str] = []

                for col in cursor.fetchall():
                    _cid, cname, ctype, _notnull, _dflt_value, pk = col
                    col_type = (ctype or "TEXT").strip() or "TEXT"
                    is_pk = bool(pk)
                    if is_pk:
                        pks.append(cname)

                    samples: List[str] = []
                    should_skip_samples = (
                        (self.skip_blob_samples and bool(_BLOB_TYPE_RE.search(col_type)))
                        or (self.skip_sensitive_samples and bool(_SENSITIVE_COL_RE.search(cname)))
                        or self.max_samples_per_col <= 0
                    )
                    if not should_skip_samples:
                        q_col = _quote_sqlite_ident(cname)
                        try:
                            cursor.execute(
                                f"SELECT DISTINCT {q_col} FROM {q_table} "
                                f"WHERE {q_col} IS NOT NULL ORDER BY {q_col} LIMIT {int(self.max_samples_per_col)};"
                            )
                            for s_row in cursor.fetchall():
                                val = s_row[0]
                                if isinstance(val, (bytes, bytearray, memoryview)):
                                    continue
                                val_str = str(val).replace("\r", " ").replace("\n", " ").strip()
                                if len(val_str) > self.max_sample_chars:
                                    val_str = val_str[: self.max_sample_chars] + "..."
                                samples.append(val_str)
                        except Exception:
                            samples = []

                    columns_info.append(
                        {
                            "name": cname,
                            "type": col_type,
                            "is_pk": is_pk,
                            "samples": samples,
                        }
                    )

                cursor.execute(f"PRAGMA foreign_key_list({q_table});")
                fks: List[Dict[str, str]] = []
                for fk in cursor.fetchall():
                    # (id, seq, table, from, to, on_update, on_delete, match)
                    fks.append(
                        {
                            "from_table": table,
                            "from_column": fk[3],
                            "to_table": fk[2],
                            "to_column": fk[4],
                        }
                    )

                schema_info["tables"][table] = {
                    "columns": columns_info,
                    "primary_keys": pks,
                }
                schema_info["foreign_keys"].extend(fks)
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        schema_info["foreign_keys"].sort(
            key=lambda fk: (fk["from_table"], fk["from_column"], fk["to_table"], fk["to_column"])
        )
        canonical_struct = {
            "tables": {
                t: {
                    "columns": [
                        {"name": c["name"], "type": c["type"], "is_pk": c["is_pk"]}
                        for c in meta["columns"]
                    ],
                    "primary_keys": meta["primary_keys"],
                }
                for t, meta in sorted(schema_info["tables"].items())
            },
            "foreign_keys": schema_info["foreign_keys"],
        }
        schema_info["schema_hash"] = hashlib.sha256(
            json.dumps(canonical_struct, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return schema_info

    @staticmethod
    def _prioritize_tables(
        schema_info: Dict[str, Any],
        question: Optional[str],
    ) -> List[str]:
        """Orders tables by relevance to question tokens while preserving connected FK join partners."""
        all_tables = sorted(schema_info["tables"].keys())
        if not question:
            return all_tables

        q_tokens = set(re.findall(r"[a-z0-9_]+", question.lower()))
        matched: Set[str] = set()
        for tname, tdata in schema_info["tables"].items():
            t_tokens = set(re.findall(r"[a-z0-9_]+", tname.lower()))
            if t_tokens & q_tokens or tname.lower() in question.lower():
                matched.add(tname)
                continue
            for col in tdata["columns"]:
                c_tokens = set(re.findall(r"[a-z0-9_]+", col["name"].lower()))
                if c_tokens & q_tokens:
                    matched.add(tname)
                    break

        # Expand 1-hop FK neighborhood so join paths stay intact under tight token budgets
        fk_expanded = set(matched)
        for fk in schema_info.get("foreign_keys", []):
            ft, tt = fk["from_table"], fk["to_table"]
            if ft in matched or tt in matched:
                fk_expanded.add(ft)
                fk_expanded.add(tt)

        prioritized = [t for t in all_tables if t in fk_expanded]
        remainder = [t for t in all_tables if t not in fk_expanded]
        return prioritized + remainder

    @staticmethod
    def _format_table_block(
        table_name: str,
        t_data: Dict[str, Any],
        include_samples: bool = True,
    ) -> str:
        lines = [f"  Table: {table_name}"]
        for c in t_data["columns"]:
            pk_tag = ", PK" if c["is_pk"] else ""
            sample_str = ""
            if include_samples and c.get("samples"):
                sample_str = f" [examples: {', '.join(repr(s) for s in c['samples'])}]"
            lines.append(f"    - {c['name']} ({c['type']}{pk_tag}){sample_str}")
        return "\n".join(lines)

    def to_compact_text(
        self,
        schema_info: Dict[str, Any],
        external_knowledge: Optional[str] = None,
        max_tokens: Optional[int] = None,
        question: Optional[str] = None,
        chars_per_token: float = 3.5,
    ) -> str:
        """Converts structured schema info into formatted M-Schema prompt text within optional token budget."""
        ordered_tables = sorted(schema_info["tables"].keys())
        knowledge_block = f"\nEXTERNAL KNOWLEDGE:\n  {external_knowledge.strip()}" if external_knowledge else ""
        char_budget = int(max_tokens * chars_per_token) if max_tokens and max_tokens > 0 else None

        def _render(table_order: List[str], include_samples: bool, keep_fks_for: Set[str]) -> str:
            lines = ["DATABASE SCHEMA:"]
            for t in table_order:
                lines.append(
                    self._format_table_block(
                        t, schema_info["tables"][t], include_samples=include_samples
                    )
                )
            fks = [
                fk
                for fk in schema_info.get("foreign_keys", [])
                if fk["from_table"] in keep_fks_for and fk["to_table"] in keep_fks_for
            ]
            if fks:
                lines.append("  Foreign Keys:")
                for fk in fks:
                    lines.append(
                        f"    - {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                    )
            if knowledge_block:
                lines.append(knowledge_block)
            return "\n".join(lines)

        full_text = _render(ordered_tables, include_samples=True, keep_fks_for=set(ordered_tables))
        if char_budget is None or len(full_text) <= char_budget:
            return full_text

        # Step 1 under tight budget: drop example values first before omitting any table/FK path
        no_samples = _render(ordered_tables, include_samples=False, keep_fks_for=set(ordered_tables))
        if len(no_samples) <= char_budget:
            return no_samples

        # Step 2: greedily include prioritized tables + FK neighbors until char_budget is reached
        prioritized = self._prioritize_tables(schema_info, question=question)
        chosen: List[str] = []
        for tbl in prioritized:
            candidate = chosen + [tbl]
            rendered = _render(candidate, include_samples=False, keep_fks_for=set(candidate))
            if chosen and len(rendered) > char_budget:
                break
            chosen.append(tbl)

        omitted_count = len(ordered_tables) - len(chosen)
        rendered_chosen = _render(chosen, include_samples=False, keep_fks_for=set(chosen))
        if omitted_count > 0:
            rendered_chosen += f"\n  [Schema truncated: {omitted_count} unprioritized table(s) omitted to fit token budget]"
        return rendered_chosen


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
        user_content = (
            f"Database dialect: SQLite\n\n{serialized_schema}\n\n"
            f"External knowledge: {external_knowledge}\n\nQuestion: {question}"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
