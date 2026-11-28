"""Shared completion extraction, batch caching, and schema-argument adapters for RLVR rewards."""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from sql_grpo.safety.ast_policy import ASTPolicyChecker, ASTValidationResult
from sql_grpo.safety.sql_parser import ParsedCompletion, SQLCompletionParser


def _extract_message_content(content: Any) -> str:
    """Extracts plain string text from a message content string or multipart chunk list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def extract_completion_text(completion: Any) -> str:
    """Normalizes TRL completion representations to plain text.

    Supports:
        - Plain strings (`"<sql>SELECT ...</sql>"`)
        - Single message dicts (`{"role": "assistant", "content": "..."}`)
        - Conversational message lists (`[{"role": "assistant", "content": "..."}]`)
        - Multipart content blocks
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return _extract_message_content(completion["content"])
        return str(completion)
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                return _extract_message_content(msg["content"])
        for msg in reversed(completion):
            if isinstance(msg, dict) and "content" in msg:
                return _extract_message_content(msg["content"])
        if completion and isinstance(completion[-1], str):
            return completion[-1]
        return "" if not completion else str(completion)
    if completion is None:
        return ""
    return str(completion)


def resolve_tables_arg(valid_tables: Optional[List[Any]], idx: int) -> Optional[Set[str]]:
    """Extracts per-sample table allowlist without discarding explicit empty allowlists."""
    if valid_tables is None or idx >= len(valid_tables):
        return None
    entry = valid_tables[idx]
    if entry is None:
        return None
    if isinstance(entry, str):
        try:
            loaded = json.loads(entry)
            if isinstance(loaded, list):
                return {str(t) for t in loaded}
        except Exception:
            pass
    return {str(t) for t in entry}


def resolve_columns_arg(
    valid_columns: Optional[List[Any]],
    valid_columns_json: Optional[List[Any]],
    idx: int,
) -> Optional[Union[Set[str], Dict[str, Set[str]]]]:
    """Extracts per-sample column allowlist from Python objects or JSON metadata column."""
    raw: Any = None
    if valid_columns is not None and idx < len(valid_columns):
        raw = valid_columns[idx]
    elif valid_columns_json is not None and idx < len(valid_columns_json):
        raw = valid_columns_json[idx]

    if raw is None:
        return None

    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            raw = json.loads(raw)
        except Exception:
            return None

    if isinstance(raw, Mapping):
        return {str(tbl): {str(col) for col in cols} for tbl, cols in raw.items()}
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {str(col) for col in raw}
    return None


def _freeze_schema_key(
    valid_tables: Optional[Set[str]],
    valid_columns: Optional[Union[Set[str], Mapping[str, Set[str]]]],
) -> Tuple[Any, Any]:
    tables_key = tuple(sorted(t.lower() for t in valid_tables)) if valid_tables is not None else None
    if valid_columns is None:
        cols_key = None
    elif isinstance(valid_columns, Mapping):
        cols_key = tuple(
            (tbl.lower(), tuple(sorted(c.lower() for c in cols)))
            for tbl, cols in sorted(valid_columns.items(), key=lambda kv: kv[0].lower())
        )
    else:
        cols_key = tuple(sorted(c.lower() for c in valid_columns))
    return tables_key, cols_key


@dataclass
class NormalizedCandidate:
    """Normalized, format-parsed, and AST-validated candidate object."""

    raw_text: str
    parsed: ParsedCompletion
    ast_result: Optional[ASTValidationResult] = None

    @property
    def is_format_valid(self) -> bool:
        return self.parsed.is_valid_format and bool(self.parsed.sql)

    @property
    def sql(self) -> Optional[str]:
        return self.parsed.sql

    @property
    def is_ast_safe(self) -> bool:
        return bool(self.ast_result and self.ast_result.is_safe)


def normalize_and_validate_candidate(
    completion: Any,
    valid_tables: Optional[Set[str]] = None,
    valid_columns: Optional[Union[Set[str], Mapping[str, Set[str]]]] = None,
    cache: Optional[Dict[Tuple[str, Any, Any], NormalizedCandidate]] = None,
) -> NormalizedCandidate:
    """Extracts text, validates tags, and runs AST allowlist checks with optional batch caching."""
    text = extract_completion_text(completion)
    if cache is not None:
        t_key, c_key = _freeze_schema_key(valid_tables, valid_columns)
        cache_key = (text, t_key, c_key)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    parsed = SQLCompletionParser.parse(text)
    ast_result: Optional[ASTValidationResult] = None
    if parsed.is_valid_format and parsed.sql:
        checker = ASTPolicyChecker(allowed_tables=valid_tables, allowed_columns=valid_columns)
        ast_result = checker.validate(parsed.sql)

    cand = NormalizedCandidate(raw_text=text, parsed=parsed, ast_result=ast_result)
    if cache is not None:
        cache[(text, t_key, c_key)] = cand
    return cand
