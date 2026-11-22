"""SQL Parser: Extracts and validates SQL tags from model completions."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedCompletion:
    is_valid_format: bool
    sql: Optional[str] = None
    is_truncated: bool = False
    error_code: Optional[str] = None
    raw_completion: str = ""


class SQLCompletionParser:
    """Extracts SQL code strictly enclosed within <sql>...</sql> tags."""

    OPEN_TAG = "<sql>"
    CLOSE_TAG = "</sql>"

    @classmethod
    def parse(cls, completion_text: str) -> ParsedCompletion:
        if not completion_text:
            return ParsedCompletion(
                is_valid_format=False,
                error_code="EMPTY_COMPLETION",
                raw_completion="",
            )

        open_count = completion_text.count(cls.OPEN_TAG)
        close_count = completion_text.count(cls.CLOSE_TAG)

        # Detect truncation: open tag exists but no closing tag
        if open_count > 0 and close_count == 0:
            return ParsedCompletion(
                is_valid_format=False,
                is_truncated=True,
                error_code="TRUNCATED_SQL_BLOCK",
                raw_completion=completion_text,
            )

        # Format violation: must have exactly 1 open tag and 1 close tag
        if open_count != 1 or close_count != 1:
            return ParsedCompletion(
                is_valid_format=False,
                error_code="MALFORMED_TAGS",
                raw_completion=completion_text,
            )

        start_idx = completion_text.find(cls.OPEN_TAG) + len(cls.OPEN_TAG)
        end_idx = completion_text.find(cls.CLOSE_TAG)

        if start_idx > end_idx:
            return ParsedCompletion(
                is_valid_format=False,
                error_code="TAG_ORDER_INVALID",
                raw_completion=completion_text,
            )

        # Check for disallowed trailing commentary after </sql>
        trailing_text = completion_text[end_idx + len(cls.CLOSE_TAG):].strip()
        if trailing_text:
            # Trailing explanations or text outside tags are format violations
            return ParsedCompletion(
                is_valid_format=False,
                error_code="DISALLOWED_TRAILING_TEXT",
                raw_completion=completion_text,
            )

        raw_sql = completion_text[start_idx:end_idx].strip()

        # Clean optional markdown code blocks if present inside tags
        if raw_sql.startswith("```"):
            lines = raw_sql.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_sql = "\n".join(lines).strip()

        if not raw_sql:
            return ParsedCompletion(
                is_valid_format=False,
                error_code="EMPTY_SQL",
                raw_completion=completion_text,
            )

        return ParsedCompletion(
            is_valid_format=True,
            sql=raw_sql,
            is_truncated=False,
            raw_completion=completion_text,
        )
