"""SQL Parser: Extracts and validates SQL tags and abstentions from model completions."""

from dataclasses import dataclass
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedCompletion:
    is_valid_format: bool
    sql: Optional[str] = None
    is_truncated: bool = False
    is_abstention: bool = False
    cleaned_markdown: bool = False
    error_code: Optional[str] = None
    raw_completion: str = ""


class SQLCompletionParser:
    """Extracts SQL code strictly enclosed within <sql>...</sql> tags."""

    OPEN_TAG = "<sql>"
    CLOSE_TAG = "</sql>"
    ABSTAIN_OPEN = "<abstain>"
    ABSTAIN_CLOSE = "</abstain>"

    @classmethod
    def parse(
        cls,
        completion_text: str,
        strip_markdown: bool = True,
    ) -> ParsedCompletion:
        if not completion_text or not completion_text.strip():
            return ParsedCompletion(
                is_valid_format=False,
                error_code="EMPTY_COMPLETION",
                raw_completion=completion_text or "",
            )

        # Detect explicit structured abstentions separately from syntax malformations
        trimmed = completion_text.strip()
        if (
            trimmed.startswith(cls.ABSTAIN_OPEN)
            and trimmed.endswith(cls.ABSTAIN_CLOSE)
            and trimmed.count(cls.OPEN_TAG) == 0
        ):
            return ParsedCompletion(
                is_valid_format=False,
                is_abstention=True,
                error_code="MODEL_ABSTENTION",
                raw_completion=completion_text,
            )

        open_count = completion_text.count(cls.OPEN_TAG)
        close_count = completion_text.count(cls.CLOSE_TAG)

        # Detect truncation: open tag exists without matching close tag
        if open_count > 0 and close_count == 0:
            return ParsedCompletion(
                is_valid_format=False,
                is_truncated=True,
                error_code="TRUNCATED_SQL_BLOCK",
                raw_completion=completion_text,
            )

        # Format violation: must contain exactly 1 opening and 1 closing tag
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

        # Disallow trailing commentary or extra SQL outside the closing </sql> tag
        trailing_text = completion_text[end_idx + len(cls.CLOSE_TAG):].strip()
        if trailing_text:
            return ParsedCompletion(
                is_valid_format=False,
                error_code="DISALLOWED_TRAILING_TEXT",
                raw_completion=completion_text,
            )

        raw_sql = completion_text[start_idx:end_idx].strip()
        cleaned_markdown = False

        if strip_markdown and raw_sql.startswith("```"):
            lines = raw_sql.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_sql = "\n".join(lines).strip()
            cleaned_markdown = True
            logger.debug("Stripped optional Markdown fences inside <sql> block.")

        if not raw_sql:
            return ParsedCompletion(
                is_valid_format=False,
                cleaned_markdown=cleaned_markdown,
                error_code="EMPTY_SQL",
                raw_completion=completion_text,
            )

        return ParsedCompletion(
            is_valid_format=True,
            sql=raw_sql,
            is_truncated=False,
            is_abstention=False,
            cleaned_markdown=cleaned_markdown,
            raw_completion=completion_text,
        )
