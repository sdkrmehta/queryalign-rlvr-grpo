"""HTTP client and presentation-state helpers for Streamlit Dashboard -> FastAPI integration."""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request


class BackendClientError(RuntimeError):
    """Raised when the FastAPI serving backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ExecutionDisplayState:
    """Explicit UI status classification preventing truncated/abstained misreporting."""

    kind: str  # "success", "empty", "truncated", "not_executed", "abstained", "error", "ast_rejected", "format_error"
    banner_level: str  # "success", "info", "warning", "error"
    message: str
    show_rows: bool


def classify_execution_display(resp: Dict[str, Any]) -> ExecutionDisplayState:
    """Classify a /generate JSON response for unambiguous Streamlit rendering."""
    status = resp.get("status", "unknown")
    validation = resp.get("validation") or {}
    execution = resp.get("execution")
    consensus = resp.get("consensus") or {}
    rows = resp.get("result_preview") or []

    if status == "format_error" or not validation.get("format_valid", False):
        err_code = validation.get("error_code") or "TAG_SYNTAX_ERROR"
        return ExecutionDisplayState(
            kind="format_error",
            banner_level="warning",
            message=f"Model output failed `<sql>` tag validation (`{err_code}`). Query was not executed.",
            show_rows=False,
        )

    if status == "ast_rejected" or not validation.get("ast_safe", False):
        err_msg = validation.get("error_message") or validation.get("error_code") or "AST policy violation"
        return ExecutionDisplayState(
            kind="ast_rejected",
            banner_level="error",
            message=f"AST Policy Violation: {err_msg}",
            show_rows=False,
        )

    if status == "abstained" or consensus.get("abstention_reason"):
        reason = resp.get("verification_label") or consensus.get("abstention_reason") or "ABSTAINED"
        return ExecutionDisplayState(
            kind="abstained",
            banner_level="warning",
            message=f"Consensus engine abstained (`{reason}`). Review candidate agreement before running manually.",
            show_rows=False,
        )

    if status == "not_execution_verified" or resp.get("execution_verified") is False and execution is None:
        return ExecutionDisplayState(
            kind="not_executed",
            banner_level="info",
            message="Preview-only mode (`execute=false`): SQL is not execution-verified.",
            show_rows=False,
        )

    if status == "truncated" or (execution and execution.get("truncated")):
        seen = execution.get("row_count", 5001) if execution else 5001
        return ExecutionDisplayState(
            kind="truncated",
            banner_level="warning",
            message=f"Result truncated after exceeding the 5,000-row safety limit ({seen} rows seen). Partial rows are not treated as complete.",
            show_rows=False,
        )

    if status == "execution_error" or (execution and not execution.get("success", False)):
        err_msg = (execution or {}).get("error_message") or (execution or {}).get("error_code") or "Execution failed"
        return ExecutionDisplayState(
            kind="error",
            banner_level="error",
            message=f"Execution Error: {err_msg}",
            show_rows=False,
        )

    if execution and execution.get("success"):
        if rows:
            return ExecutionDisplayState(
                kind="success",
                banner_level="success",
                message=f"Executed successfully in read-only sandbox ({execution.get('row_count', len(rows))} total row(s); showing preview).",
                show_rows=True,
            )
        return ExecutionDisplayState(
            kind="empty",
            banner_level="info",
            message="Query executed successfully in the read-only sandbox and returned 0 rows.",
            show_rows=False,
        )

    return ExecutionDisplayState(
        kind="not_executed",
        banner_level="warning",
        message="Query was not executed.",
        show_rows=False,
    )


class ServingAPIClient:
    """Synchronous HTTP client talking to the FastAPI Text-to-SQL service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace")
            detail = f"HTTP {exc.code}"
            try:
                parsed = json.loads(raw_err)
                if isinstance(parsed, dict) and parsed.get("detail"):
                    detail = str(parsed["detail"])
            except Exception:
                if raw_err:
                    detail = raw_err[:300]
            # Handle /ready returning 503 with valid JSON status payload
            if path == "/ready" and exc.code == 503:
                try:
                    parsed = json.loads(raw_err)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            raise BackendClientError(detail, status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise BackendClientError(
                f"Cannot reach FastAPI backend at {self.base_url}: {exc.reason}"
            ) from exc

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def ready(self) -> Dict[str, Any]:
        return self._request("GET", "/ready")

    def list_databases(self) -> List[str]:
        data = self._request("GET", "/databases")
        return list(data.get("databases") or [])

    def get_schema(self, database_id: str) -> Dict[str, Any]:
        encoded = urllib.parse.quote(database_id, safe="")
        return self._request("GET", f"/databases/{encoded}/schema")

    def generate(
        self,
        database_id: str,
        question: str,
        num_candidates: int = 1,
        execute: bool = True,
        model_choice: str = "SFT + GRPO (RLVR)",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/generate",
            payload={
                "database_id": database_id,
                "question": question,
                "num_candidates": num_candidates,
                "execute": execute,
                "model_choice": model_choice,
            },
        )
