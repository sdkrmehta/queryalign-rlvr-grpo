"""Tests for Streamlit dashboard API client and display state classification."""

import io
import json
from unittest.mock import patch
import urllib.error
import pytest

from sql_grpo.serving.client import (
    BackendClientError,
    ServingAPIClient,
    classify_execution_display,
)


def test_classify_truncated_never_reports_zero_rows():
    resp = {
        "status": "truncated",
        "sql": "SELECT * FROM big_table",
        "execution_verified": False,
        "validation": {"format_valid": True, "ast_safe": True},
        "execution": {
            "success": True,
            "complete": False,
            "row_count": 5001,
            "truncated": True,
            "error_code": "RESULT_TRUNCATED",
        },
        "result_preview": None,
    }
    state = classify_execution_display(resp)
    assert state.kind == "truncated"
    assert state.banner_level == "warning"
    assert state.show_rows is False
    assert "0 rows" not in state.message.lower()
    assert "truncated" in state.message.lower()


def test_classify_abstained_and_not_execution_verified_and_empty():
    abstained = classify_execution_display(
        {
            "status": "abstained",
            "sql": None,
            "execution_verified": False,
            "verification_label": "NO_CONSENSUS_TIE",
            "validation": {"format_valid": True, "ast_safe": True},
            "consensus": {"enabled": True, "abstention_reason": "NO_CONSENSUS_TIE"},
        }
    )
    assert abstained.kind == "abstained"
    assert abstained.show_rows is False

    preview_only = classify_execution_display(
        {
            "status": "not_execution_verified",
            "sql": "SELECT 1",
            "execution_verified": False,
            "verification_label": "not execution-verified",
            "validation": {"format_valid": True, "ast_safe": True},
            "execution": None,
        }
    )
    assert preview_only.kind == "not_executed"
    assert "not execution-verified" in preview_only.message

    empty_ok = classify_execution_display(
        {
            "status": "success",
            "sql": "SELECT name FROM employee WHERE 1=0",
            "execution_verified": True,
            "validation": {"format_valid": True, "ast_safe": True},
            "execution": {"success": True, "complete": True, "row_count": 0, "truncated": False},
            "result_preview": [],
        }
    )
    assert empty_ok.kind == "empty"
    assert "0 rows" in empty_ok.message


def test_serving_api_client_success_and_http_error():
    client = ServingAPIClient("http://127.0.0.1:8000")

    class _FakeResp:
        def __init__(self, obj):
            self._payload = json.dumps(obj).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=_FakeResp({"databases": ["test_company"]})):
        assert client.list_databases() == ["test_company"]

    err_body = io.BytesIO(json.dumps({"detail": "Local Text-to-SQL model is currently unavailable."}).encode("utf-8"))
    http_503 = urllib.error.HTTPError("http://127.0.0.1:8000/generate", 503, "Service Unavailable", {}, err_body)
    with patch("urllib.request.urlopen", side_effect=http_503):
        with pytest.raises(BackendClientError) as exc:
            client.generate(database_id="test_company", question="list names")
        assert exc.value.status_code == 503
        assert "unavailable" in str(exc.value).lower()
