"""API Acceptance Tests covering FastAPI validation, safety, readiness, and error handling."""

from unittest.mock import patch
from fastapi import HTTPException, Response
from pydantic import ValidationError
import pytest

from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.serving import api
from sql_grpo.serving.generator import SQLGenerator
from sql_grpo.serving.inference import ModelUnavailableError
from sql_grpo.serving.schemas import TextToSQLRequest
from tests.serving.test_generator import StubInferenceBackend


def _with_stub_generator(completions_by_mode=None, ready=True):
    stub = StubInferenceBackend(completions_by_mode=completions_by_mode)
    stub.is_ready = lambda: ready
    gen = SQLGenerator(
        registry=api.registry,
        sandbox=SQLiteSandbox(),
        inference_backend=stub,
    )
    return gen, stub


def test_health_and_readiness_distinguish_liveness_from_model_ready():
    h = api.health()
    assert h["status"] == "ok"

    gen_not_ready, _ = _with_stub_generator(ready=False)
    with patch.object(api, "generator", gen_not_ready):
        resp = Response()
        r = api.ready(resp)
        assert resp.status_code == 503
        assert r["status"] == "not_ready"
        assert r["model_ready"] is False

    gen_ready, _ = _with_stub_generator(ready=True)
    with patch.object(api, "generator", gen_ready):
        resp = Response()
        r = api.ready(resp)
        assert r["status"] == "ready"
        assert r["model_ready"] is True


def test_request_validation_rejects_empty_whitespace_and_unknown_db():
    for bad_q in ["", "   \n\t  ", "x" * 501]:
        with pytest.raises(ValidationError):
            TextToSQLRequest(database_id="test_company", question=bad_q)

    with pytest.raises(ValidationError):
        TextToSQLRequest(database_id="test_company", question="list", num_candidates=9)

    with pytest.raises(HTTPException) as exc:
        api.generate(TextToSQLRequest(database_id="../../etc", question="list employee"))
    assert exc.value.status_code == 400


def test_execute_false_skips_execution_and_marks_not_verified():
    gen, _ = _with_stub_generator({1: ["<sql>SELECT name FROM employee</sql>"]})
    with patch.object(api, "generator", gen):
        res = api.generate(
            TextToSQLRequest(
                database_id="test_company",
                question="list employee name",
                num_candidates=1,
                execute=False,
            )
        )
    assert res.status == "not_execution_verified"
    assert res.execution_verified is False
    assert res.verification_label == "not execution-verified"
    assert res.sql == "SELECT name FROM employee"
    assert res.execution is None
    assert res.result_preview is None
    assert res.latency_ms["execution"] == 0.0


def test_unsafe_sql_rejected_without_execution_and_no_path_leak():
    gen, _ = _with_stub_generator({1: ["<sql>DROP TABLE employee;</sql>"]})
    with patch.object(api, "generator", gen):
        res = api.generate(
            TextToSQLRequest(database_id="test_company", question="drop employee", execute=True)
        )
    assert res.status == "ast_rejected"
    assert res.sql is None
    assert res.execution is None
    assert res.validation.ast_safe is False
    assert res.validation.error_code == "DISALLOWED_OPERATION"


def test_majority8_consensus_and_abstention_status():
    # Consensus elected
    gen_ok, _ = _with_stub_generator({8: ["<sql>SELECT name FROM employee</sql>"] * 5 + ["bad"] * 3})
    with patch.object(api, "generator", gen_ok):
        res_ok = api.generate(
            TextToSQLRequest(database_id="test_company", question="list names", num_candidates=8)
        )
    assert res_ok.status == "success"
    assert res_ok.execution_verified is True
    assert res_ok.candidate_count == 8
    assert res_ok.safe_candidate_count == 5
    assert res_ok.executed_candidate_count == 5
    assert res_ok.consensus is not None and res_ok.consensus.cluster_size == 5
    assert len(res_ok.result_preview or []) == 5

    # Tie -> abstained (must not return sql or status="success")
    tie_cands = ["<sql>SELECT name FROM employee</sql>"] * 4 + ["<sql>SELECT name FROM department</sql>"] * 4
    gen_tie, _ = _with_stub_generator({8: tie_cands})
    with patch.object(api, "generator", gen_tie):
        res_tie = api.generate(
            TextToSQLRequest(database_id="test_company", question="list names", num_candidates=8)
        )
    assert res_tie.status == "abstained"
    assert res_tie.sql is None
    assert res_tie.execution_verified is False
    assert res_tie.consensus is not None and res_tie.consensus.abstention_reason == "NO_CONSENSUS_TIE"


def test_model_unavailable_oom_and_queue_full_return_503_without_leaks():
    class _BrokenBackend:
        def is_ready(self):
            return False

        def status(self):
            return {"loaded": False}

        def generate(self, **_kwargs):
            raise ModelUnavailableError("C:\\secret\\host\\path\\missing")

    gen_broken = SQLGenerator(registry=api.registry, sandbox=SQLiteSandbox(), inference_backend=_BrokenBackend())
    with patch.object(api, "generator", gen_broken):
        with pytest.raises(HTTPException) as exc:
            api.generate(TextToSQLRequest(database_id="test_company", question="list"))
        assert exc.value.status_code == 503
        assert "C:\\secret" not in str(exc.value.detail)

    class _OOMBackend(_BrokenBackend):
        def generate(self, **_kwargs):
            raise RuntimeError("CUDA out of memory while allocating tensor")

    gen_oom = SQLGenerator(registry=api.registry, sandbox=SQLiteSandbox(), inference_backend=_OOMBackend())
    with patch.object(api, "generator", gen_oom):
        with pytest.raises(HTTPException) as exc_oom:
            api.generate(TextToSQLRequest(database_id="test_company", question="list"))
        assert exc_oom.value.status_code == 503
        assert "memory" in str(exc_oom.value.detail).lower()

    # Queue full check
    class _BusySem:
        def acquire(self, timeout=None):
            return False

        def release(self):
            pass

    gen_ok, _ = _with_stub_generator()
    with patch.object(api, "generator", gen_ok), patch.object(api, "_INFERENCE_SEMAPHORE", _BusySem()):
        with pytest.raises(HTTPException) as exc_q:
            api.generate(TextToSQLRequest(database_id="test_company", question="list"))
        assert exc_q.value.status_code == 503
