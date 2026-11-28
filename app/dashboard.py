"""Streamlit Interactive Dashboard for QueryAlign (RLVR + GRPO).

Communicates with the guarded FastAPI serving backend (`API_URL`) to run local
open-weight model inference, AST safety checks, sandboxed SQLite execution,
Majority@8 consensus voting, and side-by-side model comparison.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
import streamlit as st

from sql_grpo.serving.client import (
    BackendClientError,
    ServingAPIClient,
    classify_execution_display,
)

MODEL_OPTIONS = [
    "SFT + GRPO (RLVR)",
    "QLoRA SFT Baseline",
    "Qwen2.5-Coder-1.5B (Zero-Shot)",
]

DATABASE_SAMPLE_QUESTIONS: Dict[str, List[str]] = {
    "test_company": [
        "list employee name",
        "List all departments with a budget greater than 200,000.",
        "what is the average salary of employees?",
    ],
    "concert_singer": [
        "How many singers do we have?",
        "what is the name and country of singers?",
        "find stadiums with capacity greater than 10000",
    ],
    "car_1": [
        "What are the names of all car makers?",
        "how many countries are there?",
        "what is the average horsepower of cars?",
    ],
}

st.set_page_config(
    page_title="QueryAlign — Text-to-SQL Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
client = ServingAPIClient(base_url=API_URL)

st.title("QueryAlign: Autonomous Text-to-SQL Generator (RLVR + GRPO)")
st.caption(
    "Experimental Academic Prototype — Local Qwen2.5-Coder-1.5B-Instruct + LoRA Adapter Inference via Guarded FastAPI & SQLite Sandbox"
)
st.warning(
    "⚠️ **Experimental Model Notice:** Generated SQL is produced by a local 1.5B parameter model and is **not guaranteed to be semantically correct**. Always inspect SQL, safety checks, and consensus confidence before relying on outputs."
)

# ── Sidebar: Backend Health, Model & Mode Configuration ──────────────────────
st.sidebar.header("Configuration")
st.sidebar.caption(f"FastAPI Backend: `{API_URL}`")

backend_alive = False
model_ready = False
try:
    client.health()
    backend_alive = True
    ready_info = client.ready()
    model_ready = bool(ready_info.get("model_ready", False))
except BackendClientError:
    backend_alive = False

if backend_alive and model_ready:
    st.sidebar.success("Backend: Online • Model: Loaded")
elif backend_alive:
    st.sidebar.info("Backend: Online • Model: Lazy-load on first query")
else:
    st.sidebar.error("Backend: Unreachable (start FastAPI on port 8000)")

comparison_mode = st.sidebar.checkbox(
    "Side-by-Side Model Comparison",
    value=False,
    help="Compare Zero-Shot, QLoRA SFT, and SFT+GRPO outputs for the same question.",
)

model_choice = st.sidebar.selectbox(
    "Active Model Variant",
    MODEL_OPTIONS,
    index=0,
    disabled=comparison_mode,
    help="Select the local open-weight model checkpoint served by FastAPI.",
)

inference_mode = st.sidebar.radio(
    "Inference Mode",
    ["Greedy (Pass@1)", "Majority@8 (Self-Consistency)"],
    index=0,
    help="Greedy decodes 1 query. Majority@8 samples 8 stochastic candidates (~8x generation latency) and clusters execution results.",
)
if "Majority@8" in inference_mode:
    st.sidebar.caption("ℹ️ Majority@8 runs 8 samples and sandbox executions; latency will be substantially higher than Greedy Pass@1.")

execute_sql = st.sidebar.checkbox(
    "Execute in Read-Only Sandbox",
    value=True,
    help="Disable for preview-only generation (labeled 'not execution-verified').",
)

# ── Database Selection & Schema Fetch ────────────────────────────────────────
available_databases: List[str] = []
db_fetch_error: str = ""
if backend_alive:
    try:
        available_databases = client.list_databases()
    except BackendClientError as exc:
        db_fetch_error = str(exc)

if not available_databases:
    available_databases = ["test_company"]

database_id = st.sidebar.selectbox(
    "Target Database",
    available_databases,
    index=0,
    help="Approved SQLite databases exposed by the server-side registry.",
)

st.sidebar.markdown("---")
if "Zero-Shot" in model_choice:
    st.sidebar.markdown("**Zero-Shot Baseline**\n- Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`\n- Untuned adapter disabled")
elif "SFT" in model_choice and "GRPO" not in model_choice:
    st.sidebar.markdown("**QLoRA SFT Baseline**\n- Rank 16, Alpha 16 LoRA\n- Supervised Spider training (3 epochs)")
else:
    st.sidebar.markdown("**SFT + GRPO (RLVR)**\n- Policy initialized from SFT (`G=8`, `beta=0.04`)\n- Evaluated on Spider dev subset; experimental prototype")

st.sidebar.markdown("---")
st.sidebar.info(
    "Sandbox layers: Registry allowlist, SQLGlot SELECT-only AST policy, URI `mode=ro`, `PRAGMA query_only=ON`, C-Authorizer, 2.0s VM progress deadline, 5,000-row limit."
)

schema_text = ""
schema_meta: Dict[str, Any] = {}
if backend_alive:
    try:
        schema_payload = client.get_schema(database_id)
        schema_text = schema_payload.get("schema_text", "")
        schema_meta = schema_payload.get("metadata") or {}
    except BackendClientError as exc:
        st.error(f"Failed to load schema for `{database_id}`: {exc}")

if "current_db" not in st.session_state:
    st.session_state.current_db = database_id
    st.session_state.question_input = DATABASE_SAMPLE_QUESTIONS.get(database_id, [""])[0]
elif st.session_state.current_db != database_id:
    st.session_state.current_db = database_id
    st.session_state.question_input = DATABASE_SAMPLE_QUESTIONS.get(database_id, [""])[0]
    st.session_state.last_response = None
    st.session_state.comparison_results = None

# ── Main Layout ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Natural Language Question")

    if database_id in DATABASE_SAMPLE_QUESTIONS:
        st.markdown("**Suggested Questions:**")
        samples = DATABASE_SAMPLE_QUESTIONS[database_id]
        chip_cols = st.columns(len(samples))
        for idx, q_text in enumerate(samples):
            if chip_cols[idx].button(q_text, key=f"chip_{idx}"):
                st.session_state.question_input = q_text

    question = st.text_area(
        "Enter query question (1-500 chars):",
        key="question_input",
        max_chars=500,
        height=100,
        help="Enter a natural-language question grounded in the target database schema.",
    )

    btn_col1, btn_col2 = st.columns([3, 1])
    action_label = "Compare All 3 Models" if comparison_mode else "Generate & Execute SQL"
    run_clicked = btn_col1.button(action_label, type="primary", disabled=not backend_alive)
    clear_clicked = btn_col2.button("Clear / Reset")

    if clear_clicked:
        st.session_state.question_input = ""
        st.session_state.last_response = None
        st.session_state.comparison_results = None
        st.rerun()

with col2:
    st.subheader("Database Schema Summary")
    tables_dict = schema_meta.get("tables") or {}
    fks_list = schema_meta.get("foreign_keys") or []
    col_count = sum(len(t.get("columns", [])) for t in tables_dict.values())
    st.caption(f"Database: `{database_id}` • {len(tables_dict)} table(s) • {col_count} column(s) • {len(fks_list)} FK(s)")
    st.code(schema_text or "# Schema unavailable (backend offline)", language="yaml")

# ── Execution Trigger ────────────────────────────────────────────────────────
if run_clicked:
    trimmed_q = question.strip()
    if not trimmed_q:
        st.error("Please enter a non-empty question before running inference.")
    elif db_fetch_error:
        st.error(f"Database registry unavailable: {db_fetch_error}")
    else:
        num_cands = 8 if "Majority@8" in inference_mode else 1
        if comparison_mode:
            comp_outputs: List[Dict[str, Any]] = []
            with st.spinner("Running local inference across Base, SFT, and SFT+GRPO via FastAPI..."):
                for variant in MODEL_OPTIONS:
                    try:
                        resp = client.generate(
                            database_id=database_id,
                            question=trimmed_q,
                            num_candidates=num_cands,
                            execute=execute_sql,
                            model_choice=variant,
                        )
                        comp_outputs.append({"model": variant, "response": resp, "error": None})
                    except BackendClientError as exc:
                        comp_outputs.append({"model": variant, "response": None, "error": str(exc)})
            st.session_state.comparison_results = comp_outputs
            st.session_state.last_response = None
        else:
            with st.spinner("Running local model inference, AST safety checks, and sandbox execution..."):
                try:
                    resp = client.generate(
                        database_id=database_id,
                        question=trimmed_q,
                        num_candidates=num_cands,
                        execute=execute_sql,
                        model_choice=model_choice,
                    )
                    st.session_state.last_response = resp
                    st.session_state.comparison_results = None
                    st.session_state.api_error = None
                except BackendClientError as exc:
                    st.session_state.api_error = f"API Error ({exc.status_code or 'network'}): {exc}"
                    st.session_state.last_response = None

if st.session_state.get("api_error"):
    st.error(st.session_state["api_error"])


def _render_single_response(resp: Dict[str, Any]) -> None:
    validation = resp.get("validation") or {}
    latency = resp.get("latency_ms") or {}
    execution = resp.get("execution")
    consensus = resp.get("consensus")
    display_state = classify_execution_display(resp)

    st.caption(f"Request ID: `{resp.get('request_id', 'n/a')}` • Status: `{resp.get('status', 'unknown')}`")

    # 1. Safety & Latency Metrics
    c1, c2, c3, c4 = st.columns(4)
    fmt_ok = bool(validation.get("format_valid", False))
    ast_ok = bool(validation.get("ast_safe", False))
    c1.metric("Format Validity", "PASS" if fmt_ok else "FAIL", delta=validation.get("error_code") or "OK")
    c2.metric("AST Safety", "SELECT-Only" if ast_ok else (validation.get("error_code") or "REJECTED"))
    c3.metric("Generation Latency", f"{latency.get('generation', 0.0):.1f} ms")
    c4.metric("Total Latency", f"{latency.get('total', 0.0):.1f} ms", delta=f"Exec {latency.get('execution', 0.0):.1f} ms")

    if resp.get("verification_label"):
        st.info(f"Verification Label: **{resp['verification_label']}**")

    # 2. Generated SQL
    st.markdown("#### Generated SQL")
    if resp.get("sql"):
        st.code(resp["sql"], language="sql")
    else:
        st.code("-- No SQL returned (rejected or abstained)", language="sql")

    # 3. Execution & Bounded Result Preview
    st.markdown("#### Execution & Bounded Result Preview")
    if execution:
        e1, e2, e3 = st.columns(3)
        e1.metric("Sandbox Success", "YES" if execution.get("success") else "NO")
        e2.metric("Row Count Seen", execution.get("row_count", 0))
        e3.metric("Truncated (>5,000 rows)", "YES" if execution.get("truncated") else "NO")

    if display_state.banner_level == "success":
        st.success(display_state.message)
    elif display_state.banner_level == "info":
        st.info(display_state.message)
    elif display_state.banner_level == "warning":
        st.warning(display_state.message)
    else:
        st.error(display_state.message)

    if display_state.show_rows:
        rows = resp.get("result_preview") or []
        cols = resp.get("columns") or []
        df = pd.DataFrame(rows, columns=cols if cols else None)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Showing preview of {len(rows)} row(s) (API max preview: 10; sandbox cap: 5,000).")

    # 4. Consensus Telemetry
    if consensus and consensus.get("enabled"):
        st.markdown("#### Majority@8 Consensus Telemetry")
        reason = consensus.get("abstention_reason")
        if reason:
            st.warning(
                f"**Consensus Status:** `{reason}` — Confidence: {consensus.get('confidence_score', 0.0) * 100:.1f}% "
                f"({consensus.get('cluster_size', 0)}/{consensus.get('total_candidates', 0)} candidates)."
            )
        else:
            st.success(
                f"**Consensus Status:** `CONSENSUS_ELECTED` — Confidence: {consensus.get('confidence_score', 0.0) * 100:.1f}% "
                f"({consensus.get('cluster_size', 0)}/{consensus.get('total_candidates', 0)} candidates agreed)."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidates", resp.get("candidate_count", consensus.get("total_candidates", 0)))
        m2.metric("AST-Safe", resp.get("safe_candidate_count", consensus.get("num_safe", 0)))
        m3.metric("Executed", resp.get("executed_candidate_count", consensus.get("num_executed", 0)))
        m4.metric("Largest Cluster", consensus.get("cluster_size", 0))


# ── Render Single or Side-by-Side Results ────────────────────────────────────
if st.session_state.get("last_response"):
    st.markdown("---")
    st.subheader("Execution & Safety Pipeline")
    _render_single_response(st.session_state["last_response"])

if st.session_state.get("comparison_results"):
    st.markdown("---")
    st.subheader("Side-by-Side Model Comparison")
    comp_items = st.session_state["comparison_results"]
    comp_cols = st.columns(len(comp_items))
    for col, item in zip(comp_cols, comp_items):
        with col:
            st.markdown(f"### {item['model']}")
            if item["error"]:
                st.error(item["error"])
            else:
                r = item["response"]
                state = classify_execution_display(r)
                st.caption(f"Status: `{r.get('status')}` • Total: {r.get('latency_ms', {}).get('total', 0.0):.1f} ms")
                st.code(r.get("sql") or "-- No SQL returned", language="sql")
                if state.banner_level == "success":
                    st.success(state.message)
                elif state.banner_level == "info":
                    st.info(state.message)
                elif state.banner_level == "warning":
                    st.warning(state.message)
                else:
                    st.error(state.message)
                if state.show_rows:
                    st.dataframe(
                        pd.DataFrame(r.get("result_preview") or [], columns=r.get("columns") or None),
                        use_container_width=True,
                    )

st.markdown("---")
st.caption("Academic Prototype. Never expose unauthenticated or unsandboxed SQL execution to untrusted production networks.")
