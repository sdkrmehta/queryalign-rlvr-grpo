"""Streamlit Interactive Dashboard for Text-to-SQL (RLVR + GRPO).

Full interactive prototype with live SQLite execution sandboxes,
AST static verification, dynamic database schema introspection,
model selection, and Majority@8 consensus voting.
"""

import os
import sys
import time
from pathlib import Path

# Ensure src/ is accessible
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
import streamlit as st

from sql_grpo.data.schema_serializer import SchemaSerializer
from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.serving.generator import SQLGenerator

st.set_page_config(
    page_title="SQL-GRPO Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize registry, sandbox, serializer, and generator
@st.cache_resource
def get_engine():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    registry = DatabaseRegistry(data_root=data_dir)
    sandbox = SQLiteSandbox()
    serializer = SchemaSerializer()
    generator = SQLGenerator(registry=registry, sandbox=sandbox)
    return registry, sandbox, serializer, generator


registry, sandbox, serializer, generator = get_engine()

# Title and header
st.title("Autonomous Text-to-SQL Generator (RLVR + GRPO)")
st.caption("Master Engineering Blueprint V2.0 Prototype — SQLite Sandbox & Verifiable Execution")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Model Choice
model_options = [
    "Qwen2.5-Coder-1.5B (Zero-Shot)",
    "QLoRA SFT Baseline",
    "SFT + GRPO (RLVR)",
]
model_choice = st.sidebar.selectbox(
    "Active Model",
    model_options,
    index=2,
    help="Select the alignment stage: Zero-Shot baseline (55% EX), QLoRA SFT (format-grounded), or GRPO RLVR (66% Pass@1, 80% Maj@8).",
)

# Inference Mode
inference_mode = st.sidebar.radio(
    "Inference Mode",
    ["Greedy (Pass@1)", "Majority@8 (Self-Consistency)"],
    index=0,
    help="Greedy generates a single deterministic query. Majority@8 samples 8 candidate queries and clusters result sets.",
)

# Database Selector
available_databases = registry.list_available_databases()
if not available_databases:
    available_databases = ["test_company", "concert_singer"]

database_id = st.sidebar.selectbox(
    "Target Database",
    available_databases,
    index=0,
    help="Select any approved database from the verified registry.",
)

# Model description badge in sidebar
st.sidebar.markdown("---")
if "Zero-Shot" in model_choice:
    st.sidebar.markdown(
        "**Stage 1: Zero-Shot Baseline**\n- Base Model: `Qwen2.5-Coder-1.5B`\n- Spider 1.0 EX: **55.0%**\n- Untuned greedy decoding"
    )
elif "SFT Baseline" in model_choice:
    st.sidebar.markdown(
        "**Stage 2: QLoRA SFT Baseline**\n- Rank: 16, Alpha: 16 (7K samples)\n- Strict `<sql>` tag grounding\n- Schema-aligned representation"
    )
else:
    st.sidebar.markdown(
        "**Stage 3 & 4: RLVR Policy (GRPO)**\n- G=8, 100 RL steps\n- Greedy Pass@1 EX: **66.0%**\n- Majority@8 Consensus EX: **80.0%**\n- 4-tier verifiable reward stack"
    )

st.sidebar.markdown("---")
st.sidebar.info(
    "Sandbox active: URI `mode=ro`, `PRAGMA query_only=ON`, C-Authorizer, VM Progress Handler (2.0s limit)."
)

# Dynamic Schema Introspection
db_path = registry.get_path(database_id)
schema_info = serializer.serialize_database(db_path)
schema_text = serializer.to_compact_text(schema_info)

# Sample questions per database
database_sample_questions = {
    "test_company": [
        "list employee name",
        "List all departments with a budget greater than 200,000.",
        "what is the average salary of employees?",
        "who has the highest salary?",
        "show employees and their department names",
    ],
    "concert_singer": [
        "How many singers do we have?",
        "what is the name and country of singers?",
        "find stadiums with capacity greater than 10000",
        "List all concerts held in 2014",
        "Show names and themes of all concerts",
    ],
    "car_1": [
        "list all car models",
        "how many countries are there?",
        "what is the average horsepower of cars?",
        "list car makers and their countries",
    ],
}

default_q = "list employee name"
if database_id in database_sample_questions:
    default_q = database_sample_questions[database_id][0]
elif schema_info.get("tables"):
    first_tbl = list(schema_info["tables"].keys())[0]
    default_q = f"List all records from {first_tbl}"

# Track current database to update question when database changes
if "current_db" not in st.session_state:
    st.session_state.current_db = database_id
    st.session_state.question_val = default_q
elif st.session_state.current_db != database_id:
    st.session_state.current_db = database_id
    st.session_state.question_val = default_q

# Main Area Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Natural Language Question")

    # Sample query chips
    if database_id in database_sample_questions:
        st.markdown("**Suggested Quick Questions:**")
        chip_cols = st.columns(len(database_sample_questions[database_id][:3]))
        for i, q_text in enumerate(database_sample_questions[database_id][:3]):
            if chip_cols[i].button(q_text, key=f"chip_{i}"):
                st.session_state.question_val = q_text

    question = st.text_area(
        "Enter query question:",
        value=st.session_state.get("question_val", default_q),
        height=100,
        help="Type any natural language question or SQL query for the selected database.",
    )
    generate_btn = st.button("Generate & Execute SQL", type="primary")

with col2:
    st.subheader("Database Schema Preview")
    st.caption(f"Introspected from: `{db_path.name}` ({len(schema_info['tables'])} tables)")
    st.code(schema_text, language="yaml")

# Run Pipeline on Trigger
if generate_btn or st.session_state.get("ran_once", False):
    st.session_state.ran_once = True
    st.markdown("---")
    st.subheader("Execution & Safety Pipeline")

    with st.spinner("Processing AST safety checks and executing in SQLite sandbox..."):
        result = generator.execute_and_evaluate(
            question=question,
            database_id=database_id,
            model_choice=model_choice,
            inference_mode=inference_mode,
        )

    # 1. Pipeline Metrics
    c_out1, c_out2, c_out3 = st.columns(3)

    fmt_label = "PASS" if result.format_valid else "FAIL"
    fmt_delta = "+0.20 R_fmt" if result.format_valid else "-0.50 R_fmt"
    c_out1.metric("Format Validity", fmt_label, delta=fmt_delta)

    ast_label = "SELECT-Only" if result.ast_safe else (result.ast_error_code or "REJECTED")
    ast_delta = "+0.20 R_ast" if result.ast_safe else "0.00 R_ast"
    c_out2.metric("AST Security Check", ast_label, delta=ast_delta)

    latency_str = f"{result.latency_ms:.1f} ms"
    lat_delta = "Safe" if result.latency_ms < 2000 else "High"
    c_out3.metric("Sandbox Latency", latency_str, delta=lat_delta)

    # 2. Verifiable Reward Stack (RLVR model)
    if result.rewards:
        st.markdown("##### Verifiable Reward Breakdown (GRPO Formulation)")
        r_cols = st.columns(6)
        r_cols[0].metric("R_fmt", f"{result.rewards.r_fmt:+.2f}")
        r_cols[1].metric("R_ast", f"{result.rewards.r_ast:+.2f}")
        r_cols[2].metric("R_exec", f"{result.rewards.r_exec_valid:+.2f}")
        r_cols[3].metric("R_result", f"{result.rewards.r_result:+.2f}")
        r_cols[4].metric("R_eff", f"{result.rewards.r_eff:+.2f}")
        r_cols[5].metric("Total Reward", f"{result.rewards.r_total:+.2f}")

    # 3. Generated SQL
    st.markdown("#### Generated SQL")
    st.code(result.selected_sql, language="sql")

    # 4. Bounded Result Preview
    st.markdown("#### Bounded Result Preview")
    if result.execution_result and result.execution_result.success:
        if result.rows:
            df = pd.DataFrame(result.rows, columns=result.columns)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Displaying {len(result.rows)} row(s) (Hard limit: 5,000 rows).")
        else:
            if result.columns:
                df = pd.DataFrame(columns=result.columns)
                st.dataframe(df, use_container_width=True)
            st.info(
                "ℹ️ Query executed successfully in the read-only sandbox, but returned **0 rows**.\n\n"
                "*Tip: Check the database schema preview on the right for valid sample values and IDs present in the database.*"
            )
    elif result.execution_result:
        st.error(f"Execution Error: {result.execution_result.error_message_safe or 'Unknown execution failure'}")
    elif not result.ast_safe:
        st.error(f"AST Policy Violation: {result.ast_error_message or result.ast_error_code}")
    else:
        st.warning("Query was not executed.")

    # 5. Majority@8 Consensus Telemetry (if Majority@8 selected)
    if "Majority@8" in inference_mode and result.voting_outcome:
        vo = result.voting_outcome
        st.markdown("#### Majority@8 Consensus Telemetry")

        status_text = "CONSENSUS_ELECTED" if not vo.is_abstained else (vo.abstention_reason or "ABSTAINED")
        if not vo.is_abstained:
            st.success(
                f"**Consensus Status:** `{status_text}` — Plurality Winner with **{vo.confidence_score * 100:.1f}% confidence** ({vo.cluster_size}/{vo.total_candidates} candidates agreed)."
            )
        else:
            st.warning(
                f"**Consensus Status:** `{status_text}` — Confidence: {vo.confidence_score * 100:.1f}% ({vo.cluster_size}/{vo.total_candidates} candidates)."
            )

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Candidates", vo.total_candidates)
        m_c2.metric("AST-Safe Queries", vo.num_safe)
        m_c3.metric("Successfully Executed", vo.num_executed)
        m_c4.metric("Consensus Cluster Size", vo.cluster_size)

        # Candidate Inspection Table
        if result.candidates:
            with st.expander("🔍 Inspect 8 Sampled Candidate Queries & Clusters"):
                c_data = []
                for c in result.candidates:
                    c_data.append({
                        "Sample #": c.index,
                        "Candidate SQL": c.sql or c.raw_completion,
                        "Format Valid": "✅ PASS" if c.is_format_valid else "❌ FAIL",
                        "AST Safe": "✅ SAFE" if c.is_ast_safe else f"❌ {c.ast_error}",
                        "Executed": "✅ SUCCESS" if c.is_executed else "❌ FAILED",
                        "Rows": c.row_count,
                        "Cluster": f"Cluster #{c.cluster_id}" if c.cluster_id else "N/A",
                    })
                st.dataframe(pd.DataFrame(c_data), use_container_width=True)

st.markdown("---")
st.caption("Academic Prototype. Never run un-sandboxed arbitrary SQL on production databases.")
