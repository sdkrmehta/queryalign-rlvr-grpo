$nb4_path = "c:\queryalign-rlvr-grpo\notebooks\04_kaggle_evaluation_majority8.ipynb"
$cells = [System.Collections.ArrayList]@()

# --- Cell 0 (markdown) ---
$c0 = [PSCustomObject]@{
    cell_type = "markdown"
    metadata = [PSCustomObject]@{}
    source = @(
        "# Notebook 4: Benchmark Evaluation & Majority@8 Self-Consistency (Kaggle)`n",
        "**Project**: Autonomous Text-to-SQL Generator Aligned via RLVR + GRPO  `n",
        "**Hardware**: Kaggle GPU (T4 x 1, P100, or 2xT4)  `n",
        "**Phase**: Phase 5 (AST Security Hardening, Majority@8 Voting & Confidence Handling, Final Benchmarking)`n",
        "`n",
        "---"
    )
}
$null = $cells.Add($c0)

# --- Cell 1 (code - Step 1: Pinned Dependencies) ---
$c1 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# Step 1: Install Pinned Dependencies for Evaluation`n",
        "!pip uninstall -y -q torchvision`n",
        "!pip install -q \`n",
        "    `"transformers==4.57.3`" \`n",
        "    `"trl==0.25.0`" \`n",
        "    `"peft==0.18.1`" \`n",
        "    `"datasets>=3.0.0,<5.0.0`" \`n",
        "    `"accelerate>=1.2.1,<2.0.0`" \`n",
        "    `"bitsandbytes>=0.45.2`" \`n",
        "    `"sqlglot>=26.0.0`" \`n",
        "    `"pandas`" \`n",
        "    `"tabulate`"`n",
        "print(`"Evaluation packages ready.`")"
    )
}
$null = $cells.Add($c1)

# --- Cell 2 (code - Step 2: Paths, Environment & Auto-Unpack) ---
$c2 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# Step 2: Paths & Environment Setup (With Auto-Unpack for Kaggle Inputs)`n",
        "import os`n",
        "os.environ[`"CUDA_VISIBLE_DEVICES`"] = `"0`"`n",
        "os.environ[`"PYTORCH_CUDA_ALLOC_CONF`"] = `"expandable_segments:True`"`n",
        "`n",
        "import sys`n",
        "sys.modules[`"torchvision`"] = None`n",
        "sys.modules[`"torchvision.transforms`"] = None`n",
        "sys.modules[`"torchvision.ops`"] = None`n",
        "`n",
        "import json`n",
        "import shutil`n",
        "import zipfile`n",
        "import torch`n",
        "from pathlib import Path`n",
        "`n",
        "assert torch.cuda.is_available(), `"CUDA GPU required for evaluation!`"`n",
        "major, minor = torch.cuda.get_device_capability()`n",
        "print(f`"GPU: {torch.cuda.get_device_name(0)} (Compute Capability {major}.{minor})`")`n",
        "`n",
        "WORKING_DIR = Path(`"/kaggle/working`")`n",
        "DATA_DIR = WORKING_DIR / `"data`"`n",
        "REPORTS_DIR = WORKING_DIR / `"reports`"`n",
        "REPORTS_DIR.mkdir(parents=True, exist_ok=True)`n",
        "`n",
        "# ── 1. Auto-unpack dataset (sql_grpo_data.zip) ───────────────────────────`n",
        "dev_manifest = DATA_DIR / `"processed`" / `"dev_manifest.jsonl`"`n",
        "`n",
        "def setup_dataset():`n",
        "    if dev_manifest.exists():`n",
        "        return`n",
        "    all_zips = list(Path(`"/kaggle/input`").glob(`"**/*.zip`")) + list(WORKING_DIR.glob(`"*.zip`"))`n",
        "    for z in all_zips:`n",
        "        if any(k in z.name.lower() for k in [`"sql_grpo`", `"data`", `"processed`", `"spider`"]):`n",
        "            try:`n",
        "                with zipfile.ZipFile(z, `"r`") as zf:`n",
        "                    names = zf.namelist()`n",
        "                    if any(`"dev_manifest.jsonl`" in n for n in names):`n",
        "                        print(f`"📦 Unpacking dataset from archive: {z} ...`")`n",
        "                        has_data_root = any(n.startswith(`"data/`") for n in names)`n",
        "                        extract_target = WORKING_DIR if has_data_root else DATA_DIR`n",
        "                        zf.extractall(extract_target)`n",
        "                        print(f`"Extracted into: {extract_target}`")`n",
        "                        break`n",
        "            except Exception as e:`n",
        "                print(f`"Warning inspecting {z}: {e}`")`n",
        "    if not dev_manifest.exists() and (WORKING_DIR / `"processed`" / `"dev_manifest.jsonl`").exists():`n",
        "        for folder in [`"processed`", `"raw`"]:`n",
        "            src = WORKING_DIR / folder`n",
        "            dst = DATA_DIR / folder`n",
        "            if src.exists() and not dst.exists():`n",
        "                shutil.move(src, dst)`n",
        "    if not dev_manifest.exists():`n",
        "        found_manifests = list(Path(`"/kaggle/input`").glob(`"**/dev_manifest.jsonl`"))`n",
        "        if found_manifests:`n",
        "            m_path = found_manifests[0]`n",
        "            dataset_root = m_path.parent.parent if m_path.parent.name == `"processed`" else m_path.parent`n",
        "            print(f`"📁 Found pre-extracted dataset at {dataset_root}. Syncing to {DATA_DIR} ...`")`n",
        "            src_proc = dataset_root / `"processed`" if (dataset_root / `"processed`").exists() else m_path.parent`n",
        "            dst_proc = DATA_DIR / `"processed`"`n",
        "            if src_proc.exists() and not dst_proc.exists():`n",
        "                shutil.copytree(src_proc, dst_proc)`n",
        "            src_raw = dataset_root / `"raw`" if (dataset_root / `"raw`").exists() else (dataset_root / `"data`" / `"raw`")`n",
        "            dst_raw = DATA_DIR / `"raw`"`n",
        "            if src_raw.exists() and not dst_raw.exists():`n",
        "                shutil.copytree(src_raw, dst_raw)`n",
        "`n",
        "setup_dataset()`n",
        "assert dev_manifest.exists(), `"❌ dev_manifest.jsonl missing! Please attach sql_grpo_data.zip in Kaggle.`"`n",
        "`n",
        "# ── 2. Auto-unpack models (GRPO or SFT) ──────────────────────────────────`n",
        "def setup_adapters():`n",
        "    all_zips = list(Path(`"/kaggle/input`").glob(`"**/*.zip`")) + list(WORKING_DIR.glob(`"*.zip`"))`n",
        "    for z in all_zips:`n",
        "        z_lower = z.name.lower()`n",
        "        if `"sft`" in z_lower or `"grpo`" in z_lower or `"rlvr`" in z_lower:`n",
        "            try:`n",
        "                with zipfile.ZipFile(z, `"r`") as zf:`n",
        "                    if any(`"adapter_model.safetensors`" in n for n in zf.namelist()):`n",
        "                        print(f`"📦 Unpacking adapter archive: {z.name} ...`")`n",
        "                        zf.extractall(WORKING_DIR)`n",
        "            except Exception as e:`n",
        "                print(f`"Warning inspecting {z}: {e}`")`n",
        "    # Check for uncompressed adapter folders in /kaggle/input`n",
        "    if not (WORKING_DIR / `"outputs`" / `"grpo`" / `"final_adapter`" / `"adapter_model.safetensors`").exists():`n",
        "        found_grpo = list(Path(`"/kaggle/input`").glob(`"**/final_adapter/adapter_model.safetensors`"))`n",
        "        if found_grpo:`n",
        "            (WORKING_DIR / `"outputs`" / `"grpo`" / `"final_adapter`").mkdir(parents=True, exist_ok=True)`n",
        "            for f in found_grpo[0].parent.glob(`"*`"):`n",
        "                if f.is_file():`n",
        "                    shutil.copy2(f, WORKING_DIR / `"outputs`" / `"grpo`" / `"final_adapter`" / f.name)`n",
        "    if not (WORKING_DIR / `"outputs`" / `"sft`" / `"best_adapter`" / `"adapter_model.safetensors`").exists():`n",
        "        found_sft = list(Path(`"/kaggle/input`").glob(`"**/best_adapter/adapter_model.safetensors`")) + list(WORKING_DIR.glob(`"**/best_adapter/adapter_model.safetensors`"))`n",
        "        candidates = [p for p in found_sft if p.parent != (WORKING_DIR / `"outputs`" / `"sft`" / `"best_adapter`")]`n",
        "        if candidates:`n",
        "            (WORKING_DIR / `"outputs`" / `"sft`" / `"best_adapter`").mkdir(parents=True, exist_ok=True)`n",
        "            for f in candidates[0].parent.glob(`"*`"):`n",
        "                if f.is_file():`n",
        "                    shutil.copy2(f, WORKING_DIR / `"outputs`" / `"sft`" / `"best_adapter`" / f.name)`n",
        "`n",
        "setup_adapters()`n",
        "GRPO_ADAPTER = WORKING_DIR / `"outputs`" / `"grpo`" / `"final_adapter`"`n",
        "SFT_ADAPTER = WORKING_DIR / `"outputs`" / `"sft`" / `"best_adapter`"`n",
        "print(f`"GRPO Adapter exists: {(GRPO_ADAPTER / 'adapter_model.safetensors').exists()}`")`n",
        "print(f`"SFT Adapter exists:  {(SFT_ADAPTER / 'adapter_model.safetensors').exists()}`")"
    )
}
$null = $cells.Add($c2)

# --- Cell 3 (code - Step 3: Hardened Sandbox & Verifier) ---
$c3 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# Step 3: Hardened SQLite Sandbox & AST Safety Verifier`n",
        "import sqlite3`n",
        "import time`n",
        "from collections import Counter, defaultdict`n",
        "from typing import Any, List, Optional, Tuple`n",
        "import sqlglot`n",
        "from sqlglot import exp`n",
        "`n",
        "SAFE_FUNCTIONS = {`"count`", `"sum`", `"avg`", `"min`", `"max`", `"lower`", `"upper`", `"length`", `"substr`", `"round`", `"abs`", `"coalesce`", `"strftime`", `"date`"}`n",
        "`n",
        "def create_authorizer():`n",
        "    def callback(action, arg1, arg2, db, trigger):`n",
        "        if action == sqlite3.SQLITE_SELECT:`n",
        "            return sqlite3.SQLITE_OK`n",
        "        if action == sqlite3.SQLITE_READ:`n",
        "            if (arg1 or `"`").lower() in {`"sqlite_master`", `"sqlite_schema`", `"sqlite_temp_master`"}:`n",
        "                return sqlite3.SQLITE_DENY`n",
        "            return sqlite3.SQLITE_OK`n",
        "        if action == sqlite3.SQLITE_FUNCTION:`n",
        "            if (arg2 or `"`").lower() in SAFE_FUNCTIONS:`n",
        "                return sqlite3.SQLITE_OK`n",
        "            return sqlite3.SQLITE_DENY`n",
        "        return sqlite3.SQLITE_DENY`n",
        "    return callback`n",
        "`n",
        "def execute_sandbox_query(db_path: Path, sql: str, timeout_sec: float = 2.0):`n",
        "    start = time.perf_counter()`n",
        "    deadline = start + timeout_sec`n",
        "    uri = f`"file:{db_path.resolve().as_posix()}?mode=ro`"`n",
        "    conn = None`n",
        "    try:`n",
        "        conn = sqlite3.connect(uri, uri=True, timeout=1.0)`n",
        "        conn.execute(`"PRAGMA query_only = ON`")`n",
        "        conn.set_authorizer(create_authorizer())`n",
        "        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)`n",
        "        cur = conn.cursor()`n",
        "        cur.execute(sql)`n",
        "        rows = cur.fetchmany(5001)`n",
        "        dur_ms = (time.perf_counter() - start) * 1000.0`n",
        "        if len(rows) > 5000:`n",
        "            return False, None, True, dur_ms, `"RESULT_TRUNCATED`"`n",
        "        return True, rows, False, dur_ms, None`n",
        "    except Exception as e:`n",
        "        dur_ms = (time.perf_counter() - start) * 1000.0`n",
        "        return False, None, False, dur_ms, str(e)`n",
        "    finally:`n",
        "        if conn:`n",
        "            try:`n",
        "                conn.close()`n",
        "            except Exception:`n",
        "                pass`n",
        "`n",
        "def parse_sql(text: str) -> Optional[str]:`n",
        "    if not text or `"<sql>`" not in text or `"<\/sql>`" not in text:`n",
        "        return None`n",
        "    idx1 = text.find(`"<sql>`") + len(`"<sql>`")`n",
        "    idx2 = text.find(`"<\/sql>`")`n",
        "    if idx1 >= idx2:`n",
        "        return None`n",
        "    return text[idx1:idx2].strip()`n",
        "`n",
        "def is_ast_safe(sql: str, valid_tables: List[str]) -> bool:`n",
        "    try:`n",
        "        stmts = sqlglot.parse(sql, read=`"sqlite`")`n",
        "        if not stmts or len(stmts) != 1 or not isinstance(stmts[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):`n",
        "            return False`n",
        "        ast = stmts[0]`n",
        "        cte_names = {c.alias.lower() for c in ast.find_all(exp.CTE) if c.alias}`n",
        "        tbls = {t.name.lower() for t in ast.find_all(exp.Table) if t.name} - cte_names`n",
        "        if tbls & {`"sqlite_master`", `"sqlite_schema`", `"sqlite_temp_master`"}:`n",
        "            return False`n",
        "        if valid_tables:`n",
        "            allowed = {t.lower() for t in valid_tables}`n",
        "            if tbls - allowed:`n",
        "                return False`n",
        "        return True`n",
        "    except Exception:`n",
        "        return False`n",
        "`n",
        "print(`"Sandbox verifier and AST safety check initialized.`")"
    )
}
$null = $cells.Add($c3)

# --- Cell 4 (code - Step 4: Load Base Model & Preferred Adapter) ---
$c4 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# Step 4: Load Base Model & Target Adapter (Prefers GRPO, fallbacks to SFT)`n",
        "import transformers.utils.import_utils as _imp`n",
        "_imp.is_torchvision_available = lambda: False`n",
        "`n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig`n",
        "from peft import PeftModel`n",
        "`n",
        "MODEL_ID = `"Qwen/Qwen2.5-Coder-1.5B-Instruct`"`n",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)`n",
        "if tokenizer.pad_token is None:`n",
        "    tokenizer.pad_token = tokenizer.eos_token`n",
        "tokenizer.padding_side = `"left`"`n",
        "`n",
        "major, minor = torch.cuda.get_device_capability()`n",
        "supports_bf16 = (major >= 8)`n",
        "compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16`n",
        "`n",
        "bnb_config = BitsAndBytesConfig(`n",
        "    load_in_4bit=True,`n",
        "    bnb_4bit_quant_type=`"nf4`",`n",
        "    bnb_4bit_compute_dtype=compute_dtype,`n",
        "    bnb_4bit_use_double_quant=True,`n",
        ")`n",
        "`n",
        "base_model = AutoModelForCausalLM.from_pretrained(`n",
        "    MODEL_ID,`n",
        "    quantization_config=bnb_config,`n",
        "    torch_dtype=compute_dtype,`n",
        "    device_map=`"auto`",`n",
        "    trust_remote_code=True,`n",
        ")`n",
        "`n",
        "# Select Best Available Adapter: GRPO > SFT > Base`n",
        "adapter_to_load = None`n",
        "if (GRPO_ADAPTER / `"adapter_model.safetensors`").exists():`n",
        "    adapter_to_load = GRPO_ADAPTER`n",
        "    print(f`"🚀 Evaluating GRPO-Aligned Policy from: {GRPO_ADAPTER}`")`n",
        "elif (SFT_ADAPTER / `"adapter_model.safetensors`").exists():`n",
        "    adapter_to_load = SFT_ADAPTER`n",
        "    print(f`"⚠️ GRPO adapter not found. Evaluating SFT Baseline from: {SFT_ADAPTER}`")`n",
        "else:`n",
        "    print(`"⚠️ No adapter found. Evaluating Base Qwen Model.`")`n",
        "`n",
        "if adapter_to_load:`n",
        "    model = PeftModel.from_pretrained(base_model, str(adapter_to_load))`n",
        "else:`n",
        "    model = base_model`n",
        "`n",
        "model.eval()`n",
        "print(f`"Model ready on {model.device}. Peak VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB`")"
    )
}
$null = $cells.Add($c4)

# --- Cell 5 (code - Step 5: Majority@8 Consensus Voting Engine) ---
$c5 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# Step 5: Majority@8 Consensus Voting Engine (Plurality over Multisets)`n",
        "def majority_8_inference(prompt_messages: list, db_file: Path, valid_tables: list, k: int = 8):`n",
        "    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)`n",
        "    inputs = tokenizer(prompt_text, return_tensors=`"pt`").to(model.device)`n",
        "`n",
        "    with torch.no_grad():`n",
        "        outputs = model.generate(`n",
        "            **inputs,`n",
        "            max_new_tokens=256,`n",
        "            do_sample=True,`n",
        "            temperature=0.8,`n",
        "            top_p=0.95,`n",
        "            num_return_sequences=k,`n",
        "            pad_token_id=tokenizer.pad_token_id,`n",
        "        )`n",
        "`n",
        "    in_len = inputs.input_ids.shape[1]`n",
        "    candidates = []`n",
        "    for i in range(k):`n",
        "        comp = tokenizer.decode(outputs[i][in_len:], skip_special_tokens=True)`n",
        "        sql = parse_sql(comp)`n",
        "        candidates.append(sql)`n",
        "`n",
        "    # Cluster candidates by multiset execution result`n",
        "    clusters = defaultdict(list)`n",
        "    for sql in candidates:`n",
        "        if not sql or not is_ast_safe(sql, valid_tables):`n",
        "            continue`n",
        "        ok, rows, trunc, _, _ = execute_sandbox_query(db_file, sql, timeout_sec=2.0)`n",
        "        if not ok or trunc or rows is None:`n",
        "            continue`n",
        "        # Hashable multiset signature`n",
        "        sig = tuple(sorted(Counter(rows).items()))`n",
        "        clusters[sig].append((sql, rows))`n",
        "`n",
        "    if not clusters:`n",
        "        return None, None, `"ALL_FAILED`"`n",
        "`n",
        "    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)`n",
        "    top_cluster = sorted_clusters[0]`n",
        "`n",
        "    # Tie or plurality check`n",
        "    if len(sorted_clusters) > 1 and len(sorted_clusters[1]) == len(top_cluster):`n",
        "        return top_cluster[0][0], top_cluster[0][1], `"NO_CONSENSUS_TIE`"`n",
        "`n",
        "    # Plurality threshold: at least 3 agreeing candidates`n",
        "    if len(top_cluster) < 3:`n",
        "        return top_cluster[0][0], top_cluster[0][1], `"LOW_CONFIDENCE`"`n",
        "`n",
        "    return top_cluster[0][0], top_cluster[0][1], `"CONSENSUS_ELECTED`""
    )
}
$null = $cells.Add($c5)

# --- Cell 6 (code - Step 6: Benchmark Execution on Held-Out Split) ---
$c6 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# Step 6: Benchmark Execution on Held-Out Split (Pass@1 Greedy vs Majority@8)`n",
        "import numpy as np`n",
        "import pandas as pd`n",
        "`n",
        "dev_manifest = DATA_DIR / `"processed`" / `"dev_manifest.jsonl`"`n",
        "assert dev_manifest.exists(), `"dev_manifest.jsonl missing!`"`n",
        "`n",
        "eval_data = []`n",
        "with open(dev_manifest, `"r`", encoding=`"utf-8`") as f:`n",
        "    for line in f:`n",
        "        if line.strip():`n",
        "            eval_data.append(json.loads(line))`n",
        "`n",
        "EVAL_SIZE = min(50, len(eval_data))  # 50 representative samples for Kaggle runtime`n",
        "eval_subset = eval_data[:EVAL_SIZE]`n",
        "print(f`"Benchmarking {len(eval_subset)} questions with Greedy Pass@1 and Majority@8...`")`n",
        "`n",
        "results = []`n",
        "for idx, row in enumerate(eval_subset):`n",
        "    db_id = row[`"db_id`"]`n",
        "    db_file = DATA_DIR / `"raw`" / `"spider`" / `"database`" / db_id / f`"{db_id}.sqlite`"`n",
        "    gold_raw = json.loads(row[`"gold_result_json`"])`n",
        "    gold_rows = [tuple(x) if isinstance(x, list) else (x,) for x in gold_raw]`n",
        "`n",
        "    # 1. Greedy Pass@1`n",
        "    p_text = tokenizer.apply_chat_template(row[`"prompt`"], tokenize=False, add_generation_prompt=True)`n",
        "    inp = tokenizer(p_text, return_tensors=`"pt`").to(model.device)`n",
        "    t0 = time.perf_counter()`n",
        "    with torch.no_grad():`n",
        "        g_out = model.generate(`n",
        "            **inp,`n",
        "            max_new_tokens=256,`n",
        "            do_sample=False,`n",
        "            temperature=None,`n",
        "            pad_token_id=tokenizer.pad_token_id,`n",
        "        )`n",
        "    g_lat = (time.perf_counter() - t0) * 1000.0`n",
        "    g_comp = tokenizer.decode(g_out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)`n",
        "    g_sql = parse_sql(g_comp)`n",
        "    g_correct = False`n",
        "    if g_sql and is_ast_safe(g_sql, row[`"valid_tables`"]):`n",
        "        ok, r_rows, trunc, _, _ = execute_sandbox_query(db_file, g_sql)`n",
        "        if ok and not trunc and r_rows is not None:`n",
        "            g_correct = (Counter(r_rows) == Counter(gold_rows))`n",
        "`n",
        "    # 2. Majority@8`n",
        "    m_sql, m_rows, m_status = majority_8_inference(row[`"prompt`"], db_file, row[`"valid_tables`"], k=8)`n",
        "    m_correct = False`n",
        "    if m_rows is not None:`n",
        "        m_correct = (Counter(m_rows) == Counter(gold_rows))`n",
        "`n",
        "    results.append({`n",
        "        `"id`": row.get(`"example_id`", idx),`n",
        "        `"greedy_correct`": g_correct,`n",
        "        `"majority8_correct`": m_correct,`n",
        "        `"majority8_status`": m_status,`n",
        "        `"greedy_latency_ms`": g_lat,`n",
        "    })`n",
        "    if (idx + 1) % 10 == 0:`n",
        "        print(f`"  Progress: {idx+1}/{len(eval_subset)} done.`")`n",
        "`n",
        "# Aggregate Table`n",
        "total = len(results)`n",
        "ex_greedy = sum(1 for r in results if r[`"greedy_correct`"]) / total * 100.0`n",
        "ex_maj8 = sum(1 for r in results if r[`"majority8_correct`"]) / total * 100.0`n",
        "p50_lat = np.percentile([r[`"greedy_latency_ms`"] for r in results], 50)`n",
        "p95_lat = np.percentile([r[`"greedy_latency_ms`"] for r in results], 95)`n",
        "`n",
        "summary_df = pd.DataFrame([{`n",
        "    `"Model`": `"SFT+GRPO (Ours)`",`n",
        "    `"Greedy EX (%)`": f`"{ex_greedy:.1f}%`",`n",
        "    `"Majority@8 EX (%)`": f`"{ex_maj8:.1f}%`",`n",
        "    `"Delta (pts)`": f`"{ex_maj8 - ex_greedy:+.1f}`",`n",
        "    `"p50 Latency (ms)`": f`"{p50_lat:.1f}`",`n",
        "    `"p95 Latency (ms)`": f`"{p95_lat:.1f}`",`n",
        "}])`n",
        "print(`"\n================ FINAL BENCHMARK SUMMARY ================`")`n",
        "print(summary_df.to_string(index=False))`n",
        "print(`"=========================================================`")`n",
        "`n",
        "# Save JSON report`n",
        "with open(REPORTS_DIR / `"final_benchmark_results.json`", `"w`", encoding=`"utf-8`") as f:`n",
        "    json.dump(results, f, indent=2)`n",
        "print(f`"Saved report to: {REPORTS_DIR / 'final_benchmark_results.json'}`")"
    )
}
$null = $cells.Add($c6)

# --- Cell 7 (code - Step 7: Final Zipper) ---
$c7 = [PSCustomObject]@{
    cell_type = "code"
    execution_count = $null
    metadata = [PSCustomObject]@{}
    outputs = @()
    source = @(
        "# ==========================================================================`n",
        "# FINAL CELL: MANDATORY ARTIFACT ZIPPER`n",
        "# Zips all evaluation reports, JSONL logs, and plots for download`n",
        "# ==========================================================================`n",
        "import zipfile`n",
        "`n",
        "output_zip_path = WORKING_DIR / `"sql_grpo_evaluation_results.zip`"`n",
        "print(f`"Creating evaluation artifact zip at: {output_zip_path} ...`")`n",
        "`n",
        "with zipfile.ZipFile(output_zip_path, `"w`", zipfile.ZIP_DEFLATED) as zipf:`n",
        "    for root, dirs, files in os.walk(REPORTS_DIR):`n",
        "        for file in files:`n",
        "            p = Path(root) / file`n",
        "            rel = p.relative_to(WORKING_DIR).as_posix()`n",
        "            zipf.write(p, arcname=rel)`n",
        "`n",
        "zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)`n",
        "print(f`"✅ Successfully created: {output_zip_path.name} ({zip_size_mb:.2f} MB)`")`n",
        "from IPython.display import FileLink, display`n",
        "print(`"\n👉 Click the link below to download directly in your browser:`")`n",
        "display(FileLink(`"sql_grpo_evaluation_results.zip`"))"
    )
}
$null = $cells.Add($c7)

$nb4 = [PSCustomObject]@{
    cells = $cells
    metadata = [PSCustomObject]@{
        language_info = [PSCustomObject]@{
            name = "python"
        }
    }
    nbformat = 4
    nbformat_minor = 2
}

$nb4 | ConvertTo-Json -Depth 100 | Set-Content $nb4_path -Encoding UTF8
Write-Host "Successfully generated $nb4_path with $($cells.Count) cells."
