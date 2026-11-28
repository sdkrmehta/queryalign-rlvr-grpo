"""Tests for Data Acquisition, Schema Serialization, Preprocessing, Splitting, and Validation."""

import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile
import pytest

try:
    from data.fixtures.create_fixtures import create_test_db
except ImportError:
    from sql_grpo.data.fixtures.create_fixtures import create_test_db

from sql_grpo.data.download import acquire_dataset_archive, safe_extract_zip, sha256_file
from sql_grpo.data.preprocess import DataPreprocessor
from sql_grpo.data.schema_serializer import SchemaSerializer
from sql_grpo.data.split_manifest import (
    split_records_disjoint_by_db,
    verify_no_db_leakage,
    write_immutable_manifests,
)
from sql_grpo.data.validate_dataset import DatasetValidator, load_records
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox


def test_schema_serializer_escaping_privacy_and_token_budget(tmp_path):
    db_path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT, password_hash TEXT, avatar BLOB);")
    cur.execute(
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL, "
        "FOREIGN KEY (customer_id) REFERENCES customers(id));"
    )
    cur.execute("CREATE TABLE unrelated_logs (log_id INTEGER PRIMARY KEY, payload TEXT);")
    cur.execute("INSERT INTO customers VALUES (1, 'a@example.com', 's3cr3t_hash', X'DEADBEEF');")
    cur.execute("INSERT INTO orders VALUES (10, 1, 99.5);")
    cur.execute("INSERT INTO unrelated_logs VALUES (1, 'noise line\nwith newline');")
    conn.commit()
    conn.close()

    serializer = SchemaSerializer(max_samples_per_col=3, max_sample_chars=25)
    info = serializer.serialize_database(db_path)

    cust_cols = {c["name"]: c for c in info["tables"]["customers"]["columns"]}
    # Privacy & binary controls
    assert cust_cols["email"]["samples"] == ["a@example.com"]
    assert cust_cols["password_hash"]["samples"] == []
    assert cust_cols["avatar"]["samples"] == []
    assert len(info["schema_hash"]) == 64

    # Under a tight token budget, drop sample noise & prioritize question-matching + FK neighbor tables
    compact = serializer.to_compact_text(
        info,
        max_tokens=95,
        question="What is the total of orders?",
    )
    assert "Table: orders" in compact
    assert "Table: customers" in compact  # Retained via FK edge from orders
    assert "[Schema truncated:" in compact


def test_downloader_checksum_and_zip_slip_guards(tmp_path):
    payload_bytes = b"spider benchmark test payload"
    expected_sha = hashlib.sha256(payload_bytes).hexdigest().upper()
    dest = tmp_path / "raw" / "spider.zip"

    calls = []

    def _mock_fetcher(url: str, target: Path):
        calls.append(url)
        if "broken" in url:
            raise OSError("network unreachable")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload_bytes)

    res = acquire_dataset_archive(
        urls=["https://broken.example/spider.zip", "https://mirror.example/spider.zip"],
        dest_path=dest,
        expected_sha256=expected_sha,
        fetcher=_mock_fetcher,
    )
    assert res["status"] == "downloaded_verified"
    assert res["sha256"] == expected_sha
    assert len(calls) == 2

    # Subsequent call reuses cached verified file without calling fetcher
    res_cached = acquire_dataset_archive(
        urls=["https://mirror.example/spider.zip"],
        dest_path=dest,
        expected_sha256=expected_sha,
        fetcher=_mock_fetcher,
    )
    assert res_cached["status"] == "cached_verified"
    assert len(calls) == 2

    # Checksum mismatch raises ValueError
    with pytest.raises(ValueError, match="SHA-256"):
        acquire_dataset_archive(
            urls=["https://mirror.example/spider.zip"],
            dest_path=dest,
            expected_sha256="0" * 64,
        )

    # Zip-Slip path traversal rejection
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../outside_evil.txt", "pwned")
    with pytest.raises(ValueError, match="unsafe zip entry"):
        safe_extract_zip(bad_zip, tmp_path / "extract_dir")


def test_split_manifest_disjoint_and_leakage_rejection(tmp_path):
    rows = [
        {"example_id": f"ex_{db}_{i}", "db_id": f"db_{db}", "question": f"q{i}"}
        for db in range(10)
        for i in range(2)
    ]
    tr, va, te = split_records_disjoint_by_db(rows, 0.7, 0.15, 0.15, seed=3407)
    verify_no_db_leakage(tr, va, te)

    assert len(tr) + len(va) + len(te) == 20
    assert tr[0]["split"] == "train"
    assert va[0]["split"] == "val"
    assert te[0]["split"] == "test"

    out_dir = tmp_path / "manifests"
    meta = write_immutable_manifests(out_dir, {"train": tr, "val": va, "test": te})
    assert sha256_file(out_dir / "train_manifest.jsonl") == meta["train"]["sha256"]

    with pytest.raises(FileExistsError):
        write_immutable_manifests(out_dir, {"train": tr}, overwrite=False)

    # Leakage detection must reject overlapping DBs
    with pytest.raises(ValueError, match="leakage"):
        verify_no_db_leakage([{"db_id": "shared_db"}], [{"db_id": "shared_db"}])


def test_preprocessor_classifications_and_sidecars(tmp_path):
    data_root = tmp_path / "data"
    create_test_db(data_root / "fixtures" / "test_company.sqlite")
    prep = DataPreprocessor(data_root=data_root, gold_timeout_seconds=1.0)
    sidecar_dir = data_root / "processed" / "results"

    raw_items = [
        {"example_id": "ok_1", "db_id": "test_company", "question": "List depts", "query": "SELECT name FROM department"},
        {"example_id": "emp_1", "db_id": "test_company", "question": "Zero budget depts", "query": "SELECT name FROM department WHERE budget < 0"},
        {"example_id": "ast_1", "db_id": "test_company", "question": "Drop dept", "query": "DROP TABLE department"},
        {"example_id": "badcol_1", "db_id": "test_company", "question": "Bad col", "query": "SELECT ghost FROM department"},
        {"example_id": "err_1", "db_id": "test_company", "question": "Bad arity", "query": "SELECT ROUND(1, 2, 3) FROM department"},
        {"example_id": "miss_1", "db_id": "unknown_db", "question": "Missing DB", "query": "SELECT 1"},
    ]
    retained, excluded, summary = prep.process_dataset(
        raw_items,
        dataset_name="fixture",
        split="train",
        exclude_non_success_for_train=True,
        result_sidecar_dir=sidecar_dir,
    )
    assert [r["example_id"] for r in retained] == ["ok_1", "emp_1"]
    assert retained[0]["gold_status"] == "success"
    assert retained[1]["gold_status"] == "empty" and retained[1]["gold_is_empty"] is True
    assert "valid_columns" in retained[0] and "department" in retained[0]["valid_columns"]
    assert (sidecar_dir / "ok_1.json").is_file()
    assert summary["gold_status_counts"]["ast_rejected"] == 2
    assert summary["gold_status_counts"]["error"] == 1
    assert summary["gold_status_counts"]["missing_db"] == 1
    assert len(excluded) == 4

    # Verify truncated classification explicitly
    prep.sandbox = SQLiteSandbox(row_limit=1)
    trunc_item = prep.process_raw_item("tr_1", "fixture", "test_company", "All employees", "SELECT name FROM employee")
    assert trunc_item is not None and trunc_item["gold_status"] == "truncated"


def test_real_spider_manifests_integrity_and_zero_db_leakage():
    repo_data = Path(__file__).resolve().parents[2] / "data"
    train_path = repo_data / "processed" / "train_manifest.jsonl"
    dev_path = repo_data / "processed" / "dev_manifest.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        pytest.skip("Processed Spider manifests not present")

    train_rows = load_records(train_path)
    dev_rows = load_records(dev_path)
    validator = DatasetValidator(data_root=repo_data)
    report = validator.validate_records(train_rows, eval_records=dev_rows, check_gold_ast=False)

    assert report.passed is True
    assert report.db_overlap == []
    assert report.missing_db_count == 0
    assert report.missing_fields_count == 0
    assert report.duplicate_example_ids == []
    assert len(train_rows) == 7000
    assert len(dev_rows) == 1034
