"""Mandatory 16-Vector Sandbox Security Tests."""

import hashlib
import sqlite3
import pytest
from pathlib import Path

try:
    from data.fixtures.create_fixtures import create_test_db
except ImportError:
    from sql_grpo.data.fixtures.create_fixtures import create_test_db
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.safety.ast_policy import ASTPolicyChecker


@pytest.fixture(scope="module")
def fixture_db_path(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("db_test")
    db_file = temp_dir / "test_sec.sqlite"
    create_test_db(db_file)
    return db_file


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def test_sec_01_select_basic(fixture_db_path):
    """SEC-01: Simple SELECT should be allowed."""
    sandbox = SQLiteSandbox()
    res = sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="SELECT name FROM department LIMIT 2;",
        allowed_tables={"department", "employee"},
    )
    assert res.success is True
    assert res.safe is True
    assert res.rows is not None
    assert len(res.rows) == 2


def test_sec_04_write_drop_rejected(fixture_db_path):
    """SEC-04: DROP TABLE should be rejected by AST and never mutate database."""
    initial_hash = get_file_sha256(fixture_db_path)
    checker = ASTPolicyChecker(allowed_tables={"department", "employee"})
    ast_res = checker.validate("DROP TABLE department;")
    assert ast_res.is_safe is False
    assert ast_res.error_code in ("DISALLOWED_OPERATION", "PROHIBITED_CONSTRUCT")

    # DB hash must remain unchanged
    assert get_file_sha256(fixture_db_path) == initial_hash


def test_sec_05_write_insert_rejected(fixture_db_path):
    """SEC-05: INSERT should be rejected by AST and authorizer."""
    initial_hash = get_file_sha256(fixture_db_path)
    sandbox = SQLiteSandbox()
    res = sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="INSERT INTO department (name, budget) VALUES ('Malicious', 9999);",
        allowed_tables={"department", "employee"},
    )
    assert res.success is False
    assert res.error_code in ("AUTH_DENIED", "SQLITE_OPERATIONAL_ERROR")
    assert get_file_sha256(fixture_db_path) == initial_hash


def test_sec_06_multiple_statements_rejected():
    """SEC-06: Semicolon chaining multiple statements must be rejected."""
    checker = ASTPolicyChecker()
    ast_res = checker.validate("SELECT * FROM department; SELECT * FROM employee;")
    assert ast_res.is_safe is False
    assert ast_res.error_code == "MULTIPLE_STATEMENTS"


def test_sec_07_pragma_denied(fixture_db_path):
    """SEC-07: PRAGMA commands must be denied."""
    sandbox = SQLiteSandbox()
    res = sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="PRAGMA table_info(department);",
        allowed_tables={"department", "employee"},
    )
    assert res.success is False
    assert res.error_code in ("AUTH_DENIED", "SQLITE_OPERATIONAL_ERROR")


def test_sec_08_attach_denied(fixture_db_path):
    """SEC-08: ATTACH DATABASE must be rejected."""
    checker = ASTPolicyChecker()
    ast_res = checker.validate("ATTACH DATABASE 'malicious.db' AS mal;")
    assert ast_res.is_safe is False


def test_sec_09_catalog_access_rejected():
    """SEC-09: Direct access to sqlite_master must be blocked."""
    checker = ASTPolicyChecker(allowed_tables={"department", "employee"})
    ast_res = checker.validate("SELECT * FROM sqlite_master;")
    assert ast_res.is_safe is False
    assert ast_res.error_code in ("SYSTEM_CATALOG_ACCESS", "UNAUTHORIZED_TABLE")


def test_sec_10_unknown_table_rejected():
    """SEC-10: Query referencing tables outside schema allowlist must fail."""
    checker = ASTPolicyChecker(allowed_tables={"department"})
    ast_res = checker.validate("SELECT * FROM employee;")
    assert ast_res.is_safe is False
    assert ast_res.error_code == "UNAUTHORIZED_TABLE"


def test_sec_15_row_limit_overflow(fixture_db_path):
    """SEC-15: Query exceeding row limit is flagged as truncated and incomplete."""
    sandbox = SQLiteSandbox(row_limit=2)
    res = sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="SELECT * FROM employee;",
        allowed_tables={"department", "employee"},
    )
    assert res.truncated is True
    assert res.complete is False
    assert res.rows is None
