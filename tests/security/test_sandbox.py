"""Mandatory Defense-in-Depth Sandbox & AST Security Test Matrix."""

import hashlib
from pathlib import Path
import time
import pytest

try:
    from data.fixtures.create_fixtures import create_test_db
except ImportError:
    from sql_grpo.data.fixtures.create_fixtures import create_test_db

from sql_grpo.env.db_registry import DatabaseRegistry
from sql_grpo.env.sqlite_sandbox import SQLiteSandbox
from sql_grpo.env.worker import SandboxedWorkerSupervisor
from sql_grpo.safety.ast_policy import ASTPolicyChecker

TEST_TABLES = {"department", "employee"}
TEST_COLUMNS = {
    "department": {"dept_id", "name", "budget"},
    "employee": {"emp_id", "name", "dept_id", "salary"},
}


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
        allowed_tables=TEST_TABLES,
        allowed_columns=TEST_COLUMNS,
    )
    assert res.success is True
    assert res.safe is True
    assert res.complete is True
    assert res.rows is not None
    assert len(res.rows) == 2


def test_sec_02_select_with_where_and_join(fixture_db_path):
    """SEC-02: SELECT with WHERE and FK JOIN over approved tables/columns is allowed."""
    sql = (
        "SELECT e.name, d.name FROM employee AS e "
        "JOIN department AS d ON e.dept_id = d.dept_id "
        "WHERE d.budget > 0 ORDER BY e.emp_id"
    )
    ast_res = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=TEST_COLUMNS).validate(sql)
    assert ast_res.is_safe is True

    res = SQLiteSandbox().execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql=sql,
        allowed_tables=TEST_TABLES,
        allowed_columns=TEST_COLUMNS,
    )
    assert res.success is True
    assert res.complete is True


def test_sec_03_cte_and_union_allowed(fixture_db_path):
    """SEC-03: Non-recursive CTEs and UNIONs over approved tables/columns are allowed."""
    cte_sql = (
        "WITH dept_counts AS (SELECT dept_id, COUNT(*) AS cnt FROM employee GROUP BY dept_id) "
        "SELECT dept_id, cnt FROM dept_counts ORDER BY cnt DESC"
    )
    union_sql = "SELECT name FROM department UNION SELECT name FROM employee"

    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=TEST_COLUMNS)
    assert checker.validate(cte_sql).is_safe is True
    assert checker.validate(union_sql).is_safe is True

    sandbox = SQLiteSandbox()
    assert sandbox.execute(fixture_db_path, "test_sec", cte_sql, TEST_TABLES, TEST_COLUMNS).success is True
    assert sandbox.execute(fixture_db_path, "test_sec", union_sql, TEST_TABLES, TEST_COLUMNS).success is True


def test_sec_04_write_drop_rejected(fixture_db_path):
    """SEC-04: DROP/CREATE/ALTER statements must be rejected by AST and never mutate database."""
    initial_hash = get_file_sha256(fixture_db_path)
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=TEST_COLUMNS)
    for bad_sql in (
        "DROP TABLE department;",
        "ALTER TABLE department ADD COLUMN hacked TEXT;",
        "CREATE TABLE evil (id INT);",
    ):
        ast_res = checker.validate(bad_sql)
        assert ast_res.is_safe is False
        assert ast_res.error_code in ("DISALLOWED_OPERATION", "PROHIBITED_CONSTRUCT")

    assert get_file_sha256(fixture_db_path) == initial_hash


def test_sec_05_write_insert_rejected(fixture_db_path):
    """SEC-05: INSERT/UPDATE/DELETE rejected by both AST and SQLite authorizer; file unchanged."""
    initial_hash = get_file_sha256(fixture_db_path)
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES)
    sandbox = SQLiteSandbox()

    for write_sql in (
        "INSERT INTO department (name, budget) VALUES ('Malicious', 9999);",
        "UPDATE department SET budget = 0;",
        "DELETE FROM employee;",
    ):
        assert checker.validate(write_sql).is_safe is False
        res = sandbox.execute(
            db_path=fixture_db_path,
            database_id="test_sec",
            sql=write_sql,
            allowed_tables=TEST_TABLES,
        )
        assert res.success is False
        assert res.error_code in ("AUTH_DENIED", "SQLITE_OPERATIONAL_ERROR")

    assert get_file_sha256(fixture_db_path) == initial_hash


def test_sec_06_multiple_statements_rejected(fixture_db_path):
    """SEC-06: Semicolon chaining multiple statements must be rejected by AST and sandbox."""
    initial_hash = get_file_sha256(fixture_db_path)
    payload = "SELECT * FROM department; DROP TABLE employee;"
    ast_res = ASTPolicyChecker().validate(payload)
    assert ast_res.is_safe is False
    assert ast_res.error_code == "MULTIPLE_STATEMENTS"

    res = SQLiteSandbox().execute(fixture_db_path, "test_sec", payload, TEST_TABLES)
    assert res.success is False
    assert res.safe is False
    assert res.error_code == "MULTIPLE_STATEMENTS"
    assert get_file_sha256(fixture_db_path) == initial_hash


def test_sec_07_pragma_denied(fixture_db_path):
    """SEC-07: PRAGMA statements and pragma_* table-valued function bypasses must be denied."""
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=TEST_COLUMNS)
    sandbox = SQLiteSandbox()

    res_pragma = sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="PRAGMA table_info(department);",
        allowed_tables=TEST_TABLES,
    )
    assert res_pragma.success is False
    assert res_pragma.error_code in ("AUTH_DENIED", "SQLITE_OPERATIONAL_ERROR")

    tvf_bypass = "SELECT * FROM pragma_table_info('department');"
    ast_tvf = checker.validate(tvf_bypass)
    assert ast_tvf.is_safe is False
    assert ast_tvf.error_code == "DISALLOWED_FUNCTION"

    # Even without AST or table filter, authorizer must block pragma_* virtual tables directly.
    res_tvf = sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql=tvf_bypass,
        allowed_tables=None,
    )
    assert res_tvf.success is False
    assert res_tvf.error_code == "AUTH_DENIED"


def test_sec_08_attach_denied(fixture_db_path):
    """SEC-08: ATTACH/DETACH and non-main schema qualifiers (e.g. temp.*) must be rejected."""
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES)
    assert checker.validate("ATTACH DATABASE 'malicious.db' AS mal;").is_safe is False
    assert checker.validate("DETACH DATABASE main;").is_safe is False

    cross_schema = checker.validate("SELECT * FROM temp.department;")
    assert cross_schema.is_safe is False
    assert cross_schema.error_code == "UNAUTHORIZED_SCHEMA"


def test_sec_09_catalog_access_rejected(fixture_db_path):
    """SEC-09: Direct access to sqlite_master/sqlite_schema/sqlite_sequence must be blocked."""
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES)
    sandbox = SQLiteSandbox()
    for catalog_tbl in ("sqlite_master", "sqlite_schema", "sqlite_temp_master", "sqlite_sequence"):
        sql = f"SELECT * FROM {catalog_tbl};"
        ast_res = checker.validate(sql)
        assert ast_res.is_safe is False
        assert ast_res.error_code in ("SYSTEM_CATALOG_ACCESS", "UNAUTHORIZED_TABLE")

        # Authorizer alone (without table filter) must also block catalog access.
        res = sandbox.execute(fixture_db_path, "test_sec", sql, allowed_tables=None)
        assert res.success is False
        assert res.error_code in ("AUTH_DENIED", "SQLITE_OPERATIONAL_ERROR")


def test_sec_10_unknown_table_rejected():
    """SEC-10: Unknown tables and empty allowlists (`set()`) must fail before execution."""
    ast_res = ASTPolicyChecker(allowed_tables={"department"}).validate("SELECT * FROM employee;")
    assert ast_res.is_safe is False
    assert ast_res.error_code == "UNAUTHORIZED_TABLE"

    empty_allow = ASTPolicyChecker(allowed_tables=set()).validate("SELECT * FROM department;")
    assert empty_allow.is_safe is False
    assert empty_allow.error_code == "UNAUTHORIZED_TABLE"


def test_sec_11_unknown_column_rejected_ast_and_authorizer(fixture_db_path):
    """SEC-11: Unauthorized columns must be blocked by ASTPolicyChecker and SQLite authorizer."""
    restricted_cols = {"department": {"dept_id", "name"}, "employee": {"emp_id", "name"}}
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=restricted_cols)

    bad_unqualified = checker.validate("SELECT salary FROM employee;")
    assert bad_unqualified.is_safe is False
    assert bad_unqualified.error_code == "UNAUTHORIZED_COLUMN"

    bad_aliased = checker.validate("SELECT d.budget FROM department AS d;")
    assert bad_aliased.is_safe is False
    assert bad_aliased.error_code == "UNAUTHORIZED_COLUMN"

    # Even if double-quoted to exercise authorizer directly, SQLITE_READ must deny hidden column `budget`.
    res = SQLiteSandbox().execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql='SELECT "budget" FROM department;',
        allowed_tables=TEST_TABLES,
        allowed_columns=restricted_cols,
    )
    assert res.success is False
    assert res.error_code == "AUTH_DENIED"


def test_sec_12_disallowed_function_and_recursive_cte_rejected(fixture_db_path):
    """SEC-12: Disallowed functions and recursive CTEs must be rejected by AST and Authorizer."""
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=TEST_COLUMNS)
    sandbox = SQLiteSandbox()

    for bad_fn in (
        "SELECT load_extension('evil.so');",
        "SELECT randomblob(100000000);",
        "SELECT sqlite_version();",
    ):
        ast_res = checker.validate(bad_fn)
        assert ast_res.is_safe is False
        assert ast_res.error_code == "DISALLOWED_FUNCTION"

        res = sandbox.execute(fixture_db_path, "test_sec", bad_fn, allowed_tables=TEST_TABLES)
        assert res.success is False
        assert res.error_code in ("AUTH_DENIED", "SQLITE_OPERATIONAL_ERROR")

    rec_sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r;"
    assert checker.validate(rec_sql).is_safe is False


def test_sec_13_comment_whitespace_obfuscation(fixture_db_path):
    """SEC-13: Obfuscated comments/whitespace must still follow AST policy accurately."""
    checker = ASTPolicyChecker(allowed_tables=TEST_TABLES, allowed_columns=TEST_COLUMNS)

    benign_commented = "SELECT /* ; DROP TABLE department; */ name FROM department LIMIT 1;"
    assert checker.validate(benign_commented).is_safe is True

    malicious_commented = "SELECT name FROM department /* safe */; /* evil */ DELETE FROM department;"
    assert checker.validate(malicious_commented).is_safe is False
    assert checker.validate("/* leading comment */ DELETE FROM employee;").is_safe is False


def test_sec_14_progress_handler_timeout_and_worker_kill(fixture_db_path):
    """SEC-14: VM progress handler interrupts long SQL; Worker kills hung subprocess cleanly."""
    # 11-way self-cross join (~5^11 = 48.8M iterations) triggers both short VM timeouts and hard worker kills.
    aliases = [f"e{i}" for i in range(11)]
    cross_clause = " CROSS JOIN ".join(f"employee {a}" for a in aliases)
    sum_expr = " + ".join(f"{a}.emp_id" for a in aliases)
    heavy_sql = f"SELECT COUNT(*) FROM {cross_clause} WHERE {sum_expr} > -1"

    vm_sandbox = SQLiteSandbox(timeout_seconds=0.005, progress_opcodes=200)
    res_vm = vm_sandbox.execute(fixture_db_path, "test_sec", heavy_sql, TEST_TABLES, TEST_COLUMNS)
    assert res_vm.success is False
    assert res_vm.error_code == "TIMEOUT"

    # Now test supervisor hard wall-clock termination (query_timeout > hard_timeout so child hits hard kill)
    supervisor = SandboxedWorkerSupervisor(hard_timeout_seconds=0.35)
    t0 = time.perf_counter()
    res_hard = supervisor.run_query(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql=heavy_sql,
        allowed_tables=TEST_TABLES,
        allowed_columns=TEST_COLUMNS,
        query_timeout=20.0,
        progress_opcodes=50000,
    )
    elapsed = time.perf_counter() - t0
    assert res_hard.success is False
    assert res_hard.safe is False
    assert res_hard.error_code == "HARD_TIMEOUT_KILLED"
    assert elapsed < 3.0

    # Verify healthy subprocess-isolated execution via SQLiteSandbox(isolate_process=True)
    iso_sandbox = SQLiteSandbox(isolate_process=True, timeout_seconds=2.0, hard_timeout_seconds=5.0)
    res_ok = iso_sandbox.execute(
        fixture_db_path, "test_sec", "SELECT name FROM department LIMIT 1;", TEST_TABLES, TEST_COLUMNS
    )
    assert res_ok.success is True
    assert res_ok.complete is True
    assert res_ok.rows is not None and len(res_ok.rows) == 1


def test_sec_15_row_and_byte_limit_overflow(fixture_db_path):
    """SEC-15: Exceeding either row_limit or byte_limit marks result truncated & incomplete."""
    row_cap_sandbox = SQLiteSandbox(row_limit=2)
    res_row = row_cap_sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="SELECT * FROM employee;",
        allowed_tables=TEST_TABLES,
    )
    assert res_row.truncated is True
    assert res_row.complete is False
    assert res_row.rows is None
    assert res_row.error_code == "RESULT_TRUNCATED"

    byte_cap_sandbox = SQLiteSandbox(row_limit=5000, byte_limit=20)
    res_byte = byte_cap_sandbox.execute(
        db_path=fixture_db_path,
        database_id="test_sec",
        sql="SELECT * FROM employee;",
        allowed_tables=TEST_TABLES,
    )
    assert res_byte.truncated is True
    assert res_byte.complete is False
    assert res_byte.rows is None
    assert res_byte.error_code == "RESULT_TRUNCATED"


def test_sec_16_registry_path_traversal_and_missing_db(tmp_path):
    """SEC-16: Path traversal and unapproved database paths/IDs must be rejected before opening."""
    data_root = tmp_path / "approved_data"
    fixtures = data_root / "fixtures"
    fixtures.mkdir(parents=True)
    outside_db = tmp_path / "secret.sqlite"
    create_test_db(outside_db)

    reg = DatabaseRegistry(data_root=data_root)
    with pytest.raises(ValueError):
        reg.register("secret", outside_db)
    with pytest.raises(ValueError):
        reg.register("../evil", fixtures / "evil.sqlite")
    with pytest.raises(KeyError):
        reg.get_path("../../etc/passwd")
    with pytest.raises(KeyError):
        reg.get_path("missing_database")
