"""AST Policy Enforcer: SQLGlot SQLite parser and structural allowlisting."""

from dataclasses import dataclass
from typing import List, Optional, Set
import sqlglot
from sqlglot import exp


@dataclass
class ASTValidationResult:
    is_safe: bool
    ast: Optional[exp.Expression] = None
    tables_found: Set[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


FORBIDDEN_CATALOG_TABLES = {
    "sqlite_master",
    "sqlite_schema",
    "sqlite_temp_master",
    "sqlite_temp_schema",
}


class ASTPolicyChecker:
    """Enforces SELECT-only allowlists and schema integrity via SQLGlot AST."""

    def __init__(self, allowed_tables: Optional[Set[str]] = None):
        self.allowed_tables = {t.lower() for t in allowed_tables} if allowed_tables else None

    def validate(self, sql: str) -> ASTValidationResult:
        if not sql or not sql.strip():
            return ASTValidationResult(
                is_safe=False,
                error_code="EMPTY_SQL",
                error_message="SQL string is empty",
            )

        # 1. Parse statement count
        try:
            parsed_statements = sqlglot.parse(sql, read="sqlite")
        except Exception as e:
            return ASTValidationResult(
                is_safe=False,
                error_code="PARSE_ERROR",
                error_message=f"SQLGlot failed to parse query: {str(e)}",
            )

        if len(parsed_statements) == 0 or parsed_statements[0] is None:
            return ASTValidationResult(
                is_safe=False,
                error_code="EMPTY_STATEMENT",
                error_message="No valid SQL statement found",
            )

        if len(parsed_statements) > 1:
            return ASTValidationResult(
                is_safe=False,
                error_code="MULTIPLE_STATEMENTS",
                error_message=f"Expected exactly 1 statement, got {len(parsed_statements)}",
            )

        ast = parsed_statements[0]

        # 2. Enforce SELECT-family root expression
        allowed_root_types = (
            exp.Select,
            exp.Union,
            exp.Intersect,
            exp.Except,
        )
        if not isinstance(ast, allowed_root_types):
            return ASTValidationResult(
                is_safe=False,
                error_code="DISALLOWED_OPERATION",
                error_message=f"Root AST expression {type(ast).__name__} is not in SELECT family",
            )

        # 3. Explicit check for prohibited DML / DDL / Administration anywhere in the AST
        alter_cls = getattr(exp, "AlterTable", getattr(exp, "Alter", None))
        prohibited_types = tuple(
            t
            for t in (
                exp.Insert,
                exp.Update,
                exp.Delete,
                exp.Create,
                exp.Drop,
                alter_cls,
                exp.Pragma,
                exp.Command,
                exp.Transaction,
            )
            if t is not None
        )
        for prohibited in ast.find_all(prohibited_types):
            return ASTValidationResult(
                is_safe=False,
                error_code="PROHIBITED_CONSTRUCT",
                error_message=f"Query contains prohibited construct: {type(prohibited).__name__}",
            )

        # 4. Resolve CTE aliases
        cte_aliases: Set[str] = set()
        with_clause = ast.find(exp.With)
        if with_clause:
            for cte in with_clause.find_all(exp.CTE):
                if cte.alias:
                    cte_aliases.add(cte.alias.lower())

        # 5. Extract and validate physical table references
        physical_tables: Set[str] = set()
        for table_node in ast.find_all(exp.Table):
            table_name = table_node.name.lower() if table_node.name else ""
            if not table_name:
                continue

            # If it's a defined CTE, it is not a physical table
            if table_name in cte_aliases:
                continue

            # Deny system catalog tables
            if table_name in FORBIDDEN_CATALOG_TABLES:
                return ASTValidationResult(
                    is_safe=False,
                    error_code="SYSTEM_CATALOG_ACCESS",
                    error_message=f"Access to SQLite system catalog table '{table_name}' is forbidden",
                )

            physical_tables.add(table_name)

        # Check against schema whitelist
        if self.allowed_tables is not None:
            unauthorized = physical_tables - self.allowed_tables
            if unauthorized:
                return ASTValidationResult(
                    is_safe=False,
                    error_code="UNAUTHORIZED_TABLE",
                    error_message=f"Query references unapproved table(s): {unauthorized}",
                    tables_found=physical_tables,
                )

        return ASTValidationResult(
            is_safe=True,
            ast=ast,
            tables_found=physical_tables,
        )
