"""AST Policy Enforcer: SQLGlot SQLite allowlist parser, schema validator, and function guard."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Set, Tuple, Type, Union
import sqlglot
from sqlglot import exp

from sql_grpo.env.authorizer import ALLOWED_DB_SCHEMAS, FORBIDDEN_TABLES, SAFE_SQL_FUNCTIONS


@dataclass
class ASTValidationResult:
    is_safe: bool
    ast: Optional[exp.Expression] = None
    tables_found: Optional[Set[str]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


FORBIDDEN_CATALOG_TABLES: Set[str] = set(FORBIDDEN_TABLES)

ALLOWED_ROOT_TYPES: Tuple[Type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
)

# Clauses where SELECT-level output aliases (e.g. SELECT COUNT(*) AS cnt ... ORDER BY cnt) are valid in SQLite.
_ALIAS_REF_CLAUSES: Tuple[Type[exp.Expression], ...] = (
    exp.Order,
    exp.Ordered,
    exp.Group,
    exp.Having,
)

# Structural / arithmetic / comparison AST nodes allowed inside a SELECT tree.
# Anything outside these types or approved SQL functions is rejected (default-deny).
_ALLOWED_STRUCTURAL_NODES: Tuple[Type[exp.Expression], ...] = tuple(
    t
    for t in (
        exp.Select,
        exp.Union,
        exp.Intersect,
        exp.Except,
        exp.With,
        exp.CTE,
        exp.From,
        exp.Join,
        exp.Where,
        exp.Group,
        exp.Having,
        exp.Order,
        exp.Ordered,
        exp.Limit,
        exp.Offset,
        exp.Distinct,
        exp.Window,
        getattr(exp, "WindowSpec", None),
        exp.Subquery,
        exp.Alias,
        exp.TableAlias,
        exp.Table,
        exp.Column,
        exp.Star,
        exp.Identifier,
        exp.Literal,
        exp.Boolean,
        exp.Null,
        exp.Paren,
        exp.Tuple,
        exp.DataType,
        getattr(exp, "DataTypeParam", None),
        getattr(exp, "PseudoType", None),
        getattr(exp, "Values", None),  # Note: root Values is still blocked separately below
        exp.Case,
        exp.If,
        exp.Exists,
        exp.Cast,
        getattr(exp, "TryCast", None),
        exp.Collate,
        # Predicate / logical / arithmetic operators
        exp.And,
        exp.Or,
        exp.Not,
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        getattr(exp, "NullSafeEQ", None),
        getattr(exp, "NullSafeNEQ", None),
        exp.Is,
        exp.In,
        exp.Between,
        exp.Like,
        getattr(exp, "ILike", None),
        getattr(exp, "Glob", None),
        getattr(exp, "RegexpLike", None),
        getattr(exp, "MatchAgainst", None),
        getattr(exp, "All", None),
        getattr(exp, "Any", None),
        exp.Add,
        exp.Sub,
        exp.Mul,
        exp.Div,
        getattr(exp, "IntDiv", None),
        exp.Mod,
        exp.Neg,
        getattr(exp, "DPipe", None),
        getattr(exp, "Concat", None),
        getattr(exp, "BitwiseAnd", None),
        getattr(exp, "BitwiseOr", None),
        getattr(exp, "BitwiseLeftShift", None),
        getattr(exp, "BitwiseRightShift", None),
        getattr(exp, "BitwiseNot", None),
        getattr(exp, "Slice", None),
        getattr(exp, "Filter", None),
        getattr(exp, "WithinGroup", None),
        getattr(exp, "IgnoreNulls", None),
        getattr(exp, "RespectNulls", None),
        getattr(exp, "Over", None),
        getattr(exp, "PartitionedByProperty", None),
        getattr(exp, "NullsFirst", None),
        getattr(exp, "NullsLast", None),
        getattr(exp, "Unnest", None),
    )
    if t is not None
)

# Explicit prohibited expressions anywhere in the AST for clear error reporting.
_PROHIBITED_TYPES: Tuple[Type[exp.Expression], ...] = tuple(
    t
    for t in (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        getattr(exp, "AlterTable", getattr(exp, "Alter", None)),
        exp.Pragma,
        exp.Command,
        exp.Transaction,
        getattr(exp, "Commit", None),
        getattr(exp, "Rollback", None),
        getattr(exp, "Attach", None),
        getattr(exp, "Detach", None),
        getattr(exp, "Set", None),
        getattr(exp, "Use", None),
        getattr(exp, "Analyze", None),
        getattr(exp, "Returning", None),
        getattr(exp, "Merge", None),
    )
    if t is not None
)


class ASTPolicyChecker:
    """Enforces default-deny SELECT-only AST allowlists, schemas, tables, columns, and functions."""

    def __init__(
        self,
        allowed_tables: Optional[Iterable[str]] = None,
        allowed_columns: Optional[Union[Iterable[str], Mapping[str, Iterable[str]]]] = None,
        allowed_functions: Optional[Iterable[str]] = None,
    ):
        # Explicitly distinguish `None` (no allowlist passed) from empty set `set()` (zero tables allowed).
        self.allowed_tables: Optional[Set[str]] = (
            {str(t).lower() for t in allowed_tables} if allowed_tables is not None else None
        )

        self.allowed_columns_by_table: Optional[Dict[str, Set[str]]] = None
        self.allowed_columns_flat: Optional[Set[str]] = None
        if allowed_columns is not None:
            if isinstance(allowed_columns, Mapping):
                self.allowed_columns_by_table = {
                    str(tbl).lower(): {str(c).lower() for c in cols}
                    for tbl, cols in allowed_columns.items()
                }
            else:
                self.allowed_columns_flat = {str(c).lower() for c in allowed_columns}

        self.allowed_functions: Set[str] = (
            {str(f).lower() for f in allowed_functions}
            if allowed_functions is not None
            else set(SAFE_SQL_FUNCTIONS)
        )

    @staticmethod
    def _enclosing_select(node: exp.Expression) -> Optional[exp.Select]:
        cur = node.parent
        while cur is not None:
            if isinstance(cur, exp.Select):
                return cur
            cur = cur.parent
        return None

    @staticmethod
    def _inside_alias_ref_clause(node: exp.Column) -> bool:
        cur = node.parent
        while cur is not None and not isinstance(cur, exp.Select):
            if isinstance(cur, _ALIAS_REF_CLAUSES):
                return True
            cur = cur.parent
        return False

    @staticmethod
    def _projected_columns_of_select(select_node: exp.Select) -> Tuple[Set[str], bool]:
        """Collects explicitly projected output names/aliases of a SELECT node and whether `*` is present."""
        names: Set[str] = set()
        has_star = False
        for expr in select_node.expressions:
            if isinstance(expr, exp.Star):
                has_star = True
            elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
                has_star = True
            elif isinstance(expr, exp.Alias) and expr.alias:
                names.add(expr.alias.lower())
            elif isinstance(expr, exp.Column) and expr.name:
                names.add(expr.name.lower())
        return names, has_star

    @classmethod
    def _collect_virtual_relations(
        cls, ast: exp.Expression
    ) -> Tuple[Set[str], Dict[str, Tuple[Set[str], bool]], Dict[int, Dict[str, str]], Dict[int, Set[str]]]:
        """Maps CTEs, derived subqueries, and per-SELECT table aliases & physical tables."""
        cte_aliases: Set[str] = set()
        virtual_projections: Dict[str, Tuple[Set[str], bool]] = {}

        for with_node in ast.find_all(exp.With):
            for cte in with_node.find_all(exp.CTE):
                if cte.alias:
                    alias_l = cte.alias.lower()
                    cte_aliases.add(alias_l)
                    if isinstance(cte.this, exp.Select):
                        virtual_projections[alias_l] = cls._projected_columns_of_select(cte.this)
                    else:
                        virtual_projections[alias_l] = (set(), True)

        for subq in ast.find_all(exp.Subquery):
            if subq.alias:
                alias_l = subq.alias.lower()
                if isinstance(subq.this, exp.Select):
                    virtual_projections[alias_l] = cls._projected_columns_of_select(subq.this)
                else:
                    virtual_projections[alias_l] = (set(), True)

        select_table_aliases: Dict[int, Dict[str, str]] = {}
        select_physical_tables: Dict[int, Set[str]] = {}
        for sel in ast.find_all(exp.Select):
            sid = id(sel)
            aliases: Dict[str, str] = {}
            phys: Set[str] = set()
            for tbl in sel.find_all(exp.Table):
                # Only treat tables belonging directly to this SELECT scope (not deeper nested SELECTs)
                if cls._enclosing_select(tbl) is not sel:
                    continue
                tname = tbl.name.lower() if tbl.name else ""
                if not tname or tname in cte_aliases or tname in virtual_projections:
                    continue
                phys.add(tname)
                if tbl.alias:
                    aliases[tbl.alias.lower()] = tname
            select_table_aliases[sid] = aliases
            select_physical_tables[sid] = phys

        return cte_aliases | set(virtual_projections.keys()), virtual_projections, select_table_aliases, select_physical_tables

    def _validate_columns(self, ast: exp.Expression) -> Optional[ASTValidationResult]:
        virtual_names, virtual_projections, select_table_aliases, select_physical_tables = (
            self._collect_virtual_relations(ast)
        )

        for col in ast.find_all(exp.Column):
            if isinstance(col.this, exp.Star) or not col.name:
                continue

            col_name = col.name.lower()
            tbl_ref = col.table.lower() if col.table else ""

            # SQLite historically parses double-quoted string literals (e.g. WHERE Status != "Village")
            # unless the identifier is table-qualified or actually matches a schema column.
            # Unqualified quoted values inside comparison/value positions fall through to SQLite's
            # runtime authorizer, which still denies SQLITE_READ for any real hidden column.
            if (
                not tbl_ref
                and isinstance(col.this, exp.Identifier)
                and col.this.quoted
                and isinstance(
                    col.parent,
                    (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like, exp.Between, exp.Tuple, exp.Not),
                )
            ):
                continue

            sel = self._enclosing_select(col)

            # Resolve aliases and visible physical tables walking up correlated SELECT scopes
            alias_map: Dict[str, str] = {}
            visible_phys_tables: Set[str] = set()
            visible_virtuals: Set[str] = set()
            select_output_aliases: Set[str] = set()

            cur = sel
            while cur is not None:
                if isinstance(cur, exp.Select):
                    alias_map.update({k: v for k, v in select_table_aliases.get(id(cur), {}).items() if k not in alias_map})
                    visible_phys_tables |= select_physical_tables.get(id(cur), set())
                    for expr in cur.expressions:
                        if isinstance(expr, exp.Alias) and expr.alias:
                            select_output_aliases.add(expr.alias.lower())
                    for tbl in cur.find_all(exp.Table):
                        if self._enclosing_select(tbl) is cur and tbl.name:
                            tn = tbl.name.lower()
                            if tn in virtual_names:
                                visible_virtuals.add(tn)
                    for subq in cur.find_all(exp.Subquery):
                        if self._enclosing_select(subq) is cur and subq.alias:
                            visible_virtuals.add(subq.alias.lower())
                cur = cur.parent

            if tbl_ref:
                if tbl_ref in virtual_names:
                    proj_names, has_star = virtual_projections.get(tbl_ref, (set(), True))
                    if not has_star and col_name not in proj_names:
                        return ASTValidationResult(
                            is_safe=False,
                            error_code="UNAUTHORIZED_COLUMN",
                            error_message=f"Column '{tbl_ref}.{col_name}' is not in CTE/subquery projection",
                        )
                    continue

                real_tbl = alias_map.get(tbl_ref, tbl_ref)
                if self.allowed_columns_by_table is not None:
                    allowed_cols = self.allowed_columns_by_table.get(real_tbl)
                    if allowed_cols is None or col_name not in allowed_cols:
                        return ASTValidationResult(
                            is_safe=False,
                            error_code="UNAUTHORIZED_COLUMN",
                            error_message=f"Column '{real_tbl}.{col_name}' is not in approved column allowlist",
                        )
                elif self.allowed_columns_flat is not None:
                    if col_name not in self.allowed_columns_flat:
                        return ASTValidationResult(
                            is_safe=False,
                            error_code="UNAUTHORIZED_COLUMN",
                            error_message=f"Column '{col_name}' is not in approved column allowlist",
                        )
            else:
                # Output alias reference inside ORDER BY / GROUP BY / HAVING (e.g. ORDER BY cnt)
                if col_name in select_output_aliases and self._inside_alias_ref_clause(col):
                    continue

                # Projected column from a visible CTE or derived subquery
                matched_virtual = False
                for vname in visible_virtuals:
                    proj_names, has_star = virtual_projections.get(vname, (set(), False))
                    if col_name in proj_names:
                        matched_virtual = True
                        break
                    if has_star:
                        # Star projection passes if any underlying allowed column exists; checked below if not
                        pass
                if matched_virtual:
                    continue

                if self.allowed_columns_by_table is not None:
                    if not visible_phys_tables or not any(
                        col_name in self.allowed_columns_by_table.get(tbl, set())
                        for tbl in visible_phys_tables
                    ):
                        return ASTValidationResult(
                            is_safe=False,
                            error_code="UNAUTHORIZED_COLUMN",
                            error_message=f"Column '{col_name}' is not permitted for tables in scope",
                        )
                elif self.allowed_columns_flat is not None:
                    if col_name not in self.allowed_columns_flat:
                        return ASTValidationResult(
                            is_safe=False,
                            error_code="UNAUTHORIZED_COLUMN",
                            error_message=f"Column '{col_name}' is not in approved column allowlist",
                        )

        return None

    def validate(self, sql: str) -> ASTValidationResult:
        if not sql or not sql.strip():
            return ASTValidationResult(
                is_safe=False,
                error_code="EMPTY_SQL",
                error_message="SQL string is empty",
            )

        # 1. Parse statement count cleanly under explicit SQLite dialect
        try:
            parsed_statements = sqlglot.parse(sql, read="sqlite")
        except Exception as e:
            return ASTValidationResult(
                is_safe=False,
                error_code="PARSE_ERROR",
                error_message=f"SQLGlot failed to parse query: {type(e).__name__}",
            )

        if len(parsed_statements) == 0 or any(st is None for st in parsed_statements):
            if len(parsed_statements) > 1:
                return ASTValidationResult(
                    is_safe=False,
                    error_code="MULTIPLE_STATEMENTS",
                    error_message=f"Expected exactly 1 statement, got {len(parsed_statements)}",
                )
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

        # 2. Enforce SELECT-family root expression (Values/DDL/DML roots fail here)
        if not isinstance(ast, ALLOWED_ROOT_TYPES):
            return ASTValidationResult(
                is_safe=False,
                error_code="DISALLOWED_OPERATION",
                error_message=f"Root AST expression {type(ast).__name__} is not in SELECT family",
            )

        # 3. Explicit check for prohibited constructs anywhere in the AST
        for prohibited in ast.find_all(_PROHIBITED_TYPES):
            return ASTValidationResult(
                is_safe=False,
                error_code="PROHIBITED_CONSTRUCT",
                error_message=f"Query contains prohibited construct: {type(prohibited).__name__}",
            )

        # Disallow recursive CTEs in AST policy to guard against infinite recursion DoS
        for with_node in ast.find_all(exp.With):
            if with_node.args.get("recursive"):
                return ASTValidationResult(
                    is_safe=False,
                    error_code="PROHIBITED_CONSTRUCT",
                    error_message="Recursive CTEs are disallowed by sandbox policy",
                )

        # Require at least one projected expression in every SELECT clause
        for sel in ast.find_all(exp.Select):
            if not sel.expressions:
                return ASTValidationResult(
                    is_safe=False,
                    error_code="PARSE_ERROR",
                    error_message="SELECT clause contains no projection expressions",
                )

        # 4. Default-deny AST node and SQL function allowlist inspection
        for node in ast.walk():
            if isinstance(node, exp.Anonymous):
                fn_name = (node.name or "").lower()
                if fn_name not in self.allowed_functions:
                    return ASTValidationResult(
                        is_safe=False,
                        error_code="DISALLOWED_FUNCTION",
                        error_message=f"SQL function '{fn_name}' is not in approved function allowlist",
                    )
                continue

            if isinstance(node, _ALLOWED_STRUCTURAL_NODES):
                continue

            if isinstance(node, exp.Func):
                fn_name = (node.sql_name() or "").lower()
                if fn_name in self.allowed_functions:
                    continue
                return ASTValidationResult(
                    is_safe=False,
                    error_code="DISALLOWED_FUNCTION",
                    error_message=f"SQL function '{fn_name}' is not in approved function allowlist",
                )

            return ASTValidationResult(
                is_safe=False,
                error_code="DISALLOWED_AST_NODE",
                error_message=f"AST node '{type(node).__name__}' is not on the read-only allowlist",
            )

        # 5. Collect CTE aliases and validate physical tables & schema qualifiers
        cte_aliases: Set[str] = set()
        for with_clause in ast.find_all(exp.With):
            for cte in with_clause.find_all(exp.CTE):
                if cte.alias:
                    cte_aliases.add(cte.alias.lower())

        physical_tables: Set[str] = set()
        for table_node in ast.find_all(exp.Table):
            # Reject SQLite table-valued functions (e.g. FROM pragma_table_info('t'))
            if isinstance(table_node.this, (exp.Anonymous, exp.Func)):
                fn_name = getattr(table_node.this, "name", "") or table_node.this.sql_name()
                return ASTValidationResult(
                    is_safe=False,
                    error_code="DISALLOWED_FUNCTION",
                    error_message=f"Table-valued function '{fn_name}' is not permitted",
                )

            db_qualifier = (table_node.db or "").lower()
            catalog_qualifier = (table_node.catalog or "").lower()
            if catalog_qualifier or db_qualifier not in ALLOWED_DB_SCHEMAS:
                return ASTValidationResult(
                    is_safe=False,
                    error_code="UNAUTHORIZED_SCHEMA",
                    error_message=f"Cross-database or temp schema qualifier '{table_node.db}' is forbidden",
                )

            table_name = table_node.name.lower() if table_node.name else ""
            if not table_name:
                continue

            if table_name in cte_aliases:
                continue

            if (
                table_name in FORBIDDEN_CATALOG_TABLES
                or table_name.startswith("sqlite_")
                or table_name.startswith("pragma_")
            ):
                return ASTValidationResult(
                    is_safe=False,
                    error_code="SYSTEM_CATALOG_ACCESS",
                    error_message=f"Access to internal table '{table_name}' is forbidden",
                )

            physical_tables.add(table_name)

        if self.allowed_tables is not None:
            unauthorized = physical_tables - self.allowed_tables
            if unauthorized:
                return ASTValidationResult(
                    is_safe=False,
                    error_code="UNAUTHORIZED_TABLE",
                    error_message=f"Query references unapproved table(s): {sorted(unauthorized)}",
                    tables_found=physical_tables,
                )

        # 6. Validate column references when column allowlist is provided
        if self.allowed_columns_by_table is not None or self.allowed_columns_flat is not None:
            col_err = self._validate_columns(ast)
            if col_err is not None:
                col_err.tables_found = physical_tables
                return col_err

        return ASTValidationResult(
            is_safe=True,
            ast=ast,
            tables_found=physical_tables,
        )
