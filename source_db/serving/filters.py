"""Flexible filtering over the catalog, backed by DuckDB.

The catalog is 314 columns for 198k sources, stored in `science.sqlite` as one
JSON blob per row. That shape is right for the detail page -- one primary-key
fetch returns everything -- and wrong for filtering: a predicate on anything
inside the blob is a full scan plus a JSON inflate, measured at 4.8 s. So the
classify page grew four filters, all of them on the handful of columns that
happen to be promoted, and 233 plausibly filterable columns stayed unreachable.

This module filters the *same data* out of `sources.parquet`, which stage 1 of
the build already writes and six other modules already read. Parquet is
columnar, so a predicate touches only the columns it names: filters over all
314 columns run in 3-40 ms, and the 188 MB file is never read whole because
projection pushdown means only the named columns are touched.

Measured on the real catalog, warm: 2.8 ms for an equality, 15.5 ms for an
eight-clause AND/OR with an arithmetic expression in it, 41 ms for the same
filter plus a page of 200 at offset 2000. Cross-checked against the existing
SQLite path -- identical uuid sets, and 584x faster on a column that lives in
the JSON blob (6.45 s -> 11 ms).

Nothing about the build changes. `science.sqlite` keeps serving detail rows,
light-curve payloads and cutout addresses by primary key. This answers exactly
one question -- *which uuids match* -- and hands the answer back.

    filters = CatalogFilters()
    filters.count({'column': 'wise_w1w2', 'op': '>', 'value': 0.8})
    filters.page({'op': 'and', 'clauses': [...]}, limit=200, offset=5000)

A filter is a JSON document, not a string of SQL, so it survives a URL, can be
saved and named, and can be checked before it runs. See `_PREDICATE` below for
the grammar.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from api import catalog_schema
from source_db.paths import DEFAULT_OUT_DIR

# Overridable so tests can point at a throwaway catalog, matching `api.config`.
CATALOG_PARQUET = Path(os.environ.get('LT_CATALOG_PARQUET',
                                      DEFAULT_OUT_DIR / 'sources.parquet'))

# Every source carries one; it is the join key back to `science.sqlite` and the
# tiebreaker that makes any sort a total order.
UUID = 'source_uuid'

# Caps for the embedded engine, overridable per instance. Deliberately modest:
# the whole point is that these queries are milliseconds on one core.
DEFAULT_THREADS = int(os.environ.get('LT_FILTER_THREADS', 4))
DEFAULT_MEMORY_LIMIT = os.environ.get('LT_FILTER_MEMORY_LIMIT', '2GB')


class FilterError(ValueError):
    """A filter document that cannot be compiled. Always the caller's fault."""


# ── the predicate grammar ─────────────────────────────────────────────────────
#
# _PREDICATE ::= _LEAF | _GROUP | _EXPR
#
#   _LEAF   {'column': str, 'op': str, 'value': Any, 'missing': str?}
#   _GROUP  {'op': 'and'|'or'|'not', 'clauses': [_PREDICATE, ...]}
#   _EXPR   {'expr': str, 'missing': str?}
#
# A leaf names one column and one operator, both whitelisted, with the value
# passed as a bound parameter -- so a filter document can never carry SQL.
# `_EXPR` is the escape hatch for the cuts no widget anticipates, such as
# `ZTF_gPSFMag - PSTARR_gPSFMag > 1.5`; it is validated by `_check_expr`
# rather than trusted.

# op -> SQL template. `{c}` is the quoted column, `{p}` the bound parameter.
_OPS: dict[str, str] = {
    '=':          '{c} = {p}',
    '!=':         '{c} <> {p}',
    '<':          '{c} < {p}',
    '<=':         '{c} <= {p}',
    '>':          '{c} > {p}',
    '>=':         '{c} >= {p}',
    'contains':   '{c} ILIKE {p}',
    'startswith': '{c} ILIKE {p}',
    'endswith':   '{c} ILIKE {p}',
    # `surveys` and `spec_surveys` are comma-joined lists in one text column,
    # so membership is not `contains`: 'wise' is a substring of 'neowise', and
    # a naive match would quietly fold the two surveys together. The wrapping
    # commas are what make it an element test rather than a substring test.
    'has_element': "(',' || {c} || ',') ILIKE {p}",
}
# Operators whose value is a pair or a list rather than a scalar.
_OPS_RANGE = {'between'}
_OPS_LIST = {'in', 'not_in'}
# Operators that take no value at all.
_OPS_UNARY = {
    'is_null':     '{c} IS NULL',
    'is_not_null': '{c} IS NOT NULL',
    'is_true':     '{c} = TRUE',
    'is_false':    '{c} = FALSE',
}
OPERATORS = sorted({*_OPS, *_OPS_RANGE, *_OPS_LIST, *_OPS_UNARY})

# What to do with rows where the column is NULL. This is explicit because the
# fill rates here are wildly uneven -- SIMBAD matches 26% of sources, TNS 0.6%
# -- and SQL's three-valued logic silently drops the unmatched ones. Written
# as `simbad_otype != 'Star'`, a cut meant to remove stars also removes the
# 74% of sources SIMBAD never saw. Saying which you meant is the only way to
# not get this wrong by accident.
MISSING_POLICIES = ('exclude', 'include')

# Functions the expression escape hatch may call. Deliberately tiny and
# arithmetic: enough for colours, ratios and magnitudes, nothing that reads
# another table or the filesystem.
_EXPR_FUNCTIONS = {'abs', 'log', 'log10', 'ln', 'exp', 'sqrt', 'pow',
                   'least', 'greatest', 'round', 'floor', 'ceil'}
_EXPR_KEYWORDS = {'and', 'or', 'not', 'is', 'null', 'between', 'in',
                  'true', 'false'}
# Tokens an expression may be made of. Anything else -- a semicolon, a comment
# marker, a star, a name that is not a column -- is refused rather than escaped.
_EXPR_TOKEN = re.compile(r"""
      (?P<space>\s+)
    | (?P<number>\d+\.?\d*([eE][-+]?\d+)?)
    | (?P<string>'[^']*')
    | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<op><=|>=|<>|!=|=|<|>|\+|-|\*|/|\(|\)|,)
""", re.VERBOSE)


@dataclass(frozen=True)
class ColumnInfo:
    """One filterable column, with what the UI needs to offer it sensibly."""
    name: str
    type: str                  # 'number' | 'text' | 'bool'
    group: str                 # from `api.catalog_schema`, e.g. 'wise'
    n_non_null: int
    fill: float                # 0..1; a 0.26 column needs a NULL policy

    @property
    def filterable(self) -> bool:
        """A column that is NULL for every source can only ever match nothing."""
        return self.n_non_null > 0


def _sql_type(duck_type: str) -> str:
    t = duck_type.upper()
    if t.startswith('BOOL'):
        return 'bool'
    if any(t.startswith(p) for p in
           ('TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT', 'UTINYINT',
            'USMALLINT', 'UINTEGER', 'UBIGINT', 'FLOAT', 'DOUBLE', 'DECIMAL')):
        return 'number'
    return 'text'


def _quote(column: str) -> str:
    """Quote an identifier that has already been checked against the schema.

    The check is what makes this safe; the quoting only handles the fact that
    column names here are case-sensitive and mixed (`ZTF_gPSFMag`).
    """
    return '"' + column.replace('"', '""') + '"'


class CatalogFilters:
    """Compiles filter documents against the catalog and runs them.

    One instance per process. `duckdb` connections are not safe to use from
    two threads at once -- a shared one returned crossed results in 7 of 240
    concurrent queries -- so every query runs on its own `cursor()`, which is
    DuckDB's supported way to fan out.
    """

    def __init__(self, parquet: Path | str | None = None, *,
                 threads: int | None = None, memory_limit: str | None = None):
        self.parquet = Path(parquet) if parquet is not None else CATALOG_PARQUET
        if not self.parquet.exists():
            raise FileNotFoundError(
                f'{self.parquet} not found -- run '
                f'`python -m source_db.build_catalog_tier` first')
        self._con = duckdb.connect(':memory:')
        # DuckDB defaults to every core and 80% of RAM (measured here: 14
        # threads, a 38 GiB limit). Sensible for an analysis session, wrong
        # for a web process sharing the machine with the API, the frontend and
        # two light-curve pullers. Queries here are milliseconds, so the
        # parallelism buys nothing worth that risk.
        self._con.execute(f'SET threads = {int(threads or DEFAULT_THREADS)}')
        self._con.execute(
            f"SET memory_limit = '{memory_limit or DEFAULT_MEMORY_LIMIT}'")
        # A view, not a table: the parquet stays on disk and DuckDB reads only
        # the columns a predicate names. Materialising it would cost 480 MB of
        # RAM to make 6 ms queries into 2 ms ones.
        #
        # The path is inlined because DDL cannot take a bound parameter. It is
        # a configured path rather than anything a request carries, and the
        # quote-doubling is belt-and-braces for a directory with an apostrophe.
        self._literal = str(self.parquet).replace("'", "''")
        self._attached: dict[str, list[str]] = {}
        self._derived: dict[str, str] = {}
        self._columns: dict[str, ColumnInfo] | None = None
        self._lock = threading.Lock()
        self._build_view()

    # ── extra columns ─────────────────────────────────────────────────────────

    def _build_view(self) -> None:
        """(Re)define `catalog`: the parquet, plus attachments and derivations."""
        extra_select, joins = '', ''
        for i, (name, cols) in enumerate(self._attached.items()):
            alias = f'a{i}'
            extra_select += ''.join(
                f', {alias}.{_quote(c)}' for c in cols if c != UUID)
            joins += (f' LEFT JOIN {_quote(name)} {alias} '
                      f'USING ({_quote(UUID)})')
        for name, expression in self._derived.items():
            extra_select += f', ({expression}) AS {_quote(name)}'
        self._con.execute(
            f'CREATE OR REPLACE VIEW catalog AS '
            f"SELECT p.*{extra_select} "
            f"FROM read_parquet('{self._literal}') p{joins}")
        self._columns = None

    def derive(self, name: str, expression: str) -> None:
        """Add a computed column, as though the catalog had always had it.

        For quantities the page computes from several columns -- the triage
        measure `mean_abs_dmag` is the one that matters here, being a mean of
        |ZTF - PanSTARRS| across g/r/i with flag-dependent substitutions.
        Computed in Python it can only ever be sorted or filtered within
        whatever window was already fetched, which for a *ranking* is the
        wrong answer rather than a slow one.

        Once derived it is an ordinary column: `columns()` lists it, `stats()`
        gives it a range, and it can be filtered and sorted like any other.

        `expression` is SQL and is **not** validated -- it comes from this
        codebase, not from a request. Never build one from user input; the
        `expr` predicate is the checked path for that.
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise FilterError(f'{name!r} is not a usable column name')
        if name in self._parquet_columns():
            raise FilterError(f'{name!r} already exists in the catalog')
        with self._lock:
            self._derived[name] = expression
            self._build_view()

    def attach(self, name: str, rows: Any) -> None:
        """Fold extra per-uuid columns into the catalog, keyed on `source_uuid`.

        Four columns the filter UI needs -- `n_epochs`, `surveys`, `n_spectra`,
        `spec_surveys` -- are not in the parquet. They are counted at
        serving-build time from the light-curve and spectrum tables, so they
        exist only in `science.sqlite`. DuckDB could read that file directly,
        but its sqlite extension is a network download, and putting an
        on-demand download in the serving path trades a real problem for a
        worse one.

        So they are handed over as a table instead. 198k rows of four small
        columns is 42 MB and loads once at startup; afterwards a predicate
        mixing them with catalog columns is one query over one view, which is
        what keeps pagination in a single place.

        Call before serving: this redefines the view, and DuckDB DDL is not
        something to run underneath in-flight readers.
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise FilterError(f'{name!r} is not a usable table name')
        # Registration is scoped to the connection object it was made on, so
        # the register and the DDL that reads it have to be the same one --
        # a cursor cannot see what the parent registered. Everything here runs
        # on the parent under the lock, which is right anyway: this is a
        # startup step, not something to do underneath in-flight readers.
        with self._lock:
            self._con.register('_attach_src', rows)
            try:
                columns = [r[0] for r in self._con.execute(
                    'DESCRIBE SELECT * FROM _attach_src').fetchall()]
                if UUID not in columns:
                    raise FilterError(f'an attached table needs a {UUID!r} column')
                clash = (set(columns) - {UUID}) & self._parquet_columns()
                if clash:
                    raise FilterError(f'attached columns already in the catalog: '
                                      f'{sorted(clash)}')
                self._con.execute(
                    f'CREATE OR REPLACE TABLE {_quote(name)} AS '
                    f'SELECT * FROM _attach_src')
                self._attached[name] = columns
                self._build_view()
            finally:
                self._con.unregister('_attach_src')

    def _parquet_columns(self) -> set[str]:
        return {r[0] for r in self._cursor().execute(
            f"DESCRIBE SELECT * FROM read_parquet('{self._literal}')").fetchall()}

    # ── introspection ─────────────────────────────────────────────────────────

    def _cursor(self):
        return self._con.cursor()

    def columns(self) -> list[ColumnInfo]:
        """Every column, with its type, group and fill rate.

        The fill rate is the load-bearing part: it is what lets a UI warn that
        a cut on `tns_type` is a cut on the 0.1% of sources TNS has heard of.
        Computed once -- it is a single pass and never changes for a release.
        """
        if self._columns is None:
            with self._lock:
                if self._columns is None:
                    self._columns = self._load_columns()
        return list(self._columns.values())

    def _load_columns(self) -> dict[str, ColumnInfo]:
        cur = self._cursor()
        described = cur.execute('DESCRIBE catalog').fetchall()
        names = [r[0] for r in described]
        types = {r[0]: r[1] for r in described}
        total, counts = self._fill_counts(names)
        return {
            n: ColumnInfo(name=n, type=_sql_type(types[n]),
                          group=catalog_schema.group_of(n),
                          n_non_null=int(c), fill=int(c) / (total or 1))
            for n, c in zip(names, counts)
        }

    def _fill_counts(self, names: Sequence[str]) -> tuple[int, list[int]]:
        """Non-null counts per column, from the parquet footer where possible.

        `COUNT(col)` over all 314 columns is the one query here that defeats
        projection pushdown -- it touches every column by definition -- and it
        cost 505 ms and 331 MB of resident memory. Parquet already stores a
        null count per column chunk, so the same answer is in the footer for
        5 ms and ~2 MB, and it is exact rather than an estimate.

        Statistics are optional in the format, so a file written without them
        falls back to the scan rather than reporting everything as full.
        """
        from_footer: dict[str, int] = {}
        try:
            import pyarrow.parquet as pq

            md = pq.ParquetFile(self.parquet).metadata
            nulls = [0] * md.num_columns
            for g in range(md.num_row_groups):
                group = md.row_group(g)
                for i in range(md.num_columns):
                    stats = group.column(i).statistics
                    if stats is None or stats.null_count is None:
                        raise LookupError('no null statistics in this parquet')
                    nulls[i] += stats.null_count
            from_footer = {md.schema.column(i).name: md.num_rows - nulls[i]
                           for i in range(md.num_columns)}
        except (ImportError, LookupError, AttributeError):
            from_footer = {}

        cur = self._cursor()
        total = cur.execute('SELECT COUNT(*) FROM catalog').fetchone()[0]
        # Anything the footer did not cover -- attached columns, or a parquet
        # written without statistics -- still has to be counted the slow way,
        # but only those columns rather than all 314.
        rest = [n for n in names if n not in from_footer]
        if rest:
            counted = cur.execute(
                'SELECT ' + ', '.join(f'COUNT({_quote(n)})' for n in rest)
                + ' FROM catalog').fetchone()
            from_footer.update(zip(rest, (int(c) for c in counted)))
        return total, [from_footer[n] for n in names]

    def column(self, name: str) -> ColumnInfo:
        self.columns()
        assert self._columns is not None
        if name not in self._columns:
            raise FilterError(f'unknown column {name!r}')
        return self._columns[name]

    def stats(self, name: str, top: int = 20) -> dict[str, Any]:
        """What a UI needs to offer this column: a range, or the common values.

        A slider with no idea of its own bounds is worse than no slider, and a
        free-text box for `simbad_otype_label` is a guessing game when there
        are only 96 distinct values.
        """
        info = self.column(name)
        col, cur = _quote(name), self._cursor()
        out: dict[str, Any] = {'column': name, 'type': info.type,
                               'n_non_null': info.n_non_null, 'fill': info.fill}
        if info.n_non_null == 0:
            return out
        if info.type == 'number':
            lo, hi, *qs = cur.execute(
                f'SELECT min({col}), max({col}), '
                f'quantile_cont({col}, [0.01, 0.25, 0.5, 0.75, 0.99]) '
                f'FROM catalog').fetchone()
            out.update(min=lo, max=hi,
                       quantiles=dict(zip(('p01', 'p25', 'p50', 'p75', 'p99'),
                                          qs[0] if qs else [])))
        else:
            rows = cur.execute(
                f'SELECT {col} AS v, COUNT(*) AS n FROM catalog '
                f'WHERE {col} IS NOT NULL GROUP BY v ORDER BY n DESC LIMIT ?',
                [top]).fetchall()
            n_distinct = cur.execute(
                f'SELECT COUNT(DISTINCT {col}) FROM catalog').fetchone()[0]
            out.update(n_distinct=n_distinct,
                       values=[{'value': v, 'count': n} for v, n in rows])
        return out

    # ── compiling a filter document ───────────────────────────────────────────

    def compile(self, doc: Any) -> tuple[str, list[Any]]:
        """Turn a filter document into a SQL condition plus bound parameters.

        Public because it is the thing worth testing and worth showing a user
        who asks "what did that actually do". The returned SQL is never
        concatenated with user text -- every value is a parameter.
        """
        if doc is None:
            return 'TRUE', []
        params: list[Any] = []
        sql = self._compile(doc, params)
        return sql, params

    def _compile(self, doc: Any, params: list[Any]) -> str:
        if not isinstance(doc, dict):
            raise FilterError(f'a predicate must be an object, got {type(doc).__name__}')
        if 'clauses' in doc or doc.get('op') in ('and', 'or', 'not'):
            return self._compile_group(doc, params)
        if 'expr' in doc:
            return self._compile_expr(doc, params)
        return self._compile_leaf(doc, params)

    def _compile_group(self, doc: dict, params: list[Any]) -> str:
        op = str(doc.get('op', 'and')).lower()
        if op not in ('and', 'or', 'not', 'first_of_group'):
            raise FilterError(f'unknown group operator {op!r}')
        clauses = doc.get('clauses') or []
        if not isinstance(clauses, list):
            raise FilterError("'clauses' must be a list")
        if op == 'first_of_group':
            return self._compile_first_of_group(clauses, params)
        if not clauses:
            # An empty AND matches everything and an empty OR matches nothing,
            # which is what those mean and also what a half-built UI filter
            # should do rather than erroring at the user mid-edit.
            return 'TRUE' if op in ('and', 'not') else 'FALSE'
        parts = [self._compile(c, params) for c in clauses]
        if op == 'not':
            if len(parts) != 1:
                raise FilterError("'not' takes exactly one clause")
            # `NOT x` leaves NULL as NULL, so a negated cut would silently drop
            # every unmatched source. COALESCE makes "not true" mean what a
            # person means by it.
            return f'(NOT COALESCE({parts[0]}, FALSE))'
        joiner = ' AND ' if op == 'and' else ' OR '
        return '(' + joiner.join(parts) + ')'

    def _compile_first_of_group(self, clauses: list, params: list[Any]) -> str:
        """Rows matching `clauses`, keeping one per `group_id` (several detections
        of one object): the lowest uuid among the group's rows that match.

        The match is re-evaluated inside the subquery, where unqualified columns
        bind to the group-mate `_g`, so a mate the filter drops cannot stand in.
        """
        inner: list[Any] = []
        where = self._compile({'op': 'and', 'clauses': clauses}, inner)
        params.extend(inner)
        if 'group_id' not in self._parquet_columns():
            return where
        params.extend(inner)
        u = _quote(UUID)
        return (f'({where} AND NOT EXISTS (SELECT 1 FROM catalog AS _g '
                f'WHERE catalog.group_id > 0 AND _g.group_id = catalog.group_id '
                f'AND _g.{u} < catalog.{u} AND {where}))')

    def _compile_leaf(self, doc: dict, params: list[Any]) -> str:
        name = doc.get('column')
        if not isinstance(name, str):
            raise FilterError("a predicate needs a 'column'")
        info = self.column(name)              # raises on an unknown column
        op = str(doc.get('op', '=')).lower()
        col = _quote(name)

        if op in _OPS_UNARY:
            sql = _OPS_UNARY[op].format(c=col)
            # `is_null` is already a statement about missingness; applying a
            # missing policy on top of it would be incoherent.
            return sql

        if 'value' not in doc:
            raise FilterError(f"operator {op!r} needs a 'value'")
        value = doc['value']

        if op in _OPS_RANGE:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise FilterError("'between' takes a [low, high] pair")
            params.extend(value)
            sql = f'{col} BETWEEN ? AND ?'
        elif op in _OPS_LIST:
            if not isinstance(value, (list, tuple)) or not value:
                raise FilterError(f'{op!r} takes a non-empty list')
            params.extend(value)
            holes = ', '.join('?' * len(value))
            sql = f'{col} {"NOT " if op == "not_in" else ""}IN ({holes})'
        elif op in _OPS:
            if op in ('contains', 'startswith', 'endswith', 'has_element'):
                if not isinstance(value, str):
                    raise FilterError(f'{op!r} takes a string')
                params.append({'contains': f'%{value}%',
                               'startswith': f'{value}%',
                               'endswith': f'%{value}',
                               'has_element': f'%,{value},%'}[op])
            else:
                params.append(value)
            sql = _OPS[op].format(c=col, p='?')
        else:
            raise FilterError(f'unknown operator {op!r}')

        return self._apply_missing(sql, doc, col, info)

    def _compile_expr(self, doc: dict, params: list[Any]) -> str:
        expr = doc['expr']
        if not isinstance(expr, str):
            raise FilterError("'expr' must be a string")
        if not expr.strip():
            # A blank expression is what "+ expression" adds before anything
            # is typed. Like an empty group, a half-built clause matches
            # everything rather than failing the whole page mid-edit.
            return 'TRUE'
        self._check_expr(expr)
        # No parameters: the expression has been proved to contain only
        # column names, numbers, quoted strings and arithmetic.
        return self._apply_missing(f'({expr})', doc, None, None)

    def _check_expr(self, expr: str) -> None:
        """Refuse anything that is not arithmetic over known columns.

        A whitelist rather than an escape: the expression is tokenised, and
        every token must be a number, a quoted string, an operator, a known
        function, a keyword, or a column that exists. A semicolon, a comment
        marker or a name that is not a column has no token to be, so it is
        rejected instead of sanitised.
        """
        if len(expr) > 2000:
            raise FilterError('expression too long')
        # Comment markers first, because the tokeniser below would read `--`
        # as two minus signs and `/*` as a divide and a multiply, while DuckDB
        # reads both as "ignore the rest". Nothing can be smuggled through a
        # comment -- `;` has no token, so a second statement is impossible
        # either way -- but a validator that disagrees with the parser about
        # what it just approved is not a validator.
        for marker in ('--', '/*', '*/'):
            if marker in expr:
                raise FilterError(f'{marker!r} is a comment to the query '
                                  f'engine; expressions may not contain one')
        self.columns()
        assert self._columns is not None
        pos, saw_column = 0, False
        while pos < len(expr):
            m = _EXPR_TOKEN.match(expr, pos)
            if not m:
                raise FilterError(
                    f'expression has something I will not run, at {expr[pos:pos + 20]!r}')
            pos = m.end()
            if m.lastgroup == 'name':
                word = m.group()
                low = word.lower()
                if low in _EXPR_KEYWORDS:
                    continue
                if low in _EXPR_FUNCTIONS:
                    continue
                if word in self._columns:
                    saw_column = True
                    continue
                raise FilterError(f'unknown column or function {word!r} in expression')
        if not saw_column:
            raise FilterError('an expression must mention at least one column')

    def _apply_missing(self, sql: str, doc: dict,
                       col: str | None, info: ColumnInfo | None) -> str:
        policy = str(doc.get('missing', 'exclude')).lower()
        if policy not in MISSING_POLICIES:
            raise FilterError(f'missing must be one of {MISSING_POLICIES}')
        if policy == 'exclude':
            return sql
        if col is not None:
            return f'({sql} OR {col} IS NULL)'
        # An expression can span several columns, so "include missing" means
        # "keep rows the comparison could not decide" rather than naming one.
        return f'(COALESCE({sql}, TRUE))'

    # ── running it ────────────────────────────────────────────────────────────

    def _order_by(self, sort: str, desc: bool) -> str:
        """A validated, total ordering.

        Validation comes before the name is used, not after. Ties break on
        `source_uuid` because without that two pages can disagree about a row
        that sorts equal -- which in a windowed list shows up as a source you
        see twice and one you never see. NULLS LAST in both directions, so a
        sort by redshift does not open with the 75% that have none.
        """
        # Unknown columns raise -- that catches the typo, which is what the
        # check is worth. A *known* column that happens to be NULL for every
        # source does not: that is a fact about the release, not a mistake,
        # and it orders fine (everything ties, the uuid tiebreak decides).
        # Refusing it turned "this build has no photometry" into a 422 on the
        # table's default sort.
        self.column(sort)
        direction = 'DESC' if desc else 'ASC'
        if sort == UUID:
            return f'{_quote(UUID)} {direction}'
        return f'{_quote(sort)} {direction} NULLS LAST, {_quote(UUID)} ASC'

    def count(self, doc: Any = None) -> int:
        """How many sources match. Cheap enough to call on every UI keystroke."""
        where, params = self.compile(doc)
        return self._cursor().execute(
            f'SELECT COUNT(*) FROM catalog WHERE {where}', params).fetchone()[0]

    def page(self, doc: Any = None, *, sort: str = UUID, desc: bool = False,
             limit: int = 200, offset: int = 0) -> dict[str, Any]:
        """One window of matching uuids, plus the total.

        Shaped to match the classify page's windowed navigation: an offset into
        a stable order, a slice, and the count the window needs to know how far
        it can go. Ties break on `source_uuid` so the order is total -- without
        that, two pages can disagree about a row that sorts equal.
        """
        order = self._order_by(sort, desc)
        where, params = self.compile(doc)
        cur = self._cursor()
        total = cur.execute(f'SELECT COUNT(*) FROM catalog WHERE {where}',
                            params).fetchone()[0]
        rows = cur.execute(
            f'SELECT {_quote(UUID)} FROM catalog WHERE {where} '
            f'ORDER BY {order} LIMIT ? OFFSET ?',
            [*params, limit, offset]).fetchall()
        return {'total': total, 'offset': offset, 'limit': limit,
                'uuids': [r[0] for r in rows]}

    def locate(self, target: str, doc: Any = None, *, sort: str = UUID,
               desc: bool = False) -> int | None:
        """Where `target` sits in the filtered order, or None if filtered out.

        Counted, not scanned: the number of matching rows that sort before it.
        That is one aggregate regardless of how deep the source is, where
        walking the list would cost more the further in you go -- and the
        classify page asks this on every deep link.

        Returns None when the filter excludes the target, which is a different
        answer from 0 and has to stay that way: a link to a source the current
        filter drops should say so, not land on candidate 1.
        """
        where, params = self.compile(doc)
        before, extra = self._sorts_before(target, sort, desc)
        cur = self._cursor()
        present = cur.execute(
            f'SELECT 1 FROM catalog WHERE {where} AND {_quote(UUID)} = ? LIMIT 1',
            [*params, target]).fetchone()
        if present is None:
            return None
        return cur.execute(
            f'SELECT COUNT(*) FROM catalog WHERE {where} AND ({before})',
            [*params, *extra]).fetchone()[0]

    def group_members(self, target: str) -> list[dict[str, Any]]:
        """Every source sharing `target`'s `group_id` (the same object found in
        overlapping fields), target included, ordered by uuid. Empty when it
        has no group, or the release has no `group_id` column.
        """
        if 'group_id' not in self._parquet_columns():
            return []
        cols = [UUID] + [c for c in ('coord_slug',) if c in self._parquet_columns()]
        rows = self._cursor().execute(
            f'SELECT {", ".join(_quote(c) for c in cols)} FROM catalog '
            f'WHERE group_id > 0 AND group_id = '
            f'(SELECT group_id FROM catalog WHERE {_quote(UUID)} = ?) '
            f'ORDER BY {_quote(UUID)}', [target]).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def rank_before(self, target: str, doc: Any = None, *, sort: str = UUID,
                    desc: bool = False) -> int:
        """How many matching rows sort before `target`, which need not match.

        `locate` for a source that the filter excludes: the position it *would*
        occupy. Used to carry an offset from one filtered order into another.
        """
        where, params = self.compile(doc)
        before, extra = self._sorts_before(target, sort, desc)
        return self._cursor().execute(
            f'SELECT COUNT(*) FROM catalog WHERE {where} AND ({before})',
            [*params, *extra]).fetchone()[0]

    def _sorts_before(self, target: str, sort: str,
                      desc: bool) -> tuple[str, list[Any]]:
        """The condition "this row comes before `target`", matching `_order_by`.

        It has to agree with the ORDER BY exactly or a located offset lands on
        the wrong row, and NULLS LAST is what makes that non-obvious: under a
        descending sort the nulls are still at the *end*, so a null-keyed
        target has every non-null row before it in both directions.
        """
        self._order_by(sort, desc)              # validates `sort`
        uuid_col = _quote(UUID)
        if sort == UUID:
            cmp = '<' if not desc else '>'
            return f'{uuid_col} {cmp} ?', [target]

        key = _quote(sort)
        value = self._cursor().execute(
            f'SELECT {key} FROM catalog WHERE {uuid_col} = ?', [target]).fetchone()
        if value is None:
            raise FilterError(f'no source {target!r}')
        value = value[0]
        if value is None:
            # The target is in the NULLS LAST tail: everything with a value
            # precedes it, and within the tail it is uuid order.
            return (f'({key} IS NOT NULL OR ({key} IS NULL AND {uuid_col} < ?))',
                    [target])
        cmp = '<' if not desc else '>'
        return (f'({key} {cmp} ? OR ({key} = ? AND {uuid_col} < ?))',
                [value, value, target])

    def select(self, doc: Any = None, columns: Sequence[str] = (), *,
               sort: str = UUID, desc: bool = False,
               limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """Matching rows with some columns attached, for a table view.

        Separate from `page` on purpose: the classify page wants uuids and
        nothing else, and shipping 314 columns it will not read is the habit
        this module exists to break.
        """
        for c in columns:
            self.column(c)
        wanted = [UUID, *[c for c in columns if c != UUID]]
        order = self._order_by(sort, desc)
        where, params = self.compile(doc)
        rows = self._cursor().execute(
            f'SELECT {", ".join(_quote(c) for c in wanted)} FROM catalog '
            f'WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?',
            [*params, limit, offset]).fetchall()
        return [dict(zip(wanted, r)) for r in rows]

    def explain(self, doc: Any) -> list[dict[str, Any]]:
        """How many sources each top-level clause keeps, on its own.

        The question a filter that returned 3 sources always raises is *which
        cut did that*. Answering it needs one count per clause, which is only
        affordable because each is a few milliseconds.
        """
        clauses = doc.get('clauses') if isinstance(doc, dict) else None
        if not clauses:
            clauses = [doc]
        out = []
        for clause in clauses:
            try:
                out.append({'clause': clause, 'matches': self.count(clause)})
            except FilterError as e:
                out.append({'clause': clause, 'error': str(e)})
        return out

    def uuids(self, doc: Any = None, limit: int | None = None) -> list[str]:
        """Every matching uuid, for handing to SQLite or a notebook."""
        where, params = self.compile(doc)
        sql = (f'SELECT {_quote(UUID)} FROM catalog WHERE {where} '
               f'ORDER BY {_quote(UUID)}')
        if limit is not None:
            sql += ' LIMIT ?'
            params = [*params, limit]
        return [r[0] for r in self._cursor().execute(sql, params).fetchall()]

    def close(self) -> None:
        self._con.close()


def restrict_to(uuids: Iterable[str]) -> dict[str, Any]:
    """A filter document for an explicit uuid list.

    The bridge back the other way: a selection made elsewhere -- an annotation
    query, a crossmatch done in a notebook -- becomes a filter like any other.
    """
    return {'column': UUID, 'op': 'in', 'value': list(uuids)}
