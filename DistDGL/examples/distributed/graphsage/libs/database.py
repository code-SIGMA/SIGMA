#!/usr/bin/env python3
"""Thread-safe PostgreSQL wrapper based on a psycopg3 connection pool."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import (
    Any,
    Dict,
    List,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

Row = Dict[str, Any]
Params = Optional[Union[Sequence[Any], Mapping[str, Any]]]


# ==============================================================================
# Errors
# ==============================================================================
class DatabaseError(RuntimeError):
    """Base class for every error raised deliberately by this module."""


class ConnectionFailed(DatabaseError):
    """The pool could not be opened or a connection could not be acquired."""


class QueryFailed(DatabaseError):
    """A statement was rejected by the server."""

    def __init__(self, message: str, *, sqlstate: Optional[str] = None) -> None:
        self.sqlstate = sqlstate
        super().__init__(message)


class PoolClosed(DatabaseError):
    """The wrapper was used after close(), or across a fork()."""


@contextmanager
def _translate(action: str) -> Iterator[None]:
    """Turn psycopg errors into QueryFailed, keeping the original as __cause__."""
    try:
        yield
    except psycopg.Error as exc:
        raise QueryFailed(
            f"{action}: {type(exc).__name__}: {exc}",
            sqlstate=getattr(exc, "sqlstate", None),
        ) from exc


# ==============================================================================
# Shared query surface
# ==============================================================================
class _Queries:
    """Statement helpers shared by the pooled wrapper and a single transaction."""

    @contextmanager
    def _cursor(self) -> Iterator[psycopg.Cursor]:
        raise NotImplementedError

    # -- reading ---------------------------------------------------------------
    def fetch_all(
        self, query: Union[str, sql.Composed], params: Params = None
    ) -> List[Row]:
        """Run a SELECT and return every row as a dictionary."""
        with self._cursor() as cur, _translate("fetch_all"):
            cur.execute(query, params)
            return cur.fetchall() if cur.description else []

    def fetch_one(
        self, query: Union[str, sql.Composed], params: Params = None
    ) -> Optional[Row]:
        """Run a SELECT and return the first row, or None."""
        with self._cursor() as cur, _translate("fetch_one"):
            cur.execute(query, params)
            return cur.fetchone() if cur.description else None

    def fetch_value(
        self,
        query: Union[str, sql.Composed],
        params: Params = None,
        default: Any = None,
    ) -> Any:
        """Return the first column of the first row - handy for counts and flags."""
        row = self.fetch_one(query, params)
        return default if not row else next(iter(row.values()))

    def iter_rows(
        self,
        query: Union[str, sql.Composed],
        params: Params = None,
        batch_size: int = 1000,
    ) -> Iterator[Row]:
        """
        Stream a large result set without holding it all in memory.

        Uses a server-side cursor, so a result of several million rows costs
        roughly batch_size rows of memory instead of the whole table.
        """
        with self._cursor() as cur:
            connection = cur.connection
            with connection.cursor(name="stream", row_factory=dict_row) as server_cur:
                server_cur.itersize = batch_size
                with _translate("iter_rows"):
                    server_cur.execute(query, params)
                    for row in server_cur:
                        yield row

    # -- writing ---------------------------------------------------------------
    def execute(self, query: Union[str, sql.Composed], params: Params = None) -> int:
        """Run any statement (UPDATE, DELETE, DDL) and return the affected rows."""
        with self._cursor() as cur, _translate("execute"):
            cur.execute(query, params)
            return max(cur.rowcount, 0)

    def send_data(self, table_name: str, data: Mapping[str, Any]) -> int:
        """Insert a single row. Identifiers are escaped, values are parameters."""
        if not data:
            raise ValueError("data must not be empty")

        columns = list(data)
        query = sql.SQL("INSERT INTO {table} ({fields}) VALUES ({placeholders})").format(
            table=self._table(table_name),
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        )
        with self._cursor() as cur, _translate(f"insert into {table_name}"):
            cur.execute(query, [data[c] for c in columns])
            return max(cur.rowcount, 0)

    def send_many(self, table_name: str, rows: Iterable[Mapping[str, Any]]) -> int:
        """Insert several rows in one statement, in a single transaction."""
        rows = list(rows)
        if not rows:
            return 0

        columns = list(rows[0])
        self._check_columns(rows, columns)

        query = sql.SQL("INSERT INTO {table} ({fields}) VALUES ({placeholders})").format(
            table=self._table(table_name),
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
        )
        values = [[row[c] for c in columns] for row in rows]

        with self._cursor() as cur, _translate(f"insert into {table_name}"):
            cur.executemany(query, values)
            return max(cur.rowcount, 0)

    def copy_rows(
        self,
        table_name: str,
        rows: Iterable[Mapping[str, Any]],
        columns: Optional[Sequence[str]] = None,
    ) -> int:
        """
        Bulk-load rows using COPY.

        Roughly an order of magnitude faster than send_many for large batches,
        which matters when a single run writes tens of thousands of results.
        COPY cannot handle conflicts - use upsert() when rows may already exist.
        """
        rows = iter(rows)
        first = next(rows, None)
        if first is None:
            return 0

        columns = list(columns or first)
        statement = sql.SQL("COPY {table} ({fields}) FROM STDIN").format(
            table=self._table(table_name),
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
        )

        written = 0
        with self._cursor() as cur, _translate(f"copy into {table_name}"):
            with cur.copy(statement) as copy:
                copy.write_row([first.get(c) for c in columns])
                written += 1
                for row in rows:
                    copy.write_row([row.get(c) for c in columns])
                    written += 1
        return written

    def upsert(
        self,
        table_name: str,
        data: Mapping[str, Any],
        conflict_columns: Iterable[str],
        update_columns: Optional[Iterable[str]] = None,
    ) -> int:
        """
        Atomic INSERT ... ON CONFLICT DO UPDATE.

        Doing SELECT-then-INSERT in Python races when several workers run in
        parallel; ON CONFLICT resolves that inside the database.
        """
        if not data:
            raise ValueError("data must not be empty")

        columns = list(data)
        conflict = list(conflict_columns)
        if not conflict:
            raise ValueError("conflict_columns must not be empty")

        unknown = set(conflict) - set(columns)
        if unknown:
            raise ValueError(f"conflict columns missing from data: {sorted(unknown)}")

        updates = list(update_columns) if update_columns is not None else [
            c for c in columns if c not in conflict
        ]

        if updates:
            tail = sql.SQL("DO UPDATE SET {assignments}").format(
                assignments=sql.SQL(", ").join(
                    sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
                    for c in updates
                )
            )
        else:
            tail = sql.SQL("DO NOTHING")

        query = sql.SQL(
            "INSERT INTO {table} ({fields}) VALUES ({placeholders}) "
            "ON CONFLICT ({conflict}) {tail}"
        ).format(
            table=self._table(table_name),
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            conflict=sql.SQL(", ").join(map(sql.Identifier, conflict)),
            tail=tail,
        )
        with self._cursor() as cur, _translate(f"upsert into {table_name}"):
            cur.execute(query, [data[c] for c in columns])
            return max(cur.rowcount, 0)

    def upsert_many(
        self,
        table_name: str,
        rows: Iterable[Mapping[str, Any]],
        conflict_columns: Iterable[str],
        update_columns: Optional[Iterable[str]] = None,
    ) -> int:
        """Same as upsert(), for a batch of rows sharing one set of columns."""
        rows = list(rows)
        if not rows:
            return 0
        self._check_columns(rows, list(rows[0]))
        return sum(
            self.upsert(table_name, row, conflict_columns, update_columns)
            for row in rows
        )

    # -- inspection ------------------------------------------------------------
    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        return bool(
            self.fetch_value(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                (schema, table_name),
                default=0,
            )
        )

    def ping(self) -> bool:
        """Cheap round-trip; useful before starting a long job."""
        try:
            return self.fetch_value("SELECT 1") == 1
        except DatabaseError:
            return False

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _table(name: str) -> sql.Identifier:
        """Accept both 'table' and 'schema.table'."""
        return sql.Identifier(*name.split(".")) if "." in name else sql.Identifier(name)

    @staticmethod
    def _check_columns(
        rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
    ) -> None:
        expected = set(columns)
        for index, row in enumerate(rows):
            if set(row) != expected:
                missing = sorted(expected - set(row))
                extra = sorted(set(row) - expected)
                raise ValueError(
                    f"row {index} has different columns than row 0 "
                    f"(missing: {missing}, unexpected: {extra})"
                )


# ==============================================================================
# Transaction handle
# ==============================================================================
class Transaction(_Queries):
    """
    Several statements sharing one connection and one transaction.

    Everything inside the block either lands together or not at all - the
    reason a 'delete old results, then write new ones' step cannot leave the
    table half-empty when something fails in between.
    """

    def __init__(self, cursor: psycopg.Cursor) -> None:
        self._cur = cursor

    @contextmanager
    def _cursor(self) -> Iterator[psycopg.Cursor]:
        yield self._cur


# ==============================================================================
# Database
# ==============================================================================
class Database(_Queries):
    """Thread-safe PostgreSQL wrapper. Not shareable across fork()."""

    def __init__(
        self,
        db_host: str,
        db_port: int = 5432,
        db_name: str = "postgres",
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        application_name: str = "orchestrator",
        open_now: bool = True,
    ) -> None:
        # make_conninfo quotes values properly - an f-string breaks on passwords
        # containing spaces, quotes or backslashes.
        self._conninfo = make_conninfo(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            connect_timeout=int(connect_timeout),
            application_name=application_name,
        )
        self._timeout = timeout
        self._owner_pid = os.getpid()
        self._closed = False
        self._label = f"{db_name}@{db_host}:{db_port}"

        self._pool = ConnectionPool(
            conninfo=self._conninfo,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        if open_now:
            self.open()

    def open(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Open the pool, by default waiting until a first connection succeeds.

        Waiting means bad credentials or an unreachable host surface here,
        not halfway through a job.
        """
        try:
            self._pool.open(
                wait=wait, timeout=timeout if timeout is not None else self._timeout
            )
        except Exception as exc:
            raise ConnectionFailed(
                f"could not connect to {self._label}: {type(exc).__name__}: {exc}"
            ) from exc
        logger.info("Connected to database %s", self._label)

    # -- cursors ---------------------------------------------------------------
    @contextmanager
    def _cursor(self) -> Iterator[psycopg.Cursor]:
        self._ensure_usable()
        try:
            connection = self._pool.connection()
        except Exception as exc:
            raise ConnectionFailed(
                f"no connection available from {self._label}: {exc}"
            ) from exc

        # psycopg's pool commits on a clean exit and rolls back on an exception,
        # so no manual commit/rollback is needed here.
        with connection as conn, conn.cursor() as cur:
            yield cur

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """
        Group several statements into one atomic unit.

            with db.transaction() as tx:
                tx.execute("DELETE FROM results WHERE run_id = %s", (run_id,))
                tx.copy_rows("results", rows)
        """
        with self._cursor() as cur:
            yield Transaction(cur)

    # -- lifecycle -------------------------------------------------------------
    def _ensure_usable(self) -> None:
        if self._closed:
            raise PoolClosed(f"database {self._label} is already closed")
        if os.getpid() != self._owner_pid:
            raise PoolClosed(
                "connection pool was inherited through fork(); "
                "create a new Database instance in the child process"
            )

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.close()
        logger.info("Database connection pool %s closed", self._label)

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"<Database {self._label} ({state})>"
