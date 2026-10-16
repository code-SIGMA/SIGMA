#!/usr/bin/env python3
"""
csv_writer.py

Small library for easily writing CSV files - whether creating new,
overwriting, or appending, with or without header, from dicts or lists.

Examples:

    from csv_writer import CsvWriter

    # Simplest case: append single rows (header is automatically written
    # from dict keys on the first write)
    writer = CsvWriter("data/results.csv", fieldnames=["run_id", "value", "ts"])
    writer.append_row({"run_id": 1, "value": 42, "ts": "2026-08-12"})
    writer.append_rows([
        {"run_id": 2, "value": 43, "ts": "2026-08-12"},
        {"run_id": 3, "value": 44, "ts": "2026-08-12"},
    ])

    # Rewrite the file completely (overwrites existing file)
    writer.write_rows([...], mode="overwrite")

    # As a context manager for many rows (faster, single open handle)
    with CsvWriter("data/results.csv", fieldnames=["a", "b"]) as w:
        for row in some_generator():
            w.append_row(row)

    # Plain rows (lists instead of dicts), e.g. without a fixed header
    plain = CsvWriter("data/raw.csv")
    plain.append_row(["a", "b", "c"])
    plain.append_rows([[1, 2, 3], [4, 5, 6]])
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union, Dict, List

Row = Union[Dict[str, Any], Sequence[Any]]


class CsvWriter:
    """
    Writes CSV rows to a file.

    fieldnames: column names. Only needed if you're working with dicts
                (DictWriter) or want to enforce a fixed header.
                If None and rows are lists/tuples, a plain csv.writer
                is used (no header handling).
    delimiter / encoding: configurable as usual.
    """

    def __init__(
        self,
        path: str | Path,
        fieldnames: Optional[List[str]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        newline: str = "",
    ):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.delimiter = delimiter
        self.encoding = encoding
        self.newline = newline

        self._file = None
        self._writer = None
        self._is_dict_mode: Optional[bool] = None

    # ---------- Context manager for efficiently writing many rows ----------

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._is_dict_mode = None

    # ---------- Public API ----------

    def append_row(self, row: Row) -> None:
        """Appends a single row (opens the file if needed, writes header if new)."""
        self._ensure_open(mode="append", sample_row=row)
        self._write_row(row)

    def append_rows(self, rows: Iterable[Row]) -> None:
        """Appends multiple rows."""
        rows = list(rows)
        if not rows:
            return
        self._ensure_open(mode="append", sample_row=rows[0])
        for row in rows:
            self._write_row(row)

    def write_rows(self, rows: Iterable[Row], mode: str = "overwrite") -> None:
        """
        Writes multiple rows at once.

        mode: "overwrite" (file is recreated/overwritten) or
              "append" (like append_rows).
        """
        rows = list(rows)
        if mode == "overwrite":
            self.close()  # in case something was already open
            if not rows:
                return
            self._ensure_open(mode="overwrite", sample_row=rows[0])
            for row in rows:
                self._write_row(row)
            self.close()
        elif mode == "append":
            self.append_rows(rows)
            self.close()
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    # ---------- Internals ----------

    def _ensure_open(self, mode: str, sample_row: Row) -> None:
        if self._file is not None:
            return  # already open (e.g. inside a `with` block)

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._is_dict_mode = isinstance(sample_row, dict)
        if self._is_dict_mode and self.fieldnames is None:
            self.fieldnames = list(sample_row.keys())

        file_exists = self.path.exists()
        file_has_content = file_exists and self.path.stat().st_size > 0

        if mode == "overwrite":
            file_mode = "w"
            write_header = self._is_dict_mode
        elif mode == "append":
            file_mode = "a"
            write_header = self._is_dict_mode and not file_has_content
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        self._file = open(
            self.path, file_mode, encoding=self.encoding, newline=self.newline
        )

        if self._is_dict_mode:
            self._writer = csv.DictWriter(
                self._file, fieldnames=self.fieldnames, delimiter=self.delimiter
            )
            if write_header:
                self._writer.writeheader()
        else:
            self._writer = csv.writer(self._file, delimiter=self.delimiter)

    def _write_row(self, row: Row) -> None:
        if self._is_dict_mode:
            # unknown keys in row that aren't in fieldnames would otherwise
            # raise an error -> extrasaction="ignore" would also be possible,
            # but kept strict here on purpose for data consistency.
            self._writer.writerow(row)
        else:
            self._writer.writerow(row)


# ---------- Convenience functions for quick one-off use ----------


def write_csv(
    path: str | Path,
    rows: Iterable[Row],
    fieldnames: Optional[List[str]] = None,
    mode: str = "overwrite",
    **kwargs,
) -> None:
    """One-off function without needing to instantiate a class.

    write_csv("out.csv", [{"a": 1, "b": 2}])
    write_csv("out.csv", [{"a": 3}], mode="append")
    """
    writer = CsvWriter(path, fieldnames=fieldnames, **kwargs)
    writer.write_rows(rows, mode=mode)


def append_csv_row(
    path: str | Path,
    row: Row,
    fieldnames: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """One-off function to append a single row."""
    writer = CsvWriter(path, fieldnames=fieldnames, **kwargs)
    writer.append_row(row)
    writer.close()
