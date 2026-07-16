from __future__ import annotations

import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


log = logging.getLogger(__name__)


class AtomicParquetSink:
    """Write parquet rows to a temp file and replace the target on close."""

    def __init__(self, path: Path, schema: pa.Schema, flush_rows: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._schema = schema
        self._flush_rows = flush_rows
        self._buf: list[dict] = []
        self._rows_written = 0
        self._writer: pq.ParquetWriter | None = None
        self._tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    def write(self, row: dict) -> None:
        self._buf.append(row)
        if len(self._buf) >= self._flush_rows:
            self._flush()

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        if self._tmp_path.exists():
            self._tmp_path.unlink()
        self._writer = pq.ParquetWriter(str(self._tmp_path), self._schema)
        log.info("Parquet sink opened temp file: %s", self._tmp_path)

    def _flush(self) -> None:
        if not self._buf:
            return
        self._ensure_writer()
        table = pa.Table.from_pylist(self._buf, schema=self._schema)
        self._writer.write_table(table)
        self._rows_written += len(self._buf)
        self._buf.clear()

    def close(self) -> None:
        if self._writer is None and not self._buf:
            log.info("Parquet sink closed without writes; leaving %s untouched", self._path)
            return

        # Flush remaining rows and close the writer
        self._flush()
        self._writer.close()
        # Atomic rename — if this fails, the temp file is preserved so data isn't lost
        try:
            self._tmp_path.replace(self._path)
            log.info("Parquet sink committed %d rows → %s", self._rows_written, self._path)
        except OSError:
            log.error(
                "Failed to rename %s → %s; temp file preserved for recovery",
                self._tmp_path, self._path,
            )
            raise
