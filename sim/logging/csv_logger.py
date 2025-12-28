# sim/io/logging.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CsvLogger:
    """
    Simple buffered CSV logger.

    - Call `log({...})` each step with a dict of scalar values.
    - The first call sets the column order (or you can pass fieldnames).
    - Data is buffered in memory and flushed to disk every `flush_every` rows.

    Notes:
      - This is fast enough for typical matplotlib sims when flush_every >= 100.
      - Avoid flush_every=1 (disk I/O every step will slow you down).
    """
    path: str
    flush_every: int = 200
    fieldnames: Optional[List[str]] = None

    _buffer: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _writer: Optional[csv.DictWriter] = field(default=None, init=False)
    _file: Any = field(default=None, init=False)
    _opened: bool = field(default=False, init=False)

    def log(self, row: Dict[str, Any]) -> None:
        if self.fieldnames is None:
            # Fix column order from the first row
            self.fieldnames = list(row.keys())
        else:
            # Ensure stable schema
            missing = [k for k in self.fieldnames if k not in row]
            extra = [k for k in row.keys() if k not in self.fieldnames]
            if missing:
                raise ValueError(f"CsvLogger missing keys in row: {missing}")
            if extra:
                raise ValueError(
                    f"CsvLogger got unexpected keys in row: {extra}. "
                    f"Expected only: {self.fieldnames}"
                )

        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        # Ensure parent dir exists
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if not self._opened:
            # Open once and keep handle for speed
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._opened = True

        assert self._writer is not None
        self._writer.writerows(self._buffer)
        self._buffer.clear()

        # Force write to OS buffers (optional; comment out if you want max speed)
        self._file.flush()

    def close(self) -> None:
        # Flush remaining rows and close the file
        self.flush()
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
        self._opened = False

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
