# sim/logging/csv_logger.py
from __future__ import annotations

import csv
import os
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CsvLogger:
    """
    Simple buffered CSV logger with a stable schema.

    Usage:
      - If you pass `fieldnames`, schema is fixed and enforced.
      - If `fieldnames` is None, schema is inferred from first row and enforced thereafter.
        Missing fields are filled with NaN.
        Unexpected extra fields raise ValueError (catches typos / drift).

    Notes:
      - anim.py merges base row + debug row. With sim_loop returning a fixed debug template,
        the set of keys remains constant and this logger stays happy.
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
            # Fill missing keys with NaN
            for k in self.fieldnames:
                if k not in row:
                    row[k] = math.nan

            # Strict: error on unexpected extra keys
            extra = [k for k in row.keys() if k not in self.fieldnames]
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

        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if not self._opened:
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._opened = True

        assert self._writer is not None
        self._writer.writerows(self._buffer)
        self._buffer.clear()
        self._file.flush()

    def close(self) -> None:
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
