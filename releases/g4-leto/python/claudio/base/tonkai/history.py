"""One JSONL file per invocation, sampled only at the existing report cadence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


class RunHistory:
    def __init__(self, directory, config, *, resume=None, initial_games=0, no_train=False):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.path = Path(directory) / (self.run_id + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._append({"kind": "run", "schema_version": 1, "run_id": self.run_id,
                      "config": config, "resume": resume, "initial_games": initial_games,
                      "no_train": no_train})

    def _append(self, row):
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        # Close each report: bounded I/O, and completed reports survive Ctrl+C.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def record(self, metrics, *, final=False):
        self._append({"kind": "metrics", "run_id": self.run_id,
                      "final": final, **metrics.snapshot()})
