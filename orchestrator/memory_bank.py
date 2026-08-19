"""
memory_bank.py
--------------
Cross-session, cross-week persistent context per plot — the "Memory Bank"
enterprise capability. Local JSON-backed implementation for the demo;
production swaps `_store` for Firestore (see docs/ROADMAP.md and
infra/main.tf) with zero interface change, since both are simple
document-per-plot key stores.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

STORE_PATH = Path(os.getenv("AGRISENTINEL_MEMORY_PATH", "./data/memory_store.json"))


class MemoryBank:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _load(self) -> dict[str, list]:
        return json.loads(self.path.read_text())

    def _save(self, data: dict[str, list]) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def append(self, plot_id: str, record: dict[str, Any]) -> None:
        data = self._load()
        data.setdefault(plot_id, []).append(record)
        self._save(data)

    def get(self, plot_id: str) -> list[dict]:
        return self._load().get(plot_id, [])
