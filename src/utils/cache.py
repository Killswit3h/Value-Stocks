from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional


class DiskCache:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_").replace("?", "_")[:200]
        return os.path.join(self.base_dir, safe + ".json")

    def get(self, key: str, max_age_sec: Optional[int] = None) -> Optional[dict]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if max_age_sec is not None:
                age = time.time() - data.get("_cached_at", 0)
                if age > max_age_sec:
                    return None
            return data.get("payload")
        except Exception:
            return None

    def set(self, key: str, payload: dict) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_cached_at": time.time(), "payload": payload}, f)


@dataclass
class RunLog:
    path: str
    data: dict | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

