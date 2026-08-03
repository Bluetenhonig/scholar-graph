"""Content-addressed request/response store.

A cassette is a JSON file named for the SHA-256 of the request that produced
it. Same request, same file — which is what lets `llm_mode=replay` reproduce
a run byte for byte, in CI, with no API key and no spend.

The stored payload keeps the full request alongside the response. That costs
a little disk and buys a lot: a cassette is readable as a transcript, and a
prompt change shows up as a new file rather than a silently-wrong hit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    """Stable hash of a request. Key order and unicode form are normalised."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CassetteMiss(RuntimeError):
    """Replay was asked for a request that has never been recorded."""

    def __init__(self, key: str, purpose: str, path: Path) -> None:
        self.key = key
        self.purpose = purpose
        self.path = path
        super().__init__(
            f"No cassette for {purpose!r} (key {key[:12]}…, expected {path}).\n"
            "The prompt or model changed since these cassettes were recorded. "
            "Re-record with:\n"
            "    SCHOLAR_GRAPH_LLM_MODE=record SCHOLAR_GRAPH_ANTHROPIC_API_KEY=sk-… "
            "scholar-graph research '<question>'"
        )


class CassetteStore:
    def __init__(self, root: Path, namespace: str) -> None:
        self.root = Path(root) / namespace
        self.namespace = namespace

    def path_for(self, key: str) -> Path:
        # Shard by prefix so the directory stays navigable as recordings grow.
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)
        return payload

    def save(self, key: str, request: dict[str, Any], response: dict[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"key": key, "request": request, "response": response}
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(document, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        return path

    def count(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0
