"""Test fixtures.

No test here touches a real LLM, a real travel API, or a real database. The
model is scripted so tool-calling sequences are deterministic, and MongoDB is
replaced by an in-memory stand-in that mimics the handful of operations the
favorites layer uses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Make `src` importable; the project directory name is hyphenated so it cannot
# itself be a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("REQUIRE_AUTH", "false")


from helpers import ScriptedModel  # noqa: F401  (re-exported for tests)


# --------------------------------------------------------------------------- #
# In-memory MongoDB stand-in
# --------------------------------------------------------------------------- #


class FakeCollection:
    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    @staticmethod
    def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
        return all(doc.get(key) == value for key, value in query.items())

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return next((dict(d) for d in self.docs if self._matches(d, query)), None)

    def find(self, query: dict[str, Any]) -> "FakeCursor":
        return FakeCursor([dict(d) for d in self.docs if self._matches(d, query)])

    def insert_one(self, document: dict[str, Any]) -> Any:
        self.docs.append(dict(document))
        return type("R", (), {"inserted_id": document.get("_id")})()

    def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> Any:
        matched = modified = 0
        for doc in self.docs:
            if self._matches(doc, query):
                matched = 1
                doc.update(update.get("$set", {}))
                modified = 1
                break
        return type("R", (), {"matched_count": matched, "modified_count": modified})()

    def delete_one(self, query: dict[str, Any]) -> Any:
        for index, doc in enumerate(self.docs):
            if self._matches(doc, query):
                self.docs.pop(index)
                return type("R", (), {"deleted_count": 1})()
        return type("R", (), {"deleted_count": 0})()

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        return None


class FakeCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs

    def sort(self, key: str, direction: int = 1) -> "FakeCursor":
        self.docs.sort(key=lambda d: str(d.get(key) or ""), reverse=direction < 0)
        return self

    def __iter__(self):
        return iter(self.docs)


@pytest.fixture
def fake_collection(monkeypatch: pytest.MonkeyPatch) -> FakeCollection:
    collection = FakeCollection()
    monkeypatch.setattr("src.tools.favorites.favorites_collection", lambda: collection)
    return collection


@pytest.fixture
def no_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Supabase reads/writes so nodes run without credentials."""
    from src.memory.user_memory import UserMemorySummary

    monkeypatch.setattr(
        "src.graph.nodes.get_user_memory_summary",
        lambda user_id: UserMemorySummary(user_id=user_id, available=False),
    )
    monkeypatch.setattr("src.graph.nodes.write_user_memory", lambda *a, **k: False)
