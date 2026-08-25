"""Semantic tool retrieval — bind only the tools relevant to this turn.

Without this, `AgentNode` binds all 13 tool schemas (and the system prompt
describes all of them) on every agent<->tools round of every turn, regardless
of what the request actually needs. Tool docs are embedded once into a local
Qdrant collection at startup; each turn, `retrieve_relevant_tools` does one
semantic search against the user's (sanitised) request and returns the tool
names worth binding.

Fails open everywhere: any Qdrant or embedding problem returns `None` rather
than raising, so the caller falls back to binding every tool. This mirrors the
fail-open contract already used by the scope/output guardrails
(guardrails/scope.py, guardrails/output.py) — a retrieval problem must degrade
to "slower and pricier", never to "the agent lost a capability it needed".

Favorites tools (list/save/update/remove/delete) are indexed as a single
atomic unit rather than five independently-scored points. Several of the
system prompt's favorites rules only make sense if the model can see all five
together — e.g. "use remove_trip_favorite_section, not delete_trip_favorite,
for a partial removal" requires both names to be visible — so a hit on the
group returns all five names, never a subset of them.

`request_user_selection` is never part of the index at all: it is structurally
required by `RequireSelectionNode` (see graph/nodes.py) to enforce a pause the
user explicitly asked for, so it is always bound by the caller regardless of
what retrieval returns.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from ..config import settings
from .tools import FAVORITE_TOOLS, SELECTION_TOOL, TOOLS_BY_NAME

logger = logging.getLogger(__name__)

_FAVORITES_UNIT_ID = "favorites_group"

# A short, hand-written descriptor rather than the concatenation of all five
# tools' full docstrings. Concatenating them produced a document several times
# longer than any single-tool document, and bi-encoder similarity scores skew
# toward longer, more topically-diverse documents — in testing this made the
# favorites group the top hit for essentially every query, including ones with
# nothing to do with saved trips (e.g. "what should I pack for Iceland"),
# defeating the filter entirely. Keeping it comparable in length to the other
# units' documents keeps its score meaningful relative to them.
_FAVORITES_DOCUMENT = (
    "favorites: manage the user's own saved trips in the database. List saved "
    "trips, save a destination's picked places, events and accommodations as a "
    "favorite, add more items to a saved trip, remove one section (places, "
    "events or accommodations) from a saved trip, or delete an entire saved trip."
)

# Local, on-machine embedding model (ONNX via fastembed) — no API key, no
# per-call cost. 384-dim, small enough that encoding a handful of short tool
# docs and one query per turn is effectively free.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _tool_document(tool: Any) -> str:
    name = getattr(tool, "name", "")
    description = getattr(tool, "description", "") or ""
    return f"{name}: {description}"


def _index_units() -> list[dict[str, Any]]:
    """One unit per discovery tool, plus one combined unit for all favorites
    tools. Each unit is what gets embedded and searched as a whole."""
    units: list[dict[str, Any]] = []
    for name, tool in TOOLS_BY_NAME.items():
        if name == SELECTION_TOOL or tool in FAVORITE_TOOLS:
            continue
        units.append({"key": name, "document": _tool_document(tool), "names": [name]})

    units.append(
        {
            "key": _FAVORITES_UNIT_ID,
            "document": _FAVORITES_DOCUMENT,
            "names": [getattr(tool, "name", "") for tool in FAVORITE_TOOLS],
        }
    )
    return units


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"trip_agent_tools:{key}"))


@lru_cache(maxsize=1)
def _client() -> Any:
    from qdrant_client import QdrantClient

    # check_compatibility=False: a local dev Qdrant server commonly lags a
    # minor version or two behind the client; that mismatch is harmless here
    # and shouldn't spam a warning on every startup.
    return QdrantClient(url=settings.qdrant_url, check_compatibility=False)


def ensure_tool_index() -> bool:
    """Seed the Qdrant collection. Called once at startup.

    Always deletes and recreates rather than diffing against what's already
    indexed — with 7 points and a local embedding model this is sub-second (on
    a warm fastembed model cache), and it is the simplest way to guarantee the
    index can never drift from a docstring change without needing separate
    versioning machinery.

    Returns whether the index is ready to query. Never raises.
    """
    if not settings.tool_rag_enabled:
        return False
    try:
        from qdrant_client import models

        client = _client()
        if client.collection_exists(settings.qdrant_collection):
            client.delete_collection(settings.qdrant_collection)

        size = client.get_embedding_size(EMBEDDING_MODEL)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )

        units = _index_units()
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                models.PointStruct(
                    id=_point_id(unit["key"]),
                    vector=models.Document(text=unit["document"], model=EMBEDDING_MODEL),
                    payload={"names": unit["names"]},
                )
                for unit in units
            ],
        )
        logger.info(
            "tool RAG index seeded: %d point(s) covering %d tool(s)",
            len(units),
            sum(len(unit["names"]) for unit in units),
        )
        return True
    except Exception as exc:
        logger.warning("tool RAG index unavailable, falling back to all tools: %s", exc)
        return False


def tool_index_ready() -> bool:
    """Cheap live check for /health — does not reseed, just confirms the
    collection is there and reachable."""
    if not settings.tool_rag_enabled:
        return False
    try:
        return _client().collection_exists(settings.qdrant_collection)
    except Exception:
        return False


def retrieve_relevant_tools(query: str, k: int | None = None) -> list[str] | None:
    """Semantic search for the tools most relevant to this turn's request.

    Returns `None` on any failure (Qdrant down, collection not ready,
    embedding error) or an empty query, so the caller can fail open to binding
    every tool.
    """
    if not settings.tool_rag_enabled or not (query or "").strip():
        return None
    try:
        from qdrant_client import models

        client = _client()
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=models.Document(text=query, model=EMBEDDING_MODEL),
            limit=k or settings.tool_rag_top_k,
            with_payload=True,
        )
    except Exception as exc:
        logger.warning("tool retrieval failed, falling back to all tools: %s", exc)
        return None

    names: list[str] = []
    for point in response.points:
        payload = point.payload or {}
        names.extend(payload.get("names") or [])
    return list(dict.fromkeys(names)) or None
