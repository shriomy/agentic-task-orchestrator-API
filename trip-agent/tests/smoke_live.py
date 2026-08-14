"""Live smoke check — NOT part of the pytest suite.

Runs the real graph against the real LLM and the real travel APIs to confirm the
routing behaviour in the README table: which tools the agent actually reaches for,
and whether it pauses only when asked to.

    cd trip-agent
    python tests/smoke_live.py            # routing only, no HIL
    python tests/smoke_live.py --hil      # add the pause scenarios
    python tests/smoke_live.py --case 4   # one case by number

Costs real API calls. Favorites are stubbed so nothing is written to MongoDB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from src.graph.graph import build_graph  # noqa: E402
from src.graph.tools import SELECTION_TOOL  # noqa: E402

USER = "smoke-user"

# (label, message, required tools, should it pause?, forbidden tools)
#
# `forbidden` is what enforces "do not answer more than was asked" — the spec
# says case 1 is web search ONLY, so an attraction lookup there is a failure,
# not a bonus.
CASES = [
    ("1  country + season", "What are top 5 tourist destinations in Norway during this season?", {"web_search"}, False, {"search_places", "search_events", "search_accommodations"}),
    ("2  country + places", "What are top 5 tourist destinations in Norway this season and tell me what I can go visit in those places?", {"web_search", "search_places"}, False, {"search_accommodations"}),
    ("3  country + pick + places", "What are top 5 tourist destinations in Norway this season and let me pick some of them to get more ideas on places I can go visit in those places", {"web_search", SELECTION_TOOL}, True, set()),
    ("4  city only", "What are the places to visit in Moscow?", {"search_places"}, False, {"search_events", "search_accommodations"}),
    ("5  events only", "What are the events happening in Paris this weekend?", {"search_events"}, False, {"search_accommodations"}),
    ("6  country + stays", "Tell me places to go visit in Thailand and accommodations nearby each", {"web_search", "search_accommodations"}, False, {"search_events"}),
    ("7  country + pick + stays", "Tell me places to go visit in Thailand so I can choose a few of them to decide my accommodations nearby each", {"web_search", SELECTION_TOOL}, True, set()),
    ("8  city places + events", "Send me Shanghai's best places to visit along with events next week", {"search_places", "search_events"}, False, {"search_accommodations"}),
    ("9  packing (no travel words)", "What should I pack?", set(), False, {"search_accommodations", "search_events"}),
    ("10 out of scope", "Write me a Python script to scrape hotel prices", set(), False, {"web_search", "search_places", "search_events", "search_accommodations"}),
]


def _printable(text: str) -> str:
    """Windows consoles default to cp1252, which cannot encode the em dashes and
    accented place names that come back from these APIs."""
    flattened = " ".join((text or "").split())
    encoding = sys.stdout.encoding or "utf-8"
    return flattened.encode(encoding, errors="replace").decode(encoding, errors="replace")


def run(case_index: int, allow_hil: bool) -> bool:
    label, message, expected, expect_pause, forbidden = CASES[case_index]
    if expect_pause and not allow_hil:
        print(f"SKIP {label}  (pass --hil to include)")
        return True

    graph = build_graph(checkpointer=InMemorySaver())
    thread = f"smoke-{case_index}"
    config = {"configurable": {"thread_id": thread, "auth_user_id": USER}}

    called: list[str] = []
    paused = False
    reply = ""

    try:
        for chunk in graph.stream(
            {"thread_id": thread, "user_id": USER, "message": message},
            config=config,
            stream_mode="updates",
        ):
            for node, update in (chunk or {}).items():
                if node == "__interrupt__":
                    paused = True
                    continue
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages") or []:
                    if isinstance(msg, AIMessage):
                        for call in getattr(msg, "tool_calls", None) or []:
                            called.append(call.get("name", "?"))
                if node in ("finalize", "smalltalk", "out_of_scope"):
                    reply = str(update.get("bot_response") or reply)
    except Exception as exc:
        print(f"FAIL {label}\n     error: {type(exc).__name__}: {exc}")
        return False

    used = set(called)
    missing = expected - used
    overreach = used & forbidden
    pause_ok = paused == expect_pause
    ok = not missing and not overreach and pause_ok

    print(f"{'PASS' if ok else 'FAIL'} {label}")
    print(f"     tools used : {sorted(used) or '(none)'}")
    if expected:
        print(f"     required   : {sorted(expected)}")
    print(f"     paused     : {paused}   (expected {expect_pause})")
    print(f"     reply      : {_printable(reply[:200])}")
    if missing:
        print(f"     >> under-reach, never called: {sorted(missing)}")
    if overreach:
        print(f"     >> over-reach, should not have called: {sorted(overreach)}")
    if not pause_ok:
        print(f"     >> pause behaviour wrong (expected {expect_pause}, got {paused})")
    print()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hil", action="store_true", help="include the pause scenarios")
    parser.add_argument("--case", type=int, help="run a single case (1-based)")
    args = parser.parse_args()

    # Keep favorites out of it — this script should never touch real saved trips.
    import src.graph.tools as tools

    tools.get_favorites = lambda *a, **k: {"count": 0, "favorites": []}
    tools.save_favorite = lambda *a, **k: {"action": "created", "favorite": {}}
    tools.delete_favorite = lambda *a, **k: {"action": "not_found"}
    tools.remove_favorite_section = lambda *a, **k: {"action": "not_found"}
    tools.update_favorite = lambda *a, **k: {"action": "not_found"}

    indices = [args.case - 1] if args.case else range(len(CASES))
    results = [run(index, args.hil) for index in indices]
    passed = sum(1 for result in results if result)
    print(f"{passed}/{len(results)} cases behaved as expected")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
