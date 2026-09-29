"""Per-bench world declaration (world.json), read by the scorer.

The scorer never guesses world shape: each bench declares it, and
anything undeclared makes dependent readings abstain visibly instead of
silently misreading.

Each bench adapter emits world.json next to record.jsonl. The fields the
scorer reads (none has a default):
  authored_channel  {"kind": "patches+shell_redirect"} for worlds whose
                    agent-authored material is recorded in the patch
                    channel (the graph's ARTIFACT nodes read it), or
                    {"kind": "none"} when no such channel exists
  feedback_polarity "higher_better" | "lower_better": verifier reward
                    direction, copied into the graph's outcome block
  submission        the path the task's result is taken from (the
                    coverage jury asks which acts write into it)
  measure_size      how many cases THE MEASURE holds (coverage jury)
  contract          {"source", "text", "shown_to_agent"}: the world's
                    declared rules, for worlds whose shipped task states
                    none (the reward-hacking jury reads them with the task)

A standalone world.json file, not a key in another manifest: the
declaration must live wherever the record lives. Adapters may write
further fields for other readers; no score reads them.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_world(unit: Path) -> dict:
    """The unit's world declaration; {} when the bench never emitted
    one (every dependent reading must then abstain, never guess)."""
    p = Path(unit) / "world.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}
