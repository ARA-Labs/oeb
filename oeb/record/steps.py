"""Load a unit's steps, each observation rendered as a terminal shows it.

record.jsonl is the adapter's archive and is never rewritten. The
extractor, the episode juries and the uptake jury read the step as the agent's tools
displayed it: every observation passes through
oeb.record.terminal.render, so a progress bar's thousands of
overwritten redraws read as the screen they left (each redrawn line
marked where it stands) and every other character is kept. They share
this loader so a quote taken by one verifies against the same text
wherever another checks it. The post-graph E4 juries read the raw
record instead (graph/juries/_shared.raw_steps).
"""
from __future__ import annotations

import json
from pathlib import Path

from oeb.record.gids import gidf, validate_steps
from oeb.record.terminal import render


def load_steps(unit) -> list[dict]:
    """Steps of <unit>/record.jsonl in gid order, gid as str, obs as a
    terminal shows it."""
    rp = Path(unit) / "record.jsonl"
    steps = [json.loads(line) for line in rp.open() if line.strip()]
    for s in steps:
        s["gid"] = str(s.get("gid"))
        if isinstance(s.get("obs"), str):
            s["obs"] = render(s["obs"])
    validate_steps(steps, str(rp))
    steps.sort(key=lambda s: gidf(s["gid"]))
    return steps
