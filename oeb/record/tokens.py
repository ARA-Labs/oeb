"""Token-cost axis over the record: cumulative tokens at each step, from usage.json.

Used by the graph assembler. Only the api_call grain has an exact
per-step reading (per-API-turn ledger joined to steps via record.jsonl's
api_turn stamp; unstamped steps take the nearest earlier stamped step).
Other grains carry no per-step truth and are refused here; callers
report them as totals only.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path


def token_axis(unit: Path):
    """gidf -> cumulative tokens callable (None if no usable ledger).

    The returned callable carries `.total` — the run's final
    cumulative token count."""
    up = unit / "usage.json"
    rp = unit / "record.jsonl"
    if not up.exists() or not rp.exists():
        return None
    usage = json.loads(up.read_text())
    if usage.get("grain") != "api_call":
        return None
    turns = usage.get("turns") or {}
    cum, run = {}, 0.0
    for t in sorted(turns, key=int):
        r = turns[t]
        run += (r.get("in") or 0) + (r.get("cached") or 0) \
            + (r.get("out") or 0)
        cum[int(t)] = run
    g2t = {}
    for ln in rp.read_text().splitlines():
        if not ln.strip():
            continue
        s = json.loads(ln)
        if s.get("api_turn") is not None and s.get("gid") is not None:
            g2t[float(s["gid"])] = int(s["api_turn"])
    if not cum or not g2t:
        return None
    stamped = sorted(g2t)
    cturns = sorted(cum)

    def at(gidf: float) -> float:
        # unstamped steps take the nearest earlier stamped step;
        # turns missing from the ledger take the nearest earlier one
        i = bisect.bisect_right(stamped, gidf) - 1
        turn = g2t[stamped[i]] if i >= 0 else g2t[stamped[0]]
        j = bisect.bisect_right(cturns, turn) - 1
        return cum[cturns[j]] if j >= 0 else 0.0
    at.total = run
    return at
