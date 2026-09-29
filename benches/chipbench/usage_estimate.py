"""Token-account estimates for Chip-Bench usage docs whose per-turn output is not stamped.

convert.py calls estimate() on every usage doc it writes. Two dialects
would otherwise leave T1 without a token account:

1. claude stream-json (grain api_call): every assistant line carries the
   message_start snapshot, so a turn's `out` is unknown. run_total's
   `out` and thinking are exact but cover only the session files that
   ended in a CLI result line; a file killed mid-run (the resumed runs)
   has no total. In the covered files the non-thinking output R =
   out_total - their thinking is spread over their turns by visible
   characters (prose + tool payload), then split into text/action by
   the same proportion (the rule the known-out rows and the grok
   dialect already use). R and those files' visible characters give
   this run's own chars per output token; a killed file's turns get
   thinking + visible / that ratio. No estimate when no file is
   covered or R <= 0.

2. codex (grain run_total): only run totals exist, reasoning tokens
   exact. The visible channel is char-counted from the canonical steps
   and banded 2.9 .. 4.2 chars per token with posttrainbench's
   `_usage_armtotal` (the shared run-total rule), so T1's
   arm_budget_doing renders only when both band ends agree on the pole.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
EST_TAG = "estimate"


def _armtotal():
    p = HERE.parent / "posttrainbench" / "adapter.py"
    spec = importlib.util.spec_from_file_location("_pta_adapter", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._usage_armtotal


def _in_exact_segment(r: dict) -> bool:
    """A turn whose session file ended with a CLI result line: the
    converter rescaled its thinking estimate to that file's exact total
    (or stamped it exact). A file killed before its result line (the
    resumed runs) carries raw, unrescaled estimates and is not covered
    by run_total."""
    src = r.get("thinking_source") or ""
    return "rescaled" in src or src.startswith("exact")


def _split(row: dict, think: int, nonr: float, vis: int, source: str) -> dict:
    row = dict(row)
    text = nonr * int(row.get("prose_chars") or 0) / vis if vis else 0.0
    row.update({"out": round(think + nonr), "text_tokens": round(text),
                "action_tokens": round(nonr - text), "out_source": source})
    row.pop("out_note", None)
    return row


def spread_unstamped(usage: dict) -> dict:
    """Case 1. Returns the usage doc with unknown-out turns estimated.

    Exact segments (files with a result line): run_total's non-thinking
    output R = out_total - sum(their thinking) is spread over their turns
    by visible chars. Their pooled ratio cpt = visible_chars / R is this
    run's own chars-per-output-token; turns of files killed before a
    result line get out = thinking + visible_chars / cpt."""
    if usage.get("grain") != "api_call":
        return usage
    tot = usage.get("run_total") or {}
    out_total = tot.get("out")
    turns = usage.get("turns") or {}
    unknown = {k: r for k, r in turns.items() if r.get("out") is None}
    if not unknown or not out_total:
        return usage
    if len(unknown) != len(turns):
        doc = dict(usage)
        doc["out_estimate"] = {"status": "withheld",
                               "why": "mixed stamped and unstamped turns"}
        return doc

    def vis(r):
        return int(r.get("prose_chars") or 0) + int(r.get("tool_chars") or 0)

    exact = {k: r for k, r in turns.items() if _in_exact_segment(r)}
    rest = {k: r for k, r in turns.items() if k not in exact}
    think_ex = sum(int(r.get("thinking_tokens") or 0) for r in exact.values())
    vis_ex = sum(vis(r) for r in exact.values())
    rem = out_total - think_ex
    doc = dict(usage)
    if not exact or rem <= 0 or vis_ex <= 0:
        doc["out_estimate"] = {
            "status": "withheld",
            "why": ("no turn in a file with a result line" if not exact
                    else f"non-thinking remainder {rem} <= 0" if rem <= 0
                    else "no visible characters in the exact segments")}
        return doc
    cpt = vis_ex / rem
    new_turns = {}
    for k, r in turns.items():
        think = int(r.get("thinking_tokens") or 0)
        if k in exact:
            new_turns[k] = _split(r, think, rem * vis(r) / vis_ex, vis(r),
                                  f"{EST_TAG}: exact-segment non-thinking output "
                                  "spread by visible chars")
        else:
            new_turns[k] = _split(r, think, vis(r) / cpt, vis(r),
                                  f"{EST_TAG}: killed-segment turn, visible chars / "
                                  "this run's exact-segment chars per token")
    doc["turns"] = new_turns
    doc["out_estimate"] = {
        "status": "ok",
        "rule": ("exact segments (files ending in a CLI result line; thinking "
                 "rescaled to their exact total): out = thinking + R * visible / "
                 "sum(visible), R = run_total.out - their thinking. Killed segments "
                 "(no result line, not in run_total): out = thinking (raw CLI "
                 "estimate) + visible / cpt, cpt = the exact segments' visible "
                 "chars / R. text/action = the non-thinking part split by "
                 "prose/tool chars (same chars-per-token assumed for both)"),
        "R_non_thinking_out": rem,
        "exact_turns": len(exact), "killed_turns": len(rest),
        "exact_visible_chars": vis_ex,
        "chars_per_token": round(cpt, 2),
    }
    doc["reporter"] = (usage.get("reporter") or "") + (
        f"; {EST_TAG}: unstamped turns' out estimated (see out_estimate) so "
        "T1 has an account")
    return doc


def _visible_chars(steps: list) -> tuple[int, int]:
    prose = tool = 0
    for s in steps:
        if s.get("src") != "agent":
            continue
        prose += len(s.get("msg") or "")
        for t in s.get("tools") or []:
            tool += len(str(t.get("cmd") or ""))
        for p in s.get("patches") or []:
            tool += len(str(p.get("old") or "")) + len(str(p.get("new") or ""))
    return prose, tool


def band_run_total(usage: dict, steps: list) -> dict:
    """Case 2. Adds the run-total band to a run_total doc."""
    if usage.get("grain") != "run_total":
        return usage
    tot = usage.get("run_total") or {}
    out_tok = tot.get("out")
    if not out_tok:
        return usage
    prose, tool = _visible_chars(steps)
    arm = _armtotal()(out_tok, tot.get("thinking_tokens"), prose, tool,
                      "codex turn.completed run totals")
    doc = dict(usage)
    doc.update({"grain": "arm_total", "out_tok": arm["out_tok"],
                "reasoning_tok": arm["reasoning_tok"],
                "prose_chars": prose, "tool_chars": tool, "band": arm["band"],
                "reporter": arm["reporter"] + f"; {EST_TAG}: visible chars "
                            "counted from the canonical steps"})
    if arm.get("inconsistent"):
        doc["inconsistent"] = arm["inconsistent"]
    return doc


def estimate(usage: dict, steps: list) -> dict:
    return band_run_total(spread_unstamped(usage), steps)

