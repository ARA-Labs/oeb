"""Step ids (gids): the one shared parser.

The record contract requires numeric-orderable gids, supplied by the
bench adapter; the scorer never guesses world shape.

A non-parseable gid is an ADAPTER bug: gidf() raises immediately,
naming the offender and the contract — it never returns a guessed 0.0,
which would silently mis-sort the record. validate_steps() runs the check at record
load so failure happens at ingestion, not deep in scoring.
"""
from __future__ import annotations

import re


def gidf(g, ctx: str = "") -> float:
    """Parse one gid to its numeric order value. Raises on anything
    non-numeric — the caller never receives a guess."""
    try:
        return float(str(g))
    except (TypeError, ValueError):
        raise ValueError(
            f"non-numeric gid {g!r}" + (f" [{ctx}]" if ctx else "")
            + " — the canonical record contract requires "
            "numeric-orderable gids; fix the bench adapter "
            "(benches/<bench>/), core never guesses") from None


def validate_steps(steps, ctx: str = "record") -> list:
    """Validate every step's gid at record load (ingestion-time
    failure, not deep-in-scoring failure)."""
    for s in steps:
        gidf(s.get("gid"), ctx)
    return steps


# ---- tolerant parsers for JUDGE-AUTHORED references (jury output)

def gidf_or_none(g) -> float | None:
    """Tolerant twin of gidf for judge-authored gid REFERENCES (jury
    output), where a malformed value is the judge's fault, not the
    adapter's: float, or None so the caller counts/abstains — never
    a guessed 0.0 and never a raise."""
    try:
        return float(str(g))
    except (TypeError, ValueError):
        return None


_CITE_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def gidf_cited(s) -> float | None:
    """gidf_or_none for CITATION strings a judge quotes back from a
    rendered roster. Tolerates exactly two decorations, both roster
    artifacts (a rendered-form citation must never reject a merge on
    string shape alone): a trailing
    ':<outcome>' (rosters render 'gid:outcome') is stripped first,
    then the FIRST number in what remains wins over any non-numeric
    prefix/suffix ('gid 12', 'a12', '12abc' all read 12.0). No number
    at all -> None (counted abstention, never a guess)."""
    m = _CITE_NUM.search(str(s or "").split(":", 1)[0])
    return float(m.group()) if m else None


def act_id(a, plane: str = "a") -> str:
    """ONE act id, the constructing twin of parse_act_gid. Grammar:
    '<plane><gid>.<ty>' plus the SLOT when one step set going several
    distinct things under study — 'a110.co', 'a110.co2' (launch slots,
    the same ids extract.launch_id builds).
    Every roster and jury builds ids through here so two commits born at
    one step never share an id. plane 'a' is the act plane, 'd' the
    evidence plane the pairing and settlement juries render."""
    slot = a.get("slot") or 1
    tail = ("" if a["act"] != "commit" or int(slot) == 1
            else str(int(slot)))
    return f"{plane}{a['gid']}.{a['act'][:2]}{tail}"


def parse_act_gid(act_id, ctx: str = "") -> float:
    """Gid of one TYPED ACT ID — the 'a<gid>.<ty>' / 'd<gid>.<ty>'
    grammar (one-letter plane tag + gid + optional dot-joined type
    suffix). Split on the LAST dot with the numeric-tail rule of
    assemble._evidence_act (fractional gids are legal): a
    non-numeric tail is the type suffix ('a12.5.ch' -> 12.5), a
    numeric tail belongs to the gid ('a12.5' -> 12.5), no dot means
    the whole body is the gid ('c4' -> 4.0). Raises via gidf on a
    malformed gid — act ids are built by the pipeline, so a bad one is a
    pipeline bug, never a judge abstention."""
    body = str(act_id)[1:]
    head, sep, tail = body.rpartition(".")
    if sep and gidf_or_none(tail) is None:
        body = head
    return gidf(body, ctx or f"act id {act_id!r}")


def parse_act_gid_or_none(act_id) -> float | None:
    """parse_act_gid for act-id REFERENCES a judge quotes back
    (pairing/support ids): same grammar, but a malformed id is the
    judge's fault — None so the caller counts/abstains, never a
    raise (same split as gidf/gidf_or_none)."""
    try:
        return parse_act_gid(act_id)
    except ValueError:
        return None
