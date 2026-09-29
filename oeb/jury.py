"""Shared jury infrastructure.

Home of the LLM-judging primitives every instrument shares. Every call
flows through oeb/llm.py: hardened transport, retry with backoff,
per-run call journal with replay, Refused accounting.

Refusal fallback: when the judge model refuses a prompt, judge() asks a
second model instead (gpt-5.5 by default; OEB_JUDGE_REFUSAL_FALLBACK
names another, empty turns the fallback off). Its answer is journaled
under its own model id and counted (fallback_count); a prompt both
models refuse is counted as a refusal and yields no vote.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import threading

from oeb import llm


def _norm(s: str) -> str:
    """The text normalizer for quote/identity matching; jury modules
    import this. A verified quote proves the judge read THIS text, and the
    proof lives in its letters and digits — every other character
    (quotes straight or curly, dashes of any length, escapes, markdown
    marks, punctuation, spacing, width, case) is a copyist's choice
    that no judge should be scored on. So both sides fold to Unicode
    NFKC, lowercase, and every run of non-alphanumerics becomes ONE
    space (word boundaries survive for the word-overlap and word-floor
    readers). Literal escapes (\\n/\\t/\\r) are replaced first so a
    literal backslash-n never leaves a stray "n". Applies to BOTH sides
    of every comparison (single normalizer = no drift)."""
    s = (s or "").replace("\\n", " ").replace("\\t", " ") \
                 .replace("\\r", " ")
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^0-9a-z]+", " ", s).strip()


def vote_need(votes: int) -> int:
    """The vote-acceptance threshold: strict majority of cast votes
    (2 at the default votes=3)."""
    return votes // 2 + 1


def union_by_votes(ids, pair_votes, votes):
    """Union-find fold of pairwise merge votes. pair_votes:
    {(x, y): n_votes}; pairs reaching vote_need(votes) merge. Returns
    {id: root} over ids (path-halving find), keyed in sorted id order:
    callers pass sets, and a set's order changes with each process's
    hash seed, so sorting keeps outputs byte-stable across runs."""
    ids = sorted(ids)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    need = vote_need(votes)
    for (x, y), c in pair_votes.items():
        if c >= need:
            parent[find(x)] = find(y)
    return {i: find(i) for i in ids}


# the follow-up window, counted in steps or acts: a count rather than
# tokens keeps it fair across verbose and terse agents
FOLLOW_WINDOW = 8

# ---- quote floors (single home) -----------------------------------
QUOTE_EXTRACTOR_MIN_CHARS = 6
#   extract._quote_ok floor — normalized chars; grid/sim
#   action fields run too short for word floors
QUOTE_JURY_MIN_WORDS = 5
#   connecting-jury verbatim-quote floor (>= 5 consecutive words)
QUOTE_JURY_MIN_CHARS = 20
#   connecting-jury code-span floor: code/symbol spans have few
#   whitespace words, and 20+ verbatim chars is not trivially forgeable
QUOTE_SPAN_MIN_WORDS = 3
#   short-span floor: topic, subject and reward-hacking juries


def quote_rule(floor: str) -> str:
    """Quote-acceptance disclosure for prompts: the acceptance
    mechanics are instrument fact, not judged semantics — hiding them
    would measure judge quoting style, which no dimension is supposed
    to read.
    Interpolated with each instrument's own floor so the prompt can
    never drift from the code that enforces it (floors live above)."""
    return (
        "RECEIPT ACCEPTANCE (mechanical, applied to every quote): "
        "quotes are compared on their letters and digits only — case, "
        "spacing, punctuation, quote and dash style and markup are "
        "ignored on both sides; a quote "
        f"is accepted only when it is a verbatim substring of the "
        f"named text AND {floor}. A shorter or inexact quote "
        "silently voids the emission it carries: quote generously, "
        "from the exact text shown.")
# The 6-char extractor floor and the 20-char jury floor are set
# independently (short-field tolerance vs forgeability heuristic).


def judge(model: str, prompt: str, timeout: int | None = None, salt: str = "",
          refusal_raises: bool = False) -> str:
    """Identical (model, prompt, salt) memoize to ONE sample — sampling
    loops must pass distinct salts (vote/run index).

    Per-call wall clock: the caller's explicit `timeout`, else
    OEB_JUDGE_TIMEOUT (seconds), else 600 (long extraction windows can
    need close to that).

    Refusals: a provider refusal never kills a stage. First the
    fallback judge answers in the primary's place
    (gpt-5.5 by default; OEB_JUDGE_REFUSAL_FALLBACK overrides, empty
    disables) — the stand-in call is journaled under its own model id
    and counted.
    Only when the fallback also refuses (or is disabled) does the
    refusal count bump and judge() return "" (unparseable, so the
    vote drops exactly like a parse failure). The one caller with
    FINER accounting (extract, per-window refused_windows)
    passes refusal_raises=True and gets the exception at that point
    instead."""
    if timeout is None:
        timeout = int(os.environ.get("OEB_JUDGE_TIMEOUT") or 600)
    try:
        return llm.call(model, prompt, timeout=timeout, salt=salt)
    except llm.Refused:
        fb = os.environ.get("OEB_JUDGE_REFUSAL_FALLBACK", "gpt-5.5")
        if fb and fb != model:
            try:
                out = llm.call(fb, prompt, timeout=timeout, salt=salt)
                _bump_fallbacks()
                return out
            except llm.Refused:
                pass
        if refusal_raises:
            raise
        _bump_refusals()
        return ""


# Parse-failure count: a vote whose response yields no parseable JSON
# block is counted, so a format-weak judge's lost votes are visible
# rather than indistinguishable from genuine disagreement. Same family
# as refusal accounting: the gap ships with the data (writers snapshot
# parse_failure_count() deltas into their payloads).
_PARSE_FAILURES = 0
_pf_lock = threading.Lock()    # judges parse in fan-out threads:
                               # += is not atomic


def parse_failure_count() -> int:
    return _PARSE_FAILURES


def _bump_parse_failures():
    global _PARSE_FAILURES
    with _pf_lock:
        _PARSE_FAILURES += 1


_REFUSALS = 0
_FALLBACKS = 0


def refusal_count() -> int:
    return _REFUSALS


def fallback_count() -> int:
    return _FALLBACKS


def _bump_refusals():
    global _REFUSALS
    with _pf_lock:
        _REFUSALS += 1


def _bump_fallbacks():
    global _FALLBACKS
    with _pf_lock:
        _FALLBACKS += 1


def _json_block(raw: str, arr: bool = True):
    m = re.search(r"\[.*\]" if arr else r"\{.*\}", raw, re.S)
    if not m:
        if (raw or "").strip():
            _bump_parse_failures()
        return [] if arr else {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        _bump_parse_failures()
        return [] if arr else {}


def contains(span: str, quote: str) -> bool:
    return bool(span) and _norm(span) in _norm(quote)
