"""Shared jury plumbing: floors, windowing, record loaders.

The chain order lives in the driver module, oeb.graph.connect.
"""
from __future__ import annotations

from pathlib import Path
import concurrent.futures as cf
import json
import os
from oeb.jury import (QUOTE_JURY_MIN_CHARS, QUOTE_JURY_MIN_WORDS,
                      _json_block, _norm, judge, quote_rule)
from oeb.record.gids import gidf as _gidf
from oeb.record.gids import act_id as _canon_act_id
from oeb.graph.shapes import ActRow, StanceRow


# quote-floor disclosure for the _quote_hits juries: the prompt states
# the floor its own validation code enforces. Values live in oeb.jury.
_JURY_FLOOR_RULE = quote_rule(
    f"meets the floor: >= {QUOTE_JURY_MIN_WORDS} consecutive "
    f"words, or >= {QUOTE_JURY_MIN_CHARS} verbatim characters "
    "(code/symbol spans), or one ENTIRE |-separated component of "
    "the named text")


def vote_arrays(model, prompt, votes, salt_of):
    """One formatted prompt sampled `votes` times with distinct salts;
    yields (v, parsed_array) in vote order (the votes are fetched
    concurrently). A vote that parses to nothing yields []. salt_of(v)
    must reproduce the jury's salt string exactly: call-journal replay
    keys on it. Items that are not JSON objects (a bare string in the
    array) are dropped here, so no jury has to guard against them."""
    def _one(v):
        arr = _json_block(judge(model, prompt, salt=salt_of(v)), arr=True) or []
        return [it for it in arr if isinstance(it, dict)]
    with cf.ThreadPoolExecutor(max_workers=max(1, votes)) as ex:
        arrs = list(ex.map(_one, range(votes)))
    for v, arr in enumerate(arrs):
        yield v, arr


def fresh_items(arr, ids, key="id"):
    """Per-vote dedupe: yields each dict item whose `key` is in `ids`,
    at most once per vote, so a model repeating an entry N times in
    one response does not count as N votes."""
    seen: set = set()
    for it in arr:
        iid = it.get(key) if isinstance(it, dict) else None
        if iid is None or iid not in ids or iid in seen:
            continue
        seen.add(iid)
        yield it


TASK_PROMPT_ABSENT = "(goal message absent from dump)"
#   the adapters' marker for a dump that carried no goal message; read
#   as absent here, so the fallback applies and no judge is shown the
#   marker as the task.


def _task_statement(unit: Path) -> str:
    """The full shipped task statement: record_manifest.json's
    task_prompt (stored verbatim by the adapter), else domain_note.txt,
    else "". The post-graph E4 juries use _shipped_task_prompt instead."""
    mf = unit / "record_manifest.json"
    if mf.exists():
        try:
            t = str(json.loads(mf.read_text()).get("task_prompt") or "")
            if t and t.strip() != TASK_PROMPT_ABSENT:
                return t
        except (json.JSONDecodeError, OSError):
            pass
    dn = unit / "domain_note.txt"
    return dn.read_text() if dn.exists() else ""


def raw_steps(unit: Path) -> dict:
    """{step gid (float): the step as record.jsonl stores it}. The
    post-graph E4 juries show actions and observations as stored, not in
    the terminal rendering oeb.record.steps gives the extractor."""
    out = {}
    rp = unit / "record.jsonl"
    if rp.exists():
        for line in rp.open():
            if not line.strip():
                continue
            st = json.loads(line)
            try:
                out[float(_gidf(st.get("gid")))] = st
            except ValueError:
                continue
    return out


def _shipped_task_prompt(unit: Path) -> str:
    """The task as the agent read it: record_manifest.json's task_prompt
    and nothing else; "" when the dump carried none. _task_statement's
    domain_note fallback is the bench's description of the world for
    readers, not the task: a measure read from it would lie outside
    what the agent was given."""
    mf = unit / "record_manifest.json"
    if not mf.exists():
        return ""
    try:
        t = str(json.loads(mf.read_text()).get("task_prompt") or "")
    except (json.JSONDecodeError, OSError):
        return ""
    return "" if t.strip() == TASK_PROMPT_ABSENT else t


# Roster items carry their full text; a long roster is split into
# char-budgeted windows (_roster_windows) rather than cut.
WIN = 8                        # commits/claims per prompt
ROSTER_BUDGET = 200000         # chars per roster window. Rosters hold
                               # compressed nodes (~150 chars each), not
                               # raw observations, so a realistic unit
                               # fits in one window; claim_support
                               # records the count (roster_windows).


def _roster_windows(items, render):
    """Split a roster into char-budgeted windows — every item lands in
    exactly one window; nothing is dropped."""
    wins, cur, size = [], [], 0
    for it in items:
        t = len(render(it))
        if cur and size + t > ROSTER_BUDGET:
            wins.append(cur)
            cur, size = [], 0
        cur.append(it)
        size += t
    if cur:
        wins.append(cur)
    return wins or [[]]


# fan-out width for independent (window, vote) judge calls; the llm
# layer is thread-safe and caps calls in flight (OEB_LLM_CONCURRENCY)
JURY_WORKERS = int(os.environ.get("OEB_JURY_WORKERS") or 6)


def _act_line(a, returned=None):
    """One act-roster line (id, "type | mech | expectation | quote
    | outcome | returned"), as the birth-evidence jury lists candidates.
    Pass returned = _returned(acts) over the unit's acts so a launch
    shows the result its checks show; without it an act shows only its
    own obs_receipt."""
    obs = ((returned or {}).get(_canon_act_id(a)) or a.get("obs_receipt")
           or None)
    return (_canon_act_id(a),
            " | ".join(x for x in (
                a["act"], a.get("mech"), a.get("expectation"),
                a.get("action_receipt"),
                f"outcome={a.get('outcome')}" if a.get("outcome")
                else None,
                f"returned: {obs}" if obs else None) if x))


def _acts(unit: Path) -> list[ActRow]:
    p = json.loads((unit / "acts.json").read_text())
    if p.get("status") != "ok":
        raise SystemExit(f"{unit}: acts withheld ({p.get('status')})")
    return sorted(p["acts"], key=lambda a: _gidf(a["gid"]))


def _returned(acts) -> dict:
    """act id -> what the act's execution returned, as the extractor
    filed it: the act's own obs_receipt (a check's shown result, a
    probe's gathered evidence), else, for a launch, the obs_receipts of
    the checks that show its result (reads / reads_all, in step order).
    Absent = the graph holds no returned evidence for that act.

    The shared channel for "what came back": the extractor reads every
    observation in full and quotes what it shows, and judges read those
    quotes here rather than the record's observations (the coverage jury
    and reward_hacking's measure-as-seen jury are the exceptions). A
    roster so grows with the acts, not with the bytes a tool printed.
    Accepts acts.json rows or graph ACT nodes (node id wins).
    """
    def aid(a):
        return a.get("id") or _canon_act_id(a)
    shown_by: dict = {}
    for a in sorted(acts, key=lambda a: _gidf(a["gid"])):
        r = str(a.get("obs_receipt") or "").strip()
        if a.get("act") != "check" or not r or r == "None":
            continue
        for lid in (a.get("reads_all") or ([a["reads"]] if a.get("reads") else [])):
            got = shown_by.setdefault(str(lid), [])
            if r not in got:
                got.append(r)
    out = {}
    for a in acts:
        own = str(a.get("obs_receipt") or "").strip()
        if own and own != "None":
            out[aid(a)] = own
        elif shown_by.get(aid(a)):
            out[aid(a)] = " || ".join(shown_by[aid(a)])
    return out


def _evidence(acts):
    """Full evidence roster (probes + adjudicated checks), windowed by
    char budget at prompt time rather than cut, so late commits can
    still meet their real backers. Returned data comes from _returned."""
    returned = _returned(acts)
    ev = []
    for a in acts:
        if a["act"] == "probe" or (a["act"] == "check" and a.get("outcome")):
            obs = returned.get(_canon_act_id(a))
            ev.append({"id": _canon_act_id(a, "d"),
                       "gid": a["gid"],
                       "text": " | ".join(x for x in (
                           a.get("mech"), a.get("expectation"),
                           a.get("action_receipt"),
                           f"outcome={a.get('outcome')}"
                           if a.get("outcome") else None,
                           f"returned: {obs}" if obs else None) if x)})
    return ev


def _quote_hits(rec: str, text: str) -> bool:
    """Verbatim quote acceptance for the juries: a span of the text with
    >= QUOTE_JURY_MIN_WORDS words or >= QUOTE_JURY_MIN_CHARS characters,
    OR an exact quote of one entire |-separated component (short fields,
    such as grid/sim act texts, cannot yield five words at all)."""
    if not rec:
        return False
    nt = _norm(text)
    if rec in nt and (len(rec.split()) >= QUOTE_JURY_MIN_WORDS
                      or len(rec) >= QUOTE_JURY_MIN_CHARS):
        return True     # code/symbol spans have few whitespace words;
                        # 20+ verbatim chars is not trivially forgeable
    return rec in {_norm(seg) for seg in text.split(" | ") if seg}


# stance kinds that enter proposition identity (claim-merge): the
# adjudicable three plus questions (a question opens a proposition) and
# retracts (a retraction must group with the position it withdraws).
# Reliances stay out: they stake no proposition.
MERGE_KINDS = ("belief", "conjecture", "prediction", "question",
               "retract")


def _claims(unit: Path):
    """Adjudicable stances (belief/conjecture/prediction). ALL of them
    load here; testability is the settlement juries' verdict."""
    p = json.loads((unit / "acts.json").read_text())
    return [c for c in p.get("stances") or []
            if c.get("stance", "belief") in ("belief", "conjecture",
                                             "prediction")]


def _stances(unit: Path, kinds=MERGE_KINDS) -> list[StanceRow]:
    """All stances of the given kinds, testable or not (proposition
    identity needs questions and retracts too)."""
    p = json.loads((unit / "acts.json").read_text())
    return [c for c in p.get("stances") or []
            if c.get("stance", "belief") in kinds]


def _act_roster(acts):
    """All acts as a support roster (unlike premise pairing's
    probes+checks: a revise act IS the action behind "I reverted X", a
    commit the action behind "I built it on Y"). Full roster, windowed
    by char budget at prompt time. Returned data comes from _returned."""
    returned = _returned(acts)
    out = []
    for a in acts:
        obs = returned.get(_canon_act_id(a))
        out.append({"id": _canon_act_id(a),
                    "text": " | ".join(x for x in (
                        a["act"], a.get("mech"), a.get("expectation"),
                        a.get("action_receipt"),
                        f"outcome={a.get('outcome')}"
                        if a.get("outcome") else None,
                        f"returned: {obs}" if obs else None) if x)})
    return out

