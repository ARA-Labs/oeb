"""Assembly: one unified graph built from the extraction and jury products.

The extractor (oeb.graph.extract) writes acts.json holding BOTH
planes: ACT cards (quote-verified acts) and STANCE cards (the semantic plane, typed
kinds with birth labels); the connecting juries journal their cross-item judgments
(deed_juries.json). This builder deterministically assembles them into
unified_graph.json, which the post-graph juries and the measure layer
read (the post-graph juries also re-read record.jsonl for step text).

No LLM calls: every semantic judgment was already made and journaled
upstream (extractor labels at birth; juries for cross-item relations).
The builder only joins them and derives record facts from the
extractor's pointers (reads/settles, responds_to, explains, and the
launch-channel adjudicates edges).

Node types
  ACT     a<gid>.<ty>  typed epistemic act (check/probe/commit/revise,
                       quotes, outcome, line membership)
  STANCE  k<i>         epistemic stance extracted from prose (belief |
                       conjecture | prediction | question | reliance |
                       retract | attribution; birth labels; verbatim quote)
  LINE    <key>        line of inquiry (lines-jury grouping; exact
                       mechanism identity only when <2 mechanisms)
  ARTIFACT art:<path>  a file the agent's recorded edits touched
Edge kinds
  engages      ACT -> LINE     mechanism membership
  states       ACT -> STANCE   same-step narrative anchor
  supports     ACT -> ACT      evidence act backs a commit's premise
                               (premise-pairing jury; edges exist ONLY
                               when the jury passed)
  adjudicates  ACT -> STANCE   act bears on a stated position
                               (claim-act alignment jury; same gate)
  grounds      ACT -> STANCE   the same, for a check at or before the
                               stance's birth (its basis, not its test)
  reads / settles  ACT -> ACT  a check read a launch (settles: with a
                               verdict against its stated expectation)
  builds_on / expects / responds_to / explains / touches: the
               extractor's pointers and the patch channel, copied

Proposition identity lives on STANCE nodes as stance_group (claim-
merge jury over belief/conjecture/prediction/question/retract); no
separate proposition node is stored.

Writes <unit>/unified_graph.json and <unit>/overlays/juries.json.

Record/opinion split: the graph FILE is record-only: nodes, edges,
identity, and the journal blocks that built them. The evaluative jury
(novelty_stream) is written to <unit>/overlays/juries.json, versioned
independently, so a scoring change rewrites the overlay and never the
graph.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from oeb.jury import _norm  # single normalizer
# ONE shared gid parser: raises on non-numeric gids instead of guessing 0.0
from oeb.record.gids import act_id as _act_id, gidf as _gidf, gidf_or_none as _gidf_or_none
from oeb.record.world import load_world


def _patch_paths(entry) -> tuple[list, int]:
    """Deterministic path extraction from ONE patches entry.

    Two shapes exist in recorded units: {"path", "new": <content>}
    (clean, one file) and {"path": "<possibly several names>",
    "patch": <git diff text>} (adapter shorthand; the per-file truth
    is the diff's own '+++ b/<path>' headers, a machine format with an
    exact grammar). Reading keys and parsing git's spec'd headers is
    the whole method; prose is never pattern-matched. Returns
    (paths, unparsed) where unparsed counts entries this reader
    refuses to guess about (surfaced in artifact_plane stats, never
    silently dropped)."""
    if not isinstance(entry, dict):
        return [], 1
    diff = entry.get("patch")
    if isinstance(diff, str) and "+++ " in diff:
        paths = []
        for ln in diff.splitlines():
            if ln.startswith("+++ "):
                t = ln[4:].strip()
                if t == "/dev/null":
                    continue
                paths.append(t[2:] if t.startswith(("a/", "b/")) else t)
        if paths:
            return sorted(set(paths)), 0
    # apply_patch format: the other spec'd grammar seen in
    # recorded units ("*** Update File: <path>" / Add / Delete);
    # entries arrive shell-escaped, so match on the marker anywhere
    # in the line, not at line start
    if isinstance(diff, str) and "*** " in diff:
        paths = []
        for ln in diff.replace("\\n", "\n").splitlines():
            for mk in ("*** Update File: ", "*** Add File: ",
                       "*** Delete File: "):
                i = ln.find(mk)
                if i >= 0:
                    t = ln[i + len(mk):].strip().rstrip("\\")
                    if t:
                        paths.append(t)
        if paths:
            return sorted(set(paths)), 0
    path = entry.get("path")
    if isinstance(path, str) and path and not any(
            c.isspace() for c in path):
        return [path], 0
    return [], 1


def _run_annotations(unit: Path, nodes: list) -> tuple:
    """Outcome block + axis annotations.
    Graph-level ANNOTATIONS, not nodes: facts about the run that the
    harness machine-recorded, copied out of record.jsonl so a reader
    need not dig them out of the last step's prose.

      outcome  what the run amounted to: step span, runs/levels
               reached, verifier feedback tally. The benchmark's own
               verdict on the run is never carried: the scorer does not
               see it.
      axis     phases: contiguous (run, level) intervals — harness
               facts (a level transition is the engine's, not the
               agent's); uncovered_spans: maximal step intervals the
               extractor found nothing in (no ACT/STANCE node) — the
               blank stretches said out loud instead of left as holes
               readers discover by counting. n_patch_steps rides each
               span so "invisible but busy editing" is visible.

    Zero-LLM, pure copying/counting. Returns (outcome, axis)."""
    rp = unit / "record.jsonl"
    if not rp.exists():
        na = {"status": "no record.jsonl"}
        return na, na
    rec = [json.loads(l) for l in rp.read_text().splitlines()]
    if not rec:
        na = {"status": "empty record"}
        return na, na
    fb = [(r["gid"], r["feedback"]) for r in rec if r.get("feedback")]
    runs = sorted({r.get("run") for r in rec if r.get("run") is not None})
    levels = sorted({r.get("level") for r in rec
                     if r.get("level") is not None})
    outcome = {
        "status": "ok",
        "n_steps": len(rec),
        "first_gid": rec[0]["gid"], "last_gid": rec[-1]["gid"],
        "runs": runs, "levels": levels,
        "final_run": rec[-1].get("run"),
        "final_level": rec[-1].get("level"),
        "n_feedback_events": len(fb),
        "last_feedback": ({"gid": fb[-1][0], "value": fb[-1][1]}
                          if fb else None),
        "feedback_polarity": load_world(unit).get("feedback_polarity")}

    phases = []
    for r in rec:
        key = (r.get("run"), r.get("level"))
        if phases and (phases[-1]["run"], phases[-1]["level"]) == key:
            phases[-1]["gid_end"] = r["gid"]
            phases[-1]["n_steps"] += 1
        else:
            phases.append({"run": key[0], "level": key[1],
                           "gid_start": r["gid"], "gid_end": r["gid"],
                           "n_steps": 1})
    covered = {_gidf(n["gid"]) for n in nodes
               if n["type"] in ("ACT", "STANCE")}
    spans = []
    for r in rec:
        if _gidf(r["gid"]) in covered:
            spans.append(None)             # breaker
            continue
        entry = {"gid_start": r["gid"], "gid_end": r["gid"],
                 "n_steps": 1,
                 "n_patch_steps": 1 if r.get("patches") else 0}
        if spans and spans[-1] is not None:
            spans[-1]["gid_end"] = r["gid"]
            spans[-1]["n_steps"] += 1
            spans[-1]["n_patch_steps"] += entry["n_patch_steps"]
        else:
            spans.append(entry)
    spans = [s for s in spans if s is not None]
    axis = {"status": "ok", "phases": phases,
            "uncovered_spans": spans,
            "n_uncovered_steps": sum(s["n_steps"] for s in spans)}
    return outcome, axis


def _cost_plane(unit: Path, nodes: list) -> dict:
    """Token-cost interface. Joins the usage.json ledger onto the graph
    so a reader does not need the gid -> api_turn -> ledger-turn hop by
    hand:

      grain api_call  every ACT/STANCE node gains tokens_cum — the
                      run's cumulative token spend at that step
                      (oeb.record.tokens) — and the block carries
                      total_tokens.
      other grains    no exact per-step reading exists; the block
                      says so and copies only unambiguous totals.
      no usage.json   undeclared — spend unknown, not zero.

    Zero-LLM; node stamping mutates in place before assembly."""
    up = unit / "usage.json"
    if not up.exists():
        return {"status": "undeclared (no usage.json)"}
    usage = json.loads(up.read_text())
    grain = usage.get("grain")
    if grain != "api_call":
        blk = {"status": f"unstamped (grain={grain}: no certified "
                         "per-step reading)", "grain": grain}
        if usage.get("out_tok") is not None:
            blk["out_tokens_total"] = usage["out_tok"]
        return blk
    from oeb.record.tokens import token_axis
    at = token_axis(unit)
    if at is None:
        return {"status": "unstamped (api_call ledger unusable: "
                          "missing/empty record or turns)",
                "grain": grain}
    n = 0
    for nd in nodes:
        if nd["type"] in ("ACT", "STANCE"):
            nd["tokens_cum"] = at(_gidf(nd["gid"]))
            n += 1
    return {"status": "ok", "grain": grain,
            "total_tokens": at.total, "n_stamped_nodes": n}


def _lineage(acts) -> dict:
    """act id -> lineage root launch id: a commit's root follows
    `builds_on` to the first launch; a
    check's is the launch it `reads`; a revise's is the launch it
    `answers`. None where the act points at no launch (a probe, a
    free-standing check). Identity is a chain of step-anchored ids the
    extractor chose from lists, never a wording."""
    commits = {_act_id(a): a for a in acts if a["act"] == "commit"}
    root: dict = {}

    def r(lid, seen=()):
        if lid in root:
            return root[lid]
        a = commits.get(lid)
        if a is None:
            return None
        bo = a.get("builds_on")
        val = lid if (not bo or bo in seen or bo not in commits) else r(bo, seen + (lid,))
        root[lid] = val
        return val

    out = {}
    for a in acts:
        aid = _act_id(a)
        if a["act"] == "commit":
            out[aid] = r(aid)
        elif a["act"] == "check":
            out[aid] = r(a["reads"]) if a.get("reads") else None
        elif a["act"] == "revise":
            out[aid] = r(a["answers"]) if a.get("answers") else None
        else:
            out[aid] = None
    return out


def _claim_ledger_block(p: dict, acts, edges) -> dict:
    """The stakes block: opened = launches (commits) carrying an
    expectation (mandated excluded); settled = those with a settles
    edge; topics per launch; identity = lineage."""
    opened = [a for a in acts if a["act"] == "commit" and a.get("expectation") and not a.get("mandated")]
    settled_ids = {e["dst"] for e in edges if e.get("kind") == "settles"}
    n_settled = sum(1 for a in opened if _act_id(a) in settled_ids)
    launches = [a for a in acts if a["act"] == "commit"]
    return {"status": "ok", "launches": len(launches),
            "opened": len(opened), "settled": n_settled,
            "open_at_end": len(opened) - n_settled,
            "shown_unread": sum(1 for a in acts if a["act"] == "check" and a.get("reads")
                                and not a.get("reading_receipt")),
            "topics": dict(collections.Counter(a.get("topic") or "undeclared" for a in launches)),
            "identity": p["stakes"]["identity"],
            "roster": p.get("roster") or {}}


def _settles_edges(acts, stats: dict) -> list:
    """reads: a check -> the launch (commit) whose result its OBS shows
    (the extractor's `reads`, a listed id); the edge carries the reading's
    VALENCE for that launch (favorable | adverse | mixed | none: the
    agent's own judgment of the result, claim or no claim).
    settles: the subset where the check carries a verdict against a
    stated expectation (outcome). Both record-only facts, no jury."""
    out = []
    ids = {_act_id(a) for a in acts}
    for a in acts:
        if a["act"] != "check":
            continue
        targets = a.get("reads_all") or ([a["reads"]] if a.get("reads") else [])
        for dst in targets:
            if dst not in ids or dst == _act_id(a):
                stats["n_unresolved_refs"] += 1
                continue
            valence = (a.get("valences") or {}).get(dst) or (a.get("valence") if dst == a.get("reads") else None)
            out.append({"kind": "reads", "src": _act_id(a), "dst": dst,
                        # the reading's own judgment of THIS launch's result,
                        # present whenever the launch was read
                        **({"valence": valence} if valence and a.get("reading_receipt") else {})})
            verdict = (a.get("outcomes") or {}).get(dst) or (a.get("outcome") if dst == a.get("reads") else None)
            if verdict and a.get("reading_receipt"):
                out.append({"kind": "settles", "src": _act_id(a), "dst": dst, "outcome": verdict,
                            **({"valence": valence} if valence else {})})
    return out


def _links_plane(acts, stances, stats: dict, existing_edges: list) -> tuple:
    """Backward links, all DERIVED from the extractor's step-anchored
    pointers: no judge, no window re-read.

    responds_to: the extractor's ONE response pointer: any act (probe,
    check, commit, revise) whose `answers` names a listed FAILED id,
    a launch the world went against (fail against its claim, or an
    adverse reading). Edge src = the answering act, dst = the check K
    that read the answered launch. Nothing is derived here (no window,
    no label equality, no lineage inference): the extractor names the
    failed launch, code checks it is on the list.

    explains: an attribution stance's `explains` names a failed
    launch; the edge lands on that launch's settling check.

    adjudicates (channel "launch"): the prediction a launch `expects`
    is decided by the check that settles the launch: pass -> confirms,
    fail -> contradicts. A jury edge that already decided the pair
    stands; a jury edge left undecided is overridden.
    """
    edges = []
    commits = {_act_id(a): a for a in acts if a["act"] == "commit"}
    deciding_check = {}                # launch id -> the check that read it (batch-aware)
    verdict_of = {}                    # launch id -> outcome against its claim (None: no claim)
    valence_of = {}                    # launch id -> the reading's valence
    for a in acts:
        if a["act"] != "check" or not a.get("reading_receipt"):
            continue
        for lid in (a.get("reads_all") or ([a["reads"]] if a.get("reads") else [])):
            v = (a.get("outcomes") or {}).get(lid) or (a.get("outcome") if lid == a.get("reads") else None)
            w = (a.get("valences") or {}).get(lid) or (a.get("valence") if lid == a.get("reads") else None)
            if (v or w) and lid not in deciding_check:
                deciding_check[lid] = a; verdict_of[lid] = v; valence_of[lid] = w
    n_resp = 0
    # responds_to = the extractor's `answers` pointer, on any act kind:
    # src = the answering act, dst = the check that read the
    # answered launch (or the reads-empty check named directly)
    checks_by_id = {_act_id(a): a for a in acts if a["act"] == "check"}
    for a in acts:
        tgt = a.get("answers")
        if not tgt:
            continue
        k = deciding_check.get(tgt) or checks_by_id.get(tgt)
        if k is None or _act_id(k) == _act_id(a):
            stats["n_unresolved_refs"] += 1
            continue
        edges.append({"kind": "responds_to", "src": _act_id(a), "dst": _act_id(k)})
        n_resp += 1
    n_exp = 0
    for c in stances:
        tgt = c.get("explains")
        if not tgt:
            continue
        k = deciding_check.get(tgt)
        if k is not None:
            edges.append({"kind": "explains", "src": c["id"], "dst": _act_id(k)})
            n_exp += 1
        else:
            stats["n_unresolved_refs"] += 1
    existing = {(e["src"], e["dst"]) for e in existing_edges
                if e.get("kind") in ("adjudicates", "grounds")
                and e.get("relation") in ("confirms", "contradicts")}
    undecided = {(e["src"], e["dst"]): e for e in existing_edges
                 if e.get("kind") == "adjudicates"
                 and e.get("relation") not in ("confirms", "contradicts")}
    stance_ids = {c["id"] for c in stances}
    n_claim = 0
    for lid, k in deciding_check.items():
        c = commits.get(lid)
        sid = c.get("expects") if c else None
        if not sid or sid not in stance_ids or verdict_of.get(lid) not in ("pass", "fail"):
            continue
        key = (_act_id(k), sid)
        if key in existing:
            continue
        rel = "confirms" if verdict_of[lid] == "pass" else "contradicts"
        if key in undecided:
            e = undecided[key]
            e["relation"] = rel; e["direction"] = "same"; e["channel"] = "launch (jury undecided)"
        else:
            edges.append({"kind": "adjudicates", "src": _act_id(k), "dst": sid,
                          "relation": rel, "direction": "same", "channel": "launch"})
        n_claim += 1
    return edges, {"status": "ok",
                   "responds_to": "ok (the extractor's `answers` pointer on any act kind, v9.1)",
                   "n_responds_to": n_resp, "explains": "ok",
                   "n_explains": n_exp, "n_wager_adjudications": n_claim}


def _artifact_plane(unit: Path, act_ids_by_gid: dict) -> tuple:
    """ARTIFACT nodes + touches edges from the recorded patch channel.
    Pure bookkeeping over machine-recorded facts: world.json DECLARES
    whether an authored channel exists (the scorer never guesses world shape), the
    adapter recorded every landed edit in record.jsonl's patches
    field, and this assembler joins them by gid. An edge means
    same-step co-occurrence of an act and a landed edit: a recorded
    fact, not an attribution judgment (a jury may later say which
    artifact EMBODIES which mechanism; that is opinion and belongs in
    overlays). Returns (nodes, edges, plane_status)."""
    wp = unit / "world.json"
    if not wp.exists():
        return [], [], {"status": "undeclared (no world.json)"}
    kind = ((json.loads(wp.read_text()).get("authored_channel") or {})
            .get("kind"))
    if not kind:
        return [], [], {"status": "undeclared (no authored_channel)"}
    if "patches" not in kind:
        return [], [], {"status": f"declared-absent ({kind})"}
    rp = unit / "record.jsonl"
    if not rp.exists():
        return [], [], {"status": "no record.jsonl", "channel": kind}
    touched: dict = {}                     # path -> [gid, ...]
    n_patch_steps = unparsed = 0
    for line in rp.read_text().splitlines():
        r = json.loads(line)
        if not r.get("patches"):
            continue
        n_patch_steps += 1
        for entry in r["patches"]:
            paths, bad = _patch_paths(entry)
            unparsed += bad
            for path in paths:
                touched.setdefault(path, []).append(r["gid"])
    nodes, edges, orphan_gids = [], [], set()
    for path in sorted(touched):
        gids = sorted(set(touched[path]), key=_gidf)
        nodes.append({"id": f"art:{path}", "type": "ARTIFACT",
                      "path": path, "touched": gids,
                      "n_touches": len(gids)})
        for gid in gids:
            # join on the shared gid parser, not string equality:
            # int 25 / "25" / "25.0" are one step, so a format drift
            # never silently orphans a touch (non-numeric raises)
            aids = act_ids_by_gid.get(_gidf(gid))
            if not aids:
                orphan_gids.add(str(gid))
                continue
            for aid in aids:
                edges.append({"kind": "touches", "src": aid,
                              "dst": f"art:{path}"})
    # channels_read declares partial coverage: the shell_redirect leg
    # of "patches+shell_redirect" is not read; said out loud here
    # instead of implied covered.
    plane = {"status": "ok", "channel": kind,
             "channels_read": ["patches"],
             "n_artifacts": len(nodes), "n_patch_steps": n_patch_steps,
             "n_unparsed_patch_entries": unparsed,
             # patch steps where the extractor found no act: the edit
             # is recorded on the artifact node's touched list but no
             # edge exists (no act to anchor it); counted, not hidden
             "n_actless_touch_gids": len(orphan_gids)}
    return nodes, edges, plane


def run(unit: Path) -> dict:
    ap = unit / "acts.json"
    if not ap.exists():
        return {"status": "no acts.json (run oeb.graph.extract)"}
    p = json.loads(ap.read_text())
    if p.get("status") != "ok":
        return {"status": f"withheld upstream: {p.get('status')}",
                "translator_status": p.get("status")}
    acts = sorted(p.get("acts") or [], key=lambda a: _gidf(a["gid"]))
    stances = p.get("stances")
    djp = unit / "deed_juries.json"
    dj = json.loads(djp.read_text()) if djp.exists() else {}

    # ---- identity is SEMANTIC: no lexical "normalized string = same
    # mechanism/line" fallback; no jury -> no identity -> consumers
    # withhold. Trivial exemption: with <2 distinct mechanisms no grouping
    # decision exists, so exact identity is not a semantic judgment.
    all_mechs = {_norm(a.get("mech")) for a in acts if a.get("mech")}
    trivial = len(all_mechs) < 2
    lj = dj.get("lines") or {}
    mids = lj.get("mech_ids") or {}
    groups = lj.get("groups") or {}
    line_of = {m: groups.get(mid, mid) for m, mid in mids.items()}
    lines_ran = "lines" in dj

    def line_key(mech):
        m = _norm(mech)
        if not m or not lines_ran:
            return None
        if line_of:
            return line_of.get(m)          # jury mapping only, no fallback
        return m if trivial else None

    mm = dj.get("mech_merge") or {}
    mm_ids = mm.get("mech_ids") or {}
    mm_groups = mm.get("groups") or {}
    group_of = {m: mm_groups.get(gid, gid) for m, gid in mm_ids.items()}
    mm_ran = "mech_merge" in dj
    cm_ran = "claim_merge" in dj
    cm_groups = (dj.get("claim_merge") or {}).get("groups") or {}

    # LINEAGE over the extractor's step-anchored launch pointers
    lineage = _lineage(acts)

    def mech_key(mech):
        m = _norm(mech)
        if not m or not mm_ran:
            return None
        if group_of:
            return group_of.get(m)         # jury mapping only, no fallback
        return m if trivial else None

    # the IDEA identity is the claim: the extractor's roster entry every
    # act rides, reconciled across passes by the mech-merge jury; the
    # launch id stays the EVENT identity and the lineage (builds_on
    # chains) stays a plane for lineage readers. Without a usable
    # mech-merge verdict, or on a record with no stance at all, the
    # lineage is the group.
    claim_ok = mm_ran and (bool(group_of) or trivial) and bool(stances)

    def act_group(a):
        aid = _act_id(a)
        if claim_ok:
            return mech_key(a.get("mech")) or lineage.get(aid)
        return lineage.get(aid) or (mech_key(a.get("mech")) if mm_ran else None)

    identity = {"lines": lines_ran and (bool(line_of) or trivial),
                "mech_identity": ("claim (claim roster; mech-merge jury across passes; lineage kept beside it)"
                                  if claim_ok else
                                  "lineage (builds_on / reads / answers over launch ids)"),
                "claim_merge": cm_ran}

    # jury-side references (pairing eids/cids: ids the jury layer
    # emits, contractually supposed to resolve to an act) that
    # resolve to nothing land in n_unresolved_refs (surfaced in the
    # graph's stats block) instead of vanishing. n_stances_actless
    # is DESCRIPTIVE, not an error signal: a stance extracted at a
    # step where the extractor correctly extracted no act (narration-
    # only step) is routine and must not drown the broken-reference
    # count.
    stats = {"n_unresolved_refs": 0, "n_ambiguous_pairing_ids": 0,
             "n_stances_actless": 0, "n_testable_unknown": 0}

    # jury-side gid REFERENCES parse with the ONE tolerant parser
    # (_gidf_or_none): a bad reference is counted, never raised
    nodes, edges = [], []
    lines: dict[str, list] = {}
    for a in acts:
        aid = _act_id(a)
        lk = line_key(a.get("mech"))
        nodes.append({"id": aid, "type": "ACT", "act": a["act"],
                      "gid": a["gid"], "mech": a.get("mech"),
                      "mandated": a.get("mandated", False),
                      "harness_event": a.get("harness_event", False),
                      "mech_group": act_group(a),
                      "line": lk,
                      "expectation": a.get("expectation"),
                      "outcome": a.get("outcome"),
                      "action_receipt": a.get("action_receipt"),
                      "obs_receipt": a.get("obs_receipt"),
                      # launch pointers
                      "expects": a.get("expects"), "expectation_gid": a.get("expectation_gid"),
                      "topic": a.get("topic"), "builds_on": a.get("builds_on"),
                      "reads": a.get("reads"), "reads_all": a.get("reads_all") or [],
                      "outcomes": a.get("outcomes") or {}, "reading_gid": a.get("reading_gid"),
                      "valence": a.get("valence"), "valences": a.get("valences") or {},
                      "reading_receipt": a.get("reading_receipt"),
                      "answers": a.get("answers"),
                      "votes": a.get("votes")})
        if a["act"] == "commit" and a.get("builds_on"):
            edges.append({"kind": "builds_on", "src": aid, "dst": a["builds_on"]})
        if a["act"] == "commit" and a.get("expects"):
            edges.append({"kind": "expects", "src": aid, "dst": a["expects"]})
        if lk is not None:
            lines.setdefault(lk, []).append(aid)
            edges.append({"kind": "engages", "src": aid, "dst": lk})
    # gid keys are the NUMERIC order value (_gidf): raw-string equality
    # would split "12" and "12.0"
    act_gids: dict[float, list] = {}
    for a in acts:
        act_gids.setdefault(_gidf(a["gid"]), []).append(_act_id(a))
    # a prediction a launch expects rides that launch's group
    _expected_by = {a["expects"]: act_group(a)
                    for a in acts if a["act"] == "commit" and a.get("expects")}
    # testability rides the settlement juries' verdict. A stance the
    # settlement map does not carry (a reliance, a retract, an
    # attribution) is stamped None, never False-as-if-judged, and
    # counted (stats) so consumers can tell "jury said no" from "no
    # jury data".
    _sett = dj.get("settlement") or {}

    def _testable(c):
        e = _sett.get(c["id"])
        if e is not None:
            return bool(e.get("testable"))
        stats["n_testable_unknown"] += 1
        return None
    # topic stamp: the bet_topic jury (research vs session by what
    # decides the stance's truth) lives in deed_juries.json; it is
    # copied onto STANCE nodes so graph-only consumers can filter
    # session-logistics stances ("the run ends in ~30 min") out of
    # research pools. None = no sidecar / stance unjudged, never
    # False-as-if-judged.
    _bt_topics = (dj.get("bet_topic") or {}).get("topics") or {}
    # reliance stances carry no extractor claim; the
    # reliance_wager jury places them on the mech-merge roster (canonical
    # group ids = the id space mech_key yields). wager_hearing on a
    # reliance node: assigned | null (no listed entry's shown events bear
    # on it) | undecided (no majority) | None (jury never ran). Never a
    # lexical fallback.
    _rw = dj.get("reliance_wager") or {}
    _rw_w = _rw.get("wagers") or {}
    _rw_h = _rw.get("hearing") or {}
    for c in stances or []:
        _is_rel = c.get("stance") == "reliance"
        nodes.append({"id": c["id"], "type": "STANCE",
                      "stance": c.get("stance", "belief"),
                      "stance_group": (cm_groups.get(c["id"], c["id"])
                                       if cm_ran else None),
                      # the mech-merge group of the stance's claim
                      # (extractor `wager_mech`; mech_merge ingests stance
                      # claims into the same id space); a prediction a
                      # launch expects rides that launch's group
                      "wager_group": ((mech_key(c.get("wager_mech")) if c.get("wager_mech") else None)
                                      or _expected_by.get(c["id"])
                                      or (_rw_w.get(c["id"]) if _is_rel and claim_ok else None)),
                      "wager_hearing": (_rw_h.get(c["id"]) if _is_rel else None),
                      "explains": c.get("explains"),
                      "gid": c["gid"], "text": c["text"],
                      "testable": _testable(c),
                      "topic": _bt_topics.get(c["id"]),
                      "labels": c.get("labels") or {},
                      "reasoning_receipt": c.get("reasoning_receipt"),
                      "votes": c.get("votes")})
        # states: the act(s) of the step whose reasoning made this
        # stance: narrative elements live ON the graph so consumers
        # never reach back into the record
        aids = act_gids.get(_gidf_or_none(c["gid"]), [])
        if not aids:
            # descriptive, NOT an error: a narration-only step
            # legitimately extracts a stance and no act
            stats["n_stances_actless"] += 1
        for aid in aids:
            edges.append({"kind": "states", "src": aid,
                          "dst": c["id"]})
    for lk, members in lines.items():
        nodes.append({"id": lk, "type": "LINE", "n_acts": len(members),
                      "grouped_by": ("lines jury" if line_of else
                                     "trivial (<2 mechanisms)")})

    # ---- supports edges (premise pairing; only past its gate) ------
    by_gid: dict[float, list] = {}
    for a in acts:
        by_gid.setdefault(_gidf(a["gid"]), []).append(a)

    def _evidence_act(eid_body):
        """d<gid>.<ty> jury id -> that exact evidence act. Split on
        the LAST dot (fractional gids are legal): a non-numeric tail
        is the type; a numeric tail means the dot belongs to the gid
        (untyped id). Untyped ids resolve only when unambiguous;
        multiple candidates are counted, never guessed by record
        order."""
        body = str(eid_body)
        gid_s, _, ty = body.rpartition(".")
        if not gid_s or _gidf_or_none(ty) is not None:
            gid_s, ty = body, ""
        cands = [a for a in by_gid.get(_gidf_or_none(gid_s), [])
                 if a["act"] == "probe" or (a["act"] == "check"
                                            and a.get("outcome"))]
        if ty:
            for a in cands:
                if a["act"][:2] == ty:
                    return a
        elif len(cands) == 1:
            return cands[0]
        elif len(cands) > 1:
            stats["n_ambiguous_pairing_ids"] += 1
            return None
        stats["n_unresolved_refs"] += 1
        return None

    pairing = dj.get("pairing") or {}
    # "tested before use" = the paired evidence arrives
    # before the premise's next commit or revise on the same
    # premise. Record-anchored, no window
    # constant (semantic mech group where the jury grouped it; a
    # groupless commit is its own premise, boundary = none).
    # Evidence before the commit trivially qualifies. The boundary is
    # the next commit OR revise because re-commits of a standing
    # premise are rare (continuation extracts nothing).
    _premise_acts = [a for a in acts if a["act"] in ("commit", "revise")]

    def _next_premise_gid(commit):
        g0 = _gidf(commit["gid"])
        grp = act_group(commit)
        later = [_gidf(a["gid"]) for a in _premise_acts
                 if grp is not None and _gidf(a["gid"]) > g0
                 and act_group(a) == grp]
        return min(later) if later else float("inf")

    if str(pairing.get("status", "")).startswith("ok"):
        for cid, eids in (pairing.get("tested_commits") or {}).items():
            commit = next((a for a in by_gid.get(_gidf_or_none(cid[1:]),
                                                 [])
                           if a["act"] == "commit"), None)
            if commit is None:
                stats["n_unresolved_refs"] += 1
            for eid in eids:
                ev = _evidence_act(eid[1:])
                if ev is not None and commit is not None:
                    edges.append({"kind": "supports",
                                  "src": _act_id(ev),
                                  "dst": _act_id(commit),
                                  "in_time": _gidf(ev["gid"])
                                  < _next_premise_gid(commit)})

    # ---- adjudicates edges (claim-act alignment; same gate rule) ----
    _gid_of = {n["id"]: _gidf(n["gid"]) for n in nodes
               if n.get("type") == "STANCE"}
    _act_gid = {n["id"]: _gidf(n["gid"]) for n in nodes
                if n.get("type") == "ACT"}
    cs = dj.get("claim_support") or {}
    if str(cs.get("status", "")).startswith("ok"):
        ids = {n["id"] for n in nodes}
        for kid, entries in (cs.get("support") or {}).items():
            for entry in entries:
                aid = entry.get("act")
                if aid in ids and kid in ids:
                    # a deciding check EARLIER than the stance's birth is
                    # the stance's basis (the agent read the result,
                    # then said so), not a test of it: it ships as a
                    # `grounds` edge, never as `adjudicates`.
                    _kind = "adjudicates"
                    _sg = _gid_of.get(kid)
                    if entry.get("channel") == "birth_evidence":
                        _kind = "grounds"
                    elif _sg is not None and aid in _act_gid \
                            and _act_gid[aid] < _sg:
                        _kind = "grounds"
                    e = {"kind": _kind, "src": aid,
                         "dst": kid, "relation": entry.get("relation")}
                    # direction rides the edge, so a graph-only consumer
                    # can tell "confirms the opposite" (a refutation)
                    # from a bare confirms
                    for k in ("direction", "polarity_law", "channel"):
                        if entry.get(k) is not None:
                            e[k] = entry[k]
                    edges.append(e)

    # ---- record/opinion split. The graph
    # FILE keeps only the journal blocks whose judgments BUILT
    # structure — nodes, edges, or a node attribute — plus the
    # extractor's parse diagnostics. Every evaluative jury (a
    # grade on how something was done, not a record of what
    # happened) lands in overlays/juries.json, an independently
    # versioned sidecar keyed by the same node ids, so a scoring
    # change rewrites the overlay and never the graph (oeb.measure.events
    # reads both).
    juries_core = {
        # stamps STANCE.testable (jury verdict per stance)
        "settlement": dj.get("settlement"),
        # judge-answer diagnostics (unparseable votes in the connecting juries)
        "parse_failures": dj.get("parse_failures"),
        # supports edges' source jury (summary counters only;
        # the edges themselves are first-class above)
        "pairing": {k: pairing.get(k) for k in
                    ("status", "tested_share", "n_commits")
                    if k in pairing},
        # adjudicates edges' source jury (summary counters)
        "claim_support": {k: cs.get(k) for k in
                          ("status", "backed_share",
                           "evidence_backed",
                           "asserts_verification",
                           "phantom_test_claims",
                           "n_testable_claims")
                          if k in cs},
        # LINE node grouping stamp
        "lines": ({"n_lines": lj.get("n_lines"),
                   "stamp": lj.get("stamp")} if lj else None)}
    juries_overlay = {
        "novelty_stream": dj.get("novelty_stream")}     # ONE novelty pool (claim grain): T3b + T5

    # ---- artifact plane: join the recorded patch
    # channel onto the act spine by gid: nodes for what the agent
    # built, touches edges for same-step act/edit co-occurrence.
    _by_gid: dict = {}
    for a in acts:
        _by_gid.setdefault(_gidf(a["gid"]), []).append(_act_id(a))
    art_nodes, art_edges, art_plane = _artifact_plane(unit, _by_gid)
    nodes.extend(art_nodes)
    edges.extend(art_edges)
    _stance_nodes = [n for n in nodes if n.get("type") == "STANCE"]
    edges.extend(_settles_edges(acts, stats))
    link_edges, links_plane = _links_plane(acts, _stance_nodes, stats, edges)
    edges.extend(link_edges)
    outcome, axis = _run_annotations(unit, nodes)
    cost = _cost_plane(unit, nodes)

    # graph-wide invariant, every assembly, every edge kind: an edge
    # whose end resolves to no node is counted here (like
    # n_unresolved_refs: surfaced, never silently shipped).
    _ids = {n["id"] for n in nodes}
    stats["n_unresolved_edges"] = sum(
        1 for e in edges if e["src"] not in _ids or e["dst"] not in _ids)
    # id uniqueness is a guarantee of the graph: act ids are created
    # unique (oeb.record.gids.act_id), and every assembly counts
    # violations so the guarantee is checked, not assumed
    stats["n_duplicate_ids"] = len(nodes) - len(_ids)

    g = {"version": "unified_graph_v2.11",
         "status": "ok",
         "sources": {"acts": p.get("version"),
                     "juries": dj.get("version"),
                     # instrument identity rides every hop: the
                     # extracting model is part of the measurement
                     "acts_model": p.get("model"),
                     "juries_model": dj.get("model")},
         # resolution counters: broken jury-side references
         # surfaced, never silently dropped (n_stances_actless
         # alone is descriptive)
         "stats": stats,
         "identity": identity,
         "n_acts": len(acts), "n_stances": len(stances or []),
         "n_lines": len(lines),
         "artifact_plane": art_plane,
         "links_plane": links_plane,
         "stakes": _claim_ledger_block(p, acts, edges),
         "outcome": outcome, "axis": axis, "cost": cost,
         "nodes": nodes, "edges": edges,
         "juries": juries_core}
    (unit / "unified_graph.json").write_text(json.dumps(g, indent=1))
    od = unit / "overlays"
    od.mkdir(exist_ok=True)
    (od / "juries.json").write_text(json.dumps(
        {"version": "jury_overlay_v1",
         "graph_version": g["version"],
         "sources": g["sources"],
         "blocks": juries_overlay}, indent=1))
    return g


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", nargs="+")
    a = ap.parse_args()
    for u in a.units:
        g = run(Path(u))
        print(f"{u}: {g.get('status')} nodes={len(g.get('nodes') or [])} "
              f"edges={len(g.get('edges') or [])}")


if __name__ == "__main__":
    main()
