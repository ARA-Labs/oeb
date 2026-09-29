"""Connecting juries: the cross-item judgments the unified graph is built from.

This module is the driver: the jury order in run() and the CLI. The
jury implementations live in oeb/graph/juries/*. Every jury journals its
verdicts into <unit>/deed_juries.json, which oeb.graph.assemble
assembles into the graph.

PAIRING: pairs each commit with the prior evidence acts (probes +
adjudicated checks) that bear on its premise. Verbatim >=5-word
quotes from the evidence text, verified in code; a pair needs a
majority of the votes (2 of 3 by default).

LINES: groups the acts' mechanism summaries into lines of inquiry;
feeds the graph's LINE nodes.

CLAIM SUPPORT: for each claim, lists the acts whose content bears on
it, with the same quote-verified pairing machinery as premise pairing. The
jury also votes asserts_verification per claim: does the claim ASSERT
a test/verification already happened? A claim that asserts
verification but pairs with no act is a PHANTOM TEST CLAIM.

SETTLEMENT: the unified walk (one sequential reading of the act
timeline), the direction jury over the claim-support pairs, the
polarity rule and the birth-evidence jury decide which check settles
which stance, and in which direction.

Also run here: the mechanism and claim merge juries, reliance placement,
attribution kind, subjects, the topic juries and novelty.

Usage: .venv/bin/python -m oeb.graph.connect <unit_dir>...
         [--model claude-opus-5] [--votes 3]
Writes <unit>/deed_juries.json. ALL commits, claims and evidence are
judged: rosters over the char budget get more windows, never fewer
items.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import concurrent.futures as cf
import json
from oeb import llm
from oeb.jury import _norm, fallback_count, parse_failure_count, refusal_count
from oeb.record.gids import gidf as _gidf
from oeb.graph.juries._shared import _acts, _evidence, _roster_windows, _stances
from oeb.graph.juries.behavior import _claim_topic, _line_topic, _novelty
from oeb.graph.juries.episodes import _direction_docket, _unified_episode_walk
from oeb.graph.juries.merge import _claim_merge, _mech_merge, _reliance_claim
from oeb.graph.juries.pairing import _judge_pairs, claim_support
from oeb.graph.juries.settle import apply_settlements, birth_deciding_check_docket, polarity_law
from oeb.graph.juries.structure import _topics, _lines, _subjects
from oeb.graph.juries.standing import _attribution_kind


def run(unit: Path, model: str, votes: int) -> dict:
    pf0 = parse_failure_count()
    rf0 = refusal_count()
    fb0 = fallback_count()
    acts = _acts(unit)
    commits = [a for a in acts if a["act"] == "commit"]   # ALL of them

    evidence = _evidence(acts)
    # independent juries fan out on one pool: each submitted closure
    # reads only its stated inputs (acts / unit files), none reads
    # another jury's output or the assembled dict; every future is
    # consumed at a fixed point in the chain, so votes, tallies and
    # journal replay are identical to a sequential run. The llm layer is thread-safe
    # (llm._lock/_inflight) and carries the global fresh-call gate
    # (OEB_LLM_CONCURRENCY), so nested jury fan-outs never stack
    # into an unbounded transport burst.
    jpool = cf.ThreadPoolExecutor(max_workers=7)
    f_cm = jpool.submit(_claim_merge, _stances(unit), model, votes)
    f_lines = jpool.submit(_lines, acts, model, votes)
    f_claim_topic = jpool.submit(_claim_topic, unit, model, votes)
    # claim_support surveys EVERY adjudicable stance (belief, conjecture,
    # prediction); the settlement juries below decide which are testable
    f_cs = jpool.submit(claim_support, unit, acts, model, votes)
    # mech identity FIRST: the pairing boundary, the walk and the roster
    # juries need the merge groups; mech_merge depends only on acts +
    # stance claims.
    mm = _mech_merge(acts, model, votes,
                     extra=[c.get("wager_mech") for c in _stances(unit)
                            if c.get("wager_mech")])
    mm_groups = {m: (mm.get("groups") or {}).get(g, g)
                 for m, g in (mm.get("mech_ids") or {}).items()}
    # reliance -> claim placement: reads acts + mm only, so it runs
    # beside the pairing block; consumed below
    f_rw = jpool.submit(_reliance_claim, unit, acts, mm, model, votes)
    # verdict -> uptake is a post-graph stage (oeb.graph.juries.uptake
    # over the verdict table), not run here
    f_ak = jpool.submit(_attribution_kind, unit, acts, model, votes)
    f_sj = jpool.submit(_subjects, unit, acts, mm, model, votes)
    f_tp = jpool.submit(_topics, unit, acts, mm, model, votes)
    # the unified walk reads mm only, so it runs beside the pairing block
    f_uw = jpool.submit(_unified_episode_walk, unit, acts, mm, model, votes)
    # a commit's evidence boundary is the premise's next commit or
    # revise on the same premise (a re-COMMIT of a
    # standing premise is rare, since continuation extracts nothing). Commit, verify,
    # then work the premise again earns the credit; touching it again
    # with no reading in between does not. Semantic groups where the
    # jury grouped; a groupless commit is its own premise (boundary =
    # none).
    _premise_acts = [a for a in acts if a["act"] in ("commit", "revise")]

    def _grp(a):
        return mm_groups.get(_norm(a.get("mech") or "")) or None
    premise_bound = {}
    for c in commits:
        g0, grp = _gidf(c["gid"]), _grp(c)
        later = [_gidf(o["gid"]) for o in _premise_acts
                 if _gidf(o["gid"]) > g0 and grp is not None
                 and _grp(o) == grp]
        premise_bound[f"c{c['gid']}"] = min(later) if later \
            else float("inf")
    pairing: dict = {}
    if not commits or not evidence:
        pairing["status"] = (f"insufficient (commits {len(commits)}, "
                             f"evidence {len(evidence)})")
    else:
        main, pmeta = _judge_pairs(commits, evidence, model, votes,
                                   "main", premise_bound)
        # Same-premise identity credit: an adjudicated check in the
        # commit's own mech-merge group is bearing evidence by the
        # standing identity verdict, whatever the pairing jury's
        # per-chunk recall found; code consumes that verdict, the
        # pairing jury keeps every cross-mechanism call. The
        # precedence rule applies unchanged; harness feedback events
        # stay out (not agent tests).
        ev_gids = {e["id"]: _gidf(e["gid"]) for e in evidence}
        n_identity = 0
        for c in commits:
            cid, cgrp = f"c{c['gid']}", _grp(c)
            if cgrp is None:
                continue
            for a2 in acts:
                if a2["act"] != "check" or not a2.get("outcome") \
                        or a2.get("harness_event") \
                        or _grp(a2) != cgrp:
                    continue
                eid = f"d{a2['gid']}.ch"
                if eid not in ev_gids \
                        or ev_gids[eid] >= premise_bound[cid]:
                    continue
                got = main.setdefault(cid, [])
                if eid not in got:
                    got.append(eid)
                    n_identity += 1
        pairing.update(n_identity_pairs=n_identity)
        pairing.update(n_commits=len(commits), n_evidence=len(evidence),
                       roster_windows=len(_roster_windows(
                           evidence, lambda e: f"  {e['id']}: {e['text']}")),
                       tested_commits=main,
                       tested_share=round(len(main) / len(commits), 2),
                       # the window layout, and how many pairs the
                       # precedence rule voided
                       chunk_layout={"win": pmeta["win"],
                                     "chunks": pmeta["chunks"]},
                       n_precedence_voided=pmeta[
                           "n_precedence_voided"])
        pairing["status"] = "ok"
    # ONE sequential reading (the unified walk) produces the settlement
    # map: per stance, the checks that settled it and whether it is
    # testable (a stance filed aside abstains)
    cm_j = f_cm.result()
    settlements = f_uw.result()
    n_testability_abstained = sum(
        1 for e in settlements.values() if e.get("testability") == "aside")
    cs = f_cs.result()
    # direction jury over the survey's pairs: the survey finds, the
    # closed question judges; verdicts
    # merge into settlements before they are applied anywhere
    dd_meta = _direction_docket(unit, acts, cs, settlements,
                                model, votes)
    # polarity rule + birth-evidence jury: deterministic invariants
    # over the merged map (enforcing GROUND_LAW: a shown reading
    # cannot be beaten by its own birth evidence, a pre-birth act
    # cannot beat a standing belief), then the backward-looking
    # deciding-check channel for shown readings the forward-only channels
    # leave unbacked
    pol_meta = polarity_law(unit, settlements)
    bs_meta = birth_deciding_check_docket(unit, acts, settlements, model,
                                   votes)
    apply_settlements(cs, settlements)
    out = {"version": "deed_juries_v7.5_topics",
           "model": model, "votes": votes, "pairing": pairing,
           "claim_support": cs,
           "mech_merge": mm,
           "claim_merge": cm_j,
           "settlement": settlements,
           "n_testability_abstained": n_testability_abstained,
           "settlement_channel": "qa-docket (code-"
                                 "nominated candidates, closed "
                                 "per-candidate adjudication)",
           "direction_docket": dd_meta,
           "polarity_law": pol_meta,
           "birth_settler_docket": bs_meta,
           "lines": f_lines.result()}
    # T3b and T5 read ONE pool, novelty_stream at claim grain (below).
    # Order matters: bet_topic (the claim-topic jury) before novelty_stream (session filter),
    # line_topic after lines (already journaled above).
    out["bet_topic"] = f_claim_topic.result()
    # reliance -> claim: places every claim-less RELIANCE
    # stance on the mech-merge roster under CLAIM_LAW, so the verdict
    # table can reach it (assemble stamps STANCE.wager_group /
    # wager_hearing from this journal; measure reads reliance_tested)
    out["reliance_wager"] = f_rw.result()
    # attribution kind: void vs explain, per attribution pointing at a
    # launch (measure reads experiment_yield)
    out["attribution_kind"] = f_ak.result()
    # subjects: SUBJECT identity over the claim roster
    out["subjects"] = f_sj.result()
    # topics: the topic rule, asked once per idea over the roster
    out["topics"] = f_tp.result()
    jpool.shutdown()            # every future consumed; on an
                                # exception path concurrent.futures'
                                # exit handler joins the workers
    out["line_topic"] = _line_topic(unit, acts, out, model, votes)
    out["novelty_stream"] = _novelty(unit, acts, out, model, votes)
    # measurement gaps ship with the data (parse failures, refusals,
    # fallbacks)
    out["parse_failures"] = parse_failure_count() - pf0
    out["judge_refusals"] = refusal_count() - rf0
    out["judge_fallbacks"] = fallback_count() - fb0
    (unit / "deed_juries.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", nargs="+")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--votes", type=int, default=3)
    a = ap.parse_args()
    for u in a.units:
        llm.init(Path(u) / "calls.jsonl")
        r = run(Path(u), a.model, a.votes)
        p = r["pairing"]
        cs = r["claim_support"]
        print(f"{u}: pairing {p.get('status')} tested "
              f"{p.get('tested_share')} | "
              f"claims {cs.get('status')} backed "
              f"{cs.get('backed_share')} phantom "
              f"{len(cs.get('phantom_test_claims') or [])} | "
              f"lines {r['lines'].get('n_lines')}")


if __name__ == "__main__":
    main()
