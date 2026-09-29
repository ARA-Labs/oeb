"""Analogy-transfer jury: does a new hypothesis carry structure over from another line?

One command produces <unit>/analogy.json — the sidecar consumed by the
T5 construct t5a_hypothesis_source (oeb/measure/constructs.py) and the
descriptive origin_split.

Pipeline per unit (the graph must exist):
  1. mine: zero-LLM candidate generation — one judge query per pool
     birth (the novelty_stream roster), stock = propositions settled
     earlier on OTHER lines where both lines are known (timing enforced
     here, in code).
  2. judge: the six-way judgment (transfer_same_mechanism /
     transfer_same_pattern / same_proposition / enablement / unrelated
     / insufficient), EXAMPLE-FREE prompt, verdicts must cite a shown
     stock id and quote the candidate verbatim or the vote is rejected.
  3. annotate: write <unit>/analogy.json (per-birth verdicts,
     origin_split counts and claim_merge_leaks).

Usage:
  .venv/bin/python -m oeb.graph.analogy <unit_dir> [<unit_dir> ...]
      [--model claude-opus-5] [--votes 3]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from oeb import llm
from oeb.jury import QUOTE_JURY_MIN_CHARS, _json_block, _norm, contains, judge


# ---------------------------------------------------------------- mine

def gid_num(g):
    """A step number: record gids are strings, jury steps floats; both
    parse here (None only when neither does)."""
    try:
        return int(float(str(g)))
    except (TypeError, ValueError):
        return None


def mine(unit_dir: Path) -> list:
    """The candidate pairs: every pool birth with every finding settled
    before it on another line (or on an unknown line). [] without a graph."""
    gp = unit_dir / "unified_graph.json"
    if not gp.exists():
        return []
    g = json.loads(gp.read_text())
    nodes = g.get("nodes", [])
    settlement = g.get("juries", {}).get("settlement", {}) or {}

    stances = [n for n in nodes if n.get("type") == "STANCE"]
    acts = {n["id"]: n for n in nodes if n.get("type") == "ACT"}

    # line of a stance: states-edge first (the act whose reasoning
    # extracted it carries the line), gid match as the fallback. Without a
    # line the other-lines-only stock filter cannot apply, and same-line
    # continuations would reach the jury as "transfer" candidates.
    staters: dict = {}
    for e in g.get("edges") or []:
        if e.get("kind") == "states":
            staters.setdefault(e.get("dst"), []).append(e.get("src"))

    def line_of(n):
        if n.get("line"):
            return n["line"]
        for aid in staters.get(n["id"], []):
            a = acts.get(aid)
            if a and a.get("line"):
                return a["line"]
        for a in acts.values():
            if a.get("gid") == n.get("gid") and a.get("line"):
                return a["line"]
        return None

    # ---- settled findings, with settle gid (the verdict definition the
    # measure layer uses):
    # a finding is settled when a check delivered a
    # confirming verdict on it — the stance channel (settlement jury:
    # confirms deciding checks) or the launch channel (the check that read a
    # launch pass against its claim, or favorable in the agent's own words).
    # Keyed by the IDEA (claim group) where the node names one, so a
    # birth's own idea never sits in its stock.
    edges = g.get("edges") or []
    settled = {}  # key -> {settle_gid, line, text, group}

    def _put(key, sg, line, text, group):
        cur = settled.get(key)
        if cur is None or sg < cur["settle_gid"]:
            settled[key] = {"settle_gid": sg, "line": line, "text": text, "group": group}

    for sid, s in settlement.items():
        confirm = [x for x in s.get("settlers") or [] if x.get("relation") == "confirms"]
        if not confirm:
            continue
        sgids = [gid_num(acts.get(x.get("act", ""), {}).get("gid")) for x in confirm]
        sgids = [x for x in sgids if x is not None]
        node = next((n for n in stances if n["id"] == sid), None)
        if node is None or not sgids:
            continue
        grp = node.get("wager_group") or node.get("mech_group")
        _put(grp or sid, min(sgids), line_of(node), node.get("text", ""), grp)
    for e in edges:
        if e.get("kind") not in ("settles", "reads"):
            continue
        L = acts.get(e.get("dst")); k = acts.get(e.get("src"))
        if not L or not k or L.get("act") != "commit" or not L.get("mech_group"):
            continue
        ok = (e.get("outcome") == "pass") if e.get("kind") == "settles" else (e.get("valence") == "favorable")
        sg = gid_num(k.get("gid"))
        if ok and sg is not None:
            _put(L["mech_group"], sg, L.get("line"), L.get("mech") or "", L["mech_group"])

    # ---- candidates: the jury pool IS novelty_stream's roster — one
    # birth per idea at its first claim (a launch, or a conjecture /
    # prediction), research only, session claims dropped, with at least
    # 3 findings settled before it and a far/near verdict already on
    # file. One pool, two judgments per claim (transfer jury here,
    # far/near there) -> the three-way split transferred / grown / novel
    # that T5 reads.
    try:
        _dj = json.loads((unit_dir / "deed_juries.json").read_text())
    except Exception:
        _dj = {}
    _verdicts = (_dj.get("novelty_stream") or {}).get("verdicts") or []
    by_id = {n["id"]: n for n in stances}
    # a birth is a STANCE (its node) or a CLAIM (a mech-merge group id:
    # the idea at its first claim) — a claim birth is rendered from its
    # earliest act: the claim text, that act's line and the verdict's step
    first_act = {}
    for a in sorted(acts.values(), key=lambda a: gid_num(a.get("gid")) or 0):
        if a.get("mech_group") and a["mech_group"] not in first_act:
            first_act[a["mech_group"]] = a
    births = []
    for v in _verdicts:
        n = by_id.get(v.get("id"))
        if n is None and v.get("id") in first_act:
            a = first_act[v["id"]]
            n = {"id": v["id"], "gid": v.get("step", a.get("gid")), "stance": "wager",
                 "text": a.get("mech") or "", "line": a.get("line"), "wager_group": v["id"]}
        if n is None:
            continue
        births.append(n)
    pairs = []
    for b in births:
        bg = gid_num(b.get("gid"))
        # a prose-turn birth (no act at its gid, no states edge) has no
        # line: the other-lines-only cut cannot be code-enforced, so the
        # jury runs against the FULL earlier stock (skipping it instead
        # would silently drop a sizeable share of pool claims: selection
        # bias); the prompt's rule that extending a claim is not
        # transfer covers same-line reuse
        ln = line_of(b)
        if bg is None:
            continue
        for sid, s in settled.items():
            # stock = every finding settled earlier; the other-lines-only
            # cut (same-line reuse is continuation, not analogy) is
            # enforced where BOTH lines are known
            own = b.get("wager_group") or b.get("mech_group")
            if own and s.get("group") == own:
                continue                       # its own idea is never its source
            if s["settle_gid"] < bg and (ln is None or not s["line"] or s["line"] != ln):
                pairs.append({
                    "birth": {"id": b["id"], "line": ln, "text": b.get("text", "")},
                    "source": {"id": sid, "settle_gid": s["settle_gid"], "text": s["text"]},
                })
    return pairs


# ------------------------------------------------------------- jury

VERDICTS = ("transfer_same_mechanism", "transfer_same_pattern",
            "same_proposition", "enablement", "unrelated", "insufficient")

PROMPT = """You are auditing one research record. The record is organized into \
research LINES (independent problem threads).

(A) CANDIDATE: a hypothesis stated for the first time on its line.
(B) STOCK: propositions settled (confirmed by shown evidence) on OTHER lines, \
each settled strictly before the candidate was stated, each with an id.

Question: does the CANDIDATE carry structure over from one STOCK item? \
"Carrying structure over" means: the causal relation or method-shape asserted \
by the candidate corresponds piece-by-piece to the one established by the \
stock item, with only the objects replaced — and the candidate says \
something NEW about a DIFFERENT kind of object or problem than the stock \
item's own subject. A candidate that merely expects the stock item's own \
claim to hold again for another instance of the same kind is extending that \
claim's scope, not carrying structure anywhere — that is enablement. It is \
NOT enough that both concern the same tools, the same environment, or the \
same overall goal.

Answer with exactly one verdict:
- transfer_same_mechanism: the candidate asserts the SAME established causal \
mechanism now operating in a different place.
- transfer_same_pattern: the mechanisms differ, but the candidate imports the \
method-shape by which a stock item was established.
- same_proposition: the candidate restates a stock item — same claim, and the \
same shown evidence would settle both.
- enablement: a stock item is a prerequisite, tool, or input the candidate \
builds on, or the candidate extends the stock item's own claim to further \
instances of its kind. Building on results, and expecting them to keep \
holding, is not carrying structure over.
- unrelated: none of the above holds for any stock item.
- insufficient: the material shown does not allow a decision. Answer \
insufficient only when the texts genuinely support no other reading — an \
insufficient hearing abstains and is counted, never defaulted.

For any verdict except unrelated/insufficient you MUST give the stock item id \
and quote, verbatim, the candidate wording that carries the correspondence.

CANDIDATE (line {line}):
{birth}

STOCK (settled earlier, other lines):
{stock}

Answer with ONE json object only:
{{"verdict": "<one of {verdicts}>", "stock_id": "<id or null>", \
"quote": "<verbatim candidate words or null>", "why": "<one sentence>"}}"""



def build_queries(unit_dir: Path):
    """One judge query per pool birth: the birth and its stock (every
    finding settled before it on another line), oldest first."""
    pairs = mine(unit_dir)
    births = {}
    for p in pairs:
        births.setdefault(p["birth"]["id"], p["birth"])
    queries = []
    for bid, b in births.items():
        stock = sorted((p["source"] for p in pairs if p["birth"]["id"] == bid),
                       key=lambda s: s["settle_gid"])
        stock = list({s["id"]: s for s in stock}.values())
        queries.append({"birth": b, "stock": stock})
    return queries


def run_query(h, model, votes):
    stock_txt = "\n".join(f"[{s['id']}] {s['text']}" for s in h["stock"])
    prompt = PROMPT.format(line=h["birth"]["line"],
                           birth=h["birth"]["text"], stock=stock_txt,
                           verdicts="|".join(VERDICTS))
    ids = {s["id"] for s in h["stock"]}
    cast = []
    for i in range(votes):
        v = _json_block(judge(model, prompt, salt=f"v{i}"), arr=False)
        verdict = v.get("verdict")
        if verdict not in VERDICTS:
            cast.append({"verdict": "VOID_parse"})
            continue
        if verdict not in ("unrelated", "insufficient"):
            ok_id = v.get("stock_id") in ids
            q = str(v.get("quote") or "")
            ok_q = len(_norm(q)) >= QUOTE_JURY_MIN_CHARS and contains(q, h["birth"]["text"])
            if not (ok_id and ok_q):
                cast.append({"verdict": "VOID_receipt", "raw": v})
                continue
        cast.append(v)
    good = [c["verdict"] for c in cast if not c["verdict"].startswith("VOID")]
    final = (Counter(good).most_common(1)[0][0] if good and
             Counter(good).most_common(1)[0][1] * 2 > len(cast) else None)
    if final in (None, "insufficient"):
        final = "undecided"      # abstention and no-majority leave the pool, never a pole
    return {"birth": h["birth"]["id"], "line": h["birth"]["line"],
            "text": h["birth"]["text"], "votes": cast,
            "final": final}


# ------------------------------------------------------------ annotate

TRANSFER = {"transfer_same_mechanism", "transfer_same_pattern"}


def settle_state(graph, sid):
    s = (graph.get("juries", {}).get("settlement", {}) or {}).get(sid)
    if not s or not s.get("settlers"):
        return "unsettled"
    rels = {x["relation"] for x in s["settlers"]}
    if rels == {"confirms"}:
        return "confirmed"
    if rels == {"contradicts"}:
        return "contradicted"
    return "mixed"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("units", nargs="+")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--votes", type=int, default=3)
    a = ap.parse_args()
    for u in a.units:
        unit = Path(u)
        llm.init(unit / "calls.jsonl")
        queries = build_queries(unit)
        print(f"{unit.name}: {len(queries)} judge queries, votes={a.votes}, "
              f"model={a.model}")
        results = [run_query(h, a.model, a.votes) for h in queries]
        for r in results:
            det = ""
            for c in r["votes"]:
                if c.get("stock_id"):
                    det = f" <- {c['stock_id']}"
            print(f"  {r['birth']:5s} {str(r['final']):26s} "
                  f"{r['text']}{det}")
        print("  tally:", dict(Counter(r["final"] for r in results)))
        graph = json.loads((unit / "unified_graph.json").read_text())
        split = {"transfer_born": Counter(), "other_born": Counter()}
        leaks = []
        for h in results:
            state = settle_state(graph, h["birth"])
            bucket = "transfer_born" if h["final"] in TRANSFER else "other_born"
            split[bucket][state] += 1
            if h["final"] == "same_proposition":
                cite = next((c for c in h["votes"] if c.get("stock_id")), {})
                leaks.append({"birth": h["birth"],
                              "duplicate_of": cite.get("stock_id"),
                              "why": cite.get("why", "")})
        payload = {
            "instrument": "oeb.graph.analogy (example-free prompt)",
            "scope": "descriptive annotation, counts only",
            "hearings": results,
            "origin_split": {k: dict(v) for k, v in split.items()},
            "claim_merge_leaks": leaks,
            # roster identity stamp: the pool this sidecar was built from
            "pool": "novelty_stream frozen bets roster",
            "n_pool_bets": len({h["birth"] for h in results}),
        }
        (unit / "analogy.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"  wrote {unit / 'analogy.json'} "
              f"({len(leaks)} merge leaks)")


if __name__ == "__main__":
    main()
