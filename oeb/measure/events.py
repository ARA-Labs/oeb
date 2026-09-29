"""Load a unit's graph, jury overlays and journal as indexed tables for the constructs.

It loads the unit's unified graph, the evaluative overlays (the jury
blocks, plus uptake.json, aim.json and coverage.json) and the
connecting juries' journal, and exposes them as plain indexed tables.
It makes no judgment: every label a construct reads was written
upstream by an instrument. It also defines the scoring pool and the
token account the constructs share.

Nothing here consults world.json: the world's shape reaches the
constructs only through what the record and the instruments show
(an ACT that exists, an edge that exists, a label that was judged).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from oeb.record.gids import gidf as _gid_num


def gid(x) -> float:
    return float(_gid_num(x))


@dataclass
class Unit:
    path: Path
    graph: dict
    overlay: dict            # overlays/juries.json 'blocks' (evaluative)
    journal: dict            # deed_juries.json (connecting juries' journal)
    acts: list = field(default_factory=list)
    stances: list = field(default_factory=list)
    edges: dict = field(default_factory=lambda: defaultdict(list))
    node: dict = field(default_factory=dict)

    def block(self, name: str) -> dict:
        """An instrument block by name: overlay first, journal second.
        {} when the instrument never ran — constructs reading it then
        see no opportunities and stay undefined."""
        b = self.overlay.get(name)
        if b is None:
            b = self.journal.get(name)
        return b if isinstance(b, dict) else {}

    @property
    def status_ok(self) -> bool:
        return self.graph.get("status") == "ok"


def load(unit: str | Path) -> Unit:
    p = Path(unit)
    graph = json.loads((p / "unified_graph.json").read_text())
    ov_p = p / "overlays" / "juries.json"
    overlay = json.loads(ov_p.read_text()).get("blocks", {}) if ov_p.exists() else {}
    # the verdict_uptake jury is a post-graph stage writing
    # uptake.json; it supersedes any verdict_uptake block in the journal
    up_p = p / "uptake.json"
    if up_p.exists():
        overlay = dict(overlay)
        overlay["verdict_uptake"] = json.loads(up_p.read_text())
    # the reward-hacking jury (E4) is a post-graph stage writing
    # aim.json — what each agent-credited act is aimed at
    aim_p = p / "aim.json"
    if aim_p.exists():
        overlay = dict(overlay)
        overlay["launch_aim"] = json.loads(aim_p.read_text())
    cov_p = p / "coverage.json"
    if cov_p.exists():
        overlay = dict(overlay)
        overlay["check_coverage"] = json.loads(cov_p.read_text())
    dj_p = p / "deed_juries.json"
    journal = json.loads(dj_p.read_text()) if dj_p.exists() else {}
    u = Unit(path=p, graph=graph, overlay=overlay, journal=journal)
    for n in graph.get("nodes", []):
        u.node[n["id"]] = n
        t = n.get("type")
        if t == "ACT":
            u.acts.append(n)
        elif t == "STANCE":
            u.stances.append(n)
    u.acts.sort(key=lambda n: gid(n["gid"]))
    u.stances.sort(key=lambda n: gid(n["gid"]))
    for e in graph.get("edges", []):
        u.edges[e["kind"]].append(e)
    return u


# ---------------------------------------------------------------------
# The scoring pool. Most constructs draw their items from these two
# views; the members that ask how carefully the agent works read
# engineering too (acts_all_topics, or all stances).
#   * acts:    agent-credited acts — not harness-mandated, not harness
#              events, not on a line the line-topic jury called
#              housekeeping (when that jury ran), not a logistics launch
#              or an act on one, not on an engineering idea.
#   * stances: research stances — topic == research when the bet_topic
#              jury stamped topics; every stance when it never ran
#              (undeclared is not 'off-topic'); not on an engineering idea.
# ---------------------------------------------------------------------

def _housekeeping_lines(u: Unit) -> set:
    lt = u.block("line_topic")
    return set(lt.get("housekeeping_ids") or [])


def _logistics_launches(u: Unit) -> set:
    """Launch ids the extractor declared LOGISTICS (topic is a
    per-launch choice). An act leaves the pool when it IS such a launch,
    reads one, or answers one."""
    return {a["id"] for a in u.acts if a.get("act") == "commit" and a.get("topic") == "logistics"}


def idea_research(u: Unit) -> dict:
    """idea (mech_group / wager_group) -> True when the idea is RESEARCH.
    The topic rule: topic is a property of the idea, inherited by every act
    and stance on it. The topics jury (one label per canonical claim
    over the roster) decides where it gave a label; elsewhere an idea
    is research when any of its launches is stamped research, else when
    any of its stances is, else — untagged — research (undeclared is not
    off-topic)."""
    judged = (u.block("topics") or {})
    judged = judged.get("topics") or {} if judged.get("status") == "ok" else {}
    launch_tags: dict = {}
    for a in u.acts:
        g = a.get("mech_group")
        if g and a.get("act") == "commit" and a.get("topic"):
            launch_tags.setdefault(g, set()).add(a["topic"])
    stance_tags: dict = {}
    for s in u.stances:
        g = s.get("wager_group") or s.get("stance_group")
        if g and s.get("topic"):
            stance_tags.setdefault(g, set()).add(s["topic"])
    out = {}
    for g in set(launch_tags) | set(stance_tags) | set(judged):
        if g in judged:
            out[g] = judged[g] == "research"
        elif g in launch_tags:
            out[g] = "research" in launch_tags[g]
        else:
            out[g] = "research" in stance_tags[g]
    return out


def scoped_acts(u: Unit) -> list:
    hk = _housekeeping_lines(u)
    ll = _logistics_launches(u)
    ir = idea_research(u)
    return [a for a in u.acts
            if not a.get("mandated") and not a.get("harness_event")
            and a.get("line") not in hk
            and a["id"] not in ll and a.get("answers") not in ll
            and not (set(a.get("reads_all") or ([a["reads"]] if a.get("reads") else [])) & ll)
            and ir.get(a.get("mech_group"), True)]


def scoped_stances(u: Unit) -> list:
    stamped = any(s.get("topic") is not None for s in u.stances)
    ir = idea_research(u)
    out = list(u.stances) if not stamped else [s for s in u.stances if s.get("topic") == "research"]
    return [s for s in out if ir.get(s.get("wager_group") or s.get("stance_group"), True)]


def acts_all_topics(u: Unit) -> list:
    """scoped_acts without the topic filters: the process members that ask
    how carefully the agent works — was an assertion backed, did a
    check decide something — read engineering acts too; mandated and harness acts still leave (the agent did not
    choose them)."""
    return [a for a in u.acts if not a.get("mandated") and not a.get("harness_event")]


def stance_group(s: dict) -> str:
    """The proposition a stance belongs to: its claim (the idea)
    when it named one, else its claim-merge group, else itself."""
    return s.get("wager_group") or s.get("stance_group") or s["id"]


# ---------------------------------------------------------------------
# Token account: usage.json per-api-turn sides + record api_turn joins.
# Think side = thinking + visible text (prose is deliberation); do side = action
# payloads the agent wrote. Tool RESULTS are world-imposed and enter
# neither side. Exact split when the ledger has it; else the turn's
# output tokens are split by char proportion (the one estimated split).
# ---------------------------------------------------------------------

def token_account(u: Unit):
    """(turn_sides, gid_to_turn) or (None, reason). turn_sides:
    api_turn(str) -> (think_tokens, do_tokens)."""
    up = u.path / "usage.json"
    if not up.exists():
        return None, "no usage.json"
    usage = json.loads(up.read_text())
    if usage.get("grain") != "api_call":
        return None, f"usage grain {usage.get('grain')!r} too coarse"
    turns = usage.get("turns") or {}
    rp = u.path / "record.jsonl"
    if not rp.exists():
        return None, "no record.jsonl"
    g2t = {}
    for ln in rp.read_text().splitlines():
        if ln.strip():
            s = json.loads(ln)
            if s.get("api_turn") is not None and s.get("gid") is not None:
                g2t[gid(s["gid"])] = str(s["api_turn"])
    if not g2t:
        return None, "record carries no api_turn stamps"
    sides = {}
    for k, r in turns.items():
        if r.get("action_tokens") is not None and r.get("thinking_tokens") is not None:
            sides[str(k)] = (float((r.get("thinking_tokens") or 0) + (r.get("text_tokens") or 0)),
                             float(r.get("action_tokens") or 0))
        else:
            tc = (r.get("think_chars") or 0) + (r.get("prose_chars") or 0)
            dc = r.get("tool_chars") or 0
            out = r.get("out") or 0
            tot = tc + dc
            sides[str(k)] = (out * tc / tot, out * dc / tot) if tot and out else (0.0, 0.0)
    return (sides, g2t), "ok"


def launch_claims(u: Unit) -> list:
    """Every claim made by launching: a launch (its stated expectation,
    or the claim it rides — a claim made by doing). Each with its deciding
    checks (a settles edge, or a reads edge carrying a valence) if any;
    the key is the act's identity group (mech_group)."""
    # ONE definition of a deciding check (the launch table's): the check
    # that read the launch — a settles edge (verdict against the claim) or
    # a reads edge carrying the agent's valence
    settled_by = {}
    for kind in ("settles", "reads"):
        for e in u.edges.get(kind, []):
            if kind == "reads" and not e.get("valence"):
                continue
            settled_by.setdefault(e["dst"], []).append(u.node[e["src"]])
    # a claim the bet_topic jury labelled SESSION (the run's own clock —
    # "850 optimizer steps should take about an hour") claims nothing
    # about the studied system, so it alone cannot make a launch a
    # claim. A launch that rides a claim stays a claim whatever sentence
    # sits on it; the T6 reader applies the same rule.
    _session = {k for k, v in ((u.block("bet_topic") or {}).get("topics") or {}).items()
                if v == "session"}
    out = []
    for a in scoped_acts(u):
        # a launch claims by doing — its stated expectation, or the
        # claim it rides — settled by the check that read it
        stated = bool(a.get("expectation")) and a.get("expects") not in _session
        if a.get("act") == "commit" and (stated or a.get("mech")):
            out.append({"id": a["id"], "gid": gid(a["gid"]), "key": a.get("mech_group"),
                        "stated": stated,
                        "deciding_checks": sorted(settled_by.get(a["id"], []), key=lambda x: gid(x["gid"]))})
    # One execution: an execution is identified by its showing. Two
    # launches of the same claim whose result the same check read first
    # are one execution (a dispatch step and the run it dispatched; a
    # relaunch that produced the one result); the earlier claim stands
    # for it. Launches of different claims read together (a batch) stay
    # distinct.
    # Several things launched in one step are one execution too (the
    # extractor's slots: a<gid>.co, a<gid>.co2 ...): the step's first
    # claim stands for it and carries every slot's deciding checks.
    by_step: dict = {}
    for s in sorted(out, key=lambda x: (x["gid"], x["id"])):
        if s["gid"] in by_step:
            prev = by_step[s["gid"]]
            prev["deciding_checks"] = sorted({c["id"]: c for c in prev["deciding_checks"] + s["deciding_checks"]}.values(),
                                      key=lambda x: gid(x["gid"]))
            continue
        by_step[s["gid"]] = s
    seen: dict = {}
    merged = []
    for s in sorted(by_step.values(), key=lambda x: x["gid"]):
        first = s["deciding_checks"][0]["id"] if s["deciding_checks"] else None
        key = (s["key"], first) if first else None
        if key and key in seen:
            continue
        if key:
            seen[key] = s["id"]
        merged.append(s)
    return merged
