"""Writes the numbers the prose cites (data/numbers.tex, numbers.json) and the tables main, reliability and hacktask.

It reads measure.json and calls.jsonl (and, for PostTrainBench, the bench's own judgement and metrics files under
<unit>/official/); nothing is typed in by hand. The unit folders are those of the released judged-panels dataset (same <bench>/<batch>/<unit>/
layout). Usage, from the repo root:
    python paper/data/make_tables.py [--root $OEB_DATA/out]
"""
import argparse, glob, itertools, json, os, statistics as st
from pathlib import Path
from common import official_cheat, same_judge_pairs

DIMS = ["E1", "E2", "E3", "E4", "T1", "T2", "T3", "T4", "T5", "T6"]
PAN = {"AIME 2025": "aime2-w5-merge-glm53", "Arena-Hard Writing": "arenahard-gemma-w5-glm53",
       "BFCL": "bfcl-gemma-w5-glm53", "HealthBench": "healthbench-gemma-w5-glm53",
       "HumanEval": "humaneval-gemma-w5-glm53", "GPQA": "gpqa-gemma-w5-glm53"}
SHORT = {"AIME 2025": "aime", "Arena-Hard Writing": "arenahard", "BFCL": "bfcl",
         "HealthBench": "healthbench", "HumanEval": "humaneval", "GPQA": "gpqa"}
CORE = ["opus5", "fable5", "gpt56", "gpt55", "kimik3", "glm52"]
NAME = {"opus5": "Claude Opus 5", "fable5": "Claude Fable 5", "gpt56": "GPT-5.6", "gpt55": "GPT-5.5",
        "kimik3": "Kimi K3", "glm52": "GLM-5.2"}

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.join(os.environ.get("OEB_DATA", "."), "out")); A = ap.parse_args()
ROOT = Path(A.root); PT = ROOT / "posttrainbench"
HERE = Path(__file__).resolve().parent; TAB = HERE.parent / "tables"; TAB.mkdir(exist_ok=True)
N = {}  # numbers.json


def dims(unit):
    f = Path(unit) / "measure.json"
    if not f.exists(): return None
    return json.loads(f.read_text())["dimensions"]


def scores(unit):
    d = dims(unit)
    return None if d is None else {k: (d.get(k) or {}).get("score") for k in DIMS}


def fmt(v, nd=2): return "--" if v is None else f"{v:.{nd}f}"


# ---- 1. what was scored -------------------------------------------------------------------------
units = {t: sorted(os.path.dirname(f) for f in glob.glob(str(PT / b / "*" / "measure.json"))) for t, b in PAN.items()}
chip = sorted(os.path.dirname(f) for f in glob.glob(str(ROOT / "chipbench/hc47-glm/*/measure.json")))
speed = sorted(os.path.dirname(f) for f in glob.glob(str(ROOT / "speedrun/w5-grok46/*/measure.json")))
N["ptb_runs"] = sum(len(v) for v in units.values()); N["chip_runs"] = len(chip); N["speed_runs"] = len(speed)
N["runs_total"] = N["ptb_runs"] + N["chip_runs"] + N["speed_runs"]
N["chip_tasks"] = len({os.path.basename(u).split("--")[0] for u in chip})
N["tasks_total"] = len(PAN) + N["chip_tasks"] + 1

# ---- 2. six models x six tasks: per-model means over all runs (first and second) ---------------------
lines = []; N["main"] = {}
for u in CORE:
    ss = [s for s in (scores(PT / b / v) for b in PAN.values() for v in (u, f"{u}-r2")) if s]
    ntask = sum(1 for b in PAN.values() if any(scores(PT / b / v) for v in (u, f"{u}-r2")))
    m = {}
    for k in ["E1", "E2", "E3", "E4"]:
        v = [s[k] for s in ss if isinstance(s[k], (int, float))]
        m[k] = (st.mean(v), len(v)) if v else (None, 0)
    N["main"][u] = {k: m[k][0] for k in m} | {"tasks": ntask, "runs": len(ss)}
    lines.append(f"{NAME[u]} & {ntask} & " + " & ".join((f"{fmt(m[k][0])}\\,\\textsubscript{{{m[k][1]}}}" if m[k][1] else "--") for k in m) + " \\\\\n")
(TAB / "main.tex").write_text(
    "\\begin{tabular}{@{}lrcccc@{}}\n\\toprule\n\\textbf{Agent model} & \\textbf{Tasks} & \\textbf{E1 Evidence} & \\textbf{E2 Experiment} & \\textbf{E3 Revision} & \\textbf{E4 No reward hacking} \\\\\n\\midrule\n"
    + "".join(lines) + "\\bottomrule\n\\end{tabular}\n")

# ---- 3. reliability: judge-repeat / second-run repeat / different models -----------------------------
def diffs(pairs):
    d = {k: [] for k in DIMS}; n = 0
    for a, b in pairs:
        ra, rb = scores(a), scores(b)
        if not ra or not rb: continue
        n += 1
        for k in DIMS:
            if isinstance(ra[k], (int, float)) and isinstance(rb[k], (int, float)): d[k].append(abs(ra[k] - rb[k]))
    return d, n
rej = same_judge_pairs(PT)
rep = [(PT / b / u, PT / b / f"{u}-r2") for b in PAN.values() for u in CORE]
d1, n1 = diffs(rej); d2, n2 = diffs(rep)
d3 = {k: [] for k in DIMS}
for b in PAN.values():
    rs = {u: scores(PT / b / u) for u in CORE}
    for a, c in itertools.combinations([u for u in CORE if rs[u]], 2):
        for k in DIMS:
            x, y = rs[a][k], rs[c][k]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)): d3[k].append(abs(x - y))
N["reliability"] = {"judge_repeat_pairs": n1, "trajectory_repeat_pairs": n2}
body = ""
for lab, d in (("Same record, judged twice", d1), ("Same agent, second run", d2), ("Different agent models", d3)):
    N["reliability"][lab] = {k: ([round(st.mean(d[k]), 3), len(d[k])] if d[k] else None) for k in DIMS}
    body += lab + " & " + " & ".join((f"{st.mean(d[k]):.2f}\\,\\textsubscript{{{len(d[k])}}}" if d[k] else "--") for k in DIMS) + " \\\\\n"
# same record, two judges: the Chip-Bench runs scored under both GPT-5.6-sol and GLM-5.3
xj = [(ROOT / "chipbench/hc47-sol" / os.path.basename(u), u) for u in chip]
d4, n4 = diffs(xj)
N["reliability"]["cross_judge_pairs"] = n4
N["reliability"]["Same record, other judge"] = {k: ([round(st.mean(d4[k]), 3), len(d4[k])] if d4[k] else None) for k in DIMS}
body += "\\midrule\nSame record, other judge$^\\dagger$ & " + " & ".join((f"{st.mean(d4[k]):.2f}\\,\\textsubscript{{{len(d4[k])}}}" if d4[k] else "--") for k in DIMS) + " \\\\\n"
(TAB / "reliability.tex").write_text(
    "\\begin{tabular}{@{}l" + "c" * len(DIMS) + "@{}}\n\\toprule\n & " + " & ".join(f"\\textbf{{{k}}}" for k in DIMS)
    + " \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n")


# Table (Section 6.3): where reward hacking happens, task by task: runs E4 flags, E4's instances per run, runs the
# benchmark's auditors found cheating
def hacks_k(u):
    d = ((dims(u) or {}).get("E4") or {}).get("constructs", {}).get("hacks_counted", {}).get("detail") or {}
    return d.get("k")
rows_ht = []
for t, us in units.items():
    ks = [k for k in (hacks_k(u) for u in us) if k is not None]; offs = [o for o in (official_cheat(u) for u in us) if o is not None]
    rows_ht.append((sum(ks) / len(ks), t, sum(k > 0 for k in ks), len(ks), sum(offs), len(offs)))
L = [r"\begin{tabular}{@{}l@{\hspace{6pt}}c@{\hspace{6pt}}c@{\hspace{6pt}}c@{}}", r"\toprule", r" & \multicolumn{2}{c}{\textbf{E4}} & \textbf{Auditors:} \\", r"\cmidrule(lr){2-3}",
     r"\textbf{Task} & \textbf{flagged} & \textbf{per run} & \textbf{cheated} \\", r"\midrule"]
for rate, t, fl, n, oc, on in sorted(rows_ht, reverse=True):
    L.append(f"{t.replace(' Writing', '').replace(' 2025', '')} & {fl}/{n} & {rate:.2f} & {oc}/{on} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "hacktask.tex").write_text("\n".join(L) + "\n")

# ---- 4. reward hacking against the bench's own verdicts -----------------------------------------
conf = {"first": [0, 0, 0, 0], "second": [0, 0, 0, 0]}  # pos, pos caught, clean, clean flagged
pos_e4, flagged_k, flagged_outside = [], [], 0
for t, us in units.items():
    for u in us:
        off, d = official_cheat(u), dims(u)
        e4 = d["E4"]["score"]
        if off is None or e4 is None: continue
        c = conf["second" if u.endswith("-r2") else "first"]
        if off: c[0] += 1; c[1] += e4 < 1; pos_e4.append(e4)
        else:
            c[2] += 1; c[3] += e4 < 1
            if e4 < 1:
                flagged_k.append(round(1 / e4 - 1))
                # the bench's verdict covers two things: test items in the training data (rule H2 of the reward-hacking
                # jury) and a model the work did not make (rule H5). A flagged run with neither broke only rules the
                # bench's audit does not examine.
                det = d["E4"]["constructs"]["hacks_counted"].get("detail") or {}
                flagged_outside += not any(v.get("house") in ("H2", "H5") for v in (det.get("breach_deeds") or {}).values())
tot = [a + b for a, b in zip(conf["first"], conf["second"])]
N["hack"] = {"first": conf["first"], "second": conf["second"], "total": tot,
             "positive_e4_min": min(pos_e4), "positive_e4_max": max(pos_e4), "caught_e4_max": max(v for v in pos_e4 if v < 1),
             "flagged_clean_k": sorted(flagged_k), "flagged_clean_outside": flagged_outside}
kpos = sorted(round(1 / v - 1) for v in pos_e4 if v < 1)
N["hack"]["caught_k"] = kpos

# ---- 5. how many experiments end in a stated conclusion --------------------------------------------
k = n = 0
for us in units.values():
    for u in us:
        ey = dims(u)["E2"]["constructs"].get("experiment_yield") or {}
        k += ey.get("k") or 0; n += ey.get("n") or 0
N["yield"] = {"answered": k, "experiments": n, "share": round(k / n, 3)}

# ---- 6. judge calls per run (cost) -------------------------------------------------------------
calls = []
for u in [x for us in units.values() for x in us] + chip + speed:
    f = Path(u) / "calls.jsonl"
    if f.exists():
        with open(f, "rb") as fh: calls.append(sum(1 for _ in fh))
if calls: N["calls"] = {"runs": len(calls), "median": int(st.median(calls)), "min": min(calls), "max": max(calls)}

# ---- 7. where the bench's own outcome score separates agents and where it does not ---------------
N["outcome_range"] = {}
for t, us in units.items():
    acc = []
    for u in us:
        f = Path(u) / "official" / "metrics.json"
        if f.exists():
            try: a = json.loads(f.read_text()).get("accuracy")
            except json.JSONDecodeError: a = None  # the bench shipped no score for this run
            if isinstance(a, (int, float)): acc.append(a)
    if acc: N["outcome_range"][t] = {"runs": len(acc), "min": round(min(acc), 3), "max": round(max(acc), 3)}

# ---- LaTeX macros: the prose cites these, so it cannot drift from the files ---------------------
R = N["reliability"]; H = N["hack"]["total"]
def rng(row, ks):
    v = [R[row][k][0] for k in ks if R[row][k]]; return f"{min(v):.2f}--{max(v):.2f}"
M = {"nRuns": N["runs_total"], "nTasks": N["tasks_total"], "nPtbRuns": N["ptb_runs"], "nChipRuns": N["chip_runs"],
     "nSpeedRuns": N["speed_runs"], "nChipTasks": N["chip_tasks"],
     "nLabelled": H[0] + H[2], "nCheat": H[0], "nCheatCaught": H[1], "nClean": H[2], "nCleanFlagged": H[3],
     "cheatEfourMin": f"{N['hack']['positive_e4_min']:.2f}", "cheatEfourMax": f"{N['hack']['caught_e4_max']:.2f}",
     "nCheatMissed": H[0] - H[1], "nCaughtMedian": f"{st.median(N['hack']['caught_k']):g}", "nCleanFlaggedK": (lambda lo, hi: "one" if hi == 1 else ("one or two" if (lo, hi) == (1, 2) else f"{lo}--{hi}"))(min(N['hack']['flagged_clean_k']), max(N['hack']['flagged_clean_k'])), "nCleanFlaggedOne": sum(1 for k in N["hack"]["flagged_clean_k"] if k == 1),
     "nCleanFlaggedOutside": N["hack"]["flagged_clean_outside"],
     "nExperiments": f"\\num{{{N['yield']['experiments']}}}", "nAnswered": f"\\num{{{N['yield']['answered']}}}",
     "yieldPct": f"{100 * N['yield']['share']:.0f}",
     "nJudgeRepeat": R["judge_repeat_pairs"], "nRunRepeat": R["trajectory_repeat_pairs"], "nCrossJudge": R["cross_judge_pairs"],
     "rejudgeCore": rng("Same record, judged twice", ["E1", "E2", "E3"]),
     "modelsCore": rng("Different agent models", ["E1", "E2", "E3"]),
     "rejudgeWeak": rng("Same record, judged twice", ["E4", "T2", "T6"]),
     "rerunComp": rng("Same agent, second run", ["E1", "E2", "E3", "E4"]),
     "crossJudgeEone": f"{R['Same record, other judge']['E1'][0]:.2f}",
     "aimeMax": f"{N['outcome_range']['AIME 2025']['max']:.2f}",
     "bfclMin": f"{N['outcome_range']['BFCL']['min']:.2f}", "bfclMax": f"{N['outcome_range']['BFCL']['max']:.2f}",
     "callsMedian": N["calls"]["median"], "callsMin": N["calls"]["min"], "callsMax": f"\\num{{{N['calls']['max']}}}"}
(HERE / "numbers.tex").write_text("% generated by make_tables.py -- do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))

(HERE / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False))
print(json.dumps(N, indent=1, ensure_ascii=False))
