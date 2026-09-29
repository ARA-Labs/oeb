"""Section 6.1-6.2 numbers, from the graphs of the runs the paper reports (see graph_readings.py).

Writes data/numbers_findings.tex, tables/verdicts.tex and data/findings.json (read by figures/gen_fig_winners.py).

6.1  How often an experiment ends in a verdict (E2's experiment-yield construct, per benchmark), and, where the
     benchmark logs every experiment's real result, how often the agent's verdict was right.
6.2  Each act-plane reading against the outcome: rank the runs of a task by the reading and by the outcome, pool the
     within-task ranks over tasks, Spearman's rho; the same per benchmark;
     how often the best run of a task has more than the worst; and, for a model run twice on the same task, how often
     the higher-scoring run has more. Ground truth: share of experiments and of real gain by the try's position within
     its idea, overall and within the same third of the run.
Usage: python paper/data/make_findings_numbers.py   (OEB_DATA = the extracted dataset)
"""
import json, glob, os, math, collections, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import graph_readings as G
TAB = HERE.parent / "tables"
CAP = {'c1_approx_area': 180, 'c1_pipeline_fmax': 240, 'c1_recipe_pareto': 180, 'c2_branch_pred': 240, 'c2_cache_ctrl': 360,
       'c2_chacha_tput': 300, 'c2_divider_al': 240, 'c3_mem_sched': 360, 'c3_prefetcher': 180, 'c3_replacement': 180}
BENCHES = ["PostTrainBench", "Chip-Bench", "nanoGPT speedrun"]
M = {}; OUT = {}

# ---- 6.1 verdict yield per benchmark (E2 experiment_yield: k = experiments read to a verdict never later retracted)
YB = {"PostTrainBench": [G.O + f"out/posttrainbench/{b}/*/measure.json" for b in ("aime2-w5-merge-glm53", "arenahard-gemma-w5-glm53",
      "bfcl-gemma-w5-glm53", "healthbench-gemma-w5-glm53", "humaneval-gemma-w5-glm53", "gpqa-gemma-w5-glm53")],
      "Chip-Bench": [G.O + "out/chipbench/hc47-glm/*/measure.json"], "nanoGPT speedrun": [G.O + "out/speedrun/w5-grok46/*/measure.json"]}
yl = {}; scale = {}
for b, gs in YB.items():
    k = n = runs = steps = win = read = 0
    for g in gs:
        for f in glob.glob(g):
            ey = json.load(open(f))["dimensions"]["E2"]["constructs"].get("experiment_yield") or {}; k += ey.get("k") or 0; n += ey.get("n") or 0
            u = os.path.dirname(f); runs += 1; steps += sum(1 for _ in open(u + "/record.jsonl"))
            # of the research experiments the agent read to a verdict, how many it read as an improvement
            gr, acts = G.launches(u); val = {}
            for x in gr["nodes"]:
                if x["type"] == "ACT" and x.get("valences"):
                    for kk, v in x["valences"].items(): val.setdefault(kk, v)
            for x in acts:
                if x["act"] == "commit" and x.get("topic") == "research" and val.get(x["id"]) in ("favorable", "adverse"):
                    read += 1; win += val[x["id"]] == "favorable"
    yl[b] = (k, n); scale[b] = dict(runs=runs, steps=steps, win=win, read=read)
M.update(yieldPtb=f"{100 * yl['PostTrainBench'][0] / yl['PostTrainBench'][1]:.0f}", yieldChip=f"{100 * yl['Chip-Bench'][0] / yl['Chip-Bench'][1]:.0f}",
         yieldSpeed=f"{100 * yl['nanoGPT speedrun'][0] / yl['nanoGPT speedrun'][1]:.0f}", nExpPtb=f"\\num{{{yl['PostTrainBench'][1]}}}")

# ---- runs, readings, truth
rs = G.units()
for r in rs: r.update(G.readings(r["unit"]))
for r in rs:
    if r["bench"] == "Chip-Bench":
        m = G.J(G.O + "data/chipbench/runs/" + r["name"] + "/meta.json") or {}
        if m.get("duration_seconds"): r["time_used"] = m["duration_seconds"] / 60 / CAP[r["task"]]; r["hit_budget"] = bool(m.get("hit_budget"))
TR = {r["name"]: G.truth(r) for r in rs if r["bench"] != "PostTrainBench"}

# verdict vs truth, and what the agent did next: build on the experiment or not
vt = collections.defaultdict(lambda: [0, 0]); bt = collections.defaultdict(lambda: [0, 0, 0])
for r in rs:
    tr = TR.get(r["name"])
    if not tr: continue
    g = G.J(r["unit"] + "/unified_graph.json"); val = {}
    for n in g["nodes"]:
        if n["type"] == "ACT" and n.get("valences"):
            for k, v in n["valences"].items(): val.setdefault(k, v)
    _, acts = G.launches(r["unit"]); rl = [n for n in acts if n["act"] == "commit" and n.get("topic") == "research"]; res = {n["id"] for n in rl}
    kids = collections.Counter(n["builds_on"] for n in rl if n.get("builds_on"))   # research launches that build on each launch
    for k, t in tr.items():
        if k not in res: continue  # research experiments only, as in the winners figure
        v = {"favorable": "improved", "adverse": "worse"}.get(val.get(k), "no verdict")
        c = vt[(r["bench"], v)]; c[0] += 1; c[1] += t["improve"]
        b = bt[(r["bench"], v)]; b[0] += kids[k] > 0          # the agent built on it
        b[1] += kids[k]; b[2] += kids[k] if not t["improve"] else 0   # follow-ups, and follow-ups on a result that was no real gain
truth_runs = {b: sum(1 for r in rs if r["bench"] == b and TR.get(r["name"])) for b in ("Chip-Bench", "nanoGPT speedrun")}
# one table: by the agent's own verdict, the share of research experiments that truly beat the run's best so far and
# the share that a later research launch builds on (the caption gives the experiment counts)
pct = lambda a, b: f"{100 * a / max(b, 1):.0f}\\%"
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r" & \multicolumn{2}{c}{\textbf{Truly improved}} & \multicolumn{2}{c}{\textbf{Built on by a later experiment}} \\",
     r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"\textbf{The agent read the result as} & \textbf{Speedrun} & \textbf{Chip-Bench} & \textbf{Speedrun} & \textbf{Chip-Bench} \\", r"\midrule"]
for v, lab in (("improved", "an improvement"), ("worse", "no improvement"), ("no verdict", "(no verdict stated)")):
    s_, c_ = ("nanoGPT speedrun", v), ("Chip-Bench", v)
    L.append(f"{lab} & {pct(vt[s_][1], vt[s_][0])} & {pct(vt[c_][1], vt[c_][0])} & {pct(bt[s_][0], vt[s_][0])} & {pct(bt[c_][0], vt[c_][0])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(TAB / "verdicts.tex").write_text("\n".join(L) + "\n")
for (b, v), (nb, nk, _) in bt.items():
    key = {"nanoGPT speedrun": "Speed", "Chip-Bench": "Chip"}[b] + {"improved": "Good", "worse": "Bad", "no verdict": "Silent"}[v]
    M["bt" + key + "Pct"] = f"{100 * nb / max(vt[(b, v)][0], 1):.0f}"
sk = sum(bt[("nanoGPT speedrun", v)][1] for v in ("improved", "worse", "no verdict")); sf = bt[("nanoGPT speedrun", "improved")][2]
M.update(btSpeedKids=sk, btSpeedFalseWinKids=sf, btSpeedFalseWinKidsPct=f"{100 * sf / sk:.0f}")
for (b, v), (n, w) in vt.items():
    key = {"nanoGPT speedrun": "Speed", "Chip-Bench": "Chip"}[b] + {"improved": "Good", "worse": "Bad", "no verdict": "Silent"}[v]
    M["vt" + key + "N"] = n; M["vt" + key + "Win"] = w; M["vt" + key + "Pct"] = f"{100 * w / max(n, 1):.0f}"
M.update(nTruthChip=truth_runs["Chip-Bench"], nTruthSpeed=truth_runs["nanoGPT speedrun"])
M.update(vtChipN=sum(n for (b, _), (n, _w) in vt.items() if b == "Chip-Bench"), vtSpeedN=sum(n for (b, _), (n, _w) in vt.items() if b == "nanoGPT speedrun"))

# ---- 6.2 association of readings with outcome
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); o = [0.0] * len(v); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
        for k in range(i, j + 1): o[s[k]] = (i + j) / 2 / max(len(v) - 1, 1)
        i = j + 1
    return o
def corr(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n; sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa * sb else float("nan")
for r in rs:
    if r.get("research"): r["new_ideas_2nd_half_per_step"] = r["new_ideas_2nd_half"] / r["steps"]
tasks = sorted({r["task"] for r in rs})
def assoc(f):
    X, Y = [], []; byb = collections.defaultdict(lambda: ([], [])); up = nt = 0; slope = []
    for t in tasks:
        us = [r for r in rs if r["task"] == t and isinstance(r.get(f), (int, float))]
        if len(us) < 3: continue
        a = rank([r[f] for r in us]); b = rank([r["score"] for r in us]); X += a; Y += b
        byb[us[0]["bench"]][0].extend(a); byb[us[0]["bench"]][1].extend(b)
        # best / worst run of the task; runs tied on the outcome are averaged
        hi_s, lo_s = max(r["score"] for r in us), min(r["score"] for r in us)
        best = st.mean(r[f] for r in us if r["score"] == hi_s); worst = st.mean(r[f] for r in us if r["score"] == lo_s)
        nt += 1; up += best > worst
        med = st.median(r[f] for r in us) or 1
        slope.append(dict(task=t, bench=us[0]["bench"], worst=worst, best=best, median=med))
    pu = pn = 0
    for r in rs:
        if r["second"] and isinstance(r.get(f), (int, float)):
            o = next((x for x in rs if x["task"] == r["task"] and x["name"] == r["model"]), None)
            if o and isinstance(o.get(f), (int, float)) and abs(o["score"] - r["score"]) >= 0.02 and o[f] != r[f]:
                hi, lo = (o, r) if o["score"] > r["score"] else (r, o); pn += 1; pu += hi[f] > lo[f]
    return dict(rho=corr(X, Y), n=len(X), bench={b: corr(*byb[b]) for b in byb}, up=up, nt=nt, pu=pu, pn=pn, slope=slope)
F = ["new_ideas_2nd_half", "new_ideas_2nd_half_per_step", "ideas", "exps_per_idea", "one_shot_ideas", "research", "steps", "time_used"]
A = {f: assoc(f) for f in F}
for f, a in A.items(): print(f"{f:28s} rho {a['rho']:+.2f} (n={a['n']}) " + " ".join(f"{b[:4]} {v:+.2f}" for b, v in a["bench"].items()) + f" | best>worst {a['up']}/{a['nt']} | pairs {a['pu']}/{a['pn']}")
a = A["new_ideas_2nd_half"]
M.update(rhoLate=f"{a['rho']:+.2f}", rhoLatePtb=f"{a['bench']['PostTrainBench']:+.2f}", rhoLateChip=f"{a['bench']['Chip-Bench']:+.2f}",
         rhoLateSpeed=f"{a['bench']['nanoGPT speedrun']:+.2f}", lateUp=a["up"], lateTasks=a["nt"], latePairUp=a["pu"], latePairN=a["pn"],
         rhoLatePerStep=f"{A['new_ideas_2nd_half_per_step']['rho']:+.2f}", rhoIdeas=f"{A['ideas']['rho']:+.2f}", rhoExpPerIdea=f"{A['exps_per_idea']['rho']:+.2f}",
         rhoStepsAll=f"{A['steps']['rho']:+.2f}", nFindRuns=a["n"], nFindTasks=a["nt"])
tu = [r["time_used"] for r in rs if "time_used" in r]
M.update(chipTimeUsedMedian=f"{100 * st.median(tu):.0f}", chipTimeUsedMax=f"{100 * max(tu):.0f}", chipHitBudget=sum(r.get("hit_budget", False) for r in rs), nChipTimeRuns=len(tu))

# ---- ground truth: gain by try position within an idea (Chip-Bench; the speedrun rarely gets past a second try)
pos = collections.defaultdict(lambda: [0, 0, 0.0]); third = collections.defaultdict(lambda: [0, 0.0])
for r in rs:
    tr = TR.get(r["name"])
    if not tr: continue
    _, acts = G.launches(r["unit"]); res = [n for n in acts if n["act"] == "commit" and n.get("topic") == "research"]
    steps = r["steps"]; seen = collections.Counter()
    for n in res:
        k = n.get("mech_group") or n["id"]; seen[k] += 1; i = seen[k]
        p = "first" if i == 1 else "second or third" if i <= 3 else "fourth or later"
        t = tr.get(n["id"])
        if not t: continue
        c = pos[(r["bench"], p)]; c[0] += 1; c[1] += t["improve"]; c[2] += t["gain"]
        q = min(2, int(3 * float(n["gid"]) / steps)); d = third[(r["bench"], q, i <= 3)]; d[0] += 1; d[1] += t["gain"]
POS = ["first", "second or third", "fourth or later"]
OUT["positions"] = {b: {p: dict(zip(("exps", "wins", "gain"), pos[(b, p)])) for p in POS} for b in ("Chip-Bench", "nanoGPT speedrun")}
ch = OUT["positions"]["Chip-Bench"]; te = sum(v["exps"] for v in ch.values()); tg = sum(v["gain"] for v in ch.values())
sp = OUT["positions"]["nanoGPT speedrun"]; se = sum(v["exps"] for v in sp.values()); sg = sum(v["gain"] for v in sp.values())
M.update(lateTryExpPct=f"{100 * ch['fourth or later']['exps'] / te:.0f}", lateTryGainPct=f"{100 * ch['fourth or later']['gain'] / tg:.0f}",
         lateTryWinPct=f"{100 * ch['fourth or later']['wins'] / ch['fourth or later']['exps']:.0f}", earlyTryWinPct=f"{100 * (ch['first']['wins'] + ch['second or third']['wins']) / (ch['first']['exps'] + ch['second or third']['exps']):.0f}",
         speedFirstGainPct=f"{100 * sp['first']['gain'] / sg:.0f}", speedFirstExpPct=f"{100 * sp['first']['exps'] / se:.0f}")
thirds_ok = sum(1 for q in range(3) if third[("Chip-Bench", q, True)][0] and third[("Chip-Bench", q, False)][0]
                and third[("Chip-Bench", q, True)][1] / third[("Chip-Bench", q, True)][0] > third[("Chip-Bench", q, False)][1] / third[("Chip-Bench", q, False)][0])
M.update(thirdsEarlyWins=thirds_ok)
OUT["assoc"] = {f: {k: v for k, v in a.items()} for f, a in A.items()}
(HERE / "findings.json").write_text(json.dumps(OUT, indent=1))
(HERE / "numbers_findings.tex").write_text("% generated by paper/data/make_findings_numbers.py -- do not edit by hand\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(json.dumps(M, indent=0)); print(json.dumps(OUT["positions"], indent=0))
