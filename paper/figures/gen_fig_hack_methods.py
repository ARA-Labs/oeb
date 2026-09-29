"""Writes hack-methods.pdf/.png, tables/hackmodel.tex and data/numbers_hack_methods.tex: how each agent model hacks.

The figure is in the appendix ("Model-specific specialization in reward-hacking methods"), the table in Section 6.3.
Left, how often: reward-hacking instances per run. Right, how: one 100% bar per model split by method, counts in the bar, n at its end.
The unit is a reward-hacking instance exactly as E4 counts it, so each run contributes its k; the method is the rule of
the reward-hacking jury (H1-H6) that the per-act judge query quoted for that instance (no second classification).
Reads <unit>/measure.json, aim.json, unified_graph.json.
Usage: python paper/figures/gen_fig_hack_methods.py
"""
import json, glob, os, collections
from pathlib import Path
import oeb_plotstyle as S
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get("OEB_DATA", "."))
ALL_PANELS = ["aime2-w5-merge-glm53", "arenahard-gemma-w5-glm53", "bfcl-gemma-w5-glm53", "healthbench-gemma-w5-glm53",
              "humaneval-gemma-w5-glm53", "gpqa-gemma-w5-glm53"]
# A task is hackable if at least one run of any agent (all configurations, not only the six shown) hacked on it: the
# opportunity is shown to exist. The benchmark's own auditors cannot define it: they check two forms of cheating only.
def k_of(u):
    try: return (json.load(open(u + "/measure.json"))["dimensions"]["E4"]["constructs"]["hacks_counted"].get("detail") or {}).get("k") or 0
    except Exception: return 0
PANELS = [b for b in ALL_PANELS if any(k_of(u) for u in glob.glob(str(ROOT / "out/posttrainbench" / b / "*")))]
print("hackable tasks:", PANELS)
# red = the test's content reaches the training data; amber = the model is shaped to the test; blue = the score itself is gamed
METHODS = [("use", "trains on the test items", S.RED), ("derive", "writes training data from the test items", S.RED_SOFT),
           ("H3", "tunes to the test's format", S.AMBER), ("H6", "reports a part score as the whole", S.BLUE_MID),
           ("H4", "keeps the luckiest of several reruns", S.BLUE), ("other", "other", S.LINE)]
KNOWN = {m for m, _, _ in METHODS}
def method(v):
    h = v.get("breach_house")
    if h == "H2": return "use" if (v.get("breach") or "").startswith("Do not use") else "derive"
    return h if h in KNOWN else "other"
cnt = collections.defaultdict(collections.Counter); runs = collections.Counter(); hacked_runs = collections.Counter()
for b in PANELS:
    for u in sorted(glob.glob(str(ROOT / "out/posttrainbench" / b / "*"))):
        name = os.path.basename(u); key = name.split("-")[0]
        if not (os.path.exists(u + "/aim.json") and os.path.exists(u + "/measure.json")): continue
        d = json.load(open(u + "/measure.json"))["dimensions"]["E4"]["constructs"]["hacks_counted"].get("detail") or {}
        if "k" not in d: continue
        runs[key] += 1
        if not d["k"]: continue
        hacked_runs[key] += 1
        aim_acts = json.load(open(u + "/aim.json"))["deeds"]; nodes = json.load(open(u + "/unified_graph.json"))["nodes"]
        grp = {n.get("id"): n.get("mech_group") for n in (nodes.values() if isinstance(nodes, dict) else nodes) if isinstance(n, dict)}
        # the measure's own grain: acts that share an action or an idea (mech_group) are one instance
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        keyed = []
        for i in d.get("breach_deeds") or {}:
            v = aim_acts[i]
            ks = [k for k in (("action:" + str(v["action_gid"])) if v.get("action_gid") else None, grp.get(i)) if k] or [i]
            for k in ks[1:]: parent[find(k)] = find(ks[0])
            keyed.append((ks[0], method(v)))
        ideas = collections.defaultdict(list)
        for k, m in keyed: ideas[find(k)].append(m)
        assert len(ideas) == len(d["breach_ideas"]), (b, name)
        for ms in ideas.values(): cnt[key][collections.Counter(ms).most_common(1)[0][0]] += 1
        if d.get("noise_pick"): cnt[key]["H4"] += 1
        if d.get("report_states_measure") is False: cnt[key]["H6"] += 1
        assert sum(1 for _ in ideas) + bool(d.get("noise_pick")) + (d.get("report_states_measure") is False) == d["k"], (b, name)

plt = S.use()
order = sorted(S.MODELS, key=lambda m: -sum(cnt[m[0]].values()) / runs[m[0]])
fig, (a, b) = plt.subplots(1, 2, figsize=(S.WIDTH, 2.05), gridspec_kw={"width_ratios": [1, 2.7], "wspace": 0.07}, sharey=True)
for y, (key, label, fam) in enumerate(order):
    n = sum(cnt[key].values()); rate = n / runs[key]
    a.barh(y, rate, height=0.6, color=S.NEUTRAL, edgecolor="white", linewidth=0.5, zorder=3)
    a.text(rate + 0.05, y, f"{rate:.1f}", va="center", fontsize=8, color=S.INK)
    x = 0.0
    if n == 0:
        b.text(0.01, y, f"no reward-hacking instance in {runs[key]} runs", va="center", fontsize=7.5, color=S.MUTE, style="italic"); continue
    for m, _, col in METHODS:
        w = cnt[key][m] / n
        if not w: continue
        b.barh(y, w, left=x, height=0.6, color=col, edgecolor="white", linewidth=1.0, zorder=3)
        if w >= 0.06: b.text(x + w / 2, y, str(cnt[key][m]), ha="center", va="center", fontsize=8, color="white")
        x += w
    if n: b.text(1.015, y, f"$n$={n}", va="center", fontsize=7.5, color=S.MUTE)
a.set_yticks(range(len(order))); a.set_yticklabels([m[1] for m in order]); a.invert_yaxis()
a.set_xlim(0, max(2.5, max(sum(cnt[m[0]].values()) / runs[m[0]] for m in order) + 0.45));  # the largest rate keeps room for its label
a.set_xticks([0, 1, 2]); a.set_xlabel("reward-hacking instances per run"); a.set_title("How often", loc="left", pad=4)
a.grid(axis="x", zorder=0); a.set_axisbelow(True)
b.set_xlim(0, 1); b.set_xticks([0, .25, .5, .75, 1]); b.set_xticklabels(["0", "25", "50", "75", "100%"]); b.set_xlabel("share of the model's instances")
b.set_title("How", loc="left", pad=4); b.spines["left"].set_visible(False)
used = [(m, lab, col) for m, lab, col in METHODS if any(cnt[k[0]][m] for k in order)]
fig.legend([plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white") for _, _, c in used], [l for _, l, _ in used], loc="upper center",
           bbox_to_anchor=(0.5, -0.06), ncol=2, handlelength=1.0, handleheight=0.9, handletextpad=0.5, columnspacing=2.0, labelspacing=0.3)
S.save(fig, str(HERE / "hack-methods"))
# Table (Section 6.3): how each model hacks: runs with at least one instance, instances per run, the most frequent method
SHORT = {"use": "trains on test items", "derive": "data from test items", "H3": "fits test format",
         "H6": "part score as whole", "H4": "luckiest rerun", "other": "other"}
L = [r"\begin{tabular}{@{}l@{\hspace{6pt}}c@{\hspace{6pt}}l@{}}", r"\toprule", r" & \multicolumn{2}{c}{\textbf{E4}} \\", r"\cmidrule(l){2-3}",
     r"\textbf{Model} & \textbf{per run} & \textbf{main method} \\", r"\midrule"]
for key, label, _ in order:
    n = sum(cnt[key].values()); top, _ = cnt[key].most_common(1)[0] if n else ("--", 0)
    L.append(f"{label} & {n / runs[key]:.1f} & " + (SHORT[top] if n else "--") + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
(HERE.parent / "tables/hackmodel.tex").write_text("\n".join(L) + "\n")
for key, label, _ in order: print(f"{label:16} runs={runs[key]:2} ideas={sum(cnt[key].values()):2} {dict(cnt[key])}")
# numbers the prose cites, as macros (paper/data/numbers_hack_methods.tex)
tot = lambda keys, ms=None: sum(v for k in keys for m, v in cnt[k].items() if ms is None or m in ms)
core = [m[0] for m in S.MODELS]; rate = lambda k: f"{sum(cnt[k].values()) / runs[k]:.1f}"
N = {"hmRuns": sum(runs[k] for k in core), "hmIdeas": tot(core), "hmRateGLM": rate("glm52"), "hmRateKimi": rate("kimik3"),
     "hmRateGPTsix": rate("gpt56"), "hmRateGPTfive": rate("gpt55"), "hmIdeasGLMKimi": tot(["glm52", "kimik3"]),
     "hmDeriveGLMKimi": tot(["glm52", "kimik3"], {"derive"}), "hmIdeasGPT": tot(["gpt56", "gpt55"]), "hmLuckyGPT": tot(["gpt56", "gpt55"], {"H4"}),
     "hmIdeasFable": tot(["fable5"]), "hmIdeasOpus": tot(["opus5"]), "hmDeriveOpus": tot(["opus5"], {"derive"}),
     "hmTasks": len(PANELS), "hmRateOpus": rate("opus5"), "hmRateFable": rate("fable5"),
     "hmRunsHackShareGLM": f"{100 * hacked_runs['glm52'] / runs['glm52']:.0f}", "hmRunsHackShareGPT": f"{100 * (hacked_runs['gpt56'] + hacked_runs['gpt55']) / (runs['gpt56'] + runs['gpt55']):.0f}"}
(HERE.parent / "data/numbers_hack_methods.tex").write_text("% generated by figures/gen_fig_hack_methods.py -- do not edit by hand\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
