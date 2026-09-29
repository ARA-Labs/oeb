"""Writes winners.pdf/.png (Section 6.2): what separates the best run of a task from the worst.

Left: one row per task, the number of new ideas first tested in the second
half of the run by the task's worst run (hollow) and best run (filled), log scale; rows grouped by benchmark. Right:
Chip-Bench runs with a logged real result per experiment: the share of experiments and the share of real gain by the
try's position within its idea. Reads data/findings.json (make_findings_numbers.py); counts are plotted unchanged.
Usage: python paper/figures/gen_fig_winners.py
"""
import json
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import oeb_plotstyle as S

HERE = Path(__file__).resolve().parent
F = json.loads((HERE.parent / "data/findings.json").read_text())
plt = S.use()

fig = plt.figure(figsize=(S.WIDTH, 1.95))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.42, left=0.215, right=0.995, top=0.89, bottom=0.19)
a = fig.add_subplot(gs[0, 0])
b = fig.add_subplot(gs[0, 1])

# ---- left: best vs. worst run of each task ------------------------------------------------------------------------
BENCHES = [("PostTrainBench", "PostTrainBench"), ("Chip-Bench", "Chip-Bench"), ("nanoGPT speedrun", "nanoGPT speedrun")]
SHORT = {"c1_pipeline_fmax": "pipeline_fmax", "c2_cache_ctrl": "cache_ctrl", "c2_divider_al": "divider_al",
         "c3_mem_sched": "mem_sched", "c3_replacement": "replacement", "speedrun": "training-speed record"}
SL = F["assoc"]["new_ideas_2nd_half"]["slope"]
rows = []                                     # (kind, label, record) top to bottom
for key, name in BENCHES:
    rows.append(("head", name, None))
    for s in sorted((s for s in SL if s["bench"] == key), key=lambda s: -s["best"] / max(s["worst"], 0.5)):
        rows.append(("task", SHORT.get(s["task"], s["task"]), s))
n = len(rows)
X0, X1 = 1.6, 160
lane = False
for i, (kind, label, s) in enumerate(rows):
    y = n - 1 - i
    if kind == "head":
        lane = not lane
        if lane:
            a.axhspan(y - 0.5, y + 0.5, color=S.LANE, linewidth=0, zorder=0)
        a.text(X0, y - 0.05, label, ha="left", va="center", fontsize=7.4, fontweight="bold", color=S.MUTE)
        continue
    if lane:
        a.axhspan(y - 0.5, y + 0.5, color=S.LANE, linewidth=0, zorder=0)
    lo, hi = s["worst"], s["best"]
    a.plot([lo, hi], [y, y], color=S.LINE, linewidth=1.1, zorder=1, solid_capstyle="butt")
    a.plot(lo, y, "o", markersize=4.2, markerfacecolor="white", markeredgecolor=S.MUTE, markeredgewidth=0.9, zorder=3)
    a.plot(hi, y, "o", markersize=4.2, color=S.BLUE, markeredgewidth=0, zorder=4)
    if hi <= lo:                              # the one task where the best run did not test more new ideas
        a.text(hi * 1.22, y, "tie", ha="left", va="center", fontsize=7.2, style="italic", color=S.MUTE)
a.set_xscale("log")
a.set_xlim(X0, X1)
a.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
a.xaxis.set_minor_locator(NullLocator())
a.set_xticklabels(["2", "5", "10", "20", "50", "100"])
for v in (2, 5, 10, 20, 50, 100):
    a.axvline(v, color=S.FAINT, linewidth=0.5, zorder=0)
a.set_ylim(-0.6, n - 0.4)
ty = [n - 1 - i for i, r in enumerate(rows) if r[0] == "task"]
a.set_yticks(ty)
a.set_yticklabels([r[1] for r in rows if r[0] == "task"], fontsize=7.4)
a.spines["left"].set_visible(False)
a.set_xlabel("new ideas first tested in the run's second half", fontsize=7.6, labelpad=2)
a.set_title("Best vs. worst run of each task", loc="left", pad=4, x=-0.33)
a.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=4.2, markerfacecolor="white",
                         markeredgecolor=S.MUTE, markeredgewidth=0.9, label="worst run"),
                  Line2D([], [], marker="o", linestyle="", markersize=4.2, color=S.BLUE, markeredgewidth=0,
                         label="best run")],
         loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7.2, handletextpad=0.2, borderaxespad=0.2,
         labelspacing=0.3)

# ---- right: Chip-Bench tries of an idea ---------------------------------------------------------------------------
P = F["positions"]["Chip-Bench"]
ORDER = ["first", "second or third", "fourth or later"]
COL = {"first": S.BLUE, "second or third": S.BLUE_MID, "fourth or later": S.AMBER}
LAB = {"first": "1st try", "second or third": "2nd–3rd", "fourth or later": "4th+"}
FG = {"first": "white", "second or third": "white", "fourth or later": S.INK}
nexp = sum(P[o]["exps"] for o in ORDER)
H = 0.66
for y, key in ((1, "exps"), (0, "gain")):
    tot = sum(P[o][key] for o in ORDER)
    x = 0
    for o in ORDER:
        w = P[o][key] / tot
        b.barh(y, w, left=x, color=COL[o], height=H, edgecolor="white", linewidth=0.8)
        pct = f"{100 * w:.0f}%"
        if y == 1:                            # name each segment once, inside the top bar
            b.text(x + w / 2, y, f"{LAB[o]}\n{pct}", ha="center", va="center", fontsize=7.2, color=FG[o], linespacing=1.15)
        elif w > 0.12:
            b.text(x + w / 2, y, pct, ha="center", va="center", fontsize=7.4, color=FG[o])
        else:
            b.text(x + w + 0.015, y, pct, ha="left", va="center", fontsize=7.4, color=S.INK)
        x += w
b.set_yticks([1, 0])
b.set_yticklabels([f"experiments\n(n = {nexp})", "real gain"], fontsize=7.4)
b.tick_params(axis="y", pad=2)
b.set_xlim(0, 1.1)
b.xaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
b.set_xticklabels(["0", "50", "100%"])
b.set_xlabel("share of Chip-Bench tries", fontsize=7.6, labelpad=2)
b.spines["left"].set_visible(False)
b.set_ylim(-0.5, 1.5)
b.set_title("Chip-Bench: tries of an idea", loc="left", pad=4, x=-0.02)

S.save(fig, str(HERE / "winners"))
print("ok")
