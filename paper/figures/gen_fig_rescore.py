"""Writes rescore.pdf/.png (appendix "Reliability details", cited in Section 5.2): each record's readings, scored twice.

Left: the PostTrainBench records scored twice by the same judge (GLM-5.3, empty call log). Right: 12
Chip-Bench records scored under GLM-5.3 (the judge of the Chip-Bench and PostTrainBench panels) and again under
GPT-5.6-sol. One dot per record and dimension (E1-E3 and T1-T6; E4 is checked against labels in 5.1). The grey band
is +-0.1. Reads OEB_DATA/out.
Usage: python paper/figures/gen_fig_rescore.py
"""
import json, glob, os, statistics as st
from pathlib import Path
import oeb_plotstyle as S
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import same_judge_pairs

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("OEB_DATA", "."))
PT = ROOT / "out/posttrainbench"
DIMS = ["E1", "E2", "E3", "T1", "T2", "T3", "T4", "T5", "T6"]
num = lambda v: isinstance(v, (int, float))


def M(u):
    return json.load(open(Path(u) / "measure.json"))["dimensions"]


def pts(pairs):
    out = []
    for a, c in pairs:
        if not ((Path(a) / "measure.json").exists() and (Path(c) / "measure.json").exists()):
            continue
        x, y = M(a), M(c)
        out += [(d[0], x[d]["score"], y[d]["score"]) for d in DIMS if num(x[d]["score"]) and num(y[d]["score"])]
    return out


same = pts(same_judge_pairs(PT))
other = pts([(Path(f).parent, ROOT / "out/chipbench/hc47-sol" / Path(f).parent.name)
             for f in sorted(glob.glob(str(ROOT / "out/chipbench/hc47-glm/*/measure.json")))])
plt = S.use()
fig = plt.figure(figsize=(S.WIDTH, 2.12))
gs = fig.add_gridspec(
    2, 2, height_ratios=[1.00, 0.20], hspace=0.62, wspace=0.38,
    left=0.07, right=0.995, top=0.86, bottom=0.02,
)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
ax_leg = fig.add_subplot(gs[1, :])
ax_leg.set_axis_off()
from matplotlib.lines import Line2D
ax_leg.legend(
    handles=[
        Line2D([], [], marker="o", linestyle="", markersize=4, color=S.BLUE, label="competence axes E1–E3"),
        Line2D([], [], marker="o", linestyle="", markersize=4, color=S.AMBER, label="persona traits T1–T6"),
    ],
    loc="center", ncol=2, frameon=False, fontsize=7.2,
    handletextpad=0.35, columnspacing=1.8, borderaxespad=0.0,
)
for ax, P, title, xl, yl in (
    (axes[0], same, "Same judge, scored again", "first scoring", "second scoring"),
    (axes[1], other, "Another judge", "GLM-5.3", "GPT-5.6"),
):
    ax.fill_between([0, 1], [-0.1, 0.9], [0.1, 1.1], color=S.LANE, linewidth=0, zorder=0)
    ax.plot([0, 1], [0, 1], color=S.LINE, linewidth=0.7, zorder=1)
    for kind, col in (("E", S.BLUE), ("T", S.AMBER)):
        q = [(x, y) for k, x, y in P if k == kind]
        ax.scatter([x for x, _ in q], [y for _, y in q], s=11, color=col, alpha=0.7, linewidth=0, zorder=3)
    med = st.median(abs(x - y) for _, x, y in P)
    ax.text(0.03, 0.97, f"median change {med:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color=S.INK)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    for f in (ax.set_xticks, ax.set_yticks):
        f([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_title(title, loc="left", pad=3, fontsize=8.5)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

S.save(fig, str(HERE / "rescore"))
print("ok", len(same), len(other))
