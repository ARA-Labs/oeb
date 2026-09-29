"""Writes persona.pdf/.png (appendix "The persona belongs to the model"): the persona traits T1-T6 by agent model.

One dot per run (the six PostTrainBench tasks, first and second runs), one row per model, one small panel per trait; the bar is the model's
mean, drawn when the model has at least three runs with a value. The two ends of each axis carry the names of the
trait's poles. Every value is read from <unit>/measure.json. matplotlib, shared style in oeb_plotstyle.py.
Usage: python paper/figures/gen_fig_persona.py
"""
import json, glob, os, statistics as st
from pathlib import Path
import oeb_plotstyle as S
HERE = Path(__file__).resolve().parent; ROOT = Path(os.environ.get("OEB_DATA", "."))
PANELS = ["aime2-w5-merge-glm53", "arenahard-gemma-w5-glm53", "bfcl-gemma-w5-glm53", "healthbench-gemma-w5-glm53",
          "humaneval-gemma-w5-glm53", "gpqa-gemma-w5-glm53"]
rows = []
for b in PANELS:
    for u in sorted(glob.glob(str(ROOT / "out/posttrainbench" / b / "*"))):
        if not os.path.exists(u + "/measure.json"): continue
        m = json.load(open(u + "/measure.json"))["dimensions"]
        rows.append(dict(unit=os.path.basename(u), T={t: m[t]["score"] for t in ("T1", "T2", "T3", "T4", "T5", "T6")}))
TR = [("T1", "Budget", "thinking", "doing"), ("T2", "Concurrency", "one subject", "several"), ("T3", "Search", "refine an idea", "new ideas"),
      ("T4", "After a failure", "persist", "pivot"), ("T5", "Provenance", "novel", "transferred"), ("T6", "Tempo", "run first", "hypothesis first")]
plt = S.use()
fig, axes = plt.subplots(2, 3, figsize=(S.WIDTH, 3.35), sharey=True, gridspec_kw={"wspace": 0.13, "hspace": 0.62})
for ax, (t, name, lo, hi) in zip(axes.flat, TR):
    for y, (key, label, fam) in enumerate(S.MODELS):
        col = S.FAMILY[fam]
        if y % 2 == 0: ax.axhspan(y - 0.5, y + 0.5, color=S.LANE, zorder=0, linewidth=0)
        vals = sorted(r["T"][t] for r in rows if r["unit"].split("-")[0] == key and r["T"][t] is not None)
        ax.scatter(vals, [y + ((k * 7) % 5 - 2) * 0.075 for k in range(len(vals))], s=13, color=col, alpha=0.55, linewidth=0, zorder=3)
        if len(vals) >= 3: ax.plot([st.mean(vals)] * 2, [y - 0.36, y + 0.36], color=col, linewidth=1.8, solid_capstyle="butt", zorder=4)
        else: ax.text(1.0, y, "no opportunities" if not vals else f"{len(vals)} run{'s' if len(vals) > 1 else ''}, too few", va="center", ha="right", fontsize=7, color=S.MUTE, style="italic")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(len(S.MODELS) - 0.5, -0.5); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_title(f"{t}  {name}", loc="left", pad=3)
    ax.text(0, -0.19, f"← {lo}", transform=ax.transAxes, ha="left", va="top", fontsize=7, style="italic", color=S.MUTE)
    ax.text(1, -0.19, f"{hi} →", transform=ax.transAxes, ha="right", va="top", fontsize=7, style="italic", color=S.MUTE)
for ax in axes[:, 0]:
    ax.set_yticks(range(len(S.MODELS))); ax.set_yticklabels([m[1] for m in S.MODELS])
    for tl, m in zip(ax.get_yticklabels(), S.MODELS): tl.set_color(S.FAMILY[m[2]])
S.save(fig, str(HERE / "persona")); print("wrote persona.pdf,", len(rows), "runs")
