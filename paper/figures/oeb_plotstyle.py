"""Shared matplotlib style for the paper's data figures, matched to the TikZ diagrams.

The same sans-serif type, the same ink/grey/lane neutrals and the same four accents (blue, amber,
red, green). Nothing under 7 pt, no top/right spines, a very faint solid grid, white edges between bars, PDF + PNG."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
WIDTH = 5.5                                   # ICLR text width, inches
INK, MUTE, LINE, FAINT, LANE = "#111827", "#6B7280", "#9CA3AF", "#E5E7EB", "#F3F4F6"
BLUE, BLUE_MID, AMBER, RED, RED_SOFT, NEUTRAL = "#1F4E9C", "#5B84C4", "#C99A4B", "#B3261E", "#E08A80", "#4B5563"
FAMILY = {"Claude": "#C0771A", "GPT": "#1F4E9C", "Kimi": "#1E6B45", "GLM": "#6B4E9B"}   # one colour per model family, paper-wide
MODELS = [("opus5", "Claude Opus 5", "Claude"), ("fable5", "Claude Fable 5", "Claude"), ("gpt56", "GPT-5.6", "GPT"),
          ("gpt55", "GPT-5.5", "GPT"), ("kimik3", "Kimi K3", "Kimi"), ("glm52", "GLM-5.2", "GLM")]
def use():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Liberation Sans", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
        "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5, "legend.frameon": False, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.alpha": 0.15, "grid.linestyle": "-", "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
    return plt
def save(fig, stem):
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=300); plt.close(fig)
