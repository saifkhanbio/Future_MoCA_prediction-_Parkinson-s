"""Redraw Figures 1, 5, 6, 7 and the graphical abstract from verified aggregate results.

No models are fitted and no participant-level records are read. Inputs are the
aggregate CSVs already produced and verified in codex_improv/.
Usage: python build_figures.py <output_dir>
"""
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image

HERE = Path(__file__).resolve().parent
ISCI = HERE.parents[1]
CODEX = ISCI / "codex_improv"
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)

# Categorical colours: checked with the dataviz palette validator (light surface).
TEAL, PLUM, GOLD = "#0E8A76", "#7A3C65", "#AF6522"
INK, MUTED, GRID = "#203641", "#536772", "#D5DBDD"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": INK, "axes.spines.right": False, "axes.spines.top": False,
    "pdf.fonttype": 42,
})


def save(fig, stem, dpi=450, pdf=True):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.save(OUT / f"{stem}.tiff", dpi=(dpi, dpi), compression="tiff_lzw")
    if pdf:
        fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    return im


comp = pd.read_csv(CODEX / "Stage_ABC_manuscript_comparison_v1/stage_comparison.csv")
est = pd.read_csv(CODEX / "Stage_C_verified_results_v1/verified_estimates.csv")
freq = pd.read_csv(CODEX / "Stage_J_diagnostics_v3/results/change_distribution.csv")


def n_text(stage):
    return " / ".join(str(int(v)) for v in comp[comp.stage == stage].sort_values("months").n)


# ---------------------------------------------------------------- Figure 1
def figure_1():
    fig = plt.figure(figsize=(9.1, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    X = [0.025, 0.355, 0.685]
    W = 0.29

    def box(col, top, h, head, lines, color):
        x = X[col]
        ax.add_patch(FancyBboxPatch((x, top - h), W, h, boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor="#F6F8F8", edgecolor=color, lw=1.4))
        ax.add_patch(FancyBboxPatch((x, top - h), 0.012, h, boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=color, edgecolor=color, lw=0))
        ax.text(x + 0.028, top - 0.028, head, weight="bold", size=10.5, va="top", color=color)
        ax.text(x + 0.028, top - 0.075, "\n".join(lines), size=9, va="top", linespacing=1.45)
        return (x, top - h / 2, x + W)

    def arrow(p, q, color):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                     shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=0"))

    ax.text(X[0], 0.965, "Four-score panel: development and temporal validation", size=11.5, weight="bold", va="top")
    a1 = box(0, 0.80, 0.19, "Original PPMI benchmark", ["351 participants", "18 forecasting procedures", "compared"], TEAL)
    a2 = box(1, 0.80, 0.19, "Four-score panel", ["MoCA, processing speed,", "immediate and delayed", "verbal recall"], TEAL)
    b1 = box(2, 0.905, 0.19, "Expanded PPMI", ["Panel refitted with grouped", "cross-validation", f"n = {n_text('C')}"], TEAL)
    b2 = box(2, 0.695, 0.19, "Later PPMI recruits", ["Temporal validation of the", "earlier frozen panel", f"n = {n_text('A')}"], PLUM)
    arrow((a1[2], a1[1]), (a2[0], a2[1]), TEAL)
    arrow((a2[2], a2[1] + 0.03), (b1[0], b1[1]), TEAL)
    arrow((a2[2], a2[1] - 0.03), (b2[0], b2[1]), PLUM)

    ax.plot([X[0], X[2] + W], [0.455, 0.455], color=GRID, lw=1)
    ax.text(X[0], 0.425, "Three-input equations: evaluation in independent cohorts", size=11.5, weight="bold", va="top")
    c1 = box(0, 0.26, 0.155, "PPMI clinical sample", ["Source for model fitting", "n = 301 / 292 / 281"], GOLD)
    c2 = box(1, 0.26, 0.155, "MoCA–age–sex equations", ["Fitted in PPMI, then applied", "with fixed coefficients"], GOLD)
    d1 = box(2, 0.385, 0.15, "PDBP", ["Independent cohort, nominal visits", f"n = {n_text('B')}"], GOLD)
    d2 = box(2, 0.215, 0.18, "LCC", ["Independent cohort, nominal visits", "Depression score also tested", "n = 16 / 28 / 34"], GOLD)
    arrow((c1[2], c1[1]), (c2[0], c2[1]), GOLD)
    arrow((c2[2], c2[1] + 0.025), (d1[0], d1[1]), GOLD)
    arrow((c2[2], c2[1] - 0.025), (d2[0], d2[1]), GOLD)
    ax.text(X[0], 0.004, "Participant counts are shown for 12 / 24 / 36 months. Horizon-specific samples overlap within each cohort.",
            size=8.5, color=MUTED, va="bottom")
    save(fig, "Figure_1")


# ---------------------------------------------------------------- Figure 5
def figure_5():
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.7), sharex=True, sharey=True)
    panels = [("C", "A", "Expanded PPMI", "Four scores, refitted", TEAL, "o"),
              ("A", "B", "Later PPMI recruits", "Four scores, earlier frozen model", PLUM, "s"),
              ("B", "C", "PDBP", "MoCA, age and sex equation", GOLD, "D")]
    for ax, (stage, letter, head, sub, color, mark) in zip(axs, panels):
        ax.axvline(0, color=MUTED, ls="--", lw=1)
        ax.text(0, 1.20, letter, transform=ax.transAxes, weight="bold", size=12, va="bottom")
        ax.text(0.09, 1.20, head, transform=ax.transAxes, weight="bold", size=10, va="bottom")
        ax.text(0.09, 1.09, sub, transform=ax.transAxes, size=9, color=MUTED, va="bottom")
        rows = comp[comp.stage == stage].sort_values("months")
        for i, r in enumerate(rows.itertuples()):
            y = 2 - i
            ax.errorbar(r.delta_MAE, y, xerr=[[r.delta_MAE - r.lower95], [r.upper95 - r.delta_MAE]], fmt=mark,
                        color=color, capsize=3.5, ms=7, lw=2, markeredgecolor="white", markeredgewidth=0.8)
            ax.text(-0.62, y + 0.24, f"n = {int(r.n)}", size=8.5, color=MUTED)
        ax.set_xlim(-0.64, 0.49)
        ax.set_ylim(-0.5, 2.6)
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["36 months", "24 months", "12 months"])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.76, bottom=0.26, wspace=0.12)
    fig.text(0.54, 0.105, "Model error minus error of carrying baseline MoCA forward (MAE difference, MoCA points)", ha="center")
    fig.text(0.54, 0.035, "Negative values favor the model. Bars are paired 95% intervals.", ha="center", size=8.5, color=MUTED)
    save(fig, "Figure_5")


# ---------------------------------------------------------------- Figure 6
def figure_6():
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    fig.subplots_adjust(left=0.215, right=0.97, bottom=0.21, top=0.95)
    labels = []
    for i, h in enumerate([36, 24, 12]):
        r = est[(est.horizon == h) & (est.view == "expanded") & (est.measure == "compact4-moca_only/delta_MAE")].iloc[0]
        g = -r.estimate
        ax.errorbar(g, i, xerr=[[g + r.upper95], [-r.lower95 - g]], fmt="o", color=TEAL, capsize=4, ms=8, lw=2,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.text(0.283, i, f"{g:.3f}", va="center", weight="bold")
        labels.append(f"{h} months  ·  n = {int(r.n)}")
    ax.axvline(0, ls="--", color=MUTED, lw=1)
    ax.set_yticks([0, 1, 2], labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(-0.015, 0.32)
    ax.set_xlabel("Reduction in mean absolute error versus fitted MoCA alone (MoCA points)")
    ax.spines["left"].set_visible(False)
    save(fig, "Figure_6")


# ---------------------------------------------------------------- Figure 7
def figure_7():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.16, wspace=0.10)
    out = {}
    for ax, st, letter, title, col in zip(axes, ["C", "A"], ["A", "B"], ["Expanded PPMI", "Later PPMI recruits"], [TEAL, PLUM]):
        g = freq[(freq.stage == st) & (freq.h == 12)].copy()
        g["bin"] = g.change.clip(-5, 5)
        q = g.groupby("bin").percent.sum().reindex(range(-5, 6), fill_value=0)
        r = comp[(comp.stage == st) & (comp.months == 12)].iloc[0]
        unchanged = float(g[g.change == 0].percent.sum())
        ax.bar(q.index, q.values, color=col, width=0.82)
        ax.text(0, 1.06, letter, transform=ax.transAxes, weight="bold", size=12, va="bottom")
        ax.text(0.07, 1.06, f"{title}  ·  n = {int(r.n)}", transform=ax.transAxes, size=10.5, va="bottom")
        ax.text(0.985, 0.96, f"Unchanged: {unchanged:.1f}%\nMean absolute change: {r.persistence_MAE:.2f}",
                transform=ax.transAxes, ha="right", va="top", size=9, linespacing=1.5)
        ax.set_xticks([-5, -3, 0, 3, 5], ["≤−5", "−3", "0", "+3", "≥+5"])
        ax.set_xlabel("One-year change in MoCA (points)")
        ax.set_ylim(0, 30)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        out[st] = (unchanged, float(r.persistence_MAE))
    axes[0].set_ylabel("Participants (%)")
    save(fig, "Figure_7")
    return out


# ------------------------------------------------------- Graphical abstract
def graphical_abstract():
    for name in ["arial.ttf", "arialbd.ttf"]:
        p = Path("/mnt/c/Windows/Fonts") / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    fam = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    paper, band = "#FFFCF5", "#192D38"
    with plt.rc_context({"font.family": fam}):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.add_patch(plt.Rectangle((0, 0), 100, 100, color=paper))
        # header
        ax.add_patch(plt.Rectangle((0, 84), 100, 16, color=band))
        ax.text(5, 94.2, "A brief cognitive panel", color="white", size=13.5, weight="bold", va="center")
        ax.text(5, 88.2, "improves forecasts in Parkinson disease", color="#E4EAD9", size=9.6, va="center")

        # the panel: three instruments
        def tile(x, color, head, sub):
            ax.add_patch(FancyBboxPatch((x, 70), 8.5, 8.5, boxstyle="round,pad=0,rounding_size=1.6", color=color))
            ax.text(x + 10.5, 76.3, head, size=10.5, weight="bold", va="center")
            ax.text(x + 10.5, 72.0, sub, size=8, color=MUTED, va="center")
            return x

        x = tile(4, TEAL, "Screen", "MoCA")
        for yy in (72.4, 74.3, 76.2):
            ax.plot([x + 1.6, x + 2.5, x + 3.6], [yy, yy - 0.7, yy + 0.8], color="white", lw=1.1)
            ax.plot([x + 4.6, x + 7], [yy, yy], color="white", lw=1.1)
        x = tile(37, "#BA683E", "Speed", "Symbols")
        ax.add_patch(Polygon([[x + 2.6, x * 0 + 76.9], [x + 3.7, 75.8], [x + 2.6, 74.7], [x + 1.5, 75.8]], color="white"))
        ax.add_patch(plt.Circle((x + 6.2, 75.8), 1.05, color="white"))
        ax.text(x + 4.25, 72.4, "1  2", color="white", size=6.5, weight="bold", ha="center", va="center")
        x = tile(70, PLUM, "Recall", "Word lists")
        px_, py_ = [x + 1.9, x + 4.2, x + 6.5], [74.9, 76.8, 74.6]
        ax.plot(px_, py_, color="white", lw=1.1, marker="o", ms=3.2)
        ax.annotate("", xy=(x + 6.9, 72.2), xytext=(x + 1.6, 72.2), arrowprops=dict(arrowstyle="->", color="white", lw=1.1))
        for xx in (31.5, 64.5):
            ax.text(xx, 74.2, "+", size=13, weight="bold", color=MUTED, ha="center", va="center")

        # development: forecast error, current score versus panel
        ax.add_patch(FancyBboxPatch((3, 33.5), 94, 32, boxstyle="round,pad=0,rounding_size=2", color="#E5F2EB"))
        ax.text(6, 61.6, "Lower forecast error", size=10.5, weight="bold", va="center")
        ax.text(94, 61.6, "777 participants", size=8, color=MUTED, ha="right", va="center")
        c = comp[comp.stage == "C"].sort_values("months")
        lo, hi = 1.55, 2.08
        fx = lambda v: 31 + (v - lo) / (hi - lo) * 60
        for (yy, lab), r in zip([(55, "1 year"), (48.5, "2 years"), (42, "3 years")], c.itertuples()):
            ax.text(6, yy, lab, size=9.5, va="center")
            ax.annotate("", xy=(fx(r.model_MAE) + 1.6, yy), xytext=(fx(r.persistence_MAE) - 1.2, yy),
                        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.6, mutation_scale=9, shrinkA=0, shrinkB=0))
            ax.add_patch(plt.Circle((fx(r.persistence_MAE), yy), 1.25, color="#475966", zorder=3))
            ax.add_patch(Polygon([[fx(r.model_MAE), yy + 1.7], [fx(r.model_MAE) + 1.7, yy], [fx(r.model_MAE), yy - 1.7],
                                  [fx(r.model_MAE) - 1.7, yy]], color=TEAL, zorder=3))
        ax.add_patch(Polygon([[33, 38.1], [34.5, 36.6], [33, 35.1], [31.5, 36.6]], color=TEAL))
        ax.text(36.2, 36.6, "Panel", size=8, va="center")
        ax.add_patch(plt.Circle((54, 36.6), 1.15, color="#475966"))
        ax.text(56.6, 36.6, "Current score alone", size=8, va="center")

        # temporal validation
        ax.add_patch(FancyBboxPatch((3, 19.5), 94, 11, boxstyle="round,pad=0,rounding_size=2", color="#F4E6EE"))
        ax.add_patch(plt.Circle((8.5, 25), 3.1, color=PLUM))
        ax.add_patch(plt.Circle((8.5, 25), 1.9, facecolor="none", edgecolor="white", lw=1.3))
        ax.plot([8.5, 8.5, 9.6], [26.2, 25.0, 24.4], color="white", lw=1.3, solid_capstyle="round")
        ax.text(14, 27.2, "Temporal validation", size=10.5, weight="bold", va="center")
        ax.text(14, 22.6, "Similar panel error in 213 later recruits", size=8, color=INK, va="center")

        # clinical relevance
        ax.add_patch(FancyBboxPatch((3, 3), 94, 13.5, boxstyle="round,pad=0,rounding_size=2", color="#F7E8C3"))
        ax.text(6, 12.4, "Reliable forecasts", size=9.5, weight="bold", va="center")
        ax.annotate("", xy=(43.5, 12.4), xytext=(36.5, 12.4), arrowprops=dict(arrowstyle="-|>", color="#B8860B", lw=1.6, mutation_scale=9))
        ax.text(45.5, 12.4, "Earlier support", size=9.5, weight="bold", va="center")
        ax.text(6, 6.6, "Could help preserve independence and quality of life", size=8, va="center")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    assert im.size == (1200, 1200), im.size
    im.save(OUT / "Graphical_Abstract.tiff", dpi=(300, 300), compression="tiff_lzw")
    im.save(OUT / "Graphical_Abstract.png", dpi=(300, 300))
    return fam


if __name__ == "__main__":
    figure_1()
    figure_5()
    figure_6()
    print("figure 7 annotations (unchanged %, mean absolute change):", figure_7())
    print("graphical abstract font:", graphical_abstract())
    for r in comp.itertuples():
        print(r.stage, r.months, int(r.n), round(r.model_MAE, 3), round(r.persistence_MAE, 3))
