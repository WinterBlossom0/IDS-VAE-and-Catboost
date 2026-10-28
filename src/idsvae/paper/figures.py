"""Paper figure of the Stage 1 SHAP ranking. The pipeline diagram is TikZ (paper/figures/architecture.tex)."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

from idsvae import config
from idsvae.framework import features

matplotlib.rcParams.update({
    # llncsdoc.pdf v2.25, p. 4: Springer output omits the characters of Type 3 fonts; Type 42 embeds TrueType fonts.
    "pdf.fonttype": 42,
    "font.family": "sans-serif",
    # Helvetica-metric fonts first, to match the TeX Gyre Heros text of the TikZ diagram.
    "font.sans-serif": ["Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
})

FIGURES_DIR = config.PROJECT_ROOT / "paper" / "figures"
# Kept features; reaches 3:1 against white (WCAG 2.1, non-text contrast).
BAR_COLOR = "#2a78d6"
# Dropped features differ from kept ones in lightness too, so the two stay apart in grayscale print.
DROPPED = "#bebdb5"
INK = "#2b2b29"
MUTED = "#6b6a64"
AXIS = "#c9c8c1"
GRID = "#ebeae4"
# Area washes stay light so that the line and the markers carry the data.
WASH_ALPHA = 0.12
CM_PER_INCH = 2.54
POINTS_PER_INCH = 72.0
BAR_END_RADIUS_PT = 1.6
# Springer proceedings class llncs.cls v2.25 (2026-09-03), line 55: \setlength{\textwidth}{12.2cm}.
TEXT_WIDTH = 12.2 / CM_PER_INCH
# The paper includes the figure at this share of \textwidth, its saved width, so the point sizes below hold at print size.
SHAP_WIDTH_SHARE = 1.0
SHAP_HEIGHT = 3.5 / CM_PER_INCH
# Springer proceedings guidelines: figure lettering not smaller than 6 pt.
TICK_SIZE = 6.5
LABEL_SIZE = 6.5
TITLE_SIZE = 7.0
LINE_WIDTH = 1.3
TOP_FEATURES_SHOWN = 10


def style_axes(axes, grid_axis="both"):
    axes.set_axisbelow(True)
    if grid_axis:
        axes.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=MUTED, labelcolor=INK, labelsize=TICK_SIZE)


def panel_title(axes, text):
    axes.set_title(text, fontsize=TITLE_SIZE, color=INK, loc="left", fontweight="bold", pad=4)


def save(figure, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # No tight bounding box: the saved width must equal the figure width, so that the point sizes hold at the include width.
    for suffix in ("pdf", "png"):
        # Without a creation date, a rerun writes the same PDF bytes.
        figure.savefig(FIGURES_DIR / f"{name}.{suffix}", dpi=300, metadata={"CreationDate": None} if suffix == "pdf" else None)
    plt.close(figure)


def percent_label(fraction):
    return f"{fraction * 100:g} %"


def checked_ranking(ranking_dir):
    ranking = features.load_ranking(ranking_dir)
    scores = np.array([entry["mean_abs_shap"] for entry in ranking["ranking"]])
    if np.any(np.diff(scores) > 0):
        raise ValueError(f"the ranking in {ranking_dir} is not sorted by mean |SHAP|")
    share = ranking["settings"]["cumulative_share"]
    if features.cumulative_cut(scores, share) != ranking["selected_count"]:
        raise ValueError(f"selected_count {ranking['selected_count']} in {ranking_dir} disagrees with the {share:.0%} cut of its scores")
    return ranking, scores


def rounded_hbars(axes, positions, lengths, thickness, color):
    """Horizontal bars with round data ends and a square baseline; call after the axes limits are final."""
    figure = axes.figure
    figure.canvas.draw()
    box = axes.get_window_extent()
    x_low, x_high = axes.get_xlim()
    y_low, y_high = axes.get_ylim()
    x_per_pixel = (x_high - x_low) / box.width
    # mutation_aspect turns a radius in x data units into the same radius in pixels along y.
    aspect = ((y_high - y_low) / box.height) / x_per_pixel
    radius = min(BAR_END_RADIUS_PT * figure.dpi / POINTS_PER_INCH * x_per_pixel, thickness / 2 / aspect)
    for position, length in zip(positions, lengths):
        bottom = position - thickness / 2
        axes.add_patch(FancyBboxPatch((0, bottom), length, thickness, boxstyle=f"round,pad=0,rounding_size={radius}", mutation_aspect=aspect, facecolor=color, linewidth=0))
        axes.add_patch(Rectangle((0, bottom), min(length, 2 * radius), thickness, facecolor=color, linewidth=0))


def shap_bars(axes, names, scores, runs):
    shown = min(TOP_FEATURES_SHOWN, len(names))
    positions = np.arange(shown)[::-1]
    axes.set_yticks(positions, names[:shown])
    axes.tick_params(axis="y", length=0)
    axes.set_ylim(-0.7, shown - 0.3)
    axes.set_xlim(0, scores[0] * 1.16)
    axes.set_xlabel(f"mean |SHAP| over {runs} runs (logit units)", fontsize=LABEL_SIZE, color=INK)
    style_axes(axes, grid_axis="x")
    axes.spines["left"].set_visible(False)
    panel_title(axes, f"(a) Top {shown} features")
    # Only the leading bar carries a value; the axis carries the rest.
    axes.text(scores[0] * 1.02, positions[0], f"{scores[0]:.2f}", va="center", ha="left", fontsize=TICK_SIZE, color=INK)
    rounded_hbars(axes, positions, scores[:shown], 0.58, BAR_COLOR)


def shap_cumulative(axes, cumulative, selected, share):
    counts = np.arange(1, len(cumulative) + 1)
    color = BAR_COLOR
    kept = counts <= selected
    axes.fill_between(counts[kept], 0, cumulative[kept], color=color, alpha=WASH_ALPHA, linewidth=0)
    axes.plot(counts, cumulative, color=color, linewidth=LINE_WIDTH, solid_capstyle="round", solid_joinstyle="round")
    axes.axhline(share, color=AXIS, linewidth=0.6, zorder=1)
    axes.axvline(selected + 0.5, color=INK, linewidth=0.6, zorder=3)
    axes.plot(selected, cumulative[selected - 1], marker="o", markersize=4.6, color=color, markeredgecolor="white", markeredgewidth=1.1, zorder=4)
    axes.text(selected - 1.5, 0.3, f"{percent_label(share)} of importance", ha="right", va="center", fontsize=TICK_SIZE, color=INK)
    axes.set_ylim(0, 1.05)
    axes.set_yticks([0, 0.5, 1])
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: percent_label(value)))
    axes.set_ylabel("cumulative", fontsize=LABEL_SIZE, color=INK)
    style_axes(axes)
    axes.tick_params(axis="x", labelbottom=False)
    panel_title(axes, f"(b) All {len(cumulative)} features")


def shap_profile(axes, scores, selected):
    ranks = np.arange(1, len(scores) + 1)
    kept = ranks <= selected
    axes.bar(ranks[kept], scores[kept], width=0.72, color=BAR_COLOR, linewidth=0)
    axes.bar(ranks[~kept], scores[~kept], width=0.72, color=DROPPED, linewidth=0)
    axes.axvline(selected + 0.5, color=INK, linewidth=0.6, zorder=3)
    axes.text(selected - 1.2, scores[0] * 0.62, f"{selected} kept", ha="right", va="center", fontsize=TICK_SIZE, color=INK)
    # Room right of the last bar keeps the dropped label clear of the cut line.
    right_edge = len(scores) + 6
    axes.text((selected + 0.5 + right_edge) / 2, scores[0] * 0.62, f"{len(scores) - selected}\ndropped", ha="center", va="center", fontsize=TICK_SIZE, color=MUTED)
    axes.set_xlim(0, right_edge)
    axes.set_ylim(0, scores[0] * 1.05)
    axes.set_xticks([1, 10, 20, 30, selected, len(scores)])
    axes.set_xlabel("feature rank k", fontsize=LABEL_SIZE, color=INK)
    axes.set_ylabel("mean |SHAP|", fontsize=LABEL_SIZE, color=INK)
    style_axes(axes, grid_axis="y")


def shap_ranking_figure(ranking_dir=features.STAGE1_DIR):
    ranking, scores = checked_ranking(ranking_dir)
    names = [entry["feature"] for entry in ranking["ranking"]]
    cumulative = np.cumsum(scores) / scores.sum()
    selected = ranking["selected_count"]
    figure = plt.figure(figsize=(SHAP_WIDTH_SHARE * TEXT_WIDTH, SHAP_HEIGHT), layout="constrained")
    figure.get_layout_engine().set(w_pad=0.05, h_pad=0.03, wspace=0.07, hspace=0.04)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 1.05), height_ratios=(1.0, 1.2))
    bars = figure.add_subplot(grid[:, 0])
    curve = figure.add_subplot(grid[0, 1])
    profile = figure.add_subplot(grid[1, 1], sharex=curve)
    shap_cumulative(curve, cumulative, selected, ranking["settings"]["cumulative_share"])
    shap_profile(profile, scores, selected)
    shap_bars(bars, names, scores, ranking["settings"]["runs"])
    save(figure, "shap_ranking")
    shown = min(TOP_FEATURES_SHOWN, len(names))
    return {"ranking_dir": str(ranking_dir.relative_to(config.PROJECT_ROOT)), "candidate_features": len(names), "selected_count": selected, "share_at_selected": float(cumulative[selected - 1]), "top_features": names[:shown], "top_scores": scores[:shown].tolist()}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary_path = FIGURES_DIR / "figure_data.json"
    summary = {"shap_ranking": shap_ranking_figure()}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
