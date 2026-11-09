"""Figures for the Assessor Evaluation tab (matplotlib, so they can be saved as PNG, SVG or PDF).

- distributions: one violin per rater (or rater pair) for each bone and measure, with every frame as a
  point, the median (white circle), interquartile range (thick line), mean (horizontal bar) and the
  threshold (dashed line), as in Welte et al. 2022, figure 4;
- over frames: the error at every frame for each rater, with the threshold and the frames above it
  marked, as in Welte et al. 2022, figure 5 (to find where tracking went wrong).
"""

from __future__ import annotations

import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ..core.assessment import MEASURES, AnalysisSettings, Comparison, unit_symbol

PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
           "#CCB974", "#64B5CD")


def _rater(label: str) -> str:
    """The rater a label belongs to: 'Jo s2' -> 'Jo', 'Jo s2 vs Jo s1' -> 'Jo' (pairs keep their label)."""
    first = label.split(" vs ")[0] if label.count(" vs ") == 1 and " s" in label else label
    return re.sub(r" s[^ ]+$", "", first)


def colours(labels) -> dict:
    """One colour per rater, shared by that rater's sessions."""
    raters = []
    for lab in labels:
        if _rater(lab) not in raters:
            raters.append(_rater(lab))
    return {lab: PALETTE[raters.index(_rater(lab)) % len(PALETTE)] for lab in labels}


def _dashes(labels) -> dict:
    """Solid for a rater's first session in the list, dashed / dotted for the next ones."""
    seen: dict = {}
    out = {}
    for lab in labels:
        k = seen.get(_rater(lab), 0)
        seen[_rater(lab)] = k + 1
        out[lab] = ("-", "--", ":", "-.")[k % 4]
    return out


def _labels(comps) -> list[str]:
    out = []
    for c in comps:
        if c.label not in out:
            out.append(c.label)
    return out


def _bones(comps, pool: str) -> list[str]:
    if pool == "all":
        return ["All bones"]
    out = []
    for c in comps:
        if c.bone not in out:
            out.append(c.bone)
    return out


def _values(comps, bone, label, key):
    arrs = [c.values[key] for c in comps if c.label == label and key in c.values and (bone == "All bones"
                                                                                      or c.bone == bone)]
    return np.concatenate(arrs) if arrs else np.zeros(0)


def _title(settings: AnalysisSettings) -> str:
    return {"validity": "Difference from the criterion", "inter": "Difference between raters",
            "intra": "Difference between sessions"}[settings.mode]


def draw_distributions(fig: Figure, comps: list[Comparison], settings: AnalysisSettings,
                       cap: bool = False) -> None:
    """Violins: rows = bones, columns = measures; x = raters (or pairs)."""
    fig.clear()
    measures = [k for k in settings.measures if any(k in c.values for c in comps)]
    if not comps or not measures:
        fig.text(0.5, 0.5, "No comparisons yet", ha="center", va="center", color="#666")
        return
    bones = _bones(comps, settings.pool)
    labels = _labels(comps)
    col = colours(labels)
    rng = np.random.default_rng(0)
    tilt = len(labels) > 4 or sum(len(t) for t in labels) > 30       # tilt long or many labels
    axes = fig.subplots(len(bones), len(measures), squeeze=False)
    for r, bone in enumerate(bones):
        for k, key in enumerate(measures):
            ax = axes[r, k]
            m = MEASURES[key]
            thr = settings.threshold(m)
            data = [_values(comps, bone, lab, key) for lab in labels]
            top = 3 * thr if cap else None
            hidden = 0
            for i, (lab, x) in enumerate(zip(labels, data)):
                if len(x) == 0:
                    continue
                shown = x if top is None else x[np.abs(x) <= top]
                hidden += len(x) - len(shown)
                if len(shown) > 1 and np.ptp(shown) > 0:
                    parts = ax.violinplot([shown], positions=[i], widths=0.8, showextrema=False)
                    for b in parts["bodies"]:
                        b.set_facecolor(col[lab])
                        b.set_edgecolor(col[lab])
                        b.set_alpha(0.35)
                jit = rng.uniform(-0.12, 0.12, len(shown))
                ax.scatter(i + jit, shown, s=9, color=col[lab], alpha=0.8, linewidths=0, zorder=3)
                q1, med, q3 = np.percentile(x, [25, 50, 75])
                ax.vlines(i, q1, q3, color="#222", linewidth=4, zorder=4)
                ax.hlines(np.mean(x), i - 0.18, i + 0.18, color="#222", linewidth=2, zorder=4)
                ax.scatter([i], [med], s=34, color="white", edgecolors="#222", linewidths=1, zorder=5)
            if m.signed:
                ax.axhline(0, color="#999", linewidth=0.8)
                for s in (thr, -thr):
                    ax.axhline(s, color="#222", linestyle="--", linewidth=1)
            else:
                ax.axhline(thr, color="#222", linestyle="--", linewidth=1)
                ax.set_ylim(bottom=0)
            if top is not None:
                ax.set_ylim(top=top * 1.02, bottom=-top * 1.02 if m.signed else 0)
                if hidden:
                    ax.text(0.98, 0.98, f"{hidden} frames above {top:g}{unit_symbol(m.unit)} not shown",
                            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#a33")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels if r == len(bones) - 1 else [], rotation=30 if tilt else 0,
                               ha="right" if tilt else "center", fontsize=8)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.set_ylabel(f"{m.short} ({unit_symbol(m.unit)})", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", color="#eee", linewidth=0.8)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(m.short, fontsize=9)
            if k == 0:
                ax.annotate(bone, xy=(0, 0.5), xycoords="axes fraction", xytext=(-46, 0),
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=10, fontweight="bold")
    fig.suptitle(_title(settings) + "  ·  median ○, IQR ▮, mean —, threshold - -", fontsize=9, color="#444")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9 if len(bones) > 1 else 0.85,
                        bottom=0.2 if tilt else 0.12, hspace=0.25, wspace=0.32)


def draw_over_frames(fig: Figure, comps: list[Comparison], settings: AnalysisSettings, key: str,
                     trial: str | None = None, cap: bool = False) -> None:
    """Error at each frame: rows = bones; a line per rater; frames over the threshold marked."""
    fig.clear()
    comps = [c for c in comps if key in c.values and (trial is None or c.trial == trial)]
    if not comps:
        fig.text(0.5, 0.5, "No comparisons for this measure", ha="center", va="center", color="#666")
        return
    m = MEASURES[key]
    thr = settings.threshold(m)
    bones = _bones(comps, "none")
    col = colours(_labels(comps))
    dash = _dashes(_labels(comps))
    axes = fig.subplots(len(bones), 1, squeeze=False, sharex=True)
    for r, bone in enumerate(bones):
        ax = axes[r, 0]
        for c in [c for c in comps if c.bone == bone]:
            f = c.frames + 1
            y = c.values[key]
            ax.plot(f, y, dash[c.label], color=col[c.label], linewidth=1.3, label=c.label, marker=".", markersize=3)
            over = np.abs(y) > thr
            if over.any():
                ax.scatter(f[over], y[over], s=28, facecolors="none", edgecolors="#C0392B", linewidths=1.2,
                           zorder=4)
        ax.axhline(thr, color="#222", linestyle="--", linewidth=1)
        if m.signed:
            ax.axhline(-thr, color="#222", linestyle="--", linewidth=1)
            ax.axhline(0, color="#999", linewidth=0.8)
        else:
            ax.set_ylim(bottom=0)
        if cap:
            ax.set_ylim(top=3 * thr, bottom=-3 * thr if m.signed else 0)
        ax.set_ylabel(f"{bone}\n{m.short} ({unit_symbol(m.unit)})", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(color="#eee", linewidth=0.8)
        ax.set_axisbelow(True)
        if r == 0:
            ax.legend(fontsize=7, ncol=min(4, len(col)), loc="upper left", frameon=False)
    axes[-1, 0].set_xlabel("Frame", fontsize=8)
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"{_title(settings)}: {m.short}" + (f" · {trial}" if trial else "")
                 + "  ·  ○ frames above the threshold", fontsize=9, color="#444")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.08, hspace=0.12)
