"""Redraws the trust x danger figure and the risk-coverage figure for the manuscript.

Reads only results/core/trust/trust_scores.csv; writes results/paper/ and paper/figures/.
Held-out real frames only: the trust model's own training frames (bdd_train_clear) and the
synthetic corruptions (tagged 'clear' in the source file) are excluded.

    python src_core/paper_figures.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "core" / "trust" / "trust_scores.csv"
OUT = ROOT / "results" / "paper"
PAPER_FIG = ROOT / "paper" / "figures"

# Categorical slots 1-4 (validated: CVD and normal-vision separation pass on light surface)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
T_CUT, D_CUT = 0.5, 0.4
CONDITIONS = ["clear", "night", "rain", "snow", "fog"]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def load() -> pd.DataFrame:
    d = pd.read_csv(SRC)
    d = d[(d["stratum"] != "bdd_train_clear") & ~d["stratum"].str.startswith("syn_")].copy()
    d["fail"] = d["fail"].astype(bool)
    d["under"] = (d["danger_true"] - d["danger_est"]) > 0.1
    return d


def quadrant_figure(d: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    rng = np.random.default_rng(0)
    for ax, cond in zip(axes.flat[:5], CONDITIONS):
        s = d[d["weather_gt"] == cond]
        s = s.iloc[rng.permutation(len(s))]               # no colour drawn systematically on top
        ok, bad = s[~s["fail"]], s[s["fail"]]
        ax.scatter(ok["danger_est"], ok["trust"], s=5, c=BLUE, alpha=0.45, lw=0, marker="o", label="detector correct")
        ax.scatter(bad["danger_est"], bad["trust"], s=6, c=ORANGE, alpha=0.45, lw=0, marker="^", label="detector failed")
        ax.axhline(T_CUT, color=INK2, lw=0.6, ls="--")
        ax.axvline(D_CUT, color=INK2, lw=0.6, ls="--")
        ax.add_patch(plt.Rectangle((0, 0), D_CUT, T_CUT, color="#f0efec", zorder=0))
        q = s[(s["trust"] < T_CUT) & (s["danger_est"] < D_CUT)]
        ax.text(0.02, 0.02, f"n={len(q)} ({len(q) / len(s):.0%})\nfail {q['fail'].mean():.0%}"
                if len(q) else "n=0", transform=ax.transAxes, fontsize=6.5, color=INK, va="bottom")
        ax.set_title(f"{cond} (n={len(s):,})", fontsize=8, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.4, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("apparent danger $R_{est}$")
        ax.set_ylabel("trust $T$")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, frameon=False, markerscale=3, bbox_to_anchor=(0.5, 1.0))

    # panel f: share of frames whose danger is underestimated by > 0.1, per quadrant
    ax = axes.flat[5]
    cols = [("low", "low"), ("low", "high"), ("high", "low"), ("high", "high")]
    mat = np.full((len(CONDITIONS), 4), np.nan)
    for i, cond in enumerate(CONDITIONS):
        s = d[d["weather_gt"] == cond]
        for j, (dl, tl) in enumerate(cols):
            m = ((s["danger_est"] < D_CUT) == (dl == "low")) & ((s["trust"] < T_CUT) == (tl == "low"))
            if m.sum() >= 10:
                mat[i, j] = s.loc[m, "under"].mean()
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(4):
            v = mat[i, j]
            ax.text(j, i, "n<10" if np.isnan(v) else f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                    color="white" if (not np.isnan(v) and v > 0.55) else INK)
    ax.set_xticks(range(4), ["low D\nlow T", "low D\nhigh T", "high D\nlow T", "high D\nhigh T"], fontsize=6.5)
    ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_title(r"share with $\Delta > 0.1$", fontsize=8, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k, a in enumerate(axes.flat):
        a.text(-0.12, 1.04, f"({'abcdef'[k]})", transform=a.transAxes, fontsize=8, fontweight="bold", ha="right")
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def risk_curve(trust: np.ndarray, fail: np.ndarray):
    order = np.argsort(-trust, kind="stable")
    f = fail[order].astype(float)
    n = np.arange(1, len(f) + 1)
    return n / len(f), np.cumsum(f) / n


def risk_coverage_figure(d: pd.DataFrame, path: Path) -> None:
    groups = [("BDD100K: same camera, new weather", ["bdd_val_clear", "bdd_val_night", "bdd_val_rain", "bdd_val_snow"]),
              ("ACDC: new camera, real weather", ["acdc_fog", "acdc_night", "acdc_rain", "acdc_snow"]),
              ("Cityscapes: new camera, fog levels", ["cs_clear", "cs_fog_light", "cs_fog_medium", "cs_fog_dense"])]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, sharey=True)
    axes = axes.flat
    ax = axes[0]
    y = d["fail"].to_numpy()
    for score, lab, c, ls in (("trust", "trust", BLUE, "-"), ("c_mean", "mean confidence", ORANGE, "--")):
        cov, risk = risk_curve(d[score].to_numpy(), y)
        ax.plot(cov, risk, color=c, lw=1.6, ls=ls, label=lab)
    oc, orisk = risk_curve((~y).astype(float) + 1e-9 * np.random.default_rng(0).random(len(y)), y)
    ax.plot(oc, orisk, color=INK2, lw=1.0, ls=":", label="oracle")
    ax.axhline(y.mean(), color=GRID, lw=0.8)
    ax.set_title("All real held-out frames", fontsize=8, color=INK)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    for ax, (title, strata) in zip(axes[1:], groups):
        for s_name, c in zip(strata, (BLUE, ORANGE, AQUA, YELLOW)):
            s = d[d["stratum"] == s_name]
            cov, risk = risk_curve(s["trust"].to_numpy(), s["fail"].to_numpy())
            ax.plot(cov, risk, color=c, lw=1.4, label=s_name.split("_", 1)[1].replace("_", " "))
        ax.set_title(title, fontsize=8, color=INK)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.2, ncol=2)
    for k, ax in enumerate(axes):
        ax.set_xlim(0.1, 1.0); ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.text(-0.04, 1.04, f"({'abcd'[k]})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right")
    fig.supxlabel("coverage (fraction of most-trusted frames acted on)", fontsize=8)
    fig.supylabel("failure rate among accepted frames", fontsize=8)
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    quadrant_figure(d, OUT / "quadrant_v2.pdf")
    risk_coverage_figure(d, OUT / "risk_coverage_v2.pdf")
    for name in ("quadrant_v2.pdf", "risk_coverage_v2.pdf"):
        shutil.copy(OUT / name, PAPER_FIG / name)
    print("frames used:", len(d), d.groupby("weather_gt").size().to_dict())


if __name__ == "__main__":
    main()
