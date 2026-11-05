"""Statistical re-analysis of the saved per-frame outputs for the journal manuscript.

Uses only results/core/trust/trust_scores.csv (no datasets or GPU needed) and writes results/paper/.

  * cluster-bootstrap 95% CIs (resampling whole video sequences) for AUROC of the trust score and of the
    mean-confidence baseline, and for their paired difference, per stratum and pooled;
  * selective-prediction metrics: AURC, E-AURC and residual failure at fixed coverage;
  * calibration of P(fail) = 1 - trust (ECE, Brier), raw and after a monotone (isotonic) recalibration fitted on
    the held-out clear BDD100K validation stratum only, then frozen;
  * whether low trust flags frames whose danger is underestimated (the motivation for decoupling the axes).

    python src_core/paper_stats.py [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "core" / "trust" / "trust_scores.csv"
OUT = ROOT / "results" / "paper"
TRAIN = "bdd_train_clear"
CALIB = "bdd_val_clear"
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def group_of(frame_id: str, source: str) -> str:
    """Sequence/video id, so the bootstrap resamples correlated frames together."""
    if "_frame_" in frame_id:                       # ACDC: GOPR0475_frame_000041_rgb_anon
        return frame_id.split("_frame_")[0]
    if source == "cityscapes" or frame_id.count("_") >= 3:   # aachen_000000_000019_0.02
        return "_".join(frame_id.split("_")[:2])
    return frame_id.split("-")[0]                   # BDD100K: <video>-<frame>


def auroc(y, s) -> float:
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def aurc(conf: np.ndarray, fail: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better); confidence = higher means keep."""
    order = np.argsort(-conf, kind="stable")
    f = fail[order].astype(float)
    risk = np.cumsum(f) / np.arange(1, len(f) + 1)
    return float(risk.mean())


def eaurc(conf: np.ndarray, fail: np.ndarray) -> float:
    r = fail.mean()
    optimal = r + (1 - r) * np.log(1 - r) if 0 < r < 1 else 0.0  # Geifman et al. 2019
    return aurc(conf, fail) - optimal


def risk_at(conf: np.ndarray, fail: np.ndarray, cov: float) -> float:
    k = max(1, int(round(cov * len(conf))))
    return float(fail[np.argsort(-conf, kind="stable")[:k]].mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def cluster_boot(df: pd.DataFrame, stat, B: int, rng) -> np.ndarray:
    groups = df["group"].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = stat(df.iloc[idx])
        if np.all(np.isfinite(v)):
            out.append(v)
    return np.asarray(out)


def ci(a: np.ndarray, q=(2.5, 97.5)) -> tuple[float, float]:
    lo, hi = np.percentile(a, q, axis=0)
    return float(lo), float(hi)


def auroc_stats(d: pd.DataFrame) -> np.ndarray:
    y = d["fail"].to_numpy()
    t, c = auroc(y, -d["trust"].to_numpy()), auroc(y, -d["c_mean"].to_numpy())
    return np.array([t, c, t - c])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC)
    df["fail"] = df["fail"].astype(bool)
    df["group"] = [group_of(f, s) for f, s in zip(df["frame_id"], df["source"])]
    df["p_fail"] = 1.0 - df["trust"]
    df["gap"] = df["danger_true"] - df["danger_est"]
    held = df[df["stratum"] != TRAIN]
    real_adverse = held[~held["stratum"].isin([CALIB, "cs_clear"]) & ~held["stratum"].str.startswith("syn_")]

    # ---------- 1. AUROC with cluster-bootstrap CIs ----------
    rows = []
    sets = [(s, held[held["stratum"] == s]) for s in held["stratum"].unique()]
    sets += [("POOLED held-out (all 21)", held), ("POOLED real adverse (10)", real_adverse)]
    for name, d in sets:
        point = auroc_stats(d)
        bs = cluster_boot(d, auroc_stats, a.boot, rng)
        rows.append({
            "stratum": name, "frames": len(d), "sequences": d["group"].nunique(), "fail_rate": d["fail"].mean(),
            "auroc_trust": point[0], "auroc_trust_lo": ci(bs[:, 0])[0], "auroc_trust_hi": ci(bs[:, 0])[1],
            "auroc_conf": point[1], "auroc_conf_lo": ci(bs[:, 1])[0], "auroc_conf_hi": ci(bs[:, 1])[1],
            "delta": point[2], "delta_lo": ci(bs[:, 2])[0], "delta_hi": ci(bs[:, 2])[1],
            "p_one_sided": float((bs[:, 2] <= 0).mean()),
        })
    auc = pd.DataFrame(rows)
    auc.to_csv(OUT / "auroc_ci.csv", index=False)

    # ---------- 2. selective prediction ----------
    rows = []
    for name, d in sets:
        y = d["fail"].to_numpy()
        for score_name, s in (("trust", d["trust"].to_numpy()), ("conf", d["c_mean"].to_numpy()),
                              ("oracle", (~y).astype(float) + 1e-9 * rng.random(len(y)))):
            r = {"stratum": name, "score": score_name, "aurc": aurc(s, y), "eaurc": eaurc(s, y)}
            for c in COVERAGES:
                r[f"risk@{int(c * 100)}"] = risk_at(s, y, c)
            rows.append(r)
    sel = pd.DataFrame(rows)
    sel.to_csv(OUT / "selective.csv", index=False)

    # ---------- 3. calibration: raw vs isotonic fitted on CALIB only ----------
    cal = df[df["stratum"] == CALIB]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["p_fail"], cal["fail"].astype(float))
    df["p_fail_cal"] = iso.predict(df["p_fail"])
    rows = []
    test = df[~df["stratum"].isin([TRAIN, CALIB])]
    for name, d in [(s, test[test["stratum"] == s]) for s in test["stratum"].unique()] + [("POOLED test (20)", test)]:
        y = d["fail"].to_numpy().astype(float)
        rows.append({"stratum": name, "fail_rate": y.mean(), "mean_p_raw": d["p_fail"].mean(),
                     "ece_raw": ece(d["p_fail"].to_numpy(), y), "brier_raw": float(((d["p_fail"] - y) ** 2).mean()),
                     "mean_p_cal": d["p_fail_cal"].mean(), "ece_cal": ece(d["p_fail_cal"].to_numpy(), y),
                     "brier_cal": float(((d["p_fail_cal"] - y) ** 2).mean())})
    calib = pd.DataFrame(rows)
    calib.to_csv(OUT / "calibration.csv", index=False)

    # ---------- 4. danger: estimate vs truth, and does trust flag underestimation? ----------
    rows = []
    for name, d in sets:
        under = (d["gap"] > 0.1).to_numpy()
        rho = spearmanr(d["danger_est"], d["danger_true"]).statistic
        bs = cluster_boot(d, lambda x: np.array([(x["gap"]).mean()]), a.boot, rng)
        rows.append({"stratum": name, "gap_mean": d["gap"].mean(), "gap_lo": ci(bs[:, 0])[0], "gap_hi": ci(bs[:, 0])[1],
                     "frac_under_0.1": under.mean(), "spearman_est_true": float(rho),
                     "auroc_trust_flags_under": auroc(under, -d["trust"].to_numpy()),
                     "auroc_conf_flags_under": auroc(under, -d["c_mean"].to_numpy())})
    dang = pd.DataFrame(rows)
    dang.to_csv(OUT / "danger_stats.csv", index=False)

    # quadrant: low trust (<0.5) & low apparent danger (<0.4) vs the rest, on held-out frames
    q = held.assign(low_apparent_danger=held["danger_est"] < 0.4, low_trust=held["trust"] < 0.5)
    quad = q.groupby(["low_apparent_danger", "low_trust"]).agg(
        frames=("fail", "size"), fail_rate=("fail", "mean"), missed_ped_rate=("fail_person", "mean"),
        gap_mean=("gap", "mean"), under_rate=("gap", lambda g: (g > 0.1).mean()))
    quad.to_csv(OUT / "quadrant_stats.csv")

    # ---------- figures ----------
    plot_forest(auc, OUT / "auroc_forest.pdf")
    plot_reliability(test, OUT / "reliability.pdf")

    pooled = auc.set_index("stratum").loc["POOLED held-out (all 21)"]
    summary = {
        "bootstrap": {"B": a.boot, "unit": "video/sequence", "seed": a.seed},
        "strata_trust_beats_conf_ci_excludes_0": int((auc.iloc[:-2]["delta_lo"] > 0).sum()),
        "strata_total": int(len(auc) - 2),
        "pooled": {k: float(pooled[k]) for k in ["auroc_trust", "auroc_trust_lo", "auroc_trust_hi", "auroc_conf",
                                                 "delta", "delta_lo", "delta_hi"]},
        "isotonic_fitted_on": CALIB,
        "ece_pooled_raw_vs_cal": [float(calib.iloc[-1]["ece_raw"]), float(calib.iloc[-1]["ece_cal"])],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(auc.drop(columns=["p_one_sided"]))
        print(sel[sel["stratum"].str.startswith("POOLED")])
        print(calib)
        print(dang)
        print(quad)


def plot_forest(auc: pd.DataFrame, path: Path) -> None:
    d = auc.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(d) + 1))
    ax.errorbar(d["auroc_conf"], y + 0.15, xerr=[d["auroc_conf"] - d["auroc_conf_lo"], d["auroc_conf_hi"] - d["auroc_conf"]],
                fmt="s", ms=3, color="#9a9a9a", lw=1, label="Mean confidence")
    ax.errorbar(d["auroc_trust"], y - 0.15, xerr=[d["auroc_trust"] - d["auroc_trust_lo"], d["auroc_trust_hi"] - d["auroc_trust"]],
                fmt="o", ms=3, color="#1f5aa6", lw=1, label="Trust (ours)")
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    ax.set_yticks(y, d["stratum"], fontsize=7)
    ax.set_xlabel("AUROC for predicting detector failure (95% cluster-bootstrap CI)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_reliability(test: pd.DataFrame, path: Path, bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    edges = np.linspace(0, 1, bins + 1)
    y = test["fail"].to_numpy().astype(float)
    for col, lab, c in (("p_fail", "raw", "#9a9a9a"), ("p_fail_cal", "isotonic (fit on BDD val clear)", "#1f5aa6")):
        p = test[col].to_numpy()
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        xs, ys = [], []
        for b in range(bins):
            m = idx == b
            if m.sum() >= 20:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", ms=3, color=c, label=lab)
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.set_xlabel("Predicted failure probability", fontsize=8)
    ax.set_ylabel("Observed failure rate", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
