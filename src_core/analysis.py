"""Analysis and figures: danger gap per stratum, weight sensitivity, trust x danger quadrant, enhancement ON vs OFF,
severity sweeps. Reads the records and the trust scores; writes CSV tables and PNG figures.

    python src_core/analysis.py --records results/core/records/yolo26m_off.jsonl [--records-on results/core/records/yolo26m_on.jsonl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, read_jsonl, save_json, start_log  # noqa: E402
import danger as DG  # noqa: E402
from sweep import stratum_metrics  # noqa: E402

FIG = RESULTS / "figures"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--records", type=Path, required=True, help="enhancement OFF records of the chosen detector")
    p.add_argument("--records-on", type=Path, default=None, help="enhancement ON records of the same detector")
    p.add_argument("--trust-scores", type=Path, default=RESULTS / "trust" / "trust_scores.csv")
    p.add_argument("--out", type=Path, default=RESULTS)
    return p.parse_args(argv)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def danger_gap_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    by = {}
    for r in records:
        by.setdefault(r["stratum"], []).append(r)
    for name, recs in by.items():
        gaps = np.array([r["scores"]["danger_gap"] for r in recs])
        est = np.array([r["scores"]["danger_est"] for r in recs])
        true = np.array([r["scores"]["danger_true"] for r in recs])
        rows.append({"stratum": name, "frames": len(recs), "danger_true_mean": true.mean(), "danger_est_mean": est.mean(),
                     "gap_mean": gaps.mean(), "gap_median": float(np.median(gaps)), "gap_p90": float(np.quantile(gaps, 0.9)),
                     "frac_underestimated_by_0.1": float((gaps > 0.1).mean())})
    return pd.DataFrame(rows)


def plot_gap(tab: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.bar(x, tab["gap_mean"], color=["#2f6fb3" if "clear" in s else "#c7702a" for s in tab["stratum"]])
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x, tab["stratum"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean danger gap  R_true - R_est")
    ax.set_title("Perception underestimates danger when the gap is positive", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_quadrant(scores: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    conds = [c for c in ("clear", "night", "rain", "snow", "fog") if (scores["weather_gt"] == c).any()]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.4), dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        g = scores[(scores["weather_gt"] == c) & ~scores["stratum"].str.startswith("bdd_train")]
        ok, bad = g[~g["fail"].astype(bool)], g[g["fail"].astype(bool)]
        ax.scatter(ok["danger_est"], ok["trust"], s=5, alpha=0.35, color="#2f6fb3", label="detector OK")
        ax.scatter(bad["danger_est"], bad["trust"], s=5, alpha=0.5, color="#c7702a", label="detector failed")
        ax.axhline(0.5, color="#888", linewidth=0.7, linestyle="--")
        ax.axvline(0.4, color="#888", linewidth=0.7, linestyle="--")
        ax.set_title(f"{c} (n={len(g)})", fontsize=9)
        ax.set_xlabel("danger (from detections)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("trust")
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Trust x danger. Bottom-left = low trust, low apparent danger: the corner a fused score cannot see", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_severity(metrics_off: pd.DataFrame, trust_res: pd.DataFrame | None, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    groups = {"synthetic fog": ["bdd_val_clear", "syn_fog_1", "syn_fog_3", "syn_fog_5"],
              "synthetic frost": ["bdd_val_clear", "syn_frost_1", "syn_frost_3", "syn_frost_5"],
              "synthetic motion blur": ["bdd_val_clear", "syn_motion_blur_1", "syn_motion_blur_3", "syn_motion_blur_5"],
              "Cityscapes fog (real scenes)": ["cs_clear", "cs_fog_light", "cs_fog_medium", "cs_fog_dense"]}
    m = metrics_off.set_index("stratum")
    for name, strata in groups.items():
        present = [s for s in strata if s in m.index]
        if len(present) < 2:
            continue
        axes[0].plot(range(len(present)), m.loc[present, "fail_rate"], marker="o", label=name)
        if trust_res is not None:
            t = trust_res.set_index("stratum")
            ok = [s for s in present if s in t.index]
            axes[1].plot(range(len(ok)), t.loc[ok, "auroc"], marker="o", label=name)
    axes[0].set_title("detector failure rate vs severity (0 = clear)", fontsize=9)
    axes[1].set_title("trust AUROC vs severity", fontsize=9)
    for ax in axes:
        ax.set_xlabel("severity step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def enhancement_table(off: list[dict], on: list[dict]) -> pd.DataFrame:
    a = stratum_metrics(off).set_index("stratum")
    b = stratum_metrics(on).set_index("stratum")
    cols = ["map50", "person_recall", "precision", "fail_rate_person", "mean_conf_tp", "mean_conf_fp"]
    t = pd.concat({"off": a[cols], "on": b[cols]}, axis=1)
    for c in cols:
        t[("delta", c)] = t[("on", c)] - t[("off", c)]
    return t


def main(argv=None):
    a = parse_args(argv)
    start_log(RESULTS / "log_analysis.txt")
    FIG.mkdir(parents=True, exist_ok=True)
    off = list(read_jsonl(a.records))
    gap = danger_gap_table(off)
    gap.to_csv(a.out / "danger_gap.csv", index=False)
    print("[analysis] danger gap per stratum:\n" + gap.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_gap(gap, FIG / "danger_gap.png")
    # weight sensitivity on ground-truth boxes of the held-out strata
    frames = [{"boxes": r["gt_boxes"], "hw": tuple(r["image_hw"]), "weather": r["weather_gt"],
               "fog_level": (r.get("extra") or {}).get("fog_level", "none")} for r in off if not r["stratum"].startswith("bdd_train")]
    rng = np.random.default_rng(0)
    if len(frames) > 2000:
        frames = [frames[i] for i in sorted(rng.choice(len(frames), 2000, replace=False))]
    sens = DG.weight_sensitivity(frames)
    pd.DataFrame(sens).T.to_csv(a.out / "danger_weight_sensitivity.csv")
    worst = min(v["tercile_agreement"] for v in sens.values())
    print(f"[analysis] weight sensitivity over {len(frames)} frames: minimum tercile agreement {worst:.3f}, minimum Spearman "
          f"{min(v['spearman'] for v in sens.values()):.3f} (1.0 = ranking unchanged)")
    metrics_off = stratum_metrics(off)
    trust_res = None
    if a.trust_scores.exists():
        scores = pd.read_csv(a.trust_scores)
        plot_quadrant(scores, FIG / "quadrant.png")
        q = scores[~scores["stratum"].str.startswith("bdd_train")].copy()
        q["quadrant"] = [DG.quadrant(t, d) for t, d in zip(q["trust"], q["danger_est"])]
        qt = q.groupby(["weather_gt", "quadrant"]).size().unstack(fill_value=0)
        qt.to_csv(a.out / "quadrant_counts.csv")
        print("[analysis] quadrant counts per ground-truth weather:\n" + qt.to_string())
        tr = RESULTS / "trust" / "trust_results.csv"
        trust_res = pd.read_csv(tr) if tr.exists() else None
    plot_severity(metrics_off, trust_res, FIG / "severity_sweep.png")
    if a.records_on and a.records_on.exists():
        on = list(read_jsonl(a.records_on))
        t = enhancement_table(off, on)
        t.to_csv(a.out / "enhancement_on_vs_off.csv")
        print("[analysis] enhancement ON vs OFF (delta = on - off):\n" + t[["off", "on", "delta"]].to_string(float_format=lambda x: f"{x:.3f}"))
        infl = t[(t[("delta", "mean_conf_tp")] > 0.02) & (t[("delta", "person_recall")] <= 0.0)]
        if len(infl):
            print("[analysis] WARNING confidence rose without recall improving (miscalibration) in: " + ", ".join(infl.index))
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        x = np.arange(len(t))
        ax.bar(x - 0.2, t[("off", "person_recall")], 0.4, label="enhancement OFF", color="#2f6fb3")
        ax.bar(x + 0.2, t[("on", "person_recall")], 0.4, label="enhancement ON", color="#c7702a")
        ax.set_xticks(x, t.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("pedestrian recall")
        ax.legend(fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "enhancement_on_vs_off.png", bbox_inches="tight")
        plt.close(fig)
    save_json({"records": str(a.records), "records_on": str(a.records_on), "figures": sorted(p.name for p in FIG.glob("*.png"))},
              a.out / "analysis_info.json")
    print(f"[analysis] figures -> {FIG}")


if __name__ == "__main__":
    main()
