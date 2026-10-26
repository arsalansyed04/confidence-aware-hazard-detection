"""Trust model: gradient-boosted trees over per-frame meta-features, trained on CLEAR-DAY frames only, tested on every
other stratum. Cross-condition transfer IS the experiment.

    python src_core/trust.py train --records results/core/records/yolo26m_off.jsonl
    python src_core/trust.py apply --records ... --model results/core/trust/trust_model.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS, read_jsonl, save_json, start_log  # noqa: E402
import features as F  # noqa: E402

TRAIN_STRATUM = "bdd_train_clear"
LABEL = "fail"           # primary failure label: frame F1 < 0.5 OR any missed pedestrian (see common.frame_metrics)
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--records", type=Path, required=True)
    t.add_argument("--out", type=Path, default=RESULTS / "trust")
    t.add_argument("--label", default=LABEL, choices=["fail", "fail_person", "fail_f1"])
    t.add_argument("--families", nargs="+", default=list(F.FAMILIES), choices=list(F.FAMILIES))
    t.add_argument("--seed", type=int, default=0)
    ap = sub.add_parser("apply")
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=RESULTS / "trust" / "trust_model.json")
    ap.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def table_from_records(path: Path) -> pd.DataFrame:
    rows = []
    for r in read_jsonl(path):
        row = {"frame_id": r["frame_id"], "stratum": r["stratum"], "weather_gt": r["weather_gt"], "source": r["source"],
               "fail": r["eval"]["fail"], "fail_person": r["eval"]["fail_person"], "fail_f1": r["eval"]["fail_f1"],
               "danger_est": r["scores"].get("danger_est"), "danger_true": r["scores"].get("danger_true")}
        row.update({k: r["meta"].get(k) for k in F.ALL_FEATURES})
        rows.append(row)
    df = pd.DataFrame(rows)
    for k in F.ALL_FEATURES:
        if k not in df.columns:
            df[k] = 0.0
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)
    return df


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def risk_coverage(trust: np.ndarray, fail: np.ndarray, coverages=COVERAGES) -> dict:
    """Residual failure rate when acting only on the most-trusted fraction of frames."""
    order = np.argsort(-trust)
    f = np.asarray(fail, bool)[order]
    out = {}
    for c in coverages:
        n = max(1, int(round(c * len(f))))
        out[f"fail_at_{int(c * 100)}"] = float(f[:n].mean())
    return out


def fit_xgb(X: np.ndarray, y: np.ndarray, seed: int = 0):
    import xgboost as xgb
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5, reg_lambda=2.0, scale_pos_weight=neg / pos, random_state=seed,
                              eval_metric="logloss", n_jobs=4, tree_method="hist")
    model.fit(X, y)
    return model


class TrustModel:
    """trust = 1 - P(failure). Serialised as XGBoost JSON + a feature list."""

    def __init__(self, model, features: list[str], label: str, train_stratum: str = TRAIN_STRATUM):
        self.model, self.features, self.label, self.train_stratum = model, features, label, train_stratum

    def trust(self, meta_rows: list[dict] | pd.DataFrame) -> np.ndarray:
        if isinstance(meta_rows, pd.DataFrame):
            X = meta_rows[self.features].to_numpy(dtype=np.float32)
        else:
            X = np.array([F.feature_vector(m, self.features) for m in meta_rows], dtype=np.float32)
        return 1.0 - self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        save_json({"features": self.features, "label": self.label, "train_stratum": self.train_stratum}, path.with_suffix(".meta.json"))

    @classmethod
    def load(cls, path: Path) -> "TrustModel":
        import xgboost as xgb
        path = Path(path)
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        meta = json.load(open(path.with_suffix(".meta.json"), encoding="utf-8"))
        return cls(m, meta["features"], meta["label"], meta.get("train_stratum", TRAIN_STRATUM))


def evaluate(tm: TrustModel, df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("stratum", sort=False):
        t = tm.trust(g)
        y = g[label].to_numpy(bool)
        # AUROC of FAILURE prediction: the score must rise with failure, so 1 - trust and 1 - mean confidence
        row = {"stratum": name, "frames": len(g), "fail_rate": float(y.mean()), "auroc": auroc(y, 1.0 - t),
               "auroc_conf_only": auroc(y, 1.0 - g["c_mean"].to_numpy()), "mean_trust": float(t.mean())}
        row.update(risk_coverage(t, y))
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_train(a):
    df = table_from_records(a.records)
    feats = [f for fam in a.families for f in F.FAMILIES[fam]]
    train = df[df["stratum"] == TRAIN_STRATUM]
    if train.empty:
        sys.exit(f"[trust] no '{TRAIN_STRATUM}' rows in {a.records}")
    y = train[a.label].to_numpy(bool)
    print(f"[trust] training on {len(train)} clear-day frames, failure rate {y.mean():.3f}, {len(feats)} features from {a.families}")
    if y.sum() < 10 or (~y).sum() < 10:
        sys.exit("[trust] too few failures or successes in the training stratum to learn anything")
    tm = TrustModel(fit_xgb(train[feats].to_numpy(np.float32), y, a.seed), feats, a.label)
    a.out.mkdir(parents=True, exist_ok=True)
    tm.save(a.out / "trust_model.json")
    # in-sample AUROC is reported only as a sanity ceiling; the honest numbers are the other strata
    res = evaluate(tm, df, a.label)
    res.loc[res["stratum"] == TRAIN_STRATUM, "stratum"] = TRAIN_STRATUM + " (train, in-sample)"
    res.to_csv(a.out / "trust_results.csv", index=False)
    print("[trust] AUROC of failure prediction per stratum (model trained on clear day only; conf_only = mean detection confidence baseline):")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # feature-family ablation: drop one family at a time
    abl = []
    held = df[df["stratum"] != TRAIN_STRATUM]
    for fam in a.families:
        keep = [f for ff in a.families if ff != fam for f in F.FAMILIES[ff]]
        m = TrustModel(fit_xgb(train[keep].to_numpy(np.float32), y, a.seed), keep, a.label)
        t = m.trust(held)
        abl.append({"dropped_family": fam, "auroc_adverse_pooled": auroc(held[a.label].to_numpy(bool), 1.0 - t)})
    full_t = tm.trust(held)
    abl.insert(0, {"dropped_family": "(none, full model)", "auroc_adverse_pooled": auroc(held[a.label].to_numpy(bool), 1.0 - full_t)})
    abl = pd.DataFrame(abl)
    abl.to_csv(a.out / "trust_ablation.csv", index=False)
    print("[trust] family ablation (pooled over all non-training strata):\n" + abl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    imp = pd.Series(tm.model.feature_importances_, index=feats).sort_values(ascending=False)
    imp.to_csv(a.out / "feature_importance.csv", header=["gain"])
    print("[trust] top features: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(8).items()))
    # per-frame trust scores for the analysis step
    df["trust"] = tm.trust(df)
    df[["frame_id", "stratum", "weather_gt", "source", "fail", "fail_person", "fail_f1", "trust", "danger_est", "danger_true", "c_mean"]].to_csv(
        a.out / "trust_scores.csv", index=False)
    plot_risk_coverage(df, a.label, a.out / "risk_coverage.png")
    plot_auroc(res, a.out / "trust_auroc.png")
    print(f"[trust] wrote trust_model.json, trust_results.csv, trust_ablation.csv, trust_scores.csv, figures -> {a.out}")


def cmd_apply(a):
    tm = TrustModel.load(a.model)
    df = table_from_records(a.records)
    df["trust"] = tm.trust(df)
    out = a.out or a.records.with_name(a.records.stem + "_trust.csv")
    df[["frame_id", "stratum", "weather_gt", "source", "fail", "fail_person", "fail_f1", "trust", "danger_est", "danger_true", "c_mean"]].to_csv(out, index=False)
    res = evaluate(tm, df, tm.label)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"[trust] wrote {out}")


def plot_risk_coverage(df: pd.DataFrame, label: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = np.linspace(0.3, 1.0, 15)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for name, g in df.groupby("stratum", sort=False):
        if name == TRAIN_STRATUM or len(g) < 30:
            continue
        order = np.argsort(-g["trust"].to_numpy())
        f = g[label].to_numpy(bool)[order]
        ax.plot(cov, [f[: max(1, int(round(c * len(f))))].mean() for c in cov], label=name, linewidth=1.6)
    ax.set_xlabel("coverage (fraction of frames acted on, most trusted first)")
    ax.set_ylabel("residual failure rate")
    ax.set_title("Risk-coverage: trust model trained on clear day only", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_auroc(res: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res[~res["stratum"].str.contains("train")]
    x = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.bar(x - 0.2, r["auroc"], 0.4, label="trust model (4 feature families + weather)", color="#2f6fb3")
    ax.bar(x + 0.2, r["auroc_conf_only"], 0.4, label="mean confidence only", color="#c7702a")
    ax.axhline(0.5, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, r["stratum"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("AUROC of failure prediction")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    a = parse_args(argv)
    start_log(RESULTS / "log_trust.txt")
    {"train": cmd_train, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
