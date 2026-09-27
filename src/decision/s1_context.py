"""
S1-level contextual decision layer on top of a frozen base matcher.

Leakage control:
- Every choice (rule parameters, second-stage model type, its threshold)
  is made on TRAIN S1 entities, using out-of-fold base scores
  (``src.decision.oof``) and, for the second stage, out-of-fold
  second-stage probabilities (same S1 folds).
- Validation uses the frozen base model's saved predictions and is only
  scored once per strategy. The validation threshold sweep around the base
  threshold is descriptive (that threshold was itself chosen on validation).

Context features use only scores and existing pairwise features of the
other candidates of the same S1 — never labels.

    python -m src.decision.s1_context outputs/experiments/xgb_skeleton \\
        outputs/features/sample20k_skeleton --oof outputs/experiments/s1_context \\
        --compare outputs/experiments/xgb_rarity
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.blocking.entity_cache import load_partition
from src.evaluation.name_miss_analysis import script_profile
from src.models.xgb_matcher import code_version, s1_metrics, true_link_counts


ROOT = Path(__file__).resolve().parents[2]

KEY = ["s1_id", "target_id", "target_source"]
LEVELS = (0.90, 0.95, 0.98, 0.99, 0.991)
HIGH = 0.99
EVIDENCE = [
    "address_token_set_ratio", "name_ratio", "address_number_set_jaccard",
    "translit_skeleton_ratio", "target_name_frequency",
]
ROW_FEATURES = EVIDENCE + ["house_number_match", "target_address_missing"]

# Second stage only re-decides candidates the base model scores at least
# this high; everything below stays "no match".
SECOND_STAGE_FLOOR = 0.5


# ----------------------------------------------------------------------
# Context features (no labels)
# ----------------------------------------------------------------------


def logit(p: np.ndarray | pd.Series) -> np.ndarray:

    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-aligned S1 context for candidates ``df`` (columns: s1_id,
    target_id, target_source, score and EVIDENCE). Uses only scores and
    features of the other candidates of the same S1.
    """

    order = np.lexsort((df["target_id"].to_numpy(), df["target_source"].to_numpy(),
                        -df["score"].to_numpy(), df["s1_id"].to_numpy()))
    d = df.iloc[order]
    g = d.groupby("s1_id", sort=False)

    ctx = pd.DataFrame(index=d.index)
    ctx["s1_candidate_count"] = g["score"].transform("size")
    for level in LEVELS:
        ctx[f"s1_count_ge_{level:g}"] = (d["score"] >= level).groupby(d["s1_id"], sort=False).transform("sum")

    ctx["candidate_score_rank"] = g.cumcount() + 1
    ctx["candidate_score_percentile"] = 1 - (ctx["candidate_score_rank"] - 1) / ctx["s1_candidate_count"]
    ctx["is_top"] = (ctx["candidate_score_rank"] == 1).astype(np.int8)

    top = g["score"].transform("first")
    second = d["score"].where(ctx["candidate_score_rank"] == 2).groupby(d["s1_id"], sort=False).transform("max").fillna(0.0)
    ctx["s1_top_score"] = top
    ctx["s1_second_score"] = second
    ctx["s1_score_margin"] = top - second
    ctx["candidate_score_minus_top"] = d["score"] - top
    ctx["candidate_score_minus_second"] = d["score"] - second

    high = d["score"] >= HIGH
    for code, name in ((2, "s2"), (3, "s3")):
        ctx[f"s1_high_{name}_count"] = (high & (d["target_source"] == code)).groupby(d["s1_id"], sort=False).transform("sum")
    ctx["other_source_high_count"] = np.where(
        d["target_source"] == 2, ctx["s1_high_s3_count"], ctx["s1_high_s2_count"]
    )
    ctx["other_candidates_high_count"] = ctx[f"s1_count_ge_{HIGH:g}"] - high.astype(int)
    top_source = d["target_source"].where(ctx["is_top"] == 1).groupby(d["s1_id"], sort=False).transform("max")
    ctx["top_is_s3"] = (top_source == 3).astype(np.int8)

    for col in EVIDENCE:
        top_value = d[col].where(ctx["is_top"] == 1).groupby(d["s1_id"], sort=False).transform("max")
        value = np.log1p(d[col]) if col == "target_name_frequency" else d[col]
        top_value = np.log1p(top_value) if col == "target_name_frequency" else top_value
        ctx[f"diff_top_{col}"] = value - top_value

    return ctx.reindex(df.index)


# ----------------------------------------------------------------------
# Decision strategies
# ----------------------------------------------------------------------


def decide_threshold(df: pd.DataFrame, t: float) -> np.ndarray:

    return (df["score"] >= t).to_numpy()


def decide_single_stricter(df: pd.DataFrame, t: float, t_single: float, contradiction_only: bool = False) -> np.ndarray:
    """
    Threshold ``t``; an S1 whose ONLY candidate at or above ``t`` is this
    one needs ``t_single`` instead (optionally only when the pair has an
    address contradiction).
    """

    pred = df["score"] >= t
    n = pred.groupby(df["s1_id"]).transform("sum")
    lone = pred & (n == 1)

    if contradiction_only:
        lone &= contradiction(df)

    return (pred & ~(lone & (df["score"] < t_single))).to_numpy()


def contradiction(df: pd.DataFrame) -> pd.Series:

    return (
        (df["house_number_match"] == 0)
        | (df["address_number_set_jaccard"] < 1)
        | (df["target_address_missing"] == 1)
    )


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


class Evaluator:
    """Challenge metrics for predictions over a fixed set of S1 entities."""

    def __init__(self, df: pd.DataFrame, s1_ids: np.ndarray):
        self.s1 = np.sort(s1_ids)
        self.n_true = true_link_counts(self.s1)
        self.index = np.searchsorted(self.s1, df["s1_id"].to_numpy())
        self.label = df["label"].to_numpy()

    def f05(self, pred: np.ndarray) -> float:
        return s1_metrics(self.index, self.label, pred, self.n_true)["macro_f05"]

    def metrics(self, pred: np.ndarray) -> dict:
        m = s1_metrics(self.index, self.label, pred, self.n_true)
        n_pred = m["_n_pred"]
        no_match = self.n_true == 0
        return {
            "macro_f05": m["macro_f05"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "fp": int((pred & (self.label == 0)).sum()),
            "fn_in_candidates": int((~pred & (self.label == 1)).sum()),
            "micro_precision": m["micro_precision"],
            "micro_recall": m["micro_recall"],
            "s1_with_predictions": int((n_pred > 0).sum()),
            "s1_without_predictions": int((n_pred == 0).sum()),
            "no_match_s1_predicted": int((no_match & (n_pred > 0)).sum()),
            "f05_matched_s1": m["f05_matched"],
            "f05_no_match_s1": m["f05_singletons"],
        }


def best_by(evaluator: Evaluator, candidates: list[tuple[dict, np.ndarray]]) -> tuple[dict, float]:

    scored = [(params, evaluator.f05(pred)) for params, pred in candidates]
    return max(scored, key=lambda x: x[1])


# ----------------------------------------------------------------------
# Second stage
# ----------------------------------------------------------------------


def design(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """Compact second-stage inputs (NaN kept; filled per model below)."""

    X = pd.DataFrame(index=df.index)
    X["logit_score"] = logit(df["score"])
    X["logit_top"] = logit(ctx["s1_top_score"])
    X["logit_second"] = logit(ctx["s1_second_score"].clip(lower=1e-7))
    X["logit_minus_top"] = X["logit_score"] - X["logit_top"]
    X["is_top"] = ctx["is_top"]
    X["candidate_score_rank"] = np.minimum(ctx["candidate_score_rank"], 20)
    X["log_s1_candidate_count"] = np.log1p(ctx["s1_candidate_count"])
    for level in (0.95, 0.99, 0.991):
        X[f"log_count_ge_{level:g}"] = np.log1p(ctx[f"s1_count_ge_{level:g}"])
    X["other_candidates_high_count"] = np.log1p(ctx["other_candidates_high_count"])
    X["other_source_high_count"] = np.log1p(ctx["other_source_high_count"])
    X["top_is_s3"] = ctx["top_is_s3"]
    X["candidate_is_s3"] = (df["target_source"] == 3).astype(np.int8)
    for col in EVIDENCE:
        X[f"diff_top_{col}"] = ctx[f"diff_top_{col}"]
    X["log_target_name_frequency"] = np.log1p(df["target_name_frequency"])
    X["target_address_missing"] = df["target_address_missing"]
    X["address_contradiction"] = contradiction(df).astype(np.int8)

    return X.astype(np.float32)


def lr_model(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))


def tree_model(seed: int):
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
        n_jobs=-1, random_state=seed, eval_metric="logloss",
    )


def lr_indicator_columns(columns) -> list[str]:
    """Fixed set of columns that may be NaN (same for every matrix)."""

    return [c for c in columns if c.startswith("diff_top_") or c == "log_target_name_frequency"]


def lr_matrix(X: pd.DataFrame) -> np.ndarray:
    """Logistic regression can't take NaN: fill with 0 and add indicators."""

    filled = X.fillna(0.0)
    for c in lr_indicator_columns(X.columns):
        filled[f"{c}_missing"] = X[c].isna().astype(np.float32)
    return filled.to_numpy(np.float32)


def fit_predict(kind: str, X_fit, y_fit, X_apply, seed: int) -> np.ndarray:

    if kind == "logistic":
        model = lr_model(seed)
        model.fit(lr_matrix(X_fit), y_fit)
        return model.predict_proba(lr_matrix(X_apply))[:, 1], model

    model = tree_model(seed)
    model.fit(X_fit.to_numpy(np.float32), y_fit)
    return model.predict_proba(X_apply.to_numpy(np.float32))[:, 1], model


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


def read_split_features(feature_dir: Path, split: int, columns: list[str]) -> pd.DataFrame:

    manifest = json.loads((feature_dir / "manifest.json").read_text())
    parts = [
        pq.read_table(feature_dir / p["file"], columns=KEY + ["split", "country"] + columns)
        .to_pandas().query(f"split == {split}").drop(columns="split")
        for p in manifest["parts"]
    ]
    return pd.concat(parts, ignore_index=True)


def target_script_class(df: pd.DataFrame) -> pd.Series:
    """'latin' / 'non-latin' / 'mixed' for each target name (Indic detection)."""

    indic = re.compile(r"[ऀ-෿]")
    out = pd.Series("latin", index=df.index, dtype=object)
    for (src, country), g in df.groupby(["target_source", "country"]):
        names = load_partition({2: "source2", 3: "source3"}[src], country, ["norm_name"]) \
            .set_index("id")["norm_name"].fillna("").reindex(g["target_id"].unique())
        cls = pd.Series("latin", index=names.index, dtype=object)
        non_latin = names[names.str.contains(indic)]
        cls[non_latin.index] = [script_profile(n)[1] for n in non_latin]
        out[g.index] = g["target_id"].map(cls).to_numpy()
    return out


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--oof", type=Path, required=True, help="Directory with oof_train_predictions.parquet (also the output dir).")
    parser.add_argument("--compare", type=Path, required=True)
    args = parser.parse_args()

    exp_out = args.oof
    diag_out = ROOT / "outputs" / "diagnostics" / "s1_context"
    diag_out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    meta = json.loads((args.experiment / "metadata.json").read_text())
    base_thr = meta["selected_threshold"]
    seed = meta["seed"]

    # --- Load TRAIN (OOF) and VALIDATION (frozen) ------------------------
    oof = pq.read_table(exp_out / "oof_train_predictions.parquet").to_pandas()
    train = oof.merge(read_split_features(args.feature_dir, 0, ROW_FEATURES), on=KEY, validate="one_to_one")

    val = pq.read_table(args.experiment / "validation_predictions.parquet").to_pandas().drop(columns="predicted")
    val = val.merge(read_split_features(args.feature_dir, 1, ROW_FEATURES), on=KEY, validate="one_to_one")

    split = pq.read_table(args.feature_dir / "split.parquet").to_pandas()
    ev_train = Evaluator(train, split.loc[split["split"] == 0, "s1_id"].to_numpy())
    ev_val = Evaluator(val, split.loc[split["split"] == 1, "s1_id"].to_numpy())

    ctx_train = context_features(train)
    ctx_val = context_features(val)

    # --- Score distributions: OOF (train) vs frozen (validation) ---------
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    dist = pd.DataFrame({
        "OOF train positives": train.loc[train["label"] == 1, "score"].quantile(qs),
        "frozen val positives": val.loc[val["label"] == 1, "score"].quantile(qs),
        "OOF train negatives": train.loc[train["label"] == 0, "score"].quantile(qs),
        "frozen val negatives": val.loc[val["label"] == 0, "score"].quantile(qs),
    })
    dist.index.name = "quantile"
    dist.to_csv(diag_out / "score_distribution.tsv", sep="\t", float_format="%.6f")

    # --- Threshold sweeps --------------------------------------------------
    grid = [0.985, 0.988, 0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997]
    sweep = pd.DataFrame([
        {"threshold": t, "train_oof_macro_f05": ev_train.f05(decide_threshold(train, t)),
         **{f"val_{k}": v for k, v in ev_val.metrics(decide_threshold(val, t)).items()}}
        for t in grid
    ])
    sweep.to_csv(diag_out / "threshold_comparison.tsv", sep="\t", index=False, float_format="%.6f")
    t_train = float(sweep.loc[sweep["train_oof_macro_f05"].idxmax(), "threshold"])

    # --- Simple S1-context rules (parameters chosen on TRAIN OOF) --------
    rule_candidates = {"single_stricter": [], "single_stricter_if_contradiction": []}
    for t in grid:
        for ts in [0.993, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9999]:
            if ts <= t:
                continue
            rule_candidates["single_stricter"].append(
                ({"t": t, "t_single": ts}, decide_single_stricter(train, t, ts)))
            rule_candidates["single_stricter_if_contradiction"].append(
                ({"t": t, "t_single": ts}, decide_single_stricter(train, t, ts, contradiction_only=True)))
    rules = {name: best_by(ev_train, cands) for name, cands in rule_candidates.items()}

    # --- Second stage (model + threshold chosen with train OOF) ----------
    X_train_all = design(train, ctx_train)
    X_val_all = design(val, ctx_val)
    tr_mask = (train["score"] >= SECOND_STAGE_FLOOR).to_numpy()
    va_mask = (val["score"] >= SECOND_STAGE_FLOOR).to_numpy()
    y_tr = train["label"].to_numpy()

    stage_grid = [round(x, 3) for x in np.concatenate([np.arange(0.3, 0.95, 0.05), [0.95, 0.97, 0.98, 0.99]])]
    second = {}
    for kind in ("logistic", "tree"):
        prob = np.zeros(len(train), dtype=np.float64)
        for fold in sorted(train["fold"].unique()):
            fit = tr_mask & (train["fold"] != fold).to_numpy()
            app = tr_mask & (train["fold"] == fold).to_numpy()
            p, _ = fit_predict(kind, X_train_all[fit], y_tr[fit], X_train_all[app], seed)
            prob[app] = p
        scored = [(t, ev_train.f05(prob >= t)) for t in stage_grid]
        t_best, f_best = max(scored, key=lambda x: x[1])
        second[kind] = {"threshold": t_best, "train_oof_macro_f05": f_best, "oof_prob": prob}
        print(f"[second stage] {kind}: train-OOF macro F0.5 {f_best:.4f} at {t_best}", flush=True)

    chosen_kind = max(second, key=lambda k: second[k]["train_oof_macro_f05"])
    chosen_thr = second[chosen_kind]["threshold"]
    val_prob = np.zeros(len(val))
    val_prob[va_mask], final_model = fit_predict(
        chosen_kind, X_train_all[tr_mask], y_tr[tr_mask], X_val_all[va_mask], seed
    )

    # --- Strategies on VALIDATION ----------------------------------------
    strategies = {
        f"A. base threshold {base_thr:g} (chosen on validation)": decide_threshold(val, base_thr),
        f"B'. base threshold {t_train:g} (chosen on train OOF)": decide_threshold(val, t_train),
    }
    for t in (0.990, 0.991, 0.992, 0.993, 0.994):
        strategies[f"B. threshold {t:g} (descriptive)"] = decide_threshold(val, t)
    for name, (params, f_train) in rules.items():
        strategies[f"C. rule {name} t={params['t']:g} t_single={params['t_single']:g} (train OOF {f_train:.4f})"] = \
            decide_single_stricter(val, params["t"], params["t_single"], contradiction_only=name.endswith("contradiction"))
    strategies[f"D. second stage {chosen_kind} p>={chosen_thr:g} (train OOF {second[chosen_kind]['train_oof_macro_f05']:.4f})"] = val_prob >= chosen_thr

    prev = pq.read_table(args.compare / "validation_predictions.parquet").to_pandas()
    prev = val[KEY].merge(prev[KEY + ["predicted"]], on=KEY, how="left", validate="one_to_one")
    strategies["xgb_rarity (reference)"] = prev["predicted"].to_numpy() == 1

    comparison = pd.DataFrame([{"strategy": k, **ev_val.metrics(v)} for k, v in strategies.items()])
    comparison.to_csv(exp_out / "decision_comparison.tsv", sep="\t", index=False, float_format="%.6f")

    # Best contextual strategy = best TRAIN-selected contextual candidate
    # (rules or second stage) judged on TRAIN OOF, not on validation.
    ctx_options = {
        f"C. rule {n}": (f, lambda p=p, n=n: decide_single_stricter(val, p["t"], p["t_single"], n.endswith("contradiction")))
        for n, (p, f) in rules.items()
    }
    ctx_options["D. second stage"] = (second[chosen_kind]["train_oof_macro_f05"], lambda: val_prob >= chosen_thr)
    best_ctx_name = max(ctx_options, key=lambda k: ctx_options[k][0])
    best_pred = ctx_options[best_ctx_name][1]()
    base_pred = strategies[f"A. base threshold {base_thr:g} (chosen on validation)"]

    # --- Segments (validation) -------------------------------------------
    val["target_script"] = target_script_class(val)
    s1_pred_count = pd.Series(base_pred).groupby(val["s1_id"].to_numpy()).transform("sum").to_numpy()
    no_match = val["s1_id"].map(pd.Series(ev_val.n_true, index=ev_val.s1)).to_numpy() == 0
    segments = {
        "Latin -> Latin": (val["target_script"] == "latin").to_numpy(),
        "Latin -> Indian-script": (val["target_script"] == "non-latin").to_numpy(),
        "cross-script & address token-set >= 0.75": ((val["target_script"] != "latin") & (val["address_token_set_ratio"] >= 0.75)).to_numpy(),
        "rare target name (frequency 1)": (val["target_name_frequency"] == 1).to_numpy(),
        "target address missing": (val["target_address_missing"] == 1).to_numpy(),
        "conflicting address numbers": (val["address_number_set_jaccard"] < 1).to_numpy(),
        "single-prediction S1 (base)": s1_pred_count == 1,
        "multi-prediction S1 (base)": s1_pred_count >= 2,
        "S1 with no true match": no_match,
    }
    y = val["label"].to_numpy()
    seg_rows = []
    for seg, m in segments.items():
        for name, pred in (("base 0.991", base_pred), (best_ctx_name, best_pred)):
            tp, fp, fn = int((pred & m & (y == 1)).sum()), int((pred & m & (y == 0)).sum()), int((~pred & m & (y == 1)).sum())
            seg_rows.append({"segment": seg, "strategy": name, "rows": int(m.sum()), "positives": int((m & (y == 1)).sum()),
                             "tp": tp, "fp": fp, "fn": fn,
                             "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(exp_out / "segment_comparison.tsv", sep="\t", index=False, float_format="%.4f")

    # --- Context feature analysis (validation, candidates with score >= 0.9) --
    hi = (val["score"] >= 0.9).to_numpy()
    cfa = []
    for col in ctx_val.columns:
        v = ctx_val.loc[hi, col]
        cfa.append({"feature": col,
                    "positives p25/50/75": "/".join(f"{x:.3f}" for x in v[y[hi] == 1].quantile([.25, .5, .75])),
                    "negatives p25/50/75": "/".join(f"{x:.3f}" for x in v[y[hi] == 0].quantile([.25, .5, .75]))})
    cfa = pd.DataFrame(cfa)
    if chosen_kind == "logistic":
        names = list(X_val_all.columns) + [f"{c}_missing" for c in lr_indicator_columns(X_val_all.columns)]
        coefs = final_model[-1].coef_[0]
        weights = pd.DataFrame({"feature": names[:len(coefs)], "second_stage_weight": coefs})
    else:
        gain = final_model.get_booster().get_score(importance_type="gain")
        weights = pd.DataFrame({"feature": [X_val_all.columns[int(k[1:])] for k in gain], "second_stage_weight": list(gain.values())})
    cfa.to_csv(diag_out / "context_feature_analysis.tsv", sep="\t", index=False)
    weights.sort_values("second_stage_weight", key=np.abs, ascending=False).to_csv(
        diag_out / "second_stage_weights.tsv", sep="\t", index=False)

    # --- Error cases: rows where the best contextual strategy differs from base --
    changed = best_pred != base_pred
    err = pd.concat([val.loc[changed, KEY + ["label", "score", "country", "target_script"] + ROW_FEATURES].reset_index(drop=True),
                     ctx_val.loc[changed].reset_index(drop=True)], axis=1)
    err.insert(4, "base_predicted", base_pred[changed].astype(int))
    err.insert(5, "context_predicted", best_pred[changed].astype(int))
    err.insert(6, "second_stage_prob", val_prob[changed])
    err.to_csv(diag_out / "error_cases.tsv", sep="\t", index=False, float_format="%.5f")

    # --- Artifacts ----------------------------------------------------------
    pq.write_table(pa.Table.from_pandas(pd.concat([val[KEY].reset_index(drop=True), ctx_val.reset_index(drop=True)], axis=1),
                                        preserve_index=False), exp_out / "context_features.parquet", compression="zstd")
    pq.write_table(pa.table({**{c: val[c].to_numpy() for c in KEY + ["label", "score"]},
                             "second_stage_prob": val_prob.astype(np.float32),
                             "base_predicted": base_pred.astype(np.int8),
                             "predicted": best_pred.astype(np.int8)}),
                   exp_out / "best_strategy_predictions.parquet", compression="zstd")
    if chosen_kind == "tree":
        final_model.save_model(exp_out / "second_stage_model.json")
    else:
        weights.to_csv(exp_out / "second_stage_logistic_coefficients.tsv", sep="\t", index=False)

    metadata = {
        "base_experiment": str(args.experiment), "base_threshold": base_thr,
        "oof": json.loads((exp_out / "oof_metadata.json").read_text()),
        "train_selected_base_threshold": t_train,
        "rules_selected_on_train": {k: {"params": p, "train_oof_macro_f05": f} for k, (p, f) in rules.items()},
        "second_stage": {k: {"threshold": v["threshold"], "train_oof_macro_f05": v["train_oof_macro_f05"]} for k, v in second.items()},
        "second_stage_chosen": chosen_kind, "second_stage_floor": SECOND_STAGE_FLOOR,
        "second_stage_features": list(X_val_all.columns),
        "best_contextual_strategy (selected on train OOF)": best_ctx_name,
        "code": code_version(), "seconds": round(time.perf_counter() - started, 1),
    }
    (exp_out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))

    fmt = lambda x: f"{x:.4f}"
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 95)
    report = [
        "# S1-context decision experiment\n",
        f"Base: {args.experiment.name} (threshold {base_thr:g}). Train-OOF-selected base threshold: {t_train:g}.\n",
        "## Decision strategies (validation)\n```\n" + comparison.to_string(index=False, float_format=fmt) + "\n```\n",
        f"Best contextual strategy by TRAIN-OOF F0.5: **{best_ctx_name}**\n",
        "## Segments: base vs best contextual strategy (validation)\n```\n" + seg_df.to_string(index=False, float_format=fmt) + "\n```\n",
        "## Threshold sweep (train OOF vs validation)\n```\n" + sweep[["threshold", "train_oof_macro_f05", "val_macro_f05", "val_fp", "val_fn_in_candidates", "val_no_match_s1_predicted"]].to_string(index=False, float_format=fmt) + "\n```\n",
        "## Score distributions (OOF train vs frozen validation)\n```\n" + dist.to_string(float_format=lambda x: f"{x:.5f}") + "\n```\n",
        "## Second-stage weights (chosen model)\n```\n" + weights.sort_values("second_stage_weight", key=np.abs, ascending=False).head(15).to_string(index=False, float_format=fmt) + "\n```\n",
    ]
    (diag_out / "summary.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
