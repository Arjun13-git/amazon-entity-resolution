"""
False-positive analysis of a matcher, focused on the transliterated
skeleton feature and on S1 entities that have no true match.

Analysis only: predictions, features and models are read, never changed.
The saved model is used for one *counterfactual* re-scoring:
``translit_skeleton_ratio`` is replaced by ``translit_name_ratio`` (what
the skeleton would contribute if it added nothing beyond plain
transliteration). A false positive whose counterfactual score falls
below the threshold is attributed to the skeleton.

    python -m src.evaluation.fp_skeleton_analysis \\
        outputs/experiments/xgb_skeleton outputs/features/sample20k_skeleton \\
        --compare outputs/experiments/xgb_rarity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from src.evaluation.error_analysis import KEY, load_validation, with_text
from src.evaluation.name_miss_analysis import script_profile
from src.models.xgb_matcher import true_link_counts


ROOT = Path(__file__).resolve().parents[2]

FREQ_EDGES = [0, 1.5, 2.5, 5.5, 20.5, np.inf]
FREQ_LABELS = ["1", "2", "3-5", "6-20", ">20"]


# ----------------------------------------------------------------------
# Pure helpers (unit tested)
# ----------------------------------------------------------------------


def s1_score_summary(s1_id: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """
    Per S1: candidate count, top and second-best score, margin and number of
    predictions. ``second_score`` is 0 when the S1 has a single candidate.
    """

    df = pd.DataFrame({"s1_id": s1_id, "score": score, "predicted": predicted})
    df = df.sort_values(["s1_id", "score"], ascending=[True, False])
    rank = df.groupby("s1_id").cumcount()

    top = df[rank == 0].set_index("s1_id")["score"]
    second = df[rank == 1].set_index("s1_id")["score"].reindex(top.index).fillna(0.0)

    return pd.DataFrame(
        {
            "candidate_count": df.groupby("s1_id").size(),
            "top_score": top,
            "second_score": second,
            "score_margin": top - second,
            "n_predicted": df.groupby("s1_id")["predicted"].sum(),
        }
    )


def best_name(df: pd.DataFrame) -> pd.Series:

    return np.fmax(df["name_ratio"], df["translit_name_ratio"])


# (name, rule) — first matching rule wins. Rules only use pairwise
# features and the target's script class.
CATEGORY_RULES = [
    ("transliteration collision (cross-script)",
     lambda d: d["cross_script"] & (d["translit_skeleton_ratio"] >= 0.7) & (d["name_ratio"] < 0.5)),
    ("skeleton collision (Latin, spelling differs)",
     lambda d: ~d["cross_script"] & (d["translit_skeleton_ratio"] >= 0.8) & (best_name(d) < 0.7)),
    ("near-duplicate record: name & address ≥0.9, numbers agree",
     lambda d: (best_name(d) >= 0.9) & (d["address_token_set_ratio"] >= 0.9)
     & ~(d["address_number_set_jaccard"] < 1)),
    ("address near-duplicate, numbers conflict",
     lambda d: (d["address_token_set_ratio"] >= 0.9) & (d["address_number_set_jaccard"] < 1)),
    ("same/near name, weak or missing address — common name (freq > 5)",
     lambda d: (best_name(d) >= 0.9) & ~(d["address_token_set_ratio"] >= 0.75) & (d["target_name_frequency"] > 5)),
    ("same/near name, weak or missing address — rare name (freq ≤ 5)",
     lambda d: (best_name(d) >= 0.9) & ~(d["address_token_set_ratio"] >= 0.75)),
    ("same/near address, different name",
     lambda d: (d["address_token_set_ratio"] >= 0.9) & (best_name(d) < 0.9)),
    ("house+city collision, weak name & address",
     lambda d: (d["house_city_match"] == 1) & (best_name(d) < 0.9) & (d["address_token_set_ratio"] < 0.9)),
    ("partial evidence on both name and address",
     lambda d: (best_name(d) >= 0.7) & (d["address_token_set_ratio"] >= 0.75)),
]


def categorize(df: pd.DataFrame) -> pd.Series:
    """Exclusive primary category per row; 'other' when no rule matches."""

    out = pd.Series("other", index=df.index, dtype=object)
    free = pd.Series(True, index=df.index)

    for name, rule in CATEGORY_RULES:
        hit = free & rule(df).fillna(False).astype(bool)
        out[hit] = name
        free &= ~hit

    return out


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(score of a random positive > random negative), ties counted half."""

    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if not len(positive) or not len(negative):
        return float("nan")

    values = np.concatenate([positive, negative])
    ranks = pd.Series(values).rank(method="average").to_numpy()
    r_pos = ranks[: len(positive)].sum()

    return float((r_pos - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


def counterfactual_scores(experiment: Path, df: pd.DataFrame) -> np.ndarray:
    """Model scores with translit_skeleton_ratio := translit_name_ratio."""

    features = json.loads((experiment / "features.json").read_text())["features"]
    model = xgb.XGBClassifier()
    model.load_model(experiment / "model.json")

    X = df[features].to_numpy(dtype=np.float32, copy=True)
    X[:, features.index("translit_skeleton_ratio")] = df["translit_name_ratio"].to_numpy(np.float32)

    return model.predict_proba(X)[:, 1].astype(np.float32)


def quantiles(series: pd.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> str:

    s = series.dropna()
    if s.empty:
        return "—"
    return "/".join(f"{v:.3f}" for v in s.quantile(list(qs)))


def rate(mask: pd.Series) -> float:

    return float(mask.mean()) if len(mask) else float("nan")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--compare", type=Path, required=True)
    args = parser.parse_args()

    out = ROOT / "outputs" / "diagnostics" / f"error_analysis_{args.experiment.name}"
    out.mkdir(parents=True, exist_ok=True)

    model_features = json.loads((args.experiment / "features.json").read_text())["features"]
    df, thr = load_validation(args.experiment, args.feature_dir, extra_features=model_features)
    prev = pq.read_table(args.compare / "validation_predictions.parquet").to_pandas()
    prev_thr = json.loads((args.compare / "metadata.json").read_text())["selected_threshold"]
    df = df.merge(
        prev[KEY + ["score", "predicted"]].rename(columns={"score": "prev_score", "predicted": "prev_predicted"}),
        on=KEY, validate="one_to_one",
    )

    # Recompute the current prediction from the score so both runs use
    # their own selected threshold consistently.
    df["predicted"] = (df["score"] >= thr).astype(np.int8)
    df["cf_score"] = counterfactual_scores(args.experiment, df)
    df["cf_predicted"] = (df["cf_score"] >= thr).astype(np.int8)

    split = pq.read_table(args.feature_dir / "split.parquet").to_pandas()
    val_s1 = np.sort(split.loc[split["split"] == 1, "s1_id"].to_numpy())
    n_true = pd.Series(true_link_counts(val_s1), index=val_s1)
    df["s1_true_links"] = df["s1_id"].map(n_true)

    s1 = s1_score_summary(df["s1_id"].to_numpy(), df["score"].to_numpy(), df["predicted"].to_numpy())
    s1["true_links"] = n_true.reindex(s1.index)
    s1["tp"] = df[(df["predicted"] == 1) & (df["label"] == 1)].groupby("s1_id").size().reindex(s1.index).fillna(0).astype(int)
    s1["fp"] = df[(df["predicted"] == 1) & (df["label"] == 0)].groupby("s1_id").size().reindex(s1.index).fillna(0).astype(int)
    df = df.join(s1[["candidate_count", "top_score", "second_score", "score_margin", "n_predicted"]], on="s1_id")

    fp = df[(df["predicted"] == 1) & (df["label"] == 0)].copy()
    tp = df[(df["predicted"] == 1) & (df["label"] == 1)].copy()
    prev_fp_keys = set(map(tuple, df.loc[(df["prev_predicted"] == 1) & (df["label"] == 0), KEY].to_numpy()))
    fp["new_fp"] = [tuple(k) not in prev_fp_keys for k in fp[KEY].to_numpy()]
    removed_fp = df[(df["prev_predicted"] == 1) & (df["label"] == 0) & (df["predicted"] == 0)]

    # Text + scripts for FP rows and for rows the previous model got wrong.
    text_rows = pd.concat([fp, removed_fp])
    text_rows = with_text(text_rows[~text_rows.index.duplicated()])
    prof = text_rows["target_name"].fillna("").map(script_profile)
    text_rows["target_script"] = [p[0] for p in prof]
    text_rows["cross_script"] = [p[1] != "latin" for p in prof]
    for col in ("s1", "target", "s1_name", "target_name", "s1_address", "target_address", "target_script", "cross_script"):
        fp[col] = text_rows.loc[fp.index, col]
        removed_fp = removed_fp.assign(**{col: text_rows.loc[removed_fp.index, col]})

    # Scripts for TPs are only needed as a class: Indic letters in the target name.
    tp_text = with_text(tp)
    tp["cross_script"] = [script_profile(n)[1] != "latin" for n in tp_text["target_name"].fillna("")]

    fp["category"] = categorize(fp)
    tp["category"] = categorize(tp)
    fp["skeleton_attributable"] = fp["cf_predicted"] == 0
    fp["crossed_threshold"] = fp["prev_score"] < prev_thr
    fp["skeleton_gain_over_translit"] = fp["translit_skeleton_ratio"] - fp["translit_name_ratio"]
    fp["no_match_s1"] = fp["s1_true_links"] == 0
    fp["multiple_predictions_in_s1"] = fp["n_predicted"] >= 2
    fp["freq_bucket"] = pd.cut(fp["target_name_frequency"], FREQ_EDGES, labels=FREQ_LABELS).astype(object).fillna("missing")
    tp["freq_bucket"] = pd.cut(tp["target_name_frequency"], FREQ_EDGES, labels=FREQ_LABELS).astype(object).fillna("missing")

    tp_attrib = int(((tp["cf_predicted"] == 0)).sum())

    # ---------------- 2. No-match S1 false positives ----------------
    cols_case = [
        "s1", "target", "target_source", "score", "prev_score", "cf_score", "category",
        "skeleton_attributable", "crossed_threshold", "new_fp",
        "candidate_count", "n_predicted", "top_score", "second_score", "score_margin",
        "s1_name", "target_name", "s1_address", "target_address", "target_script", "cross_script",
        "name_ratio", "translit_name_ratio", "translit_skeleton_ratio", "name_token_set_ratio",
        "target_name_frequency", "address_ratio", "address_token_set_ratio",
        "address_number_set_jaccard", "address_number_sequence_similarity", "secondary_number_match",
        "house_number_match", "city_match", "house_city_hit", "target_address_missing",
        "exact_name_hit", "name_char_hit", "address_char_hit", "n_channels", "name_char_rank", "address_char_rank",
    ]
    nomatch = fp[fp["no_match_s1"]].sort_values(["s1_id", "score"], ascending=[True, False])
    nomatch[cols_case].to_csv(out / "fp_no_match_s1.tsv", sep="\t", index=False)

    # ---------------- 3. FP comparison vs previous model --------------
    fp.sort_values("score", ascending=False)[cols_case + ["no_match_s1"]].to_csv(
        out / "fp_comparison.tsv", sep="\t", index=False
    )

    groups = {
        "all FP": fp,
        "new FP (not FP in previous model)": fp[fp["new_fp"]],
        "Latin-Latin FP": fp[~fp["cross_script"]],
        "no-match-S1 FP": nomatch,
    }
    comp_rows = []
    for name, g in groups.items():
        comp_rows.append({
            "group": name, "rows": len(g),
            "skeleton_attributable (counterfactual)": int(g["skeleton_attributable"].sum()),
            "crossed threshold vs previous model": int(g["crossed_threshold"].sum()),
            "skeleton > translit + 0.15": int((g["skeleton_gain_over_translit"] > 0.15).sum()),
            "median score / prev / counterfactual": f"{g['score'].median():.4f} / {g['prev_score'].median():.4f} / {g['cf_score'].median():.4f}",
            "median skeleton / translit / name": f"{g['translit_skeleton_ratio'].median():.2f} / {g['translit_name_ratio'].median():.2f} / {g['name_ratio'].median():.2f}",
        })
    comparison = pd.DataFrame(comp_rows)

    # ---------------- Categories (FP vs TP in the same region) --------
    cat_rows = []
    for name in [c for c, _ in CATEGORY_RULES] + ["other"]:
        f, t = fp[fp["category"] == name], tp[tp["category"] == name]
        cat_rows.append({
            "category": name,
            "fp_rows": len(f), "fp_pct": len(f) / max(len(fp), 1),
            "fp_no_match_s1": int(f["no_match_s1"].sum()),
            "fp_new": int(f["new_fp"].sum()),
            "fp_skeleton_attributable": int(f["skeleton_attributable"].sum()),
            "tp_rows_same_region": len(t),
            "precision_in_region": len(t) / max(len(t) + len(f), 1),
            "fp_median_score": f["score"].median(), "tp_median_score": t["score"].median(),
            "fp_median_margin": f["score_margin"].median(), "tp_median_margin": t["score_margin"].median(),
        })
    categories = pd.DataFrame(cat_rows)
    categories.to_csv(out / "category_counts.tsv", sep="\t", index=False)

    # ---------------- 4. Score margins -------------------------------
    s1["group"] = np.select(
        [(s1["true_links"] == 0) & (s1["n_predicted"] > 0),
         (s1["true_links"] == 0),
         (s1["true_links"] > 0) & (s1["fp"] > 0),
         (s1["true_links"] > 0) & (s1["tp"] > 0)],
        ["no-match S1, predicted (FP)", "no-match S1, correctly empty",
         "matched S1 with an FP", "matched S1, ≥1 TP, no FP"],
        default="matched S1, no prediction",
    )
    margin_rows = []
    for name, g in s1.groupby("group"):
        margin_rows.append({
            "group": name, "s1": len(g),
            "top_score p10/25/50/75/90": quantiles(g["top_score"]),
            "margin p10/25/50/75/90": quantiles(g["score_margin"]),
            "candidate_count p10/25/50/75/90": quantiles(g["candidate_count"]),
            "n_predicted mean": g["n_predicted"].mean(),
        })
    margins = pd.DataFrame(margin_rows)

    predicted_s1 = s1[s1["n_predicted"] > 0]
    fp_s1 = predicted_s1[predicted_s1["true_links"] == 0]
    ok_s1 = predicted_s1[(predicted_s1["true_links"] > 0) & (predicted_s1["fp"] == 0)]
    auc = {
        "top_score": rank_auc(ok_s1["top_score"], fp_s1["top_score"]),
        "score_margin": rank_auc(ok_s1["score_margin"], fp_s1["score_margin"]),
        "second_score (lower = cleaner)": rank_auc(-ok_s1["second_score"], -fp_s1["second_score"]),
        "candidate_count (lower)": rank_auc(-ok_s1["candidate_count"], -fp_s1["candidate_count"]),
    }
    # Row-level: FP rows vs TP rows by the S1's margin.
    row_auc = rank_auc(tp["score_margin"], fp["score_margin"])
    margins.to_csv(out / "score_margin_analysis.tsv", sep="\t", index=False)
    with open(out / "score_margin_analysis.tsv", "a") as fh:
        fh.write("\n# AUC: correctly-predicted matched S1 vs no-match S1 with a prediction (0.5 = no signal)\n")
        for k, v in auc.items():
            fh.write(f"# {k}\t{v:.3f}\n")
        fh.write(f"# row-level AUC of S1 margin, TP rows vs FP rows\t{row_auc:.3f}\n")

    # ---------------- 5. Name frequency ------------------------------
    prev_fp = df[(df["prev_predicted"] == 1) & (df["label"] == 0)]
    prev_fp_b = pd.cut(prev_fp["target_name_frequency"], FREQ_EDGES, labels=FREQ_LABELS).astype(object).fillna("missing")
    freq = pd.DataFrame({
        "TP": tp["freq_bucket"].value_counts(normalize=True),
        "FP (current)": fp["freq_bucket"].value_counts(normalize=True),
        "FP (previous model)": prev_fp_b.value_counts(normalize=True),
        "no-match-S1 FP": nomatch["freq_bucket"].value_counts(normalize=True),
        "skeleton-attributable FP": fp.loc[fp["skeleton_attributable"], "freq_bucket"].value_counts(normalize=True),
        "FP count": fp["freq_bucket"].value_counts(),
        "TP count": tp["freq_bucket"].value_counts(),
    }).reindex(FREQ_LABELS + ["missing"]).fillna(0)
    freq["precision_in_bucket"] = freq["TP count"] / (freq["TP count"] + freq["FP count"]).replace(0, np.nan)
    hi_fp, hi_tp = fp[fp["translit_skeleton_ratio"] >= 0.8], tp[tp["translit_skeleton_ratio"] >= 0.8]
    freq["precision_high_skeleton"] = [
        (hi_tp["freq_bucket"] == b).sum() / max((hi_tp["freq_bucket"] == b).sum() + (hi_fp["freq_bucket"] == b).sum(), 1)
        for b in freq.index
    ]
    freq.index.name = "target_name_frequency"
    freq.to_csv(out / "name_frequency_analysis.tsv", sep="\t")

    # ---------------- 6. Address evidence ----------------------------
    def evidence(g: pd.DataFrame) -> dict:
        return {
            "rows": len(g),
            "weak address (token-set < 0.75)": rate(g["address_token_set_ratio"] < 0.75),
            "conflicting house number (house_number_match = 0)": rate(g["house_number_match"] == 0),
            "numbers conflict (number-set Jaccard < 1)": rate(g["address_number_set_jaccard"] < 1),
            "secondary number mismatch": rate(g["secondary_number_match"] == 0),
            "different city (city_match = 0)": rate(g["city_match"] == 0),
            "identical address (address_ratio ≥ 0.99)": rate(g["address_ratio"] >= 0.99),
            "near-identical address (token-set ≥ 0.9)": rate(g["address_token_set_ratio"] >= 0.9),
            "target address missing": rate(g["target_address_missing"] == 1),
            "median address token-set": g["address_token_set_ratio"].median(),
            "median number-set Jaccard": g["address_number_set_jaccard"].median(),
        }

    ev = pd.DataFrame({
        "FP, skeleton ≥ 0.8": evidence(hi_fp),
        "TP, skeleton ≥ 0.8": evidence(hi_tp),
        "FP, skeleton-attributable": evidence(fp[fp["skeleton_attributable"]]),
        "FP, Latin-Latin": evidence(fp[~fp["cross_script"]]),
        "TP, Latin-Latin": evidence(tp[~tp["cross_script"]]),
        "FP, no-match S1": evidence(nomatch),
        "all FP": evidence(fp),
        "all TP": evidence(tp),
    })
    ev.to_csv(out / "address_evidence_analysis.tsv", sep="\t")

    # ---------------- 7. FP vs TP distributions ----------------------
    dist_cols = ["translit_skeleton_ratio", "name_ratio", "address_token_set_ratio", "address_number_set_jaccard",
                 "target_name_frequency", "score", "score_margin", "candidate_count", "n_channels"]
    dist = pd.DataFrame({
        c: {"TP p10/25/50/75/90": quantiles(tp[c]), "FP p10/25/50/75/90": quantiles(fp[c]),
            "no-match FP p10/25/50/75/90": quantiles(nomatch[c])}
        for c in dist_cols
    }).T

    # ---------------- Summary ----------------------------------------
    latin_fp_now = int((~fp["cross_script"]).sum())
    latin_fp_prev = int((~text_rows.loc[prev_fp.index.intersection(text_rows.index), "cross_script"]).sum()) \
        if len(prev_fp.index.intersection(text_rows.index)) == len(prev_fp) else None

    summary = {
        "threshold": thr, "previous_threshold": prev_thr,
        "no_match_s1_with_fp": int(nomatch["s1_id"].nunique()),
        "no_match_s1_total": int((n_true == 0).sum()),
        "no_match_fp_rows": int(len(nomatch)),
        "fp_rows": int(len(fp)), "previous_fp_rows": int(len(prev_fp)),
        "new_fp_rows": int(fp["new_fp"].sum()), "removed_fp_rows": int(len(removed_fp)),
        "fp_skeleton_attributable": int(fp["skeleton_attributable"].sum()),
        "tp_skeleton_attributable": tp_attrib,
        "latin_latin_fp": latin_fp_now,
        "latin_latin_fp_previous": latin_fp_prev,
        "auc": auc, "row_auc_margin": row_auc,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 70)
    fmt = lambda x: f"{x:.3f}"
    report = []
    report.append(f"# Error analysis: {args.experiment.name} (vs {args.compare.name})\n")
    report.append(f"Threshold {thr:g} (previous {prev_thr:g}). Validation: {len(val_s1):,} S1.\n")
    report.append("## Counts\n```\n" + json.dumps({k: v for k, v in summary.items() if k not in ('auc',)}, indent=2, default=float) + "\n```\n")
    report.append("## FP groups vs previous model (counterfactual = skeleton replaced by translit ratio)\n```\n" + comparison.to_string(index=False) + "\n```\n")
    report.append("## FP categories (exclusive; TP rows in the same rule region for comparison)\n```\n" + categories.to_string(index=False, float_format=fmt) + "\n```\n")
    report.append("## Score margins per S1\n```\n" + margins.to_string(index=False, float_format=fmt) + "\n```\n")
    report.append("AUC (correctly-predicted matched S1 vs no-match S1 with a prediction): "
                  + ", ".join(f"{k} {v:.3f}" for k, v in auc.items()) + f"; row-level margin AUC TP vs FP {row_auc:.3f}\n")
    report.append("## Target name frequency (shares; precision per bucket)\n```\n" + freq.to_string(float_format=fmt) + "\n```\n")
    report.append("## Address evidence\n```\n" + ev.to_string(float_format=fmt) + "\n```\n")
    report.append("## TP vs FP distributions\n```\n" + dist.to_string() + "\n```\n")
    (out / "summary.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
