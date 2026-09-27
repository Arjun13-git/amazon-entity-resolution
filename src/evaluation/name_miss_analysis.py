"""
Analysis of missed true links with low name similarity (cross-script /
multilingual diagnosis). Analysis only: nothing is retrained or changed.

Population (validation, candidate set):
- FN_LOW: label = 1, score < selected threshold, original name_ratio < 0.5
  (``best_below_0_5`` marks the stricter subset where the transliterated
  ratio is also < 0.5, i.e. best name similarity < 0.5).
- For separability, ALL validation candidates with name_ratio < 0.5 are
  profiled by script pair, transliterated similarity and address similarity.

Outputs go to ``outputs/diagnostics/name_miss_analysis_<experiment>/``.

    python -m src.evaluation.name_miss_analysis outputs/experiments/xgb_rarity \\
        outputs/features/sample20k_rarity
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.blocking.entity_cache import cache_path
from src.preprocessing.transliterate import transliterate_text


ROOT = Path(__file__).resolve().parents[2]

KEY = ["s1_id", "target_id", "target_source"]
FEATURES = [
    "name_ratio", "name_token_set_ratio", "name_token_jaccard",
    "translit_name_ratio", "translit_name_token_jaccard",
    "address_ratio", "address_token_set_ratio", "address_token_jaccard",
    "address_number_set_jaccard", "target_address_missing",
    "name_char_hit", "address_char_hit", "house_city_hit", "exact_name_hit",
]

SOURCE = {1: "source1", 2: "source2", 3: "source3"}
PREFIX = {1: "S1", 2: "S2", 3: "S3"}


def script_profile(text: str) -> tuple[str, str]:
    """
    (dominant script, class) of a name's letters.

    Script = first word of the Unicode character name (LATIN, DEVANAGARI...).
    Class: 'latin', 'non-latin', 'mixed' (>= 20% of letters in a second
    script), or 'none' (no letters).
    """

    counts: dict[str, int] = {}
    for ch in text or "":
        if unicodedata.category(ch).startswith("L"):
            script = unicodedata.name(ch, "UNKNOWN").split()[0]
            counts[script] = counts.get(script, 0) + 1

    if not counts:
        return "NONE", "none"

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    dominant = ranked[0][0]

    if len(ranked) > 1 and ranked[1][1] / total >= 0.2:
        return f"{dominant}+{ranked[1][0]}", "mixed"

    return dominant, "latin" if dominant == "LATIN" else "non-latin"


def fetch_names(ids_by_source: dict[int, np.ndarray]) -> dict[int, pd.DataFrame]:
    """Raw business name + normalized name/address for the given ids."""

    out = {}
    for code, ids in ids_by_source.items():
        if len(ids) == 0:
            continue
        norm = ds.dataset(cache_path(SOURCE[code])).to_table(
            columns=["id", "norm_name", "norm_address"],
            filter=ds.field("id").isin(ids.tolist()),
        ).to_pandas().set_index("id")
        raw = ds.dataset(ROOT / "outputs" / "parquet" / f"train_{SOURCE[code]}.parquet").to_table(
            columns=["entity_id", "business_name"],
            filter=ds.field("entity_id").isin([f"{PREFIX[code]}-{i}" for i in ids]),
        ).to_pandas()
        raw["id"] = raw["entity_id"].str[3:].astype("int64")
        out[code] = norm.join(raw.set_index("id")["business_name"])
    return out


def load(experiment: Path, feature_dir: Path) -> tuple[pd.DataFrame, float]:

    meta = json.loads((experiment / "metadata.json").read_text())
    pred = pq.read_table(experiment / "validation_predictions.parquet").to_pandas()

    manifest = json.loads((feature_dir / "manifest.json").read_text())
    parts = [
        pq.read_table(feature_dir / p["file"], columns=KEY + ["split"] + FEATURES)
        .to_pandas().query("split == 1").drop(columns="split")
        for p in manifest["parts"]
    ]
    df = pred.merge(pd.concat(parts), on=KEY, how="left", validate="one_to_one")

    return df, meta["selected_threshold"]


def attach_text(rows: pd.DataFrame) -> pd.DataFrame:

    rows = rows.copy()
    names = fetch_names(
        {1: rows["s1_id"].unique(), **{c: rows.loc[rows["target_source"] == c, "target_id"].unique() for c in (2, 3)}}
    )

    s1 = names[1]
    rows["s1_raw_name"] = rows["s1_id"].map(s1["business_name"])
    rows["s1_norm_name"] = rows["s1_id"].map(s1["norm_name"])
    rows["s1_norm_address"] = rows["s1_id"].map(s1["norm_address"])

    for col in ("target_raw_name", "target_norm_name", "target_norm_address"):
        rows[col] = ""
    for code in (2, 3):
        m = rows["target_source"] == code
        if m.any():
            t = names[code]
            rows.loc[m, "target_raw_name"] = rows.loc[m, "target_id"].map(t["business_name"])
            rows.loc[m, "target_norm_name"] = rows.loc[m, "target_id"].map(t["norm_name"])
            rows.loc[m, "target_norm_address"] = rows.loc[m, "target_id"].map(t["norm_address"])

    rows["s1_translit_name"] = rows["s1_norm_name"].fillna("").map(transliterate_text)
    rows["target_translit_name"] = rows["target_norm_name"].fillna("").map(transliterate_text)

    s1_prof = rows["s1_norm_name"].fillna("").map(script_profile)
    t_prof = rows["target_norm_name"].fillna("").map(script_profile)
    rows["s1_script"] = [p[0] for p in s1_prof]
    rows["s1_script_class"] = [p[1] for p in s1_prof]
    rows["target_script"] = [p[0] for p in t_prof]
    rows["target_script_class"] = [p[1] for p in t_prof]
    rows["different_scripts"] = rows["s1_script"] != rows["target_script"]

    rows["script_relation"] = np.select(
        [
            (rows["s1_script_class"] == "latin") & (rows["target_script_class"] == "latin"),
            (rows["s1_script_class"] == "non-latin") & (rows["target_script_class"] == "non-latin") & ~rows["different_scripts"],
            (rows["s1_script_class"] == "latin") & (rows["target_script_class"] == "non-latin"),
            (rows["s1_script_class"] == "non-latin") & (rows["target_script_class"] == "latin"),
            (rows["s1_script_class"] == "mixed") | (rows["target_script_class"] == "mixed"),
        ],
        [
            "both Latin",
            "same non-Latin script",
            "Latin S1 -> non-Latin target",
            "non-Latin S1 -> Latin target",
            "mixed-script name",
        ],
        default="other (different non-Latin / no letters)",
    )

    delta = rows["translit_name_ratio"] - rows["name_ratio"]
    rows["translit_effect"] = np.select([delta > 0.05, delta < -0.05], ["helps", "worse"], default="no change")

    rows["target"] = rows["target_source"].map(PREFIX) + "-" + rows["target_id"].astype(str)
    rows["s1"] = "S1-" + rows["s1_id"].astype(str)

    return rows


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    out = ROOT / "outputs" / "diagnostics" / f"name_miss_analysis_{args.experiment.name}"
    out.mkdir(parents=True, exist_ok=True)

    df, threshold = load(args.experiment, args.feature_dir)
    fn_all = df[(df["label"] == 1) & (df["score"] < threshold)]
    low = attach_text(fn_all[fn_all["name_ratio"] < 0.5])
    low["best_below_0_5"] = np.fmax(low["name_ratio"], low["translit_name_ratio"]) < 0.5

    summary = {
        "experiment": args.experiment.name,
        "threshold": threshold,
        "false_negatives_in_candidates": int(len(fn_all)),
        "fn_name_ratio_below_0_5": int(len(low)),
        "fn_name_ratio_below_0_5_pct_of_fn": len(low) / len(fn_all),
        "fn_best_name_below_0_5": int(low["best_below_0_5"].sum()),
        "by_source": {PREFIX[k]: int(v) for k, v in low["target_source"].value_counts().items()},
        "target_address_missing": int((low["target_address_missing"] == 1).sum()),
    }

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)

    print(f"=== {args.experiment.name}: threshold {threshold:g} ===")
    print(json.dumps(summary, indent=2))

    # --- Script relation & script pairs ------------------------------------
    rel = low.groupby("script_relation").agg(
        count=("label", "size"),
        best_below_0_5=("best_below_0_5", "sum"),
        mean_name_ratio=("name_ratio", "mean"),
        mean_translit_ratio=("translit_name_ratio", "mean"),
        translit_helps=("translit_effect", lambda s: (s == "helps").mean()),
        translit_worse=("translit_effect", lambda s: (s == "worse").mean()),
        mean_address_token_set=("address_token_set_ratio", "mean"),
        target_address_missing=("target_address_missing", "mean"),
    ).sort_values("count", ascending=False)
    rel["pct"] = rel["count"] / len(low)
    print("\n=== SCRIPT RELATION (FN with name_ratio < 0.5) ===")
    print(rel.to_string(float_format=lambda x: f"{x:.3f}"))

    pairs = low.groupby(["s1_script", "target_script"]).agg(
        count=("label", "size"),
        mean_name_ratio=("name_ratio", "mean"),
        mean_translit_ratio=("translit_name_ratio", "mean"),
        mean_translit_jaccard=("translit_name_token_jaccard", "mean"),
        mean_address_token_set=("address_token_set_ratio", "mean"),
    ).sort_values("count", ascending=False)
    pairs["pct"] = pairs["count"] / len(low)
    print("\n=== SCRIPT PAIRS (top 15) ===")
    print(pairs.head(15).to_string(float_format=lambda x: f"{x:.3f}"))

    # --- Transliteration effect ---------------------------------------------
    print("\n=== TRANSLITERATION EFFECT (translit_name_ratio − name_ratio, ±0.05) ===")
    print(pd.crosstab(low["script_relation"], low["translit_effect"], margins=True).to_string())

    print("\n=== HYPOTHETICAL RECOVERY by translit_name_ratio (analysis only) ===")
    rec = []
    for t in (0.5, 0.6, 0.7, 0.8):
        m = low["translit_name_ratio"] >= t
        rec.append({"translit >=": t, "FN reached": int(m.sum()), "pct of low-name FN": m.mean(),
                    "of which cross-script": int((m & low["different_scripts"]).sum())})
    rec = pd.DataFrame(rec)
    print(rec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # --- Separability: all validation candidates with name_ratio < 0.5 -------
    cand = df[(df["name_ratio"] < 0.5) & (df["translit_name_ratio"] >= 0.5)]
    cand = attach_text(cand)
    cand["addr_bucket"] = pd.cut(cand["address_token_set_ratio"], [0, 0.5, 0.75, 0.9, 1.01],
                                 labels=["<0.50", "0.50-0.75", "0.75-0.90", ">=0.90"], right=False).astype(object).where(
        cand["address_token_set_ratio"].notna(), "missing")
    cand["translit_bucket"] = pd.cut(cand["translit_name_ratio"], [0.5, 0.6, 0.7, 0.8, 1.01],
                                     labels=["0.5-0.6", "0.6-0.7", "0.7-0.8", ">=0.8"], right=False).astype(str)
    cand["cross_script"] = np.where(cand["different_scripts"], "cross-script", "same script")

    sep = cand.groupby(["cross_script", "translit_bucket"]).agg(
        candidates=("label", "size"), positives=("label", "sum"),
        currently_predicted=("predicted", "sum"),
        missed_positives=("label", lambda s: ((s == 1) & (cand.loc[s.index, "predicted"] == 0)).sum()),
        false_positives=("label", lambda s: ((s == 0) & (cand.loc[s.index, "predicted"] == 1)).sum()),
    )
    sep["positive_rate"] = sep["positives"] / sep["candidates"]
    print("\n=== SEPARABILITY: all validation candidates, name_ratio < 0.5 AND translit >= 0.5 ===")
    print(sep.to_string(float_format=lambda x: f"{x:.3f}"))

    sep_addr = cand[cand["different_scripts"]].groupby(["translit_bucket", "addr_bucket"]).agg(
        candidates=("label", "size"), positives=("label", "sum"),
        missed_positives=("label", lambda s: ((s == 1) & (cand.loc[s.index, "predicted"] == 0)).sum()),
    )
    sep_addr["positive_rate"] = sep_addr["positives"] / sep_addr["candidates"]
    print("\n=== CROSS-SCRIPT candidates (translit >= 0.5): positive rate by address token-set similarity ===")
    print(sep_addr.to_string(float_format=lambda x: f"{x:.3f}"))

    fp_low = df[(df["predicted"] == 1) & (df["label"] == 0) & (df["name_ratio"] < 0.5)]
    fp_low = attach_text(fp_low) if len(fp_low) else fp_low
    if len(fp_low):
        fp_low["best_below_0_5"] = np.fmax(fp_low["name_ratio"], fp_low["translit_name_ratio"]) < 0.5
    summary["fp_with_name_ratio_below_0_5"] = int(len(fp_low))
    if len(fp_low):
        print(f"\n=== FALSE POSITIVES with name_ratio < 0.5: {len(fp_low)} ===")
        print(fp_low.groupby("script_relation").agg(
            count=("label", "size"), mean_translit=("translit_name_ratio", "mean"),
            mean_address_token_set=("address_token_set_ratio", "mean")).to_string(float_format=lambda x: f"{x:.3f}"))

    # --- Examples ------------------------------------------------------------
    cols = ["s1", "target", "score", "script_relation", "s1_script", "target_script",
            "s1_raw_name", "target_raw_name", "s1_translit_name", "target_translit_name",
            "name_ratio", "translit_name_ratio", "name_token_jaccard", "translit_name_token_jaccard",
            "address_token_set_ratio", "address_ratio", "target_address_missing",
            "s1_norm_address", "target_norm_address", "translit_effect", "best_below_0_5"]
    examples = []
    for pair in pairs.head(6).index:
        g = low[(low["s1_script"] == pair[0]) & (low["target_script"] == pair[1])]
        examples.append(g.sample(min(args.examples, len(g)), random_state=0)[cols])
    examples = pd.concat(examples, ignore_index=True)

    print("\n=== EXAMPLES (first 4 per top script pair; full set saved) ===")
    for (s1s, ts), g in examples.groupby(["s1_script", "target_script"], sort=False):
        print(f"\n  [{s1s} -> {ts}]")
        for r in g.head(4).itertuples():
            print(f"    name {r.name_ratio:.2f} / translit {r.translit_name_ratio:.2f} / addr_set {r.address_token_set_ratio if r.address_token_set_ratio == r.address_token_set_ratio else float('nan'):.2f}  score {r.score:.3f}")
            print(f"      S1: {r.s1_raw_name!r}  ->  {r.s1_translit_name!r}")
            print(f"      T : {r.target_raw_name!r}  ->  {r.target_translit_name!r}")

    # --- Save ----------------------------------------------------------------
    low[cols].to_csv(out / "fn_low_name_similarity.csv", index=False)
    rel.to_csv(out / "script_relation.csv")
    pairs.to_csv(out / "script_pairs.csv")
    rec.to_csv(out / "hypothetical_translit_recovery.csv", index=False)
    sep.to_csv(out / "separability_translit.csv")
    sep_addr.to_csv(out / "separability_cross_script_by_address.csv")
    examples.to_csv(out / "examples_by_script_pair.csv", index=False)
    if len(fp_low):
        fp_low[cols].to_csv(out / "fp_low_name_similarity.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
