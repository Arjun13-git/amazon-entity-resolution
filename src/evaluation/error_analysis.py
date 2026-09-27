"""
Error analysis of a matcher experiment's validation predictions.

Reads ``validation_predictions.parquet`` of an experiment, joins the stored
feature rows (no recomputation, no retraining) and characterizes:

- false positives: predicted (score >= threshold) but not a true link
- false negatives: true link in the candidate set, scored below threshold

For every category it also reports the *oracle gain*: the validation
macro F0.5 if exactly those errors were fixed and nothing else changed.

Detailed rows go to ``--out`` (default ``outputs/diagnostics/error_analysis_<experiment>``);
only aggregates are printed.

    python -m src.evaluation.error_analysis outputs/experiments/xgb_numeric \\
        outputs/features/sample20k_numeric --compare outputs/experiments/xgb_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking.candidate_pipeline import SOURCE_BITS
from src.evaluation.f05 import f05_from_counts
from src.features.audit_features import fetch_text
from src.models.xgb_matcher import true_link_counts


ROOT = Path(__file__).resolve().parents[2]

KEY = ["s1_id", "target_id", "target_source"]

FEATURES = [
    "name_exact", "name_ratio", "name_token_set_ratio", "translit_name_ratio",
    "address_ratio", "address_token_set_ratio",
    "address_number_set_jaccard", "address_number_sequence_similarity",
    "secondary_number_match", "house_number_match", "city_match", "house_city_match",
    "exact_name_hit", "name_char_hit", "address_char_hit", "house_city_hit", "n_channels",
    "name_char_rank", "address_char_rank", "s1_candidate_count",
    "s1_name_missing", "target_name_missing", "s1_address_missing", "target_address_missing",
]

CHANNELS = ["exact_name_hit", "name_char_hit", "address_char_hit", "house_city_hit"]
CHANNEL_SHORT = {"exact_name_hit": "exact", "name_char_hit": "name", "address_char_hit": "addr", "house_city_hit": "hc"}

SOURCE_FILES = {2: "source2", 3: "source3"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_validation(
    experiment: Path,
    feature_dir: Path,
    extra_features: list[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Validation predictions joined with their stored feature rows."""

    meta = json.loads((experiment / "metadata.json").read_text())
    threshold = meta["selected_threshold"]

    pred = pq.read_table(experiment / "validation_predictions.parquet").to_pandas()

    manifest = json.loads((feature_dir / "manifest.json").read_text())
    available = set(manifest["feature_columns"])
    wanted = list(dict.fromkeys(FEATURES + (extra_features or [])))
    columns = KEY + ["split", "label"] + [c for c in wanted if c in available]

    parts = []
    for part in manifest["parts"]:
        t = pq.read_table(feature_dir / part["file"], columns=columns).to_pandas()
        parts.append(t[t["split"] == 1].drop(columns="split"))
    feats = pd.concat(parts, ignore_index=True)

    df = pred.merge(feats, on=KEY, how="left", suffixes=("", "_feat"), validate="one_to_one")
    if df["label_feat"].isna().any() or (df["label"] != df["label_feat"]).any():
        raise ValueError("predictions and feature rows disagree on keys or labels")

    df = df.drop(columns="label_feat")
    df["predicted"] = (df["score"] >= threshold).astype(np.int8)
    # Same bits as the candidate files (SOURCE_BITS).
    df["mask"] = sum(
        df[f"{name}_hit"].astype(int) * bit for name, bit in SOURCE_BITS.items()
    )
    df["channels"] = channel_labels(df)

    return df, threshold


def channel_labels(df: pd.DataFrame) -> pd.Series:

    parts = [np.where(df[c] == 1, CHANNEL_SHORT[c], "") for c in CHANNELS]
    joined = ["+".join(p for p in row if p) for row in zip(*parts)]

    return pd.Series(joined, index=df.index)


# ----------------------------------------------------------------------
# Categories
# ----------------------------------------------------------------------


def best_name(df: pd.DataFrame) -> pd.Series:

    return np.fmax(df["name_ratio"], df["translit_name_ratio"])


def fp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Non-exclusive deterministic categories for false-positive rows."""

    name_same = df["name_exact"] == 1
    name_near = ~name_same & (best_name(df) >= 0.90)
    addr_same = df["address_ratio"] == 1
    addr_near = ~addr_same & (df["address_ratio"] >= 0.90)
    num_mismatch = df["address_number_set_jaccard"] < 1

    return pd.DataFrame(
        {
            "identical name": name_same,
            "near-identical name (≥0.90, not identical)": name_near,
            "identical address": addr_same,
            "near-identical address (≥0.90, not identical)": addr_near,
            "house+city agree": df["house_city_match"] == 1,
            "address numeric mismatch (number-set Jaccard < 1)": num_mismatch,
            "identical name + numeric mismatch": name_same & num_mismatch,
            "identical name AND identical address": name_same & addr_same,
            "different city (city_match = 0)": df["city_match"] == 0,
            "house_city-only candidate": (df["house_city_hit"] == 1) & (df["n_channels"] == 1),
        }
    )


FP_PRIMARY = [
    ("identical name + identical address", lambda d: (d["name_exact"] == 1) & (d["address_ratio"] == 1)),
    ("identical name + numeric mismatch", lambda d: (d["name_exact"] == 1) & (d["address_number_set_jaccard"] < 1)),
    ("near-identical name + numeric mismatch", lambda d: (best_name(d) >= 0.9) & (d["address_number_set_jaccard"] < 1)),
    ("identical/near name, numbers agree", lambda d: best_name(d) >= 0.9),
    ("different city", lambda d: d["city_match"] == 0),
    ("house_city-only", lambda d: (d["house_city_hit"] == 1) & (d["n_channels"] == 1)),
]


def primary_category(df: pd.DataFrame, rules) -> pd.Series:
    """First matching rule (exclusive categories); 'other' if none."""

    out = pd.Series("other", index=df.index, dtype=object)
    unassigned = pd.Series(True, index=df.index)

    for name, rule in rules:
        hit = unassigned & rule(df).fillna(False).astype(bool)
        out[hit] = name
        unassigned &= ~hit

    return out


def bucket(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:

    out = pd.cut(values, edges, labels=labels, right=False, include_lowest=True).astype(object)

    return out.where(values.notna(), "missing")


def fn_breakdowns(fn: pd.DataFrame) -> dict[str, pd.Series]:

    sim_edges = [0, 0.5, 0.75, 0.9, 1.0, 1.01]
    sim_labels = ["<0.50", "0.50-0.75", "0.75-0.90", "0.90-<1", "1.0"]

    return {
        "score": bucket(fn["score"], [0, 0.1, 0.5, 0.9, 0.99, 1.01], ["<0.10", "0.10-0.50", "0.50-0.90", "0.90-0.99", "0.99-<thr"]),
        "best name ratio (orig/translit)": bucket(best_name(fn), sim_edges, sim_labels),
        "address ratio": bucket(fn["address_ratio"], sim_edges, sim_labels),
        "address number-set Jaccard": bucket(fn["address_number_set_jaccard"], sim_edges, sim_labels),
        "channels": fn["channels"].replace("", "none"),
        "name_char_rank": bucket(fn["name_char_rank"], [0, 1, 5, 20, 50, 100], ["0", "1-4", "5-19", "20-49", "50-99"]),
        "address_char_rank": bucket(fn["address_char_rank"], [0, 1, 5, 10, 25], ["0", "1-4", "5-9", "10-24"]),
        "n_channels": fn["n_channels"].astype(int).astype(str),
        "missing name/address": np.select(
            [fn["target_address_missing"] == 1, fn["target_name_missing"] == 1, fn["s1_address_missing"] == 1],
            ["target address missing", "target name missing", "S1 address missing"],
            default="none missing",
        ),
        "house_number_match": fn["house_number_match"].map({1.0: "1", 0.0: "0"}).fillna("missing"),
        "city_match": fn["city_match"].map({1.0: "1", 0.0: "0"}).fillna("missing"),
        "house_city_match": fn["house_city_match"].map({1.0: "1", 0.0: "0"}).fillna("missing"),
    }


# ----------------------------------------------------------------------
# F0.5 accounting
# ----------------------------------------------------------------------


class S1Accounting:
    """Per-S1 counts over the full validation S1 set (incl. blocking misses)."""

    def __init__(self, df: pd.DataFrame, feature_dir: Path):
        split = pq.read_table(feature_dir / "split.parquet").to_pandas()
        self.s1 = np.sort(split.loc[split["split"] == 1, "s1_id"].to_numpy())
        self.n_true = true_link_counts(self.s1)
        self.index = np.searchsorted(self.s1, df["s1_id"].to_numpy())
        self.label = df["label"].to_numpy()
        self.pred = df["predicted"].to_numpy() == 1

    def macro_f05(self, pred: np.ndarray) -> float:
        n = len(self.s1)
        n_pred = np.bincount(self.index[pred], minlength=n)
        tp = np.bincount(self.index[pred & (self.label == 1)], minlength=n)
        return float(f05_from_counts(tp, n_pred, self.n_true).mean())

    def oracle_gain(self, fix: np.ndarray) -> float:
        """Macro F0.5 change if the rows in ``fix`` were decided correctly."""

        corrected = self.pred.copy()
        corrected[fix] = self.label[fix] == 1

        return self.macro_f05(corrected) - self.macro_f05(self.pred)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


def with_text(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized S1 / target name and address (pushed-down reads)."""

    rows = rows.copy()
    s1 = fetch_text("source1", rows["s1_id"].unique())
    rows["s1_name"] = rows["s1_id"].map(s1["norm_name"])
    rows["s1_address"] = rows["s1_id"].map(s1["norm_address"])

    rows["target_name"] = ""
    rows["target_address"] = ""
    for code, source in SOURCE_FILES.items():
        m = rows["target_source"] == code
        if m.any():
            t = fetch_text(source, rows.loc[m, "target_id"].unique())
            rows.loc[m, "target_name"] = rows.loc[m, "target_id"].map(t["norm_name"])
            rows.loc[m, "target_address"] = rows.loc[m, "target_id"].map(t["norm_address"])

    rows["target"] = "S" + rows["target_source"].astype(str) + "-" + rows["target_id"].astype(str)
    rows["s1"] = "S1-" + rows["s1_id"].astype(str)

    return rows


def analyze(experiment: Path, feature_dir: Path) -> dict:

    df, threshold = load_validation(experiment, feature_dir)
    acct = S1Accounting(df, feature_dir)

    fp = df[(df["predicted"] == 1) & (df["label"] == 0)].copy()
    fn = df[(df["predicted"] == 0) & (df["label"] == 1)].copy()
    tp = df[(df["predicted"] == 1) & (df["label"] == 1)]

    # S1 context of each error.
    tp_s1 = set(tp["s1_id"])
    true_s1 = set(acct.s1[acct.n_true > 0])
    pred_s1 = set(df.loc[df["predicted"] == 1, "s1_id"])

    fp["s1_context"] = np.select(
        [~fp["s1_id"].isin(true_s1), fp["s1_id"].isin(tp_s1)],
        ["S1 has no true match (singleton)", "extra prediction next to a correct one"],
        default="S1 has matches, none predicted correctly",
    )
    fn["s1_context"] = np.where(
        fn["s1_id"].isin(pred_s1), "S1 has other predictions", "S1 has no predictions"
    )
    fn["rank_in_s1"] = df.groupby("s1_id")["score"].rank(ascending=False, method="first").loc[fn.index]

    fp["category"] = primary_category(fp, FP_PRIMARY)
    fp_non_exclusive = fp_flags(fp)

    result = {
        "experiment": str(experiment),
        "threshold": threshold,
        "macro_f05": acct.macro_f05(acct.pred),
        "counts": {
            "predicted": int(df["predicted"].sum()),
            "true_positives": int(len(tp)),
            "false_positives": int(len(fp)),
            "false_negatives_in_candidates": int(len(fn)),
            "true_links_missed_by_blocking": int(acct.n_true.sum() - df["label"].sum()),
        },
        "oracle_gain": {
            "fix all false positives": acct.oracle_gain(((df["predicted"] == 1) & (df["label"] == 0)).to_numpy()),
            "fix all false negatives": acct.oracle_gain(((df["predicted"] == 0) & (df["label"] == 1)).to_numpy()),
        },
    }

    def gain_for(rows: pd.DataFrame) -> float:
        mask = np.zeros(len(df), bool)
        mask[df.index.get_indexer(rows.index)] = True
        return acct.oracle_gain(mask)

    result["fp_flags"] = {
        name: {"count": int(col.sum()), "pct": float(col.mean()), "oracle_gain": gain_for(fp[col])}
        for name, col in fp_non_exclusive.items()
    }
    result["fp_primary"] = {
        name: {"count": int(len(g)), "pct": len(g) / len(fp), "oracle_gain": gain_for(g)}
        for name, g in fp.groupby("category")
    }
    result["fp_s1_context"] = {
        name: {"count": int(len(g)), "pct": len(g) / len(fp), "oracle_gain": gain_for(g)}
        for name, g in fp.groupby("s1_context")
    }
    result["fn_s1_context"] = {
        name: {"count": int(len(g)), "pct": len(g) / len(fn), "oracle_gain": gain_for(g)}
        for name, g in fn.groupby("s1_context")
    }
    result["fn_rank_in_s1"] = bucket(
        fn["rank_in_s1"], [1, 2, 4, 11, 1e9], ["1 (top of S1)", "2-3", "4-10", ">10"]
    ).value_counts().to_dict()

    result["fn_breakdowns"] = {}
    for name, values in fn_breakdowns(fn).items():
        groups = {}
        for value, g in fn.groupby(values):
            groups[str(value)] = {"count": int(len(g)), "pct": len(g) / len(fn), "oracle_gain": gain_for(g)}
        result["fn_breakdowns"][name] = dict(sorted(groups.items(), key=lambda kv: -kv[1]["count"]))

    return {"summary": result, "df": df, "fp": fp, "fn": fn}


def save_details(analysis: dict, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    df, fp, fn = analysis["df"], analysis["fp"], analysis["fn"]

    detail_cols = [
        "s1", "target", "score", "label", "predicted", "channels", "mask",
        "s1_name", "target_name", "s1_address", "target_address",
    ] + [c for c in FEATURES if c in df.columns]

    top_fp = with_text(fp.nlargest(100, "score"))
    top_fp[["category", "s1_context"] + detail_cols].to_csv(out_dir / "top100_false_positives.csv", index=False)

    low_tp = with_text(df[df["label"] == 1].nsmallest(100, "score"))
    low_tp[detail_cols].to_csv(out_dir / "bottom100_true_positives.csv", index=False)

    fp.to_parquet(out_dir / "false_positives.parquet", index=False)
    fn.to_parquet(out_dir / "false_negatives.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(analysis["summary"], indent=2, default=float))


def print_table(title: str, table: dict, total: int | None = None) -> None:

    print(f"\n  {title}")
    for name, v in table.items():
        print(f"    {name:<52} {v['count']:>6,}  {v['pct']:6.1%}   oracle ΔF0.5 {v['oracle_gain']:+.4f}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="Another experiment to compare categories with.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = {"current": args.experiment}
    if args.compare:
        runs["compare"] = args.compare

    analyses = {}
    for role, path in runs.items():
        analyses[role] = analyze(path, args.feature_dir)
        out = args.out if (role == "current" and args.out) else (
            ROOT / "outputs" / "diagnostics" / f"error_analysis_{path.name}"
        )
        save_details(analyses[role], out)
        print(f"[{role}] {path.name}: details -> {out}")

    for role, a in analyses.items():
        s = a["summary"]
        print(f"\n=== {Path(s['experiment']).name}: threshold {s['threshold']:g}, macro F0.5 {s['macro_f05']:.4f} ===")
        print("  counts:", s["counts"])
        print("  oracle:", {k: round(v, 4) for k, v in s["oracle_gain"].items()})
        print_table("FALSE POSITIVES — non-exclusive flags", s["fp_flags"])
        print_table("FALSE POSITIVES — exclusive primary category (first rule wins)", s["fp_primary"])
        print_table("FALSE POSITIVES — S1 context", s["fp_s1_context"])
        print_table("FALSE NEGATIVES — S1 context", s["fn_s1_context"])
        print(f"\n  FALSE NEGATIVES — rank of the true target within its S1: {s['fn_rank_in_s1']}")
        if role == "current":
            for name, table in s["fn_breakdowns"].items():
                print_table(f"FALSE NEGATIVES — {name}", table)


if __name__ == "__main__":
    main()
