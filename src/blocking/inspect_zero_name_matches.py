from __future__ import annotations

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from rapidfuzz.fuzz import ratio

from src.data.loader import load_ground_truth
from src.preprocessing.normalize import normalize_business_name
from src.preprocessing.transliterate import transliterate_text


ROOT = Path(__file__).resolve().parents[2]

PARQUET_DIR = ROOT / "outputs" / "parquet"

GROUND_TRUTH = (
    ROOT.parent
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

ENTITY_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

SAMPLE_LINKS = 200
MAX_EXAMPLES = 100
RANDOM_STATE = 42


def load_true_links() -> pd.Series:
    """Explode ground truth into target IDs, indexed by S1 entity_id."""

    gt = load_ground_truth(GROUND_TRUTH)

    return (
        gt.set_index("source1_entity_id")["matched_entity_ids"]
        .str.split(",")
        .explode()
    )


def report_link_composition(targets: pd.Series) -> None:
    """Show which source each ground-truth target belongs to."""

    prefixes = targets.fillna("").str[:3].replace("", "<empty>")

    print("\n=== GROUND-TRUTH TARGET COMPOSITION ===")

    for prefix, share in prefixes.value_counts(normalize=True).items():
        print(f"{prefix:>8}: {share:.2%}")


def load_rows(
    parquet_path: Path,
    entity_ids: list[str],
) -> pd.DataFrame:
    """Load only the requested entity rows via a pushed-down filter."""

    dataset = ds.dataset(parquet_path)

    return dataset.to_table(
        columns=ENTITY_COLUMNS,
        filter=ds.field("entity_id").isin(entity_ids),
    ).to_pandas().rename(
        columns={
            "business_name": "name",
            "business_address": "address",
        }
    )


def fuzzy_ratio(left: str, right: str) -> float:
    """RapidFuzz ratio; 0 when either side is empty."""

    if not left or not right:
        return 0.0

    return ratio(left, right) / 100.0


def dominant_script(text: str) -> str:
    """Most common Unicode script prefix among letters (LATIN, CJK, ...)."""

    scripts = Counter(
        unicodedata.name(char, "UNKNOWN").split()[0]
        for char in text
        if unicodedata.category(char).startswith("L")
    )

    if not scripts:
        return ""

    return scripts.most_common(1)[0][0]


def is_abbreviation(short: str, long: str) -> bool:
    """True when `short` equals the initials of the tokens in `long`."""

    short = short.replace(" ", "")
    tokens = long.split()

    if len(short) < 2 or len(tokens) < 2:
        return False

    return short == "".join(token[0] for token in tokens)


def is_noisy(raw: str, normalized: str) -> bool:
    """Name dominated by digits/symbols, or containing replacement chars."""

    if "�" in raw:
        return True

    letters = sum(char.isalpha() for char in normalized)

    return letters < max(2, len(normalized.replace(" ", "")) // 2)


def categorize(row: pd.Series) -> str:
    """Heuristic category for a zero-name-similarity true match.

    Translation cannot be told apart from a genuinely different name
    by string statistics alone, so both land in
    `translation_or_different` and need manual review.
    """

    if pd.isna(row["s2_name"]):
        return "unjoined_target"

    if not row["s1_norm_name"] or not row["s2_norm_name"]:
        return "missing_name"

    if is_noisy(row["s1_name"], row["s1_norm_name"]) or is_noisy(
        row["s2_name"], row["s2_norm_name"]
    ):
        return "noisy_corrupted"

    s1_script = dominant_script(row["s1_norm_name"])
    s2_script = dominant_script(row["s2_norm_name"])

    if s1_script != s2_script:
        return "different_script"

    s1 = row["s1_translit_name"]
    s2 = row["s2_translit_name"]

    if is_abbreviation(s1, s2) or is_abbreviation(s2, s1):
        return "abbreviation"

    if len(s1) <= 3 or len(s2) <= 3:
        return "other"

    return "translation_or_different"


def print_example(index: int, row: pd.Series) -> None:

    print(f"\n--- [{index}] {row['category']} ---")
    print(f"S1 id        : {row['source1_entity_id']}")
    print(f"S2 id        : {row['target_id']}")
    print(f"country      : {row['s1_country']!r} | {row['s2_country']!r}")
    print(f"S1 name      : {row['s1_name']!r}")
    print(f"S2 name      : {row['s2_name']!r}")
    print(f"S1 norm      : {row['s1_norm_name']!r}")
    print(f"S2 norm      : {row['s2_norm_name']!r}")
    print(f"S1 translit  : {row['s1_translit_name']!r}")
    print(f"S2 translit  : {row['s2_translit_name']!r}")
    print(f"S1 address   : {row['s1_address']!r}")
    print(f"S2 address   : {row['s2_address']!r}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Inspect true S1->S2 matches with zero name similarity."
    )
    parser.add_argument("--sample", type=int, default=SAMPLE_LINKS)
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    print("Loading ground truth...")

    targets = load_true_links()

    report_link_composition(targets)

    s2_links = targets[targets.str.startswith("S2-", na=False)]

    links = (
        s2_links.sample(
            n=min(args.sample, len(s2_links)),
            random_state=args.seed,
        )
        .rename("target_id")
        .rename_axis("source1_entity_id")
        .reset_index()
    )

    del targets, s2_links

    print(f"\nSampled S1->S2 links: {len(links):,}")

    s1 = load_rows(
        PARQUET_DIR / "train_source1.parquet",
        links["source1_entity_id"].unique().tolist(),
    ).add_prefix("s1_")

    s2 = load_rows(
        PARQUET_DIR / "train_source2.parquet",
        links["target_id"].unique().tolist(),
    ).add_prefix("s2_")

    pairs = links.merge(
        s1,
        left_on="source1_entity_id",
        right_on="s1_entity_id",
        how="left",
    ).merge(
        s2,
        left_on="target_id",
        right_on="s2_entity_id",
        how="left",
    )

    print(
        "Unjoined S1 rows: "
        f"{pairs['s1_entity_id'].isna().sum():,}; "
        "unjoined S2 rows: "
        f"{pairs['s2_entity_id'].isna().sum():,}"
    )

    for side in ("s1", "s2"):
        names = pairs[f"{side}_name"].fillna("")
        pairs[f"{side}_norm_name"] = names.map(normalize_business_name)
        pairs[f"{side}_translit_name"] = pairs[f"{side}_norm_name"].map(
            transliterate_text
        )

    pairs["name_ratio"] = [
        fuzzy_ratio(left, right)
        for left, right in zip(pairs["s1_norm_name"], pairs["s2_norm_name"])
    ]

    pairs["translit_name_ratio"] = [
        fuzzy_ratio(left, right)
        for left, right in zip(
            pairs["s1_translit_name"],
            pairs["s2_translit_name"],
        )
    ]

    zero = pairs[
        (pairs["name_ratio"] == 0) & (pairs["translit_name_ratio"] == 0)
    ].copy()

    print(
        "\nZero original AND transliterated name ratio: "
        f"{len(zero):,} / {len(pairs):,} "
        f"({len(zero) / max(len(pairs), 1):.2%})"
    )

    if zero.empty:
        print("No zero-similarity examples in this sample.")
        return

    zero["category"] = zero.apply(categorize, axis=1)

    print("\n=== CATEGORY BREAKDOWN (heuristic) ===")

    for category, count in zero["category"].value_counts().items():
        print(f"{category:>26}: {count:,} ({count / len(zero):.1%})")

    shown = zero.head(args.max_examples)

    print(f"\n=== {len(shown)} EXAMPLES ===")

    for index, (_, row) in enumerate(shown.iterrows(), start=1):
        print_example(index, row)


if __name__ == "__main__":
    main()
