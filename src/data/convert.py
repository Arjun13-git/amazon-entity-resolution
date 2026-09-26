from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT.parent
    / "student_resource"
    / "dataset"
)

OUTPUT_DIR = ROOT / "outputs" / "parquet"


FILES = {
    "train_source1": DATASET_DIR / "train" / "train_source1.tsv",
    "train_source2": DATASET_DIR / "train" / "train_source2.tsv",
    "train_source3": DATASET_DIR / "train" / "train_source3.tsv",
}


def convert_file(
    name: str,
    source: Path,
) -> None:
    destination = OUTPUT_DIR / f"{name}.parquet"

    if destination.exists():
        print(f"[SKIP] {destination}")
        return

    print(f"[LOAD] {source}")
    
    df = pd.read_csv(
        source,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )

    print(
        f"[WRITE] {destination} "
        f"({len(df):,} rows)"
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        destination,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    print(
        f"[DONE] {name}: "
        f"{destination.stat().st_size / 1024**2:.1f} MB"
    )


def main() -> None:
    for name, source in FILES.items():
        convert_file(name, source)


if __name__ == "__main__":
    main()