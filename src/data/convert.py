from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import dataset_dir


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = dataset_dir()

OUTPUT_DIR = ROOT / "outputs" / "parquet"


FILES = {
    f"{split}_source{i}": DATASET_DIR / split / f"{split}_source{i}.tsv"
    for split in ("train", "test")
    for i in (1, 2, 3)
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

    # Write to a temporary file and rename, so an interrupted run never
    # leaves a truncated Parquet file that a later run would [SKIP].
    tmp = destination.with_suffix(".tmp")
    df.to_parquet(
        tmp,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    tmp.replace(destination)

    print(
        f"[DONE] {name}: "
        f"{destination.stat().st_size / 1024**2:.1f} MB"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert challenge TSVs to Parquet.")
    parser.add_argument("--datasets", nargs="+", default=["train"], choices=["train", "test"])
    args = parser.parse_args()

    for name, source in FILES.items():
        if name.split("_")[0] in args.datasets:
            convert_file(name, source)


if __name__ == "__main__":
    main()