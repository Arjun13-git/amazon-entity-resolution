"""
Streaming pairwise feature extraction + labeling for candidate outputs.

Reads a candidate directory written by ``src.blocking.candidate_pipeline``
one part at a time and writes one feature part per candidate part:

    <out>/target=S2/country=us/part-00000.parquet

Each row: s1_id, target_id, target_source, country, split (0 train /
1 validation), label, then float32/int feature columns. Candidates are
never modified or re-generated here.

    python -m src.features.pairwise outputs/candidates/sample20k_a \\
        --out outputs/features/sample20k
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking.address_keys import extract_keys
from src.blocking.benchmark_candidates import TARGET_FILES, locate
from src.blocking.entity_cache import load_partition
from src.evaluation.labels import TruthIndex
from src.evaluation.split import make_s1_validation_split
from src.features.address_features import address_features, structured_features
from src.features.retrieval_features import retrieval_features
from src.features.text_features import name_features
from src.preprocessing.transliterate import transliterate_text


ID_COLUMNS = ["s1_id", "target_id", "target_source", "country", "split", "label"]


class EntityLookup:
    """Normalized name/address of one source's country partition, by id."""

    def __init__(self, source: str, country: str, ids: np.ndarray | None = None):
        df = load_partition(source, country, ["norm_name", "norm_address"])

        if ids is not None:
            df = df[np.isin(df["id"].to_numpy(), ids)].reset_index(drop=True)

        self.country = country
        self.ids = df["id"].to_numpy()
        self.name = df["norm_name"].fillna("").to_numpy(dtype=object)
        self.address = df["norm_address"].fillna("").to_numpy(dtype=object)

    def positions(self, ids: np.ndarray) -> np.ndarray:

        pos = locate(self.ids, ids)
        if (pos < 0).any():
            raise ValueError(
                f"{(pos < 0).sum()} candidate ids missing from the {self.country} partition"
            )

        return pos

    def derived(self, pos: np.ndarray, lexicon: frozenset[str]) -> dict[str, np.ndarray]:
        """Translit name and address keys for the given rows (computed once per unique row)."""

        uniq, inverse = np.unique(pos, return_inverse=True)

        translit = np.array(
            [transliterate_text(n) for n in self.name[uniq]], dtype=object
        )
        keys = extract_keys(
            pd.Series(self.address[uniq]), self.country, lexicon
        )

        return {
            "translit": translit[inverse],
            **{k: keys[k].to_numpy(dtype=object)[inverse] for k in ("house", "city", "house_city")},
        }


def build_split(s1_ids: np.ndarray, validation_fraction: float, seed: int) -> pd.DataFrame:

    train, valid = make_s1_validation_split(
        pd.Series(s1_ids),
        validation_fraction=validation_fraction,
        random_state=seed,
    )

    split = pd.DataFrame(
        {
            "s1_id": np.r_[train.to_numpy(), valid.to_numpy()].astype(np.int32),
            "split": np.r_[np.zeros(len(train)), np.ones(len(valid))].astype(np.int8),
        }
    )

    return split.sort_values("s1_id").reset_index(drop=True)


def part_features(
    candidates: pa.Table,
    s1: EntityLookup,
    tg: EntityLookup,
    lexicon: frozenset[str],
) -> dict[str, np.ndarray]:

    s1_pos = s1.positions(candidates["s1_id"].to_numpy())
    t_pos = tg.positions(candidates["target_id"].to_numpy())

    s1_name, t_name = s1.name[s1_pos], tg.name[t_pos]
    s1_addr, t_addr = s1.address[s1_pos], tg.address[t_pos]

    s1_derived = s1.derived(s1_pos, lexicon)
    t_derived = tg.derived(t_pos, lexicon)

    # Both lookups are the part's country partition and every id was
    # found in them, so country agreement holds for every row.
    country_match = np.ones(len(s1_pos), dtype=np.float32)

    features = {}
    features.update(
        name_features(s1_name, t_name, s1_derived["translit"], t_derived["translit"])
    )
    features.update(address_features(s1_addr, t_addr))
    features.update(structured_features(s1_derived, t_derived, country_match))
    features.update(retrieval_features(candidates))
    features.update(
        {
            "s1_name_missing": (s1_name == "").astype(np.int8),
            "target_name_missing": (t_name == "").astype(np.int8),
            "s1_address_missing": (s1_addr == "").astype(np.int8),
            "target_address_missing": (t_addr == "").astype(np.int8),
        }
    )

    return features


def run(candidate_dir: Path, out_dir: Path, validation_fraction: float, seed: int) -> dict:

    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    cand_manifest = json.loads((candidate_dir / "manifest.json").read_text())

    lexicons = {
        country: frozenset((candidate_dir / info["file"]).read_text().split())
        for country, info in cand_manifest["lexicons"].items()
    }

    s1_path = candidate_dir / "s1_ids.parquet"
    s1_ids = pq.read_table(s1_path)["s1_id"].to_numpy() if s1_path.exists() else None

    if s1_ids is None:
        s1_ids = np.concatenate(
            [load_partition("source1", c, [])["id"].to_numpy() for c in lexicons]
        )

    split = build_split(s1_ids, validation_fraction, seed)
    pq.write_table(pa.Table.from_pandas(split, preserve_index=False), out_dir / "split.parquet")
    split_ids = split["s1_id"].to_numpy()
    split_flag = split["split"].to_numpy()

    truth = {
        target: TruthIndex(target, s1_ids)
        for target in cand_manifest["config"]["targets"]
    }

    manifest = {
        "candidate_dir": str(candidate_dir),
        "candidate_code": cand_manifest["code"],
        "split": {
            "validation_fraction": validation_fraction,
            "seed": seed,
            "train_s1": int((split_flag == 0).sum()),
            "validation_s1": int((split_flag == 1).sum()),
        },
        "parts": [],
    }

    parts = sorted(cand_manifest["parts"], key=lambda p: (p["target"], p["country"], p["file"]))
    current = None

    for meta in parts:
        key = (meta["target"], meta["country"])

        if key != current:
            current = key
            target, country = key
            s1 = EntityLookup("source1", country, s1_ids)
            tg = EntityLookup(TARGET_FILES[target], country)
            print(f"[{target} | {country}] lookups: S1 {len(s1.ids):,}, targets {len(tg.ids):,}", flush=True)

        part_start = time.perf_counter()
        candidates = pq.read_table(candidate_dir / meta["file"])

        features = part_features(candidates, s1, tg, lexicons[country])

        s1_id = candidates["s1_id"].to_numpy()
        target_id = candidates["target_id"].to_numpy()
        target_source = candidates["target_source"].to_numpy()

        split_pos = locate(split_ids, s1_id)
        if (split_pos < 0).any():
            raise ValueError("candidate S1 without a split assignment")

        columns = {
            "s1_id": s1_id,
            "target_id": target_id,
            "target_source": target_source,
            "country": candidates["country"],
            "split": split_flag[split_pos],
            "label": truth[target].label(s1_id, target_id, target_source),
            **features,
        }

        table = pa.table(columns)
        path = out_dir / meta["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")

        elapsed = time.perf_counter() - part_start
        manifest["parts"].append(
            {
                "file": meta["file"],
                "target": target,
                "country": country,
                "rows": table.num_rows,
                "positives": int(columns["label"].sum()),
                "bytes": path.stat().st_size,
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"  {meta['file']}: {table.num_rows:,} rows, "
            f"{int(columns['label'].sum()):,} positives, {elapsed:.1f}s",
            flush=True,
        )

        del candidates, features, table, columns

    manifest["feature_columns"] = [c for c in pq.read_schema(out_dir / parts[0]["file"]).names if c not in ID_COLUMNS]
    manifest["schema"] = pq.read_schema(out_dir / parts[0]["file"]).to_string()
    manifest["runtime_seconds"] = round(time.perf_counter() - started, 1)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = run(args.candidate_dir, args.out, args.validation_fraction, args.seed)

    print(
        f"\nDone: {sum(p['rows'] for p in manifest['parts']):,} rows, "
        f"{sum(p['bytes'] for p in manifest['parts']) / 2**20:,.1f} MB, "
        f"{manifest['runtime_seconds']:,.0f}s"
    )


if __name__ == "__main__":
    main()
