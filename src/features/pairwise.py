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
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking.address_keys import extract_keys
from src.blocking.benchmark_candidates import TARGET_FILES, locate
from src.blocking.candidate_pipeline import atomic_write_json, atomic_write_parquet
from src.blocking.entity_cache import load_partition
from src.evaluation.labels import TruthIndex
from src.evaluation.split import make_s1_validation_split
from src.features.address_features import (
    address_features,
    extract_numbers,
    numeric_features,
    structured_features,
)
from src.features.retrieval_features import retrieval_features
from src.features.text_features import (
    name_features,
    name_frequency,
    name_rarity_features,
    skeleton_features,
    skeleton_from_translit,
)
from src.preprocessing.transliterate import transliterate_text


ID_COLUMNS = ["s1_id", "target_id", "target_source", "country", "split", "label"]


class EntityLookup:
    """Normalized name/address of one source's country partition, by id."""

    def __init__(self, source: str, country: str, ids: np.ndarray | None = None, dataset: str = "train"):
        df = load_partition(source, country, ["norm_name", "norm_address"], dataset)

        if ids is not None:
            df = df[np.isin(df["id"].to_numpy(), ids)].reset_index(drop=True)

        self.country = country
        self.complete = ids is None
        self._name_frequency = None
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

    @property
    def name_frequency(self) -> np.ndarray:
        """
        Entities in this (source, country) partition sharing each entity's
        normalized name; NaN for empty names. Computed once, from the
        entity population only (no labels).
        """

        if not self.complete:
            raise ValueError("name frequency needs the complete partition, not an id subset")

        if self._name_frequency is None:
            self._name_frequency = name_frequency(self.name)

        return self._name_frequency

    def derived(self, pos: np.ndarray, lexicon: frozenset[str]) -> dict[str, np.ndarray]:
        """Translit name, address keys and address numbers for the given rows (once per unique row)."""

        uniq, inverse = np.unique(pos, return_inverse=True)

        translit = np.array(
            [transliterate_text(n) for n in self.name[uniq]], dtype=object
        )
        keys = extract_keys(
            pd.Series(self.address[uniq]), self.country, lexicon
        )

        numbers = np.empty(len(uniq), dtype=object)
        numbers[:] = [extract_numbers(a) for a in self.address[uniq]]

        return {
            "translit": translit[inverse],
            "skeleton": np.array(
                [skeleton_from_translit(t) for t in translit], dtype=object
            )[inverse],
            "numbers": numbers[inverse],
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


def prefilter(
    candidates: pa.Table,
    s1: EntityLookup,
    tg: EntityLookup,
) -> np.ndarray:
    """
    Fast pre-filter: drop obvious non-matches before the model sees them.

    Keeps a candidate if ANY of:
    - exact name match (normalized)
    - name ratio >= 0.40 (catches typos, minor variations)
    - house number match AND (address ratio >= 0.30 OR city match)
    - address ratio >= 0.60 (strong address overlap)
    - at least 2 matching numbers in address (strong signal)
    - postal code match (exact)

    Returns a boolean mask of candidates to KEEP.
    """
    s1_pos = s1.positions(candidates["s1_id"].to_numpy())
    t_pos = tg.positions(candidates["target_id"].to_numpy())

    s1_name = s1.name[s1_pos]
    t_name = tg.name[t_pos]
    s1_addr = s1.address[s1_pos]
    t_addr = tg.address[t_pos]

    n = len(s1_pos)
    keep = np.zeros(n, dtype=bool)

    # 1. Exact name match
    keep |= (s1_name == t_name) & (s1_name != "")

    # 2. Name ratio >= 0.40
    from rapidfuzz.fuzz import ratio as fuzz_ratio
    from rapidfuzz.process import cpdist
    name_sim = cpdist(s1_name, t_name, scorer=fuzz_ratio, dtype=np.float32, workers=-1) / 100.0
    keep |= name_sim >= 0.40

    # 3. House number match + address/city signal
    s1_derived = s1.derived(s1_pos, frozenset())
    t_derived = tg.derived(t_pos, frozenset())
    house_match = (s1_derived["house"] == t_derived["house"]) & (s1_derived["house"] != "")
    addr_sim = cpdist(s1_addr, t_addr, scorer=fuzz_ratio, dtype=np.float32, workers=-1) / 100.0
    city_match = (s1_derived["city"] == t_derived["city"]) & (s1_derived["city"] != "")
    keep |= house_match & ((addr_sim >= 0.30) | city_match)

    # 4. Address ratio >= 0.60
    keep |= addr_sim >= 0.60

    # 5. At least 2 matching numbers
    s1_nums = s1_derived["numbers"]
    t_nums = t_derived["numbers"]
    for i in range(n):
        if not keep[i] and s1_nums[i] and t_nums[i]:
            if len(set(s1_nums[i]) & set(t_nums[i])) >= 2:
                keep[i] = True

    # 6. Postal code match
    s1_postal = s1_derived["house_city"].str.split("|").str[0]  # not postal, use address_keys
    # Actually use extract_keys for postal
    from src.blocking.address_keys import extract_keys
    import pandas as pd
    s1_keys = extract_keys(pd.Series(s1_addr), "us", frozenset())
    t_keys = extract_keys(pd.Series(t_addr), "us", frozenset())
    postal_match = (s1_keys["postal"].to_numpy() == t_keys["postal"].to_numpy()) & (s1_keys["postal"].to_numpy() != "")
    keep |= postal_match

    return keep


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
    features.update(skeleton_features(s1_derived["skeleton"], t_derived["skeleton"]))
    features.update(name_rarity_features(tg.name_frequency[t_pos]))
    features.update(address_features(s1_addr, t_addr))
    features.update(structured_features(s1_derived, t_derived, country_match))
    features.update(
        numeric_features(
            s1_derived["numbers"],
            t_derived["numbers"],
            s1_addr == "",
            t_addr == "",
        )
    )
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


def run(
    candidate_dir: Path,
    out_dir: Path,
    validation_fraction: float,
    seed: int,
    resume: bool = False,
) -> dict:
    """
    Feature parts for every candidate part. Restartable: parts are written
    atomically with a JSON sidecar and skipped on ``resume``.

    Training data (``dataset == "train"``) gets labels and the S1 split.
    Other datasets (test) have no ground truth: ``label`` and ``split`` are
    written as -1 and the truth tables are never consulted.
    """

    started = time.perf_counter()

    cand_manifest = json.loads((candidate_dir / "manifest.json").read_text())
    dataset = cand_manifest["config"].get("dataset", "train")
    labelled = dataset == "train"

    run_config = {
        "candidate_dir": str(candidate_dir),
        "candidate_manifest_sha256": hashlib.sha256((candidate_dir / "manifest.json").read_bytes()).hexdigest(),
        "dataset": dataset,
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    if out_dir.exists():
        if not resume:
            raise FileExistsError(f"{out_dir} exists (use --resume to continue it)")
        if json.loads((out_dir / "config.json").read_text()) != run_config:
            raise ValueError("feature run config differs from the run being resumed")
    else:
        out_dir.mkdir(parents=True)
        atomic_write_json(out_dir / "config.json", run_config)

    lexicons = {
        country: frozenset((candidate_dir / info["file"]).read_text().split())
        for country, info in cand_manifest["lexicons"].items()
    }

    s1_path = candidate_dir / "s1_ids.parquet"
    s1_ids = pq.read_table(s1_path)["s1_id"].to_numpy() if s1_path.exists() else None
    countries = sorted({p["country"] for p in cand_manifest["parts"]})

    manifest = {
        "candidate_dir": str(candidate_dir),
        "candidate_code": cand_manifest["code"],
        "dataset": dataset,
        "parts": [],
    }

    if labelled:
        if s1_ids is None:
            s1_ids = np.concatenate(
                [load_partition("source1", c, [], dataset)["id"].to_numpy() for c in countries]
            )
        split = build_split(s1_ids, validation_fraction, seed)
        if not (out_dir / "split.parquet").exists():
            atomic_write_parquet(pa.Table.from_pandas(split, preserve_index=False), out_dir / "split.parquet")
        split_ids = split["s1_id"].to_numpy()
        split_flag = split["split"].to_numpy()
        truth = {target: TruthIndex(target, s1_ids) for target in cand_manifest["config"]["targets"]}
        manifest["split"] = {
            "validation_fraction": validation_fraction,
            "seed": seed,
            "train_s1": int((split_flag == 0).sum()),
            "validation_s1": int((split_flag == 1).sum()),
        }

    parts = sorted(cand_manifest["parts"], key=lambda p: (p["target"], p["country"], p["file"]))
    current = None

    for meta in parts:
        path = out_dir / meta["file"]
        sidecar = path.with_suffix(".json")

        if sidecar.exists():
            manifest["parts"].append(json.loads(sidecar.read_text()))
            continue

        key = (meta["target"], meta["country"])
        if key != current:
            current = key
            target, country = key
            s1 = EntityLookup("source1", country, s1_ids, dataset)
            tg = EntityLookup(TARGET_FILES[target], country, None, dataset)
            print(f"[{target} | {country}] lookups: S1 {len(s1.ids):,}, targets {len(tg.ids):,}", flush=True)

        part_start = time.perf_counter()
        candidates = pq.read_table(candidate_dir / meta["file"])

        # Pre-filter: drop obvious non-matches before feature extraction
        keep_mask = prefilter(candidates, s1, tg)
        n_before = candidates.num_rows
        candidates = candidates.filter(keep_mask)
        n_after = candidates.num_rows
        if n_before != n_after:
            print(f"    prefilter: {n_before:,} -> {n_after:,} ({n_after/n_before:.1%} kept)", flush=True)

        features = part_features(candidates, s1, tg, lexicons[country])

        s1_id = candidates["s1_id"].to_numpy()
        target_id = candidates["target_id"].to_numpy()
        target_source = candidates["target_source"].to_numpy()

        if labelled:
            split_pos = locate(split_ids, s1_id)
            if (split_pos < 0).any():
                raise ValueError("candidate S1 without a split assignment")
            split_col = split_flag[split_pos]
            label_col = truth[target].label(s1_id, target_id, target_source)
        else:
            split_col = np.full(len(s1_id), -1, dtype=np.int8)
            label_col = np.full(len(s1_id), -1, dtype=np.int8)

        columns = {
            "s1_id": s1_id,
            "target_id": target_id,
            "target_source": target_source,
            "country": candidates["country"],
            "split": split_col,
            "label": label_col,
            **features,
        }

        table = pa.table(columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(table, path)

        elapsed = time.perf_counter() - part_start
        record = {
            "file": meta["file"],
            "target": meta["target"],
            "country": meta["country"],
            "part": meta.get("part"),
            "rows": table.num_rows,
            "positives": int(label_col.sum()) if labelled else None,
            "bytes": path.stat().st_size,
            "seconds": round(elapsed, 1),
        }
        atomic_write_json(sidecar, record)
        manifest["parts"].append(record)
        print(
            f"  {meta['file']}: {table.num_rows:,} rows"
            + (f", {record['positives']:,} positives" if labelled else "")
            + f", {elapsed:.1f}s",
            flush=True,
        )

        del candidates, features, table, columns

    first = out_dir / parts[0]["file"]
    manifest["feature_columns"] = [c for c in pq.read_schema(first).names if c not in ID_COLUMNS]
    manifest["schema"] = pq.read_schema(first).to_string()
    manifest["runtime_seconds"] = round(time.perf_counter() - started, 1)

    atomic_write_json(out_dir / "manifest.json", manifest)

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
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run in --out.")
    args = parser.parse_args()

    manifest = run(args.candidate_dir, args.out, args.validation_fraction, args.seed, resume=args.resume)

    print(
        f"\nDone: {sum(p['rows'] for p in manifest['parts']):,} rows, "
        f"{sum(p['bytes'] for p in manifest['parts']) / 2**20:,.1f} MB, "
        f"{manifest['runtime_seconds']:,.0f}s"
    )


if __name__ == "__main__":
    main()
