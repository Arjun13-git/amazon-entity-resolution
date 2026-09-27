"""
Production pipeline: exact S1 grouping, output writing, and an end-to-end
run (candidates -> features -> base scores -> decisions -> output files)
on a tiny synthetic US + France dataset, including resume.

The end-to-end test needs the frozen model artifacts under outputs/
(git-ignored); it is skipped when they are absent.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking import candidate_pipeline
from src.blocking.validate_candidates import tables_equal
from src.decision.s1_context import context_features
from src.features import pairwise
from src.inference import decide, predict_base, write_outputs
from src.inference.decide import decision_groups
from src.inference.write_outputs import group_s1_ids, id_lists


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "experiments" / "xgb_skeleton"
SECOND = ROOT / "outputs" / "experiments" / "s1_context"


def part(target, country, part, lo, hi, rows):
    return {"target": target, "country": country, "part": part, "s1_id_min": lo, "s1_id_max": hi,
            "s1_rows": rows, "file": f"target={target}/country={country}/part-{part:05d}.parquet"}


class DecisionGroupsTest(unittest.TestCase):

    def manifest(self, parts):
        return {"config": {"targets": ["S2", "S3"]}, "parts": parts}

    def test_pairs_parts_by_country_and_index(self):
        groups = decision_groups(self.manifest([
            part("S2", "us", 0, 1, 10, 5), part("S3", "us", 0, 1, 10, 5),
            part("S2", "us", 1, 11, 20, 5), part("S3", "us", 1, 11, 20, 5),
            part("S2", "france", 0, 3, 9, 4), part("S3", "france", 0, 3, 9, 4),
        ]))
        self.assertEqual([(g["country"], g["part"]) for g in groups], [("france", 0), ("us", 0), ("us", 1)])
        self.assertEqual(groups[1]["files"]["S3"], "target=S3/country=us/part-00000.parquet")

    def test_rejects_different_s1_sets(self):
        with self.assertRaises(ValueError):
            decision_groups(self.manifest([part("S2", "us", 0, 1, 10, 5), part("S3", "us", 0, 1, 12, 6)]))

    def test_rejects_missing_target(self):
        with self.assertRaises(ValueError):
            decision_groups(self.manifest([part("S2", "us", 0, 1, 10, 5)]))

    def test_rejects_overlapping_groups(self):
        with self.assertRaises(ValueError):
            decision_groups(self.manifest([
                part("S2", "us", 0, 1, 10, 5), part("S3", "us", 0, 1, 10, 5),
                part("S2", "us", 1, 10, 20, 5), part("S3", "us", 1, 10, 20, 5),
            ]))


class GroupedContextExactnessTest(unittest.TestCase):

    def frame(self, n_s1=40, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for s1 in range(1, n_s1 + 1):
            for src in (2, 3):
                for t in range(rng.integers(0, 5)):
                    rows.append({"s1_id": s1, "target_id": s1 * 100 + src * 10 + t, "target_source": src,
                                 "score": float(rng.random() ** 0.2),
                                 "address_token_set_ratio": rng.random(), "name_ratio": rng.random(),
                                 "address_number_set_jaccard": rng.random(), "translit_skeleton_ratio": rng.random(),
                                 "target_name_frequency": float(rng.integers(1, 9))})
        return pd.DataFrame(rows)

    def test_context_per_group_equals_global_context(self):
        df = self.frame()
        full = context_features(df)
        # Groups = contiguous S1 ranges containing BOTH sources' candidates.
        pieces = [context_features(g) for _, g in df.groupby(df["s1_id"] // 7)]
        grouped = pd.concat(pieces).reindex(df.index)
        pd.testing.assert_frame_equal(full, grouped)

    def test_splitting_an_s1_across_chunks_would_change_context(self):
        # Why decision_groups insists on complete S1s: S2-only chunks give a
        # different (wrong) context for S1 with S3 candidates.
        df = self.frame()
        wrong = pd.concat([context_features(g) for _, g in df.groupby("target_source")]).reindex(df.index)
        self.assertFalse(context_features(df).equals(wrong))


class OutputWriterTest(unittest.TestCase):

    def test_id_lists_are_ordered_and_empty_when_absent(self):
        df = pd.DataFrame({"s1_id": [5, 5, 5, 7], "target_id": [30, 10, 20, 1], "target_source": [3, 3, 2, 2]})
        self.assertEqual(id_lists(df, np.array([5, 6, 7])), ["S2-20,S3-10,S3-30", "", "S2-1"])
        self.assertEqual(id_lists(df.iloc[:0], np.array([1, 2])), ["", ""])

    def test_group_s1_ids_checks_count(self):
        ids = np.array([1, 3, 5, 7, 9])
        g = {"country": "us", "part": 0, "s1_id_min": 3, "s1_id_max": 7, "s1_rows": 3}
        np.testing.assert_array_equal(group_s1_ids(ids, g, None), [3, 5, 7])
        with self.assertRaises(ValueError):
            group_s1_ids(ids, {**g, "s1_rows": 4}, None)


# ----------------------------------------------------------------------
# End-to-end on tiny synthetic data
# ----------------------------------------------------------------------

def tiny_data() -> dict:
    s1 = {
        "us": [
            (11, "acme traders", "12 main street springfield il"),
            (12, "blue tech", "40 oak road portland or"),
            (13, "zenith labs", ""),                       # missing address
            (14, "", "7 elm street springfield il"),       # missing name
            (15, "orphan co", "999 nowhere lane springfield il"),
        ],
        "france": [
            (21, "mas pharmacie", "25 mail pablo picasso nantes pays de la loire"),
            (22, "pg finance", "75 rue brunet pessac nouvelle aquitaine"),
            (23, "tourcoing maternelle", "40 rue bonne nouvelle tourcoing hauts de france"),
        ],
    }
    targets = {}
    for src, offset in (("source2", 1000), ("source3", 2000)):
        targets[src] = {
            "us": [
                (offset + 1, "acme traders", "12 main st springfield illinois"),
                (offset + 2, "blue tech", "40 oak rd portland oregon"),
                (offset + 3, "zenith labs", "5 pine st springfield il"),
                (offset + 4, "someone else", "7 elm street springfield il"),
                (offset + 5, "acme", "13 main street springfield il"),
                (offset + 6, "", ""),
            ],
            "france": [
                (offset + 11, "mas pharmacie", "25 mail pablo picasso nantes"),
                (offset + 12, "pg finance sas", "75 rue brunet pessac"),
                (offset + 13, "maternelle de tourcoing", "40 rue bonne nouvelle tourcoing"),
                (offset + 14, "boulangerie", "3 rue de lille roubaix"),
            ],
        }
    return {"source1": s1, **targets}


def fake_load_partition(data):
    def load(source, country, columns, dataset="train"):
        rows = data[source].get(country, [])
        df = pd.DataFrame(rows, columns=["id", "norm_name", "norm_address"])
        df["id"] = df["id"].astype("int32")
        return df[["id", *columns]].sort_values("id").reset_index(drop=True)
    return load


@unittest.skipUnless((BASE / "model.json").exists() and (SECOND / "second_stage_model.json").exists(),
                     "frozen model artifacts not available")
class EndToEndTinyTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = tiny_data()
        lex = self.tmp / "lexicons"
        lex.mkdir()
        (lex / "lexicon_us.txt").write_text("springfield\nportland\n")   # no lexicon for france
        self.cfg = candidate_pipeline.PipelineConfig(
            chunk_size=3, workers=1, dataset="test", lexicon_dir=str(lex), chunk_work=10_000,
        )
        load = fake_load_partition(self.data)
        self.patches = [
            mock.patch.object(candidate_pipeline, "load_partition", load),
            mock.patch.object(candidate_pipeline, "list_countries", lambda source, dataset="train": ["france", "us"]),
            mock.patch.object(pairwise, "load_partition", load),
            mock.patch.object(write_outputs, "load_partition", load),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def test_end_to_end_and_resume(self):
        run_dir = self.tmp / "run"
        cand = run_dir / "candidates"
        m = candidate_pipeline.run(self.cfg, cand)

        parts = {(p["target"], p["country"], p["part"]): pq.read_table(cand / p["file"]) for p in m["parts"]}
        self.assertEqual(len(parts), 2 * (1 + 2))   # france: 3 S1 -> 1 part; us: 5 S1 -> 2 parts

        france = pd.concat([t.to_pandas() for (_, c, _), t in parts.items() if c == "france"])
        self.assertGreater(len(france), 0)
        self.assertFalse(france["house_city_hit"].any())                      # no structured keys
        self.assertTrue(((france["candidate_source_mask"] & 1) > 0).any())    # exact name works
        self.assertTrue(((france["candidate_source_mask"] & 2) > 0).any())    # name char works
        self.assertTrue(((france["candidate_source_mask"] & 4) > 0).any())    # address char works

        us = pd.concat([t.to_pandas() for (_, c, _), t in parts.items() if c == "us"])
        self.assertTrue(us["house_city_hit"].any())                           # US structured keys work

        # Resume: remove one finished part, rerun, get identical content.
        victim = cand / m["parts"][1]["file"]
        victim.unlink()
        victim.with_suffix(".json").unlink()
        candidate_pipeline.run(self.cfg, cand, resume=True)
        for p in m["parts"]:
            # NaN-aware comparison (Table.equals treats NaN != NaN).
            self.assertTrue(tables_equal(pq.read_table(cand / p["file"]), parts[(p["target"], p["country"], p["part"])]))

        with self.assertRaises(FileExistsError):
            candidate_pipeline.run(self.cfg, cand)
        with self.assertRaises(ValueError):
            candidate_pipeline.run(replace(self.cfg, chunk_size=4), cand, resume=True)

        # Features in test mode: no labels, all 35 model features present.
        fm = pairwise.run(cand, run_dir / "features", 0.2, 42)
        feats = pd.concat([pq.read_table(run_dir / "features" / p["file"]).to_pandas() for p in fm["parts"]])
        self.assertEqual(set(feats["label"]), {-1})
        self.assertEqual(set(feats["split"]), {-1})
        model_features = json.loads((BASE / "features.json").read_text())["features"]
        self.assertTrue(set(model_features) <= set(feats.columns))
        fr = feats[feats["country"] == "france"]
        self.assertTrue(fr["house_city_match"].isna().all() and fr["city_match"].isna().all())
        self.assertTrue(fr["translit_skeleton_ratio"].notna().any())

        predict_base.run(run_dir / "features", BASE, run_dir / "scores")
        dm = decide.run(run_dir, SECOND)
        self.assertEqual(dm["candidates"], len(feats))

        out = write_outputs.run(run_dir)
        matching = pd.read_csv(out["files"]["matching_results.tsv"], sep="\t", dtype=str, keep_default_na=False)
        cands = pd.read_csv(out["files"]["candidate_pairs.tsv"], sep="\t", dtype=str, keep_default_na=False)

        expected = sorted(f"S1-{r[0]}" for c in self.data["source1"].values() for r in c)
        self.assertEqual(sorted(matching["source1_entity_id"]), expected)       # every S1 exactly once
        self.assertEqual(sorted(cands["source1_entity_id"]), expected)
        for m_ids, c_ids in zip(matching["matched_entity_ids"], cands["candidate_entity_ids"]):
            self.assertTrue(set(filter(None, m_ids.split(","))) <= set(filter(None, c_ids.split(","))))

        # Re-running the output stage is deterministic.
        first = Path(out["files"]["matching_results.tsv"]).read_bytes()
        write_outputs.run(run_dir)
        self.assertEqual(Path(out["files"]["matching_results.tsv"]).read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
