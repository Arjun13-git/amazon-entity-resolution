"""
Dataset location resolution (development layout and $ER_DATASET_DIR override).

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from src.data.loader import DATASET_ENV, dataset_dir


CODE_ROOT = Path(__file__).resolve().parents[1]


class DatasetDirTest(unittest.TestCase):

    def test_default_is_student_resource_next_to_code_root(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(DATASET_ENV, None)
            self.assertEqual(dataset_dir(), CODE_ROOT.parent / "student_resource" / "dataset")

    def test_environment_override(self):
        with mock.patch.dict(os.environ, {DATASET_ENV: "/data/challenge/dataset"}):
            self.assertEqual(dataset_dir(), Path("/data/challenge/dataset").resolve())

    def test_empty_override_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {DATASET_ENV: ""}):
            self.assertEqual(dataset_dir(), CODE_ROOT.parent / "student_resource" / "dataset")


if __name__ == "__main__":
    unittest.main()
