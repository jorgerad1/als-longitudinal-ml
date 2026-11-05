"""Synthetic-only regression tests for the v2 corrected T1/T2 descriptors."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from temporal_v2 import (
    ALL_BASES, DOMAINS, CHANGED_BASES, METRIC_BASES,
    FEATURE_COLUMNS, COVERAGE_COLUMNS, TemporalInputError, build_temporal_features,
    validate_schema,
)

FIXTURE=Path(__file__).resolve().parents[1]/"synthetic_data"/"v2_temporal_pairs.json"


def get_source():
    records=json.loads(FIXTURE.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


class TestTemporalV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=get_source()
        cls.result=build_temporal_features(cls.source)
        cls.feat=cls.result.features.set_index("ID")
        cls.cov=cls.result.coverage_audit.set_index("ID")

    def test_exact_inventory(self):
        validate_schema()
        self.assertEqual(len(ALL_BASES),43)
        self.assertEqual(len(set(ALL_BASES)),43)
        self.assertEqual(len(CHANGED_BASES),14)
        self.assertEqual(len(METRIC_BASES),3)
        self.assertEqual(len(FEATURE_COLUMNS),24)
        self.assertEqual(len(COVERAGE_COLUMNS),7)
        self.assertEqual(set(ALL_BASES)-set.union(*(set(v) for v in DOMAINS.values())),{"IMC_CAT"})
        self.assertEqual(list(self.result.features.columns),["ID",*FEATURE_COLUMNS])
        self.assertEqual(list(self.result.coverage_audit.columns),["ID",*COVERAGE_COLUMNS])

    def test_all_43_pairs_drive_rates_not_14_flags(self):
        a=self.feat.loc["SYN_A"]
        self.assertAlmostEqual(a.T12_RATE_CHANGED_ALL,6/43)
        self.assertAlmostEqual(a.T12_RATE_CHANGED_SOCIO,1/6)
        self.assertAlmostEqual(a.T12_RATE_CHANGED_ACTIVITY,1/2)
        self.assertAlmostEqual(a.T12_RATE_CHANGED_BODY,2/7)
        self.assertAlmostEqual(a.T12_RATE_CHANGED_DIET,1/15)
        self.assertEqual(self.cov.loc["SYN_A","T12_N_COMPARABLE_ALL"],43)
        # The IMC_CAT change counts in ALL but none of the six domains.
        self.assertEqual(a.T12_CHANGED_DEPORTEFREC,1.0)
        self.assertEqual(a.T12_CHANGED_PESO,1.0)
        self.assertEqual(a.T12_CHANGED_ALTURA,1.0)
        self.assertEqual(a.T12_CHANGED_COMIDAS_DIA,1.0)
        self.assertEqual(a.T12_CHANGED_ALIMENTACION,0.0)
        self.assertEqual(a.T12_CHANGED_FASTFOOD,0.0)

    def test_numeric_deltas_only_for_three_bases(self):
        a=self.feat.loc["SYN_A"]
        self.assertEqual(a.T12_METRIC_DELTA_PESO,5.0)
        self.assertEqual(a.T12_METRIC_DELTA_ALTURA,1.0)
        self.assertEqual(a.T12_METRIC_DELTA_COMIDAS_DIA,-1.0)
        self.assertFalse("T12_METRIC_DELTA_DEPORTEFREC" in self.feat.columns)

    def test_missing_not_zero_and_noncomparable(self):
        b=self.feat.loc["SYN_B"]
        self.assertEqual(self.cov.loc["SYN_B","T12_N_COMPARABLE_ALL"],39)
        for col in ("T12_CHANGED_PESO", "T12_CHANGED_ALTURA", "T12_CHANGED_COMIDAS_DIA",
                    "T12_METRIC_DELTA_PESO", "T12_METRIC_DELTA_ALTURA",
                    "T12_METRIC_DELTA_COMIDAS_DIA"):
            self.assertTrue(np.isnan(b[col]))
        self.assertEqual(b.T12_RATE_CHANGED_ALL,0.)
        z=self.feat.loc["SYN_ZERO"]
        for col in FEATURE_COLUMNS:
            self.assertTrue(np.isnan(z[col]),col)
        for col in COVERAGE_COLUMNS:
            self.assertEqual(self.cov.loc["SYN_ZERO",col],0)

    def test_one_unassigned_pair_missing(self):
        c=self.feat.loc["SYN_C"]
        self.assertAlmostEqual(c.T12_RATE_CHANGED_ALL,1/42)
        self.assertEqual(self.cov.loc["SYN_C","T12_N_COMPARABLE_ALL"],42)
        self.assertAlmostEqual(c.T12_RATE_CHANGED_SOCIO,1/6)
        self.assertFalse(np.isnan(c.T12_METRIC_DELTA_COMIDAS_DIA))

    def test_explicit_id_order_no_sorting(self):
        order=["SYN_ZERO", "SYN_C", "SYN_B", "SYN_A"]
        got=build_temporal_features(self.source.sample(frac=1,random_state=7),participant_order=order)
        self.assertEqual(got.features.ID.tolist(),order)
        self.assertTrue(got.features.set_index("ID").sort_index().equals(self.feat.sort_index()))

    def test_reject_duplicate_missing_unknown_and_label_columns(self):
        duplicate=pd.concat([self.source,self.source.iloc[[0]]],ignore_index=True)
        bad=[duplicate,self.source.iloc[1:].copy(),self.source.assign(base=lambda d:d.base.mask(d.index==0,"UNEXPECTED")),self.source.assign(diagnosis=0)]
        for frame in bad:
            with self.subTest(shape=frame.shape):
                with self.assertRaises(TemporalInputError):build_temporal_features(frame)

    def test_reject_non_numeric_when_present(self):
        s=self.source.copy();s["T1"]=s["T1"].astype(object);s.loc[0,"T1"]="unknown"
        with self.assertRaises(TemporalInputError):build_temporal_features(s)


if __name__ == '__main__':unittest.main()
