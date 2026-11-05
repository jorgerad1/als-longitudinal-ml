#!/usr/bin/env python3
"""Run corrected temporal-feature demonstration on invented input only.

No API, no model fitting, no individual clinical inputs or artifacts are read.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from temporal_v2 import FEATURE_COLUMNS, COVERAGE_COLUMNS, build_temporal_features


def main():
    path=ROOT/'synthetic_data/v2_temporal_pairs.json'
    records=json.loads(path.read_text(encoding='utf8'))
    assert len(records)==4*43 and all(str(x['ID']).startswith('SYN_') for x in records)
    result=build_temporal_features(pd.DataFrame(records))
    assert result.features.shape == (4,25) and result.coverage_audit.shape == (4,8)
    print('=== V2_TEMPORAL_SYNTHETIC_DEMO ===')
    print('INPUT: 4 invented participants, 43 source pairs each; no patient data')
    print('FEATURE_COLUMNS: 24 = 14 change flags + 7 observed-pair proportions + 3 numerical differences')
    print('COVERAGE_AUDIT: 7 counts, excluded from predictor matrix')
    print('MISSING_HANDLING: absent pair -> NaN, zero observed pairs -> NaN ratio')
    print('SCHEMA: exact 43 pairs; IMC_CAT appears only in ALL domain')
    print('NO_API=1 NO_ML=1 NO_CLINICAL_DATA=1 NO_OUTPUT_FILES=1')
    print('=== END_V2_TEMPORAL_SYNTHETIC_DEMO ===')

if __name__=='__main__':main()
