# v2 corrected temporal descriptors: synthetic-first public module

**Status: component demonstration only; not a complete paperNLP v2.0.0 release.**

This independent, self-contained module reproduces the *arithmetic and column
semantics* of the study's corrected T1/T2 temporal descriptors. It does not
include the private workbook, source clinical records, any OpenAI API calls,
a training pipeline, human annotations, or the clinical results of the paper.

## Input contract

An explicit long-format table of **exactly 43 source-defined pairs per subject**:
`ID`, `base`, `T1`, `T2`. Both observation values must already be harmonized to
numeric codes or missing (`null` / `NaN`). The original clinical workbook's
column harmonization, age/regime eligibility, and correction history are
**not replicated** by this standalone component. The `ID` is used only for
alignment; labels/outcomes are explicitly rejected from the input contract.

## Output

- **14** `T12_CHANGED_<BASE>` binary indicators: 1 for observed unequal T1/T2,
  0 for observed equal T1/T2, missing for an unobserved pair. Numeric category
  codes are compared for equality but not subtracted.
- **7** `T12_RATE_CHANGED_<DOMAIN>` proportions: changed, observed pairs divided
  by observed pairs for ALL and six prespecified domains. These values are **not
  per-year rates**, and the result is missing where no pair is comparable.
- **3** `T12_METRIC_DELTA_<BASE>` differences for PESO, ALTURA and COMIDAS_DIA;
  differences are T2 minus T1 in source units, not annualized rates.
- **7** `T12_N_COMPARABLE_<DOMAIN>` coverage counts are returned for an independent
  audit, **never included in the 24-predictor feature matrix**.

The six named domains use 42 nonoverlapping pairs, while the 43rd pair
`IMC_CAT` belongs to `ALL` only. Hence each proportion must be computed from
**all 43 paired source measurements**, not from the 14 indicator columns.

## Reproducibility checks

```bash
python -m unittest discover -s tests -p 'test_temporal_v2.py' -v
python scripts/v2_temporal_demo.py
```

The JSON fixture contains exactly four independently invented `SYN_*` records
with 43 pairs each. Tests check changes, missingness, domain denominators,
negative numeric differences, zero-coverage cases, participant order and
rejection of malformed, extra or outcome-bearing inputs.

## Publication boundary

This component documents an independently authored port of rules audited in
the private project's Stage25/34 outputs. It was tested **only on the supplied
invented data**. It does not establish cross-environment equivalence against
restricted participant-level results; that requires a separate private,
aggregate-only verification step and an independent code review.

**Release v2.0.0 remains blocked** until extraction, pool construction,
nested evaluation, matching, correlation sensitivity, plotting, environment,
public smoke tests and security gates are completed.
