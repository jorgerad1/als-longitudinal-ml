# ALS Longitudinal ML — v2.0.0

Research software for processing mixed-format lifestyle questionnaires and evaluating compact longitudinal feature representations in ALS-versus-control classification.

## What's included

- **Temporal representations:** 43 paired T1/T2 questionnaire variables, summarized as 24 predictors (14 change indicators, seven observed-pair change proportions and three numeric deltas). Seven coverage counts are reserved for auditing.
- **Text-to-table processing:** offline parsing of schema-guided extraction JSON, participant-ID realignment, preservation of source structured values, 36 canonical habit aliases, five activity groups and eight compact text descriptors.
- **Predictive evaluation:** train-only preprocessing, repeated nested five-fold cross-validation with three-fold inner tuning, Elastic Net logistic regression, linear SVC and Random Forest.
- **Sensitivity and inference:** T2-duration pair-aware evaluation, analysis without correlation filtering, subject-/pair-grouped bootstrap, paired permutation comparisons and Holm multiple-testing adjustment.
- **Tests and examples:** fully invented JSON/questionnaire fixtures, 52 unit tests and five runnable offline demonstrations.

## Quick start

Use Python 3.11 and the packages declared in `environment.yml`. From the repository root:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
python -B -m unittest discover -s tests -p 'test_*_v2.py' -q
python -B scripts/v2_temporal_demo.py
python -B scripts/v2_extraction_pools_demo.py
python -B scripts/v2_canonical_maps_demo.py
python -B scripts/v2_nested_cv_demo.py
python -B scripts/v2_paired_inference_demo.py
```

The demonstrations operate offline on **fully synthetic, invented data**; no OpenAI API key or real participant information is needed. The clinical questionnaire dataset is subject to institutional access controls and is not distributed with the software.

## Version history and documentation

Version 2.0.0 adds the revised temporal representation, offline extraction examples, nested evaluation and paired inferential utilities. See `docs/V2_CHANGELOG.md`, `docs/V2_RELEASE_SCOPE.md`, `docs/V2_REPRODUCIBILITY.md` and `docs/V2_PRIVACY_AND_LIMITATIONS.md`.

Version 1.0.0 is preserved at https://doi.org/10.5281/zenodo.22864706. The earlier README is archived for context in `docs/README_V1_HISTORICAL.md`.

## Citation and license

See `CITATION.cff` for the software authors and `LICENSE` for the distribution terms. The version-specific v2.0.0 DOI can be added when its Zenodo record is created.
