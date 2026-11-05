# How to run the v2.0.0 software

## Requirements

- Python 3.11 and the scientific Python dependencies listed in `environment.yml`.
- Access to this repository only; the included examples run without a clinical-data connection or OpenAI API credentials.

## Validation suite

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
python -B -m unittest discover -s tests -p 'test_*_v2.py' -q
```

## Demonstrations

```bash
python -B scripts/v2_temporal_demo.py
python -B scripts/v2_extraction_pools_demo.py
python -B scripts/v2_canonical_maps_demo.py
python -B scripts/v2_nested_cv_demo.py
python -B scripts/v2_paired_inference_demo.py
```

These examples demonstrate temporal features, offline extraction-output parsing, classifier training/validation and paired inference on wholly invented data. They are suitable for testing the software installation and exploring the analytical procedures.

`config/v2_CANONICAL_alias_groups.json` contains the original publicly documented alias/group definitions. Any additional illustrative fixture is used solely in the synthetic demonstration. The private clinical dataset is not part of this repository.
