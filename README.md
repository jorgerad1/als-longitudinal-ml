# ALS Longitudinal ML

Reproducibility code for the study:

**Compact longitudinal representations derived from mixed-format lifestyle questionnaires outperform static text-derived features for ALS-versus-control classification**

This repository contains the public computational implementation of the final leakage-free machine-learning workflow used in the study.

## Overview

The study compares three increasingly enriched representations of mixed-format lifestyle questionnaires:

- **Pool1:** structured baseline variables.
- **Pool2:** Pool1 plus structured first-time-point (T1) variables and compact descriptors derived from T1 free-text content.
- **Pool3:** Pool2 plus compact descriptors of longitudinal change between T1 and T2. Raw paired T2 variables are removed after the temporal descriptors are generated.

The public implementation includes:

- schema-guided LLM text-to-table extraction;
- row realignment and preservation of original structured values;
- canonicalization and compact grouping of text-derived habits;
- compact T1-to-T2 longitudinal feature construction;
- train-only missingness filtering, imputation, encoding, scaling and feature filtering;
- Logistic Regression, Linear SVC and Random Forest hyperparameter tuning;
- repeated stratified holdout and repeated stratified cross-validation;
- Pool2 and Pool3 ablation analyses;
- aggregate results used in the manuscript;
- scripts for reproducing Figures 5 and 6;
- a fully synthetic offline demonstration dataset.

## Restricted clinical data

The patient-level questionnaire dataset used in the study is **not distributed with this repository**.

The original workbook contains sensitive information from patients and healthy volunteers and is subject to ethical and privacy restrictions. Neither the original workbook nor patient-level derived feature matrices, instantiated LLM prompts, or patient-level model outputs are included here.

The synthetic dataset included under `synthetic_data/` was independently generated for software testing. It is not an anonymized, perturbed, transformed, or distribution-matched version of the clinical cohort.

Consequently:

- the public repository reproduces the computational methodology;
- the synthetic example permits end-to-end software testing;
- the synthetic example does **not** reproduce the numerical clinical results reported in the article;
- exact numerical reproduction of the published clinical results requires access to the restricted source data.

## Repository structure

```text
als-longitudinal-ml/
├── config/
│   ├── final.yaml
│   └── prompt_template_es.txt
├── figures/
│   ├── Figure_5.pdf
│   └── Figure_6.pdf
├── provenance/
├── results/
│   ├── feature_counts_by_stage.csv
│   ├── published_ablation_metrics.csv
│   ├── published_confusion_matrices.json
│   └── published_final_metrics.csv
├── scripts/
│   ├── make_figure5_confusions.py
│   ├── make_figure6_dimensionality.py
│   └── run_synthetic_demo.py
├── src/
│   ├── ablation.py
│   ├── evaluation.py
│   ├── models.py
│   ├── pool_construction.py
│   ├── preprocessing.py
│   ├── temporal_features.py
│   ├── text_extraction.py
│   └── text_features.py
├── synthetic_data/
│   ├── generate_synthetic.py
│   ├── README.md
│   └── synthetic_questionnaire.xlsx
├── tests/
├── environment.yml
├── LICENSE
└── CITATION.cff
```

## Software environment

The published analyses were executed with:

- Python 3.11.14
- pandas 3.0.0
- NumPy 2.4.1
- scikit-learn 1.8.0
- SciPy 1.17.0
- joblib 1.5.3
- threadpoolctl 3.6.0
- openpyxl 3.1.5
- Matplotlib 3.10.8
- seaborn 0.13.2
- OpenAI Python SDK 2.15.0
- JupyterLab 4.5.3
- Linux x86_64

The principal reproducibility environment is provided in `environment.yml`. Additional environment snapshots are stored under `provenance/`.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate als-longitudinal-ml
```

## Synthetic offline demonstration

Generate the fully synthetic workbook:

```bash
python synthetic_data/generate_synthetic.py
```

Run the offline demonstration:

```bash
python scripts/run_synthetic_demo.py
```

This demonstration constructs synthetic Pool1, Pool2 and Pool3 feature matrices and runs a leakage-free classifier pipeline without accessing the restricted clinical dataset or making an LLM API call.

The resulting synthetic model accuracy is not scientifically meaningful and should not be compared with the published clinical results.

## LLM extraction

The original analyses used the OpenAI Chat Completions API with:

- model identifier: `gpt-4o-mini`
- temperature: `0.2`
- chunk size: `10`

The literal schema-guided prompt template is provided in:

```text
config/prompt_template_es.txt
```

The public implementation reads the API credential only from the environment:

```bash
export OPENAI_API_KEY="..."
```

No API credential is included in this repository.

The synthetic offline demonstration does not require an API key.

## Machine-learning evaluation

The final evaluation uses ten fixed seeds:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

For each configuration:

- repeated stratified holdout uses a 75/25 split for each seed;
- repeated stratified cross-validation uses 5 shuffled folds for each of the 10 seeds, yielding 50 outer folds;
- hyperparameters are optimized using inner 3-fold stratified cross-validation;
- macro-F1 is used as the tuning metric;
- all data-dependent preprocessing is fitted exclusively on each training partition.

The classifiers are:

- Elastic-Net Logistic Regression;
- Linear SVC;
- Random Forest.

Full search spaces are defined in `src/models.py` and `config/final.yaml`.

## Ablation analysis

The explicit compact text block is identified as:

```text
TXTG_*
TXT_ANY_HABIT
TXT_HABIT_COUNT
```

The temporal block is identified as:

```text
DELTA_*
T12_*
```

Pool2 evaluates the full representation and a version without the compact text-descriptor block.

Pool3 evaluates:

- full representation;
- without the compact text-descriptor block;
- without the temporal block;
- without both blocks.

## Reproducing Figures 5 and 6

Figure 5 is generated entirely from the published aggregate confusion matrices:

```bash
python scripts/make_figure5_confusions.py
```

Figure 6 is generated from the published aggregate feature counts:

```bash
python scripts/make_figure6_dimensionality.py
```

The source aggregate values are stored in `results/`.

No patient-level data are required to reproduce either figure.

## Published aggregate outputs

The `results/` directory contains aggregate values corresponding to the manuscript:

- final model metrics;
- ablation metrics;
- aggregated holdout confusion matrices;
- retained feature counts used for Figure 6.

These aggregate files contain no participant-level records.

## Provenance

`provenance/hashes.tsv` records SHA-256 hashes of the final analysis notebooks used to generate the reported results.

The original notebooks themselves are not distributed because they were executed in the restricted clinical environment and may contain patient-level material.

## Citation

Citation metadata are provided in `CITATION.cff`.

A versioned archival DOI will be added after creation of the corresponding Zenodo release.

## License

The software in this repository is released under the MIT License.

The license applies to the source code and does not grant access to or rights over the restricted clinical dataset.
