# v2 extraction-output and feature-pool contracts — synthetic demonstration

**Status: STAGING ONLY — not the v2.0.0 release; no clinical equivalence established.**

## What is represented

1. The revised clinical workflow specified a schema-guided OpenAI Chat Completions extraction with `model=gpt-6-sol`, `temperature=0.2`, and `reasoning_effort=none`. Its application-level response had `pacientes_actualizados` (each containing `datos`) and `habitos_comunes_nuevos`. This module reads a **fictitious saved response** with that same shape. It makes **no API call** and does not claim to emulate GPT-6 Sol outputs. The provider did not enforce JSON schema.
2. Returned rows are realigned by the **explicit participant ID**. Missing/duplicate/unexpected IDs are rejected. Nonmissing original structured fields take precedence; output proposals are accepted for empty structured fields only. This is structural preservation, **not** a test of whether the proposed value is clinically supported by the text.
3. Habit names are canonicalized, aliases deduplicated, and five compact activity-group indicators plus `TXT_ANY_HABIT`, `TXT_HABIT_COUNT`, `TXTG_N_GROUPS` derived. A zero in an extracted habit indicator means *not explicitly mentioned*, not evidence the person did not undertake the activity.
4. The T1-only and the T1+T2 passes are processed as **separate sets of source-aligned outputs**, not obtained by deleting T2 columns from a merged full-run response. Pool2 uses T1-only compact text, and the full Pool3 uses T1+T2 compact text plus the 24 corrected temporal features. Removing the temporal features from Pool3 still leaves **T1+T2 text** and is not automatically identical to Pool2.
5. Four fictitious participants and wholly fabricated `DEMO_BASE_01..24` and `DEMO_T1_01..45` features illustrate the manuscript's **24 / 77 / 101** input-feature counts. These 45 T1 synthetic fields do **not** establish the number or identities of the study's actual structured T1 predictors. Patient outcomes are never added to the predictor matrices. All seven `T12_N_COMPARABLE_*` fields remain audit-only.

## Explicit gaps to close before v2 publication

- The alias/group map in `config/v2_ILLUSTRATIVE_alias_groups.json` is **not** the clinically reviewed `HABIT_ALIAS_MAP=36` / `TEXT_GROUP_MAP=5` from the private analysis. Port the **approved** mapping through private code review, without leaking patient text.
- Independently port the real source spreadsheet harmonization and eligibility rules from the verified Stage52 pipeline; match actual columns/names and participant-order semantics, not merely dimensions.
- Check offline saved outputs (restricted) and the revised private matrices for the same feature values in a **private parity test**. Report only aggregate mismatch counts and hashes, never patient data or text.
- Add frozen-vocabulary and cumulative order scenarios, including audited model configuration and request template (redacted, generic). Do not replay clinical prompts through a public package.
- Add the nested model selection/validation, duration-matching, correlation-filter sensitivity, inference, and final figures in following stages, then audit complete public package.

## Run on synthetic-only data

```bash
python -m unittest discover -s tests -p 'test_extraction_pools_v2.py' -v
python scripts/v2_extraction_pools_demo.py
```

No OpenAI credentials, account connections, patient-level data, inference calls, trained models, or output files are required. The saved demo response contains only fictional identifiers `SYN_*` and invented activities.
