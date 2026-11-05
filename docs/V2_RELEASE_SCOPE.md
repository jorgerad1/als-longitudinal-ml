# v2.0.0 — Scope and study methods

This release provides source code, configuration, tests and synthetic demonstrations for the revised longitudinal-questionnaire analysis workflow.

The corresponding research dataset included 103 ALS/control participants (60 controls and 43 ALS); the main T2-eligible analysis used 95 participants (52 controls and 43 ALS). The input feature sets contained 24, 77 and 101 predictors in Pools 1, 2 and 3.

The software implements the following analytical procedures:

1. Harmonization of 43 retrospective T1/T2 variable pairs and construction of 24 temporal descriptors, with explicit handling of missing comparisons and seven audit-only coverage counts.
2. Schema-guided extraction-output processing, canonical activity grouping and compact text representation.
3. Leakage-controlled type-aware preprocessing and repeated nested cross-validation with three classifier families.
4. Duration-matched group-aware evaluation, no-correlation sensitivity and exploratory paired statistical contrasts.

The accompanying tests and demonstrations use fabricated data and enable public inspection of these procedures. Participant-level questionnaires, original model responses and clinical matrices remain under access restrictions.
