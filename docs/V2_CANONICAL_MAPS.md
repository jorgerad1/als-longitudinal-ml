# Public canonical habit maps (v2 draft)

This draft adds the actual **36 aliases** and **5 compact groups** already
published in the historical repository's `src/text_features.py`. The config
`config/v2_CANONICAL_alias_groups.json` is generated during isolated staging
by parsing those public Python literal dictionaries, not by reading any
private annotation, dataset, prompts, checkpoints, or manifest.

The adapter converts the original group-to-list format into the
canonical-habit-to-group(s) contract expected by `merge_llm_response`. It preserves
multi-group membership, so one mentioned habit can activate more than one
compact group without being double-counted as an individual habit. It checks
ambiguities and does not modify the original `src/text_features.py` or the
Stage67 illustrative example. The synthetic demo exercises one invented
positive mention from each group.

**Remaining scientific gate:** this proves parity of map *definitions*
with the already-public historical source. It does NOT prove that the revised
Stage52 clinical matrices contain identical columns or feature values, nor
does it reproduce the remote GPT-6 Sol extraction. That private, aggregate-only
parity audit is still required before a release.

**Not publishable:** inherited staging manifests named `*_PRIVATE.json`,
draft notes, and historical artifacts require an independent privacy/license
review and curated final packaging. No staging folder is intended for direct
upload to GitHub or Zenodo.
