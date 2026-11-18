# Update 18 November 2026

- Added methods and selected code for clinical DAT-SPECT, GA28 DAT-SPECT, and ridge/EBM/TabPFN comparisons.
- Added independent-verification implementations and a public synthetic kernel check without study-data access.
- Made new source-directory references journal-neutral; exposed the GA protocol as a template without fitted selections.
- Updated README and reproduction guidance to distinguish original, expanded, imaging, algorithm and externally evaluated models.
- Retained all previously designated aggregate supplements unchanged. No new results, participant records, predictions, weights, private bindings or run logs were uploaded.

Repository updates

## 2026-11-15

- Use the current study title: A brief cognitive panel improves forecasts of future cognition in Parkinson disease: Development and temporal validation.
- Add scripts, locked protocols and independent verification code for the expanded-PPMI, temporal-validation, PDBP and diagnostic analyses, and the scripts that draw the manuscript figures. Sources are byte-identical.
- Replace the supplementary appendix with Document S1 and add aggregate Tables S15–S21 and the Figure S4 source table. Tables S3–S14 are unchanged.
- Fix character-encoding errors in the README and reproducibility guide, and refresh the checksum manifest.
- Keep the release journal-neutral: later-analysis code lives in `analysis/later_analyses_2026-11-15/`; local folder names and labels that named a journal were replaced by neutral labels (path-only and report-text changes, recorded in the manifest).
- Update Document S1 so its code-availability text and reference [S1] cite the journal-neutral release (commit d40c454) and `analysis/later_analyses_2026-11-15/`.

The update does not retrain models, score participants or modify frozen study results. Source datasets, participant-level artifacts, eligibility ledgers, predictions, credentials and submission documents remain outside the public release.

## 2026-11-07

- Use the current study title and journal-neutral documentation and source labels.
- Describe the independent LCC clinical-model evaluation alongside PPMI internal and temporal analyses.
- Add three frozen LCC implementation files, retaining their execution and integrity guards.
- Synchronize the designated aggregate supplement with the current verified manuscript material, including Tables S11–S14.
- Add synthetic LCC numerical checks and refresh the source/public checksum manifest.

The repository update does not retrain models, score participants or modify frozen study results. Source datasets, participant-level artifacts, credentials and submission documents remain outside the public release.
