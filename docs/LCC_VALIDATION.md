# Independent LCC clinical-model evaluation

This stage evaluates future education-adjusted MoCA at nominal 12, 24 and 36 months in the Michael J. Fox Foundation LRRK2 Cohort Consortium (LCC). It follows a separate locked protocol after the PPMI benchmarking work.

## Models and evaluation

Two clinical predictor sets are evaluated at each horizon:

- Baseline MoCA, completed age and sex (code identifier: `reduced3`).
- The same predictors plus GDS15 (code identifier: `reduced4`).

These six models were separately fitted using PPMI only. Five source folds selected the ridge penalty from 0.001, 0.01, 0.1, 1, 10 and 100, using pooled mean squared error; ties favor the larger penalty. Preprocessing is source-trained, and the objective is mean squared error plus the penalty times the squared coefficient norm. The intercept is unpenalized. Predictions are not clipped to the MoCA range.

Source-only fitting involved 186 ridge fits. This number describes the completed study stage, not a command to repeat it. LCC predictions used the verified, fixed models without refitting or recalibration. Persistence and a PPMI-derived typical-change reference are retained.

The matched LCC samples contain 16, 28 and 34 participants by nominal horizon (46 distinct people). Known PPMI links and participants without adequate identity mapping were excluded. Follow-up labels provide nominal timing; exact visit dates were unavailable. Both clinical models and references use identical participants within each horizon.

The primary contrasts compare addition of GDS15 at the three horizons. Conditional participant-component bootstrap intervals use 20,000 draws; the three primary contrasts also have Bonferroni-adjusted 98.33% intervals. The supplementary methods and tables retain outcome rules, cohort provenance, all comparisons and sensitivity definitions.

The external evaluation concerns these specified clinical models. The compact cognitive panel and final GA models were not scored in LCC. The separate later-calendar PPMI evaluation is temporal evaluation within PPMI.

## Public implementation

[`analysis/lcc_nominal_validation_2026-11-05/src/`](../analysis/lcc_nominal_validation_2026-11-05/src) contains three byte-identical source files:

| File | Purpose |
| --- | --- |
| `core.py` | I/O-free ridge fitting, prediction, tuning, metrics and bootstrap calculations |
| `io_contract.py` | File hashes, identity/outcome binding, source seals and authorization guards |
| `launcher.py` | Guarded preflight, source training and target evaluation entry points |

The original private protocol directory, manifests, participant mappings, source records, fitted-model bundles and authorization seals are not distributed. Keeping the `src/` layout does not supply those inputs. Do not remove integrity checks or invent bindings to make a public clone run.

The public smoke check imports only `core.py` and uses synthetic in-memory arrays. It checks ridge predictions against an independent augmented least-squares reference, source-only standardization, constant-column handling, penalty tie-breaking and metric arithmetic. It does not train study models or score LCC.

Supplementary Tables S11–S14 contain aggregate metrics, paired contrasts, cohort summaries and the six model equations. Table S14 is a permitted aggregate parameter table, not a participant-bound model bundle.
