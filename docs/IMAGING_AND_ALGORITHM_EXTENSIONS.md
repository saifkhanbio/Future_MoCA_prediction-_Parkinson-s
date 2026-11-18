# Imaging and algorithm extensions

This is a methods-and-code release of three separately contracted exploratory internal comparisons. It does not replace the original benchmark, constitute a fresh test set or extend independent validation to additional models. No new numerical study results are distributed with this update.

## Packages and evaluation scope

| Package | Scope | Prespecified priority |
| --- | --- | --- |
| DAT_SPECT_readiness_v1 and DAT_SPECT_launcher_v1 | Four-score and enhanced clinical models, with and without bilateral caudate/putamen binding ratios; 12, 24, 36 and 48 months; matched participants and inherited grouped folds | 48-month four-score imaging-minus-clinical MAE |
| DAT_SPECT_GA28_v1 | Historical unrestricted28 GA fold-specific selections and full28 reference, each with and without DAT-SPECT; 12, 24 and 36 months | 36-month GA-selected imaging-minus-clinical R² |
| EBM_TabPFN_comparison_v1 | Ridge, EBM and TabPFN v2 on four-score/enhanced/full28 complete-panel views and four-score/enhanced expanded views; 12, 24 and 36 months | Two matched complete-panel four-score MAE contrasts against ridge at 12 months |

The four-score panel is baseline MoCA, SDMT and HVLT immediate/delayed recall, with form controls and missing indicators as specified. Full28 includes genetic and blood-count-derived predictors as well as clinical assessments. All learned preprocessing and configuration selection remain within the training folds. The imaging protocols retain their distinct timing views; same-month records do not certify day-level ordering. The GA imaging comparison reuses historical fold-specific selections rather than rerunning genetic search inside the imaging subset.

## Archived code is not a turnkey study run

Paths preserve package relationships within `analysis/imaging_algorithms_2026-11-18/`. The historical study launchers intentionally require private participant matrices, identity-grouped memberships, endpoints, original splits, source hashes, validation seals and execution authorizations. These are not shipped. Do not disable missing-input or integrity guards to make a public clone run.

Some source scripts perform verification or create output directories at module import; do not import the study verifiers for exploration. Inspect their text or use the safe synthetic check below. Historical preparation metadata may say training was not authorized: it records that stage, not the status of a later authorized local execution. This code release does not authorize a new study run.

The GA28 protocol is published as `Protocol.template.json`. Its `frozen_fold_panels` mapping is deliberately empty because selected masks, fitted feature sets and historical penalties are study outputs. Restore the authorized original private protocol and its seals for reproduction; a newly invented mask would define a different experiment. The two other protocol files contain methods and aggregate eligibility counts, not outcome metrics.

Two readiness scripts refer to `clinical_extensions/Stage_C_protocol_v1` in the private working tree; this replaces a journal-named directory. Public earlier Stage C scripts remain under `analysis/later_analyses_2026-11-15/`. Original 48-month inputs and historical GA bindings are additional private dependencies. `CODE_MANIFEST.json` records byte-identical copies versus path normalization or template redaction. Original study source files were not modified.

## EBM and TabPFN execution

The protocol uses five outer and three inner folds. Ridge and EBM are selected by pooled inner MAE; TabPFN v2 uses a fixed local checkpoint and eight estimators without weight fine-tuning. Its fit calls attach training context, not train new neural weights. No hosted prediction API is used. Checkpoint provenance gives the public source and SHA-256, without distributing weights. Obtain the model separately under its provider terms.

Use a separate Python 3.10 environment matching `RuntimeVersions.json` (interpret-core 0.7.4 and tabpfn 2.2.1). The original Python 3.9 environment and root requirements serve the older kernels; they are not the complete runtime specification for this extension. No environment installation or checkpoint download is performed by the public checks.

The archived execution code routes ridge to validated FP64 GPU calculations and EBM/TabPFN to CPU. The proposed TabPFN GPU path was not used for study inference after its synthetic agreement gate failed. `execution_parallel.py` changes CPU scheduling only; `verify_stream_v3.py` supports incremental verification with a privately configured worker count. Machine-specific process-management, recruitment and stop scripts are intentionally omitted.

The two co-primary contrasts use Bonferroni-adjusted 97.5% paired bootstrap intervals. Other contrasts have descriptive 95% intervals. Intervals condition on fixed predictions. The independent verifier reconstructs preprocessing and ridge by SVD, refits selected outer EBM/TabPFN models, audits inner selection traces and recalculates metrics. It does not independently rerun every nonlinear inner fit.

## Safe public checks

From the repository root, using an existing NumPy/pandas environment:

```bash
python scripts/smoke_check.py
python scripts/check_extension_kernels.py
```

The first check exercises the earlier numerical and binding kernels and verifies the release manifest. The second checks copied imaging/algorithm ridge kernels against an independent synthetic SVD reference, training-only preprocessing, paired metric calculations, tuning criteria and tie handling. It extracts only required pure function definitions through Python AST, avoiding study-module initialization, private inputs and model checkpoints. It does not validate installed EBM/TabPFN backends or reproduce study performance.

No original data, participant predictions, fitted objects, analysis locks or execution logs are distributed here. Existing designated aggregate supplements remain at their earlier revision. New manuscript supplements should be obtained from the authors; their results have not been added to this repository.
