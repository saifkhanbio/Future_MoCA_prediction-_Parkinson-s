# Future MoCA prediction in Parkinson's disease

Analysis code and the designated aggregate supplementary material accompanying **Cognitive assessment and feature selection for future MoCA prediction in Parkinson's disease**, prepared for *Parkinsonism & Related Disorders*.

This repository documents **Clinical_hallmark**, an exploratory retrospective PPMI benchmark of future Montreal Cognitive Assessment (MoCA) scores at **12, 24 and 36 months**. The endpoint is the continuous education-adjusted future MoCA level, not a diagnosis of dementia or MCI and not necessarily within-person decline. This is a manuscript code release, not a clinical prediction service or a claim of peer-reviewed publication.

## What is included

- The original analysis, numerical, launcher and independent-verification scripts relevant to the manuscript.
- Methods-only contracts, predictor definitions and search profiles. Three contracts replace the author's local filesystem root with `<PRIVATE_STUDY_ROOT>`; their numerical settings are unchanged.
- The **12 files from the designated manuscript `supplement` folder**, including its appendix and aggregate CSV tables. These are the explicitly included supplementary results.
- Documentation and a small, data-free synthetic numerical check.

**No raw or participant-level source data, individual predictions, participant/fold manifests, fitted models, search databases, checkpoints, credentials, or other study results are included.** Access to PPMI data must be obtained independently through [PPMI](https://www.ppmi-info.org/access-data-specimens/download-data). Repository access does not grant access to controlled data or permit their redistribution.

## Study design

The source frame is an RNA-linked PPMI population retained for matched comparisons. Clinical prediction and feature selection use five outer folds and three inner folds, grouped by known participant identity. The same master fold allocation is subsetted across horizons. Imputation, scaling, penalty tuning and feature selection are performed within the appropriate training partitions. Numeric missingness indicators and fixed categorical dictionaries are specified in the contracts.

The manuscript compares:

| Family | Predictors / purpose |
| --- | --- |
| Persistence and typical change | Baseline MoCA; baseline MoCA plus a training-source median change |
| Core clinical | Age, sex, education, baseline MoCA, diagnosis duration, MDS-UPDRS II/III, GDS-15, medication status and motor examination state |
| Enhanced clinical | Core plus SDMT and HVLT immediate and delayed recall |
| Compact cognitive | Baseline MoCA, SDMT, HVLT immediate and delayed recall |
| Expanded and feature-only recipes | Core plus 18 optional features; MoCA plus 18; or the optional 18 alone |
| GA and matched random search | Earlier core-mandatory searches; MoCA-mandatory optional-feature search; final v4 unrestricted 28-feature and MoCA-mandatory 27-bit searches |
| Original RNA extension | Fixed five-gene and 11-Hallmark representations added to the clinical comparators, including an RNA-off option |

The optional panel includes cognitive, sleep, autonomic and neuropsychiatric assessments, APOE dosage and neutrophil-to-lymphocyte ratio. It is not exclusively a questionnaire panel. Form controls and missing indicators are additional design columns, not extra substantive predictor counts. See [feature definitions](analysis/model_comparison_v4_2026-11-01/config/Feature_Definitions.json).

MAE- and MSE-tuned recipes are distinct. For a fixed inner evaluation set, minimizing pooled MSE is equivalent to maximizing pooled R². Acquisition-group burden is the second GA objective; the final selected model follows the version-specific rule, not an assumed common rule across all searches. Final v4 uses bounded stochastic search with convergence checks; neither a large mask budget nor convergence establishes a global optimum. See the [final v4 contract](analysis/ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/Analysis_Contract.json) and [search profile](analysis/ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/r2_expanded_v1.json).

A separate later-calendar PPMI evaluation applies the earlier frozen core/enhanced clinical ensembles. It is **within-PPMI temporal evaluation**, not external validation of the final GA or compact models. Those models were not evaluated in that temporal cohort.

## Repository map

| Location | Contents |
| --- | --- |
| [`analysis/`](analysis) | Versioned original scripts and selected methods configurations; package names preserve historical dependencies |
| [`supplement/`](supplement) | The designated aggregate supplementary files, copied unchanged |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Workflow map, controlled-input requirements and reproduction limits |
| [`docs/SUPPLEMENT_INVENTORY.md`](docs/SUPPLEMENT_INVENTORY.md) | Description of each supplementary file |
| [`CODE_MANIFEST.json`](CODE_MANIFEST.json) | Original/publication SHA-256 hashes and any path-only normalization |
| [`scripts/smoke_check.py`](scripts/smoke_check.py) | Synthetic kernel, preprocessing, identity-binding and checksum checks |

## Quick start: synthetic checks only

The study used Python 3.9.23 in Linux/WSL (`agpu_env`), NumPy 1.26.4, pandas 2.3.1 and SciPy 1.13.1. The hybrid search used PyTorch 2.8.0+cu128. These are observed environment versions; no patient analysis is launched by the commands below.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_check.py
```

The smoke check fits only tiny synthetic ridge examples and writes no model or study output. PyTorch/GPU access is not needed for it. The guarded historical hybrid launchers require Linux/WSL, compatible PyTorch/CUDA, controlled inputs and private bindings; they are not a one-command public reproduction. Do not bypass their missing-input or integrity guards. Installing dependencies is optional if an appropriate environment already exists.

## Interpretation and reproducibility limits

This is an exploratory benchmarking study using participants/outcomes exposed during earlier analyses. Nested cross-validation controls fold-level tuning leakage; it does not erase that research history. The original primary analysis and subsequent exploratory comparisons retain their different status. The 12-month GA findings are exploratory, not an independently confirmed new primary endpoint.

Paired bootstrap intervals condition on frozen predictions, are unadjusted for the many exploratory comparisons, and do not capture full refitting or research-selection uncertainty. Acquisition groups are a proxy for burden, not measured minutes or cost. Numerical superiority alone is not proof of clinical utility. The scripts do not establish a universally optimal feature set or generalizability beyond the evaluated population.

A public clone can inspect methods, verify the supplied file checksums and exercise synthetic numerical kernels. It **cannot reproduce participant-level results without separately authorized data and the exact private source, cohort, split and integrity bindings**. Selected source preprocessing code is included; the complete historical data-acquisition and audit workspace is not. Some historical readiness files say training was not yet authorized: these describe their creation stage and do not override later study execution records, which are not distributed.

## Attribution and contact

Study/repository maintainer: **Saif Khan**, University of Ha'il. Contact: **sf.khan@uoh.edu.sa**.

PPMI is the source study; its access and acknowledgement requirements continue to apply. The original gene-set membership files and instrument/source exports are not redistributed. No software license grant is added by this release; contact the maintainer regarding reuse permissions. Cite the final manuscript when its publication details become available; no DOI is claimed here.
