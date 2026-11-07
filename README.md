# Future MoCA prediction in Parkinson's disease

Analysis code, methods documentation and designated aggregate supplements for **Cognitive prognosis in Parkinson disease: Model benchmarking and independent validation**. This repository is independent of any journal or submission destination.

This repository documents **Clinical_hallmark**: exploratory PPMI benchmarking of future Montreal Cognitive Assessment (MoCA) scores at **12, 24 and 36 months**, within-PPMI temporal evaluation, and a separately specified independent clinical-model evaluation in the **LRRK2 Cohort Consortium (LCC)**. The endpoint is the continuous education-adjusted future MoCA level, not a diagnosis of dementia or MCI and not necessarily within-person decline. This is a manuscript code release, not a clinical prediction service or a claim of peer-reviewed publication.

## What is included

- The original analysis, numerical, launcher and independent-verification scripts relevant to the manuscript.
- Methods-only contracts, predictor definitions and search profiles. Three contracts replace the author's local filesystem root with `<PRIVATE_STUDY_ROOT>`; their numerical settings are unchanged.
- The current supplementary methods and **13 aggregate CSV tables**, including the independent LCC evaluation. The appendix is presented without journal branding; CSV values are copied unchanged from the verified manuscript supplement.
- The frozen LCC numerical core and guarded I/O/launcher code, with an execution-scope guide.
- Documentation and a small, data-free synthetic numerical check.

**No raw or participant-level source data, individual predictions, participant/fold manifests, participant-bound fitted-model bundles, search databases, checkpoints, credentials, or results outside the designated supplement are included.** Access to PPMI data must be obtained independently through [PPMI](https://www.ppmi-info.org/access-data-specimens/download-data). LCC and AMP-PD access must likewise be obtained through their respective providers. Repository access does not grant access to controlled data or permit their redistribution. The aggregate equations in supplementary Table S14 are the explicit model-parameter exception; they contain no participant lists or training records.

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

MAE- and MSE-tuned recipes are distinct. For a fixed inner evaluation set, minimizing pooled MSE is equivalent to maximizing pooled RÂ². Acquisition-group burden is the second GA objective; the final selected model follows the version-specific rule, not an assumed common rule across all searches. Final v4 uses bounded stochastic search with convergence checks; neither a large mask budget nor convergence establishes a global optimum. See the [final v4 contract](analysis/ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/Analysis_Contract.json) and [search profile](analysis/ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/r2_expanded_v1.json).

A separate later-calendar PPMI evaluation applies the earlier frozen core/enhanced clinical ensembles. It is **within-PPMI temporal evaluation**, not external validation of the final GA or compact models. Those models were not evaluated in that temporal cohort.

### Independent external clinical validation

A separately locked experiment trained six horizon-specific clinical models using **PPMI only**, then evaluated them in LCC without refitting or recalibration. The two predictor sets were baseline MoCA, age and sex, with or without GDS15. Known PPMI links and unresolved identities were excluded before evaluation. Nominal LCC visits defined 12-, 24- and 36-month endpoints; these are not exact calendar windows. This external experiment evaluates these specified clinical models, while the compact cognitive panel and final GA recipes retain their internal evaluation status.

PPMI supplies development and temporal data, LCC supplies the external cohort, and AMP-PD harmonized records and identity/QC metadata support reconciliation and overlap checks. AMP-PD copies of PPMI records do not create an independent cohort. See [LCC methods and execution scope](docs/LCC_VALIDATION.md).

## Repository map

| Location | Contents |
| --- | --- |
| [`analysis/`](analysis) | Versioned original scripts and selected methods configurations; package names preserve historical dependencies |
| [`supplement/`](supplement) | Current aggregate tables and a journal-neutral methods appendix |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Workflow map, controlled-input requirements and reproduction limits |
| [`docs/SUPPLEMENT_INVENTORY.md`](docs/SUPPLEMENT_INVENTORY.md) | Description of each supplementary file |
| [`CODE_MANIFEST.json`](CODE_MANIFEST.json) | Source/public SHA-256 hashes, logical source labels and recorded editorial normalization |
| [`scripts/smoke_check.py`](scripts/smoke_check.py) | Synthetic PPMI/LCC kernels, preprocessing, identity-binding and checksum checks |

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

## PPMI data source and acknowledgement

Data used in the preparation of this article were obtained on 2026-10-20 (clinical and RNA data), with a MoCA update obtained on 2026-10-28 for the temporal evaluation, from the Parkinsonâ€™s Progression Markers Initiative (PPMI) database (https://www.ppmi-info.org/access-data-specimens/download-data), RRID:SCR_006431. For up-to-date information on the study, visit https://www.ppmi-info.org. Data were accessed under approved PPMI data-access permissions; the investigator holds Tier 2 approval.

PPMI â€“ a public-private partnership â€“ is funded by the Michael J. Fox Foundation for Parkinsonâ€™s Research and funding partners, including AbbVie, Alamar Biosciences, Aligning Science Across Parkinsonâ€™s (ASAP), Arrowhead Pharma, Arvinas, AskBio, BIAL, BioArctic, Biohaven, BlueRock Therapeutics, Bristol Myers Squibb, Calico Labs, Capsida Biotherapeutics, Critical Path Institute, DaCapo Brainscience, Denali, Edmond J. Safra Foundation, Eli Lilly, Gain Therapeutics, GE Healthcare, Genentech, GSK, Insitro, Johnson & Johnson Innovative Medicine, Lundbeck, Merck, Neumora, Neuron23, Novartis, Olink, Regeneron, Roche, Sanofi, Tenvie, UCB, Vanqua Bio, Voyager Therapeutics, The Weston Family Foundation.

The RNA download date was confirmed by the corresponding investigator. Clinical export provenance identifies 20 October 2026; the later-calendar evaluation uses the MoCA refresh dated 28 October 2026. These are download/export dates, not participant visit dates. The preserved scripts were created during Octoberâ€“November 2026; versioned package names and the code manifest identify their provenance. The evolving PPMI database may require code adaptation for a different export vintage.

PPMI study reference: K. Marek, D. Jennings, S. Lasch, A. Siderowf, C. Tanner, T. Simuni, et al., *The Parkinson Progression Marker Initiative (PPMI)*, Progress in Neurobiology 95 (2011) 629â€“635. [doi:10.1016/j.pneurobio.2011.09.005](https://doi.org/10.1016/j.pneurobio.2011.09.005).

Attribution was checked on 4 November 2026 against the [PPMI Publication Policy (September 2024)](https://www.ppmi-info.org/sites/default/files/docs/ppmi-publication-policy.pdf) and the [current funding-partner list (May 2026)](https://www.ppmi-info.org/sites/default/files/docs/PPMI%20Funding%20Partners.pdf). Data-access approval does not imply PPMI consortium authorship or institutional ethics approval. No participant-level source data are redistributed here.

## LCC and AMP-PD data sources

The LCC Ashkenazi Jewish New York longitudinal release dated 26 June 2017 was obtained through approved MJFF LCC access on 4 November 2026. We thank its participants, investigators and staff. LCC coordination and funding are provided by The Michael J. Fox Foundation for Parkinson's Research. See the [LCC study](https://www.michaeljfox.org/lcc) and [investigator resources](https://www.michaeljfox.org/lccinvestigators).

The locally retained AMP-PD release `2023_v4release_1027` supplied harmonized clinical records and identity/QC metadata. AMP-PD is a public-private program managed by the Foundation for the National Institutes of Health, with NINDS, FDA, NIA, ASAP and industry/nonprofit partners. Provider attribution is retained in the manuscript; [AMP-PDRD](https://amp-pdrd.org) provides program and access information. These source roles do not imply consortium authorship or endorsement.

## Attribution and contact

Study/repository maintainer: **Saif Khan**, University of Ha'il. Contact: **sf.khan@uoh.edu.sa**.

PPMI is the source study; its access and acknowledgement requirements continue to apply. The original gene-set membership files and instrument/source exports are not redistributed. No software license grant is added by this release; contact the maintainer regarding reuse permissions. Cite the final manuscript when its publication details become available; no DOI is claimed here.
