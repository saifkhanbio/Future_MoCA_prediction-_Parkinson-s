# Reproduction scope and workflow

## Public release boundary

The scripts are selected historical snapshots, not a newly fitted or refactored prediction pipeline. Their algorithmic source is byte-identical to the locally frozen scripts. The manifest identifies path-only changes in three JSON contracts. The publication directory was created separately; frozen analyses were not overwritten.

Only the named aggregate supplementary files are included as results. Raw PPMI/AMP-PD records, individual outcomes/predictions, identifiers, identity components, actual fold assignments, participant-bound model bundles, source bindings, execution authorizations, historical seals and checkpoint/search databases are deliberately absent. Table S14 includes the six LCC clinical-model equations and source offsets as an explicit aggregate supplement. These do not supply controlled inputs, identity/fold manifests, provenance or execution seals. Gene-set names do not supply membership definitions; obtain the exact licensed release independently.

## Workflow map

All package names below are children of `analysis/`. Keep this sibling layout: some original modules import code from their earlier package. Run scripts from their intended package entry point; do not mix same-named `engine`, `support` or `binding` modules from different versions.

| Stage | Package(s) | Relevant entry points / purpose |
| --- | --- | --- |
| Original eligibility | `participant_eligibility_2026-10-24` | `scripts/run_audit.py`, audit verification and rule-edge checks |
| Clinical/RNA main comparison | `model_readiness_2026-10-24`, `main_comparison_2026-10-28` | `scripts/prepare_design.py`, `scripts/launcher.py`, numerical/reference and independent verification scripts |
| Earlier MAE GA | `ga_readiness_2026-10-30`, `ga_hybrid_launcher_resume_fix_2026-10-30` | `src/ga_engine.py`, `src/launch_hybrid.py`, cache/search/backend modules |
| Earlier RÂ² GA | `ga_r2_readiness_2026-10-30` | `src/launch_hybrid.py`, `r2_engine.py`, `binding.py` |
| Fixed feature recipes | `feature_only_readiness_2026-10-31` | `src/launch.py`, `engine.py`, `numerics.py` |
| MoCA-mandatory GA | `ga_moca_readiness_v2_2026-10-31` | `src/launch_hybrid.py` and versioned search/binding modules |
| Final v4 GA/random search | `ga_unrestricted_readiness_v4_10kgen_2026-11-01` | `src/launch_hybrid.py`, `search_core.py`, guarded CPU/GPU backend and resume handling |
| Independent v4 verification | `ga_unrestricted_verification_v4_2026-11-01` | Context, metric, completion-gate and historical WAL verification scripts |
| Frozen model comparison | `model_comparison_2026-10-31`, `model_comparison_v4_2026-11-01` | Comparison and independently implemented verification; require existing private predictions |
| Independent LCC evaluation | `lcc_nominal_validation_2026-11-05` | Frozen `src/core.py`, `src/io_contract.py` and `src/launcher.py`; see [the LCC guide](LCC_VALIDATION.md) before use |
| Later-calendar inference | `temporal_validation_readiness_2026-10-28` | `src/clinical_inference.py`, guarded launcher and numerical/pipeline checks; execution subfolder contains verification code only |

The final v4 profile records 20 main and five confirmation seeds, population/offspring size 1,024, a 200,000-distinct-mask ceiling per restart, a 10,000-generation ceiling, minimum-search and plateau checks, and a global active-time safeguard. Ceilings do not imply every run evaluated every possible mask. The recorded profile uses 16 CPU workers plus one GPU worker, with 18 CPU-only fallback workers. Earlier hybrid profiles differ; use each version's own configuration. A new reproduction must size resources appropriately and disclose any changes, rather than silently claiming the frozen configuration.

## Inputs required for a controlled reproduction

1. Obtain authorized PPMI records with the exact export vintage and source definitions; keep source tables outside the public repository. The original RNA-linked sampling frame must not be silently replaced with a larger clinical cohort.
2. Reconstruct and verify participant identity, enrollment/diagnosis rules, source provenance, RNA month-end landmarks, baseline predictor timing and education-adjusted MoCA endpoints. Monthly precision does not resolve within-month ordering.
3. Rebuild the matched per-horizon manifests and the original outcome-independent grouped fold allocation. No actual participant assignments are published here. Changing source vintages or participant/fold membership creates a new analysis, not a byte-for-byte reproduction.
4. For RNA comparisons, obtain the bound counts, sample mappings, selected gene versions and licensed Hallmark memberships. Fit the gene-expression eligibility/scaling rules within training partitions; do not substitute globally fitted transformations.
5. Recreate private input bindings, locks and authorization records in a separate controlled workspace. The public files are insufficient to satisfy the historical integrity guards. Never fabricate hashes, disable guards, or commit these private files to make a launcher run.
6. Validate the numerical and binding checks; execute only the intended, separately authorized analysis. Independent verification uses the same controlled artifacts. No study training, scoring, data download or cloud query is part of preparing this public release.

## Numerical check scope

`scripts/smoke_check.py` verifies publication hashes, compiles all released Python sources in memory, checks frozen training-only imputation/scaling on synthetic rows, compares two released ridge kernels with an independent augmented least-squares solution, checks metric arithmetic, verifies the v4 mandatory-MoCA mapping, and checks temporal identity binding rejects shuffled identity mismatches. It reads only the repository and synthetic in-memory inputs. It does not test CUDA execution, checkpoint recovery, full historical data bindings, or reproduce the study estimates.

## Interpretation boundaries

The source clinical/RNA, earlier search, final v4 search, temporal evaluation and LCC evaluation have separate historical contracts. They must not be presented as a single prospectively registered experiment. The final model-comparison scripts summarize existing predictions and do not perform a new external evaluation. Temporal evaluation concerns the earlier core/enhanced clinical ensembles only. Study-level conclusions must retain prior-use, multiplicity, uncertainty and cohort-selection limitations described in the manuscript and appendix.

The LCC source code is byte-identical to the validated implementation. Its public placement preserves `src/` structure, but the adjacent private protocol, manifests, source bindings and authorization seals are absent. No bypass or dummy authorization is supplied. The repository smoke test imports only the I/O-free LCC numerical core and evaluates synthetic arrays.
