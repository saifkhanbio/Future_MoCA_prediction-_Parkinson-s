# Supplementary material

Document S1 integrates the original and expanded PPMI benchmarks, temporal validation, and the independent PDBP and LCC clinical-model evaluations. Its journal name has been removed from the document metadata; text, tables and figures are unchanged from the verified manuscript supplement. CSVs are copied without numerical edits. Source and released hashes are recorded in `CODE_MANIFEST.json`; source labels identify logical artifacts without journal-specific directory names.

| File | Content |
| --- | --- |
| `Document_S1.docx` | Editable supplementary methods, Tables S1–S2 and supplementary figures |
| `Document_S1.pdf` | PDF of Document S1 |
| `Document_S1.txt` | Paragraph and table text from Document S1 |
| `Table_S3_All_54_Internal_Metrics.csv` | Metrics for 18 internal model recipes at three horizons |
| `Table_S4_All_459_Paired_Contrasts.csv` | All paired internal exploratory contrasts |
| `Table_S5_Feature_Selection_Frequencies.csv` | Feature selection frequencies |
| `Table_S5b_All_270_Feature_Sets.csv` | Fold-level selected features, burden and penalties |
| `Table_S6_Temporal_Metrics_All_Sensitivities.csv` | Temporal metrics and sensitivity evaluations |
| `Table_S7_Original_RNA_Contrasts.csv` | Original RNA-extension contrasts |
| `Table_S8_Analysis_Lineage.csv` | Distinct questions, primary contrasts and analysis status |
| `Table_S9_Descriptive_Calibration.csv` | Aggregate descriptive calibration |
| `Table_S10_Tuning_Comparison.csv` | MAE-versus-MSE tuning comparison |
| `Table_S11_LCC_Metrics.csv` | Independent LCC evaluation metrics with pointwise intervals |
| `Table_S12_LCC_Paired_GDS.csv` | GDS15 contrasts with pointwise and adjusted intervals |
| `Table_S13_Reduced_Cohort_Characteristics.csv` | Source/target cohort summaries by horizon |
| `Table_S14_Frozen_Reduced_Model_Equations.csv` | Six clinical-model equations, penalties and source offsets |
| `Table_S15_Frozen_A_B.csv` | Temporal-validation and PDBP metrics and paired contrasts |
| `Table_S16_Expanded_C.csv` | Expanded-PPMI metrics and paired contrasts |
| `Table_S17_Stage_comparison.csv` | Model versus carrying baseline forward, by cohort and horizon |
| `Table_S18_Calibration.csv` | Calibration and mean signed error |
| `Table_S19_Followup_stability.csv` | Follow-up availability and score stability |
| `Table_S20_Change_distribution.csv` | Distribution of observed MoCA change |
| `Table_S21_Baseline_bands.csv` | Paired error by baseline MoCA band |
| `Figure_S4_source.csv` | Aggregate values plotted in Figure S4 |

The three frozen LCC implementation scripts are in `analysis/lcc_nominal_validation_2026-11-05/src/`, with [execution guidance](LCC_VALIDATION.md).

These are designated aggregate supplementary results. They contain no participant records, identity mappings, individual predictions or fold-membership assignments. Fold numbers describe model evaluations. Approved source access is required separately.
