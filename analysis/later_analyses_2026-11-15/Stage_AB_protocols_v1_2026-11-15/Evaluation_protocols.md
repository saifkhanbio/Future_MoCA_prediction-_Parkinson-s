# Locked evaluation protocols for Stage A and Stage B

The two evaluations ask whether a frozen model improves on predicting that a person's MoCA score will stay at baseline. Each has one leading contrast at 12 months, the model's mean absolute error minus the baseline forecast's mean absolute error, in MoCA points. Negative values favor the model. The 24- and 36-month contrasts are supporting estimates. This hierarchy was set after the eligibility audit and before these models were scored in either target population.

| Locked population or comparison | 12 months | 24 months | 36 months |
|---|---:|---:|---:|
| A, four scores versus persistence in later PPMI | 213 | 175 | 39 |
| A, complete inputs for all three panels | 69 | 62 | 24 |
| A, complete inputs for the compact and 19-predictor panels | 70 | 62 | 24 |
| A, center/year collection-supported comparison of all three panels | 131 | 115 | 37 |
| A, center/year collection-supported compact and 19-predictor comparison | 132 | 115 | 37 |
| B, Reduced3 versus persistence at the scheduled PDBP visit | 180 | 153 | 119 |
| B, demographic education category above 16 years | 89 | 72 | 56 |

**Stage A.** The four-score model uses baseline MoCA, SDMT and HVLT immediate and delayed recall. Exactly five previously fitted PPMI members contribute equally per model and horizon. Every member uses its own stored design column order, imputation parameters, scaling, category indicators and coefficients. The 19-predictor and 28-predictor comparisons use identical participants across arms within each specified view. The principal secondary panel comparison is compact versus 28 on the fully observed sample. The sensitivity samples permit sporadic missing inputs under the audited center/year rule. Persistence and the frozen source typical-change forecast serve as benchmarks in every view.

The clinical landmark and refreshed MoCA ledger bind the outcome to calendar windows of 9–15, 21–27 and 33–39 months. The 227 later PPMI participants are separate from development but have already contributed outcomes to earlier core/enhanced temporal tests. This evaluation therefore addresses exploratory temporal transport of the selected panel. The original 36-month independent-cohort question remains open.

**Stage B.** Use the separately verified PPMI equation for each horizon with baseline MoCA, completed age and sex. Use the source model's exact preprocessing and coefficients. Compare it with baseline persistence and the horizon-specific mean change from the PPMI reduced-model training cohort. The 201 PDBP participants satisfy the frozen baseline PD, identity, age, sex, education and MoCA quality rules. M0 and the unique annual visit define the landmark and outcome. The selected MoCA totals agree with their domain sums and education codes. The source training landmark was anchored to the baseline GDS assessment. Actual visit dates are unavailable in PDBP, so inference concerns nominal annual visits.

**Estimation.** Evaluate raw predictions on the observed 0–30 MoCA scale. Report MAE, RMSE, predictive R², signed bias, descriptive calibration intercept and slope, and the fraction of predictions outside the scale. Every pair of methods within each view receives paired MAE and R² differences in the model order listed in the stage JSON. At each horizon and view, resample identity components together across models with replacement. Use 5,000 draws for A and 20,000 for B. Report pointwise 95% percentile intervals conditional on the frozen model and observed cohort. Resamples with undefined estimates are counted. Report an interval only when the point estimate is defined and at least 95% of draws are valid. No pooling across horizons, target recalibration or formal equivalence claim is prespecified.

**Execution.** The locked package binds exact feature rows, outcome rows, model states, model specifications, audited memberships and source hashes. Study inputs are in the private restricted folder. The launcher joins predictions and outcomes by component and horizon. For synthetic testing run `python -B launch.py --stage A --mode synthetic` or select stage B. The preflight mode checks file integrity without computing predictions. Real evaluation is a separate action that requires explicit authorization and the sealed protocol digest. Study outputs are published once to a new stage-specific file.

Detailed machine-readable definitions and seeds are in Protocol_A.json and Protocol_B.json. LOCK.json identifies the exact inputs and executable code. Synthetic_validation.json reports the subsequent numerical and integrity checks.
