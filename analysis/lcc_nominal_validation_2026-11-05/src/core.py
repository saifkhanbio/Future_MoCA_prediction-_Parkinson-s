"""Numerical core for the locked reduced-model LCC transport protocol.
No file I/O, data discovery, network access, or automatic study execution.
"""
import numpy as np

HORIZONS = (12, 24, 36)
FEATURES = ("baseline_moca", "age_years", "sex_male", "gds15")
RECIPES = {"reduced3": FEATURES[:3], "reduced4": FEATURES}
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
TIE_TOL = 1e-12
BOOTSTRAPS = 20000

def finite_array(x, ndim):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != ndim or not np.all(np.isfinite(x)):
        raise ValueError("Wrong shape or nonfinite numerical input")
    return x

def ridge_fit(x, y, penalty):
    x, y = finite_array(x, 2), finite_array(y, 1)
    if len(x) != len(y) or len(y) < 2 or x.shape[1] < 1:
        raise ValueError("Training shapes")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Penalty must be positive")
    mean, raw_scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    constant = raw_scale == 0
    scale = np.where(constant, 1.0, raw_scale)
    z = (x - mean) / scale
    z[:, constant] = 0.0
    intercept = float(y.mean())
    # MSE + lambda*||beta||^2, not SSE + lambda*||beta||^2.
    gram = z.T @ z / len(y) + penalty * np.eye(x.shape[1])
    rhs = z.T @ (y - intercept) / len(y)
    chol = np.linalg.cholesky(gram)
    coef = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "constant": constant.tolist(), "coefficient": coef.tolist(),
            "intercept": intercept, "lambda": float(penalty), "n_train": len(y)}

def ridge_predict(model, x):
    x = finite_array(x, 2)
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    coefficient = np.asarray(model["coefficient"])
    constant = np.asarray(model["constant"], dtype=bool)
    if x.shape[1] != len(mean) or scale.shape != mean.shape or coefficient.shape != mean.shape:
        raise ValueError("Model/feature dimensions differ")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Invalid saved scale")
    z = (x - mean) / scale
    z[:, constant] = 0.0
    prediction = float(model["intercept"]) + z @ coefficient
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Nonfinite prediction")
    return prediction  # no clipping

def choose_penalty(mses):
    a = finite_array(mses, 1)
    if len(a) != len(GRID):
        raise ValueError("Fixed grid length mismatch")
    best = float(a.min())
    return max(i for i, x in enumerate(a) if x <= best + TIE_TOL)

def tune_and_fit(x, y, folds, components, ids):
    x, y = finite_array(x, 2), finite_array(y, 1)
    folds, components, ids = np.asarray(folds), np.asarray(components), np.asarray(ids)
    if not (len(y) == len(x) == len(folds) == len(components) == len(ids)):
        raise ValueError("Tuning arrays differ")
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Exactly five folds required")
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated source participant")
    if len(set(components)) != len(components):
        raise ValueError("Source requires one person per known component")
    errors = np.zeros(len(GRID), dtype=np.float64)
    audit = []
    count = 0
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if test.sum() < 10 or train.sum() < 10:
            raise ValueError("Fold below locked minimum")
        if set(components[train]) & set(components[test]):
            raise ValueError("Identity leakage")
        for j, penalty in enumerate(GRID):
            model = ridge_fit(x[train], y[train], penalty)
            error = ridge_predict(model, x[test]) - y[test]
            errors[j] += float(error @ error)
            count += 1
        # Save fold-local preprocessing evidence once per fold.
        audit.append({"fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum()),
                      "train_ids": sorted(ids[train].tolist()), "test_ids": sorted(ids[test].tolist()),
                      "training_mean": model["mean"], "training_scale": model["scale"],
                      "training_y_mean": model["intercept"]})
    mses = errors / len(y)
    chosen = choose_penalty(mses)
    final = ridge_fit(x, y, GRID[chosen])
    return final, {"lambda_grid": list(GRID), "pooled_cv_mse": mses.tolist(),
                   "selected_lambda": GRID[chosen], "ridge_fits": count + 1,
                   "cv_is_tuning_only": True, "fold_audit": audit}

def metrics(y, prediction):
    y, prediction = finite_array(y, 1), finite_array(prediction, 1)
    if len(y) < 1 or len(y) != len(prediction):
        raise ValueError("Metric shape mismatch")
    e = prediction - y
    sse = float(e @ e)
    centered = y - y.mean()
    sst = float(centered @ centered)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e * e))),
            "R2": None if sst == 0 else float(1 - sse / sst),
            "bias": float(e.mean()),
            "outside_0_30_fraction": float(np.mean((prediction < 0) | (prediction > 30)))}

def cluster_draws(components, horizon, repetitions=BOOTSTRAPS):
    # Stable ordering makes file row order irrelevant.
    if horizon not in HORIZONS or repetitions < 1:
        raise ValueError("Bootstrap settings")
    groups = np.array(sorted(set(str(x) for x in components)))
    if len(groups) < 2:
        raise ValueError("At least two known identity groups required")
    rng = np.random.Generator(np.random.PCG64(20261105 + int(horizon)))
    return groups, rng.integers(0, len(groups), size=(repetitions, len(groups)))

def bootstrap_statistics(y, predictions, components, horizon, repetitions=BOOTSTRAPS):
    """Resample entire components; all arms share draws, including unequal clusters."""
    y = finite_array(y, 1)
    components = np.asarray(components, dtype=str)
    if len(components) != len(y) or set(predictions) != {"reduced3", "reduced4", "persistence", "typical_change"}:
        raise ValueError("Bootstrap membership/arms")
    predictions = {k: finite_array(v, 1) for k, v in predictions.items()}
    if any(len(v) != len(y) for v in predictions.values()):
        raise ValueError("Prediction alignment")
    groups, draws = cluster_draws(components, horizon, repetitions)
    sizes = np.array([sum(components == g) for g in groups], dtype=np.float64)
    per = {k: np.full((repetitions, 5), np.nan) for k in predictions}
    # Compute centered outcome moments for numerical stability.
    center = float(y.mean())
    y0 = y - center
    sums_y = np.array([y0[components == g].sum() for g in groups])
    sums_yy = np.array([(y0[components == g] ** 2).sum() for g in groups])
    totals_n = sizes[draws].sum(axis=1)
    sy = sums_y[draws].sum(axis=1)
    syy = sums_yy[draws].sum(axis=1)
    sst = syy - sy * sy / totals_n
    # Zero-variance bootstrap draws only; no truncation of genuine small positive variance.
    sst[np.abs(sst) <= np.finfo(float).eps * np.maximum(syy, 1) * 8] = 0
    if np.any(sst < 0):
        raise ValueError("Numerically invalid variance")
    for name, prediction in predictions.items():
        err = prediction - y
        grouped = np.array([[np.abs(err[components == g]).sum(),
                             (err[components == g] ** 2).sum(), err[components == g].sum(),
                             np.sum((prediction[components == g] < 0) | (prediction[components == g] > 30))]
                            for g in groups])
        acc = grouped[draws].sum(axis=1)
        vals = per[name]
        vals[:, 0] = acc[:, 0] / totals_n
        vals[:, 1] = np.sqrt(acc[:, 1] / totals_n)
        np.divide(acc[:, 1], sst, out=vals[:, 2], where=sst > 0)
        vals[:, 2] = np.where(sst > 0, 1 - vals[:, 2], np.nan)
        vals[:, 3] = acc[:, 2] / totals_n
        vals[:, 4] = acc[:, 3] / totals_n
    delta = per["reduced4"][:, 0] - per["reduced3"][:, 0]
    return per, delta

def percentile_interval(values, tail=0.025, required_fraction=1.0):
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if not len(values) or good.mean() < required_fraction:
        return {"interval": None, "valid_replicates": int(good.sum()), "replicates": len(values)}
    return {"interval": np.quantile(values[good], [tail, 1-tail], method="linear").tolist(),
            "valid_replicates": int(good.sum()), "replicates": len(values)}

def evaluate_arrays(y, predictions, components, horizon):
    point = {k: metrics(y, v) for k, v in predictions.items()}
    boot, delta = bootstrap_statistics(y, predictions, components, horizon)
    names = ("MAE", "RMSE", "R2", "bias", "outside_0_30_fraction")
    intervals = {}
    for name, array in boot.items():
        intervals[name] = {}
        for j, metric in enumerate(names):
            if metric != "R2" and not np.all(np.isfinite(array[:, j])):
                raise ValueError("Invalid bootstrap metric")
            intervals[name][metric] = percentile_interval(array[:, j], required_fraction=0.95 if metric == "R2" else 1.0)
    return {"horizon_months": horizon, "n": len(y), "known_components": len(set(components)),
            "point_metrics": point, "pointwise_95_intervals": intervals,
            "delta_MAE_reduced4_minus_reduced3": point["reduced4"]["MAE"] - point["reduced3"]["MAE"],
            "delta_MAE_pointwise_95": percentile_interval(delta),
            "delta_MAE_Bonferroni_98_333333": percentile_interval(delta, tail=0.05 / (2*3)),
            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": 20261105+horizon,
            "conditional_on_fixed_source_models": True}, boot, delta
