import numpy as np

def design(x, train, test, panel):
    nums = panel['numeric']
    a = x.loc[train, nums].to_numpy(dtype=float)
    b = x.loc[test, nums].to_numpy(dtype=float)
    am = np.isnan(a)
    bm = np.isnan(b)
    med = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0 for z in a.T])
    a = np.where(am, med, a)
    b = np.where(bm, med, b)
    mean = a.mean(0)
    sd = a.std(0)
    sd[sd <= 1e-12] = 1
    aa = [(a - mean) / sd, am]
    bb = [(b - mean) / sd, bm]
    for (k, levels) in panel['categories'].items():
        lv = np.array(levels)
        aa.append(x.loc[train, k].to_numpy()[:, None] == lv)
        bb.append(x.loc[test, k].to_numpy()[:, None] == lv)
    return (np.column_stack(aa), np.column_stack(bb))

def ridge(a, b, y, penalty):
    mean = a.mean(0)
    d = a - mean
    coef = np.linalg.solve(d.T @ d + np.eye(a.shape[1]) * len(a) * penalty, d.T @ (y - y.mean()))
    return y.mean() + (b - mean) @ coef

def predict_fold(x, target, split, j, panel, grid):
    training = split.index[split.outer.ne(j)]
    test = split.index[split.outer.eq(j)]
    assignments = split.loc[training, 'inner_for_outer_' + str(j)]
    errors = np.zeros(len(grid))
    for k in range(3):
        ti = training[assignments.ne(k)]
        vi = training[assignments.eq(k)]
        (a, b) = design(x, ti, vi, panel)
        for (i, penalty) in enumerate(grid):
            errors[i] += np.abs(ridge(a, b, target.loc[ti].to_numpy(), penalty) - target.loc[vi].to_numpy()).sum()
    errors /= len(training)
    selected = max((z for (z, e) in zip(grid, errors) if e <= min(errors) + 1e-12))
    (a, b) = design(x, training, test, panel)
    return (test, ridge(a, b, target.loc[training].to_numpy(), selected), selected, errors.tolist())
