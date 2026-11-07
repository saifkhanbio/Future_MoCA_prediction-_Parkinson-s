"""Data-free release checks. Does not read controlled data or launch a study run."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def must_reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid synthetic input was accepted')

def main():
    manifest = json.loads((ROOT / 'CODE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        actual = hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest()
        assert actual == item['published_sha256'], item['path']
    scripts = list(ROOT.rglob('*.py'))
    for path in scripts:
        if '.venv' not in path.parts:
            compile(path.read_text(encoding='utf-8-sig'), str(path.relative_to(ROOT)), 'exec')
    fixed = load('analysis/feature_only_readiness_2026-10-31/src/numerics.py', 'release_fixed_numerics')
    ga = load('analysis/ga_unrestricted_readiness_v4_10kgen_2026-11-01/src/r2_engine.py', 'release_ga_numerics')
    temporal = load('analysis/temporal_validation_readiness_2026-10-28/src/clinical_inference.py', 'release_temporal_numerics')

    # Independent centered augmented-least-squares reference (synthetic only).
    rng = np.random.default_rng(21983)
    X = rng.normal(size=(17, 5))
    X[:, 4] = X[:, 0]  # include a collinear design
    y = rng.normal(size=17)
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        centered = X - X.mean(axis=0)
        aug = np.vstack([centered, np.sqrt(len(y) * alpha) * np.eye(X.shape[1])])
        target = np.r_[y - y.mean(), np.zeros(X.shape[1])]
        coef = np.linalg.lstsq(aug, target, rcond=None)[0]
        expected = centered @ coef + y.mean()
        for kernel in (fixed, ga):
            pred = kernel.predict(X, kernel.ridge(X, y, alpha))
            np.testing.assert_allclose(pred, expected, atol=1e-10, rtol=1e-10)
    must_reject(lambda: ga.ridge(X, y, 0))

    # The held-out extreme value must not refit the training median/scaling.
    rows = [{'x': 1, 'empty': '', 'category': 'a'},
            {'x': '', 'empty': '', 'category': 'unknown'},
            {'x': 3, 'empty': '', 'category': 'a'}]
    schema = {'numeric': ['x', 'empty'], 'categories': {'category': ['a', 'unknown']}}
    state = fixed.prepare(rows, schema)
    before = copy.deepcopy(state)
    assert state['numeric']['x']['median'] == 2
    assert state['numeric']['empty']['median'] == 0
    transformed, labels = fixed.transform([{'x': 1000, 'empty': '', 'category': 'a'}], state)
    assert state == before
    assert transformed[0, labels.index('empty')] == 0
    assert transformed[0, labels.index('empty__missing')] == 1
    np.testing.assert_allclose(transformed[0, labels.index('x')], (1000 - 2) / np.sqrt(2 / 3))
    must_reject(lambda: fixed.transform([{'x': 1, 'empty': '', 'category': 'invalid'}], state))

    observed = np.array([1., 2., 4., 5.])
    predicted = np.array([1.5, 2.5, 3.5, 5.5])
    for kernel in (fixed, ga, temporal):
        metrics = kernel.metrics(observed, predicted)
        np.testing.assert_allclose(metrics['MAE'], 0.5)
        np.testing.assert_allclose(metrics['RMSE'], 0.5)
        np.testing.assert_allclose(metrics['R2'], 0.9)

    contract = json.loads((ROOT / 'analysis/ga_unrestricted_readiness_v4_10kgen_2026-11-01/config/Analysis_Contract.json').read_text())
    original = copy.deepcopy(contract)
    mandatory = ga.for_arm(contract, 'moca27')
    assert contract == original
    assert len(contract['features']['optional']) == 28
    assert len(mandatory['features']['optional']) == 27
    assert mandatory['features']['core']['numeric'] == ['baseline_moca']
    assert 'baseline_moca' not in [x['name'] for x in mandatory['features']['optional']]
    must_reject(lambda: ga.selected(2 ** 28, contract))

    # Deliberately artificial identities; no real participant information.
    predictions = [{'component_id': 'synthetic_A', 'PATNO': 'synthetic_A', 'horizon_months': 12},
                   {'component_id': 'synthetic_B', 'PATNO': 'synthetic_B', 'horizon_months': 12}]
    outcomes = [dict(r, endpoint_QC_pass='true', education_adjusted_MoCA=24 + i) for i, r in enumerate(predictions)]
    bound = temporal.bind_outcomes(predictions, outcomes[::-1])
    assert [value for _, value in bound] == [24, 25]
    bad = copy.deepcopy(outcomes)
    bad[0]['PATNO'] = 'synthetic_B'
    must_reject(lambda: temporal.bind_outcomes(predictions, bad))
    must_reject(lambda: temporal.bind_outcomes(predictions, outcomes + outcomes[:1]))
    # Independent LCC clinical kernel, synthetic inputs only.
    lcc = load('analysis/lcc_nominal_validation_2026-11-05/src/core.py', 'release_lcc_core')
    x_lcc = rng.normal(size=(19, 4))
    x_lcc[:, 2] = 1.0
    y_lcc = rng.normal(size=19)
    for alpha in lcc.GRID:
        model = lcc.ridge_fit(x_lcc, y_lcc, alpha)
        frozen_state = copy.deepcopy(model)
        scale = x_lcc.std(axis=0)
        z = (x_lcc - x_lcc.mean(axis=0)) / np.where(scale == 0, 1.0, scale)
        z[:, scale == 0] = 0.0
        augmented = np.vstack([z, np.sqrt(len(y_lcc) * alpha) * np.eye(4)])
        target = np.r_[y_lcc - y_lcc.mean(), np.zeros(4)]
        coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
        expected = y_lcc.mean() + z @ coefficient
        np.testing.assert_allclose(lcc.ridge_predict(model, x_lcc), expected, atol=1e-10, rtol=1e-10)
        shifted = x_lcc.copy()
        shifted[:, 2] = 900.0
        np.testing.assert_allclose(lcc.ridge_predict(model, shifted), expected, atol=1e-10, rtol=1e-10)
        assert model == frozen_state
    assert lcc.choose_penalty(np.ones(len(lcc.GRID))) == len(lcc.GRID) - 1
    must_reject(lambda: lcc.ridge_fit(x_lcc, y_lcc, 0))
    np.testing.assert_allclose(lcc.metrics(observed, predicted)['R2'], 0.9)

    print('PASS: publication checksums, Python syntax, synthetic PPMI/LCC ridge/preprocessing/metrics, mask mapping and identity guards.')
    print('No participant data, study training, GPU execution or cloud queries were used.')

if __name__ == '__main__':
    main()