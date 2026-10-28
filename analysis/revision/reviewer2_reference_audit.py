"""Verify regression references and existing CV evidence without replacing research artefacts.

Run from the repository root: python -m analysis.revision.reviewer2_reference_audit
Outputs are confined to results/reviewer2_reference_audit. Fresh RF fits are
verification fits only: they are not saved or selected using external data.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
from importlib.metadata import version

import numpy as np
import pandas as pd

from analysis.revision import data as data_api
from analysis.revision import supervised as api


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/reviewer2_reference_audit'
CV = ROOT / 'results/revision_1/supervised/cv'
KEYS = ['Indicator', 'Gender_Encoded', 'Age (months)']
COEFFICIENTS = ['L', 'M', 'S']
METRICS = ['mae', 'rmse', 'r2']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_who_table(path):
    table = pd.read_excel(path)
    table.columns = table.columns.astype(str).str.strip()
    return table


def canonical_reference():
    manifest = read_json(ROOT / 'dataset/revision/references/manifest.json')
    frames = []
    for name, metadata in manifest['raw_sources'].items():
        path = ROOT / f'dataset/revision/raw/who/{name}.xlsx'
        assert digest(path) == metadata['sha256'], name
        assert path.stat().st_size == metadata['bytes'], name
        table = read_who_table(path).rename(columns={'Month': 'Age (months)'})
        table = table[['Age (months)', *COEFFICIENTS]].copy()
        table['Indicator'] = 'WAZ' if name.startswith('wfa') else 'HAZ'
        table['Gender_Encoded'] = 1 if name.endswith('boys') else 0
        if name.startswith('lfa'):
            table = table.loc[table['Age (months)'] < 24]
        elif name.startswith('hfa'):
            table = table.loc[table['Age (months)'] >= 24]
        frames.append(table)
    rebuilt = pd.concat(frames)[KEYS + COEFFICIENTS].sort_values(KEYS).reset_index(drop=True)
    cached_path = ROOT / 'dataset/revision/references/who_monthly_corrected.csv'
    assert digest(cached_path) == manifest['corrected_sha256']
    cached = pd.read_csv(cached_path).sort_values(KEYS).reset_index(drop=True)
    assert len(rebuilt) == 244 and not rebuilt.duplicated(KEYS).any()
    pd.testing.assert_frame_equal(rebuilt, cached, check_exact=True)
    return rebuilt, manifest


def scalar_score(value, l, median, spread, restricted):
    score = math.log(value / median) / spread if l == 0 else (
        (math.pow(value / median, l) - 1) / (l * spread))
    if restricted and abs(score) > 3:
        def measurement(z):
            return median * (math.exp(spread * z) if l == 0 else
                             math.pow(1 + l * spread * z, 1 / l))
        if score > 3:
            score = 3 + (value - measurement(3)) / (measurement(3) - measurement(2))
        else:
            score = -3 + (value - measurement(-3)) / (measurement(-2) - measurement(-3))
    return score


def original_notebook_data():
    notebook = read_json(ROOT / 'Stunting_Regression_Analysis.ipynb')
    namespace = {'pd': pd, 'np': np, 'LabelEncoder': __import__(
        'sklearn.preprocessing', fromlist=['LabelEncoder']).LabelEncoder}
    with contextlib.redirect_stdout(io.StringIO()):
        for index in [3, 5, 9]:
            exec(compile(''.join(notebook['cells'][index]['source']),
                         f'Stunting_Regression_Analysis.ipynb:cell{index}', 'exec'), namespace)
    frame = namespace['df_final'].copy()
    frame['Source_Row'] = np.arange(1, len(frame) + 1)
    original_model_namespace = {name: getattr(api, name) for name in [
        'LinearRegression', 'Ridge', 'RandomForestRegressor', 'XGBRegressor',
        'LGBMRegressor', 'CatBoostRegressor', 'KFold']}
    parsed = ast.parse(''.join(notebook['cells'][11]['source']))
    assignments = [node for node in parsed.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ['models_reg', 'cv']
                           for t in node.targets)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), 'original_model_definitions', 'exec'),
         original_model_namespace)
    submitted = {}
    target = None
    for output in notebook['cells'][11].get('outputs', []):
        if output.get('output_type') != 'stream':
            continue
        for line in ''.join(output.get('text', [])).splitlines():
            found = re.search(r'Evaluating (HAZ|WAZ)', line)
            if found:
                target = found.group(1)
            match = re.search(r'- (.+) R2=([\d.]+) \| MAE=([\d.]+) \| RMSE=([\d.]+)', line)
            if match:
                submitted[(target, match.group(1))] = dict(zip(
                    ['r2', 'mae', 'rmse'], match.groups()[1:]))
    assert len(submitted) == 12
    return frame, original_model_namespace, submitted, notebook['metadata'].get('language_info', {})


def main():
    assert Path.cwd() == ROOT, 'Run this audit from the repository root'
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / 'Stunting_Regression_Analysis.ipynb',
                 ROOT / 'Stunting_Classification_Analysis.ipynb',
                 ROOT / 'dataset/Stunting Wasting Dataset (Synthetic)/stunting_wasting_dataset.csv',
                 ROOT / 'dataset/revision/references/who_monthly_corrected.csv',
                 ROOT / 'dataset/revision/references/manifest.json',
                 ROOT / 'results/regression/stunting_random_forest_reg_haz.joblib',
                 ROOT / 'results/regression/stunting_random_forest_reg_waz.joblib',
                 ROOT / 'results/classification/stunting_random_forest_classifier.joblib',
                 ROOT / 'results/shap_revised/summary.json',
                 ROOT / 'results/revision_1/supervised/cv_metrics.csv']
    protected += sorted((ROOT / 'dataset/revision/raw/who').glob('*.xlsx'))
    protected += sorted((ROOT / 'dataset/Stunting Wasting Dataset (Synthetic)/z-scores who').glob('*.csv'))
    protected += sorted(CV.glob('*random*json'))
    protected += sorted((ROOT / 'results/revision_1/supervised').glob('folds_*random*.csv'))
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in protected}

    rebuilt, manifest = canonical_reference()
    original, definitions, submitted, notebook_language = original_notebook_data()
    legacy_reference = data_api._legacy_references(ROOT)
    legacy = data_api._enrich(original, legacy_reference, corrected=False)
    corrected = data_api._enrich(original, rebuilt, corrected=True)
    assert len(original) == 100000
    assert original['Stunting'].equals(corrected['Stunting'])
    for task in ['HAZ', 'WAZ']:
        np.testing.assert_array_equal(original[task], legacy[task])
        pd.testing.assert_frame_equal(original[api.FEATURES[task]], corrected[api.FEATURES[task]])

    comparison = legacy_reference.merge(rebuilt, on=KEYS, suffixes=('_old', '_corrected'), validate='one_to_one')
    changed_coefficients = np.column_stack([
        comparison[f'{c}_old'].to_numpy() != comparison[f'{c}_corrected'].to_numpy() for c in COEFFICIENTS])
    changed = comparison.loc[changed_coefficients.any(axis=1)].copy()
    assert len(changed) == 2 and int(changed_coefficients.sum()) == 4
    assert set(changed.Indicator) == {'HAZ'} and set(changed['Age (months)']) == {24}
    for sex, encoded in [('boys', 1), ('girls', 0)]:
        low = read_who_table(ROOT / f'dataset/revision/raw/who/lfa_{sex}.xlsx').set_index('Month').loc[24]
        high = read_who_table(ROOT / f'dataset/revision/raw/who/hfa_{sex}.xlsx').set_index('Month').loc[24]
        old = changed.loc[changed.Gender_Encoded == encoded].iloc[0]
        for c in COEFFICIENTS:
            np.testing.assert_allclose(old[f'{c}_old'], (low[c] + high[c]) / 2, rtol=0, atol=1e-12)
    changed.to_csv(OUT / 'reference_differences.csv', index=False)

    target_rows, independent_errors = [], {}
    for task, measurement in [('HAZ', 'Height (cm)'), ('WAZ', 'Weight (kg)')]:
        table = rebuilt.loc[rebuilt.Indicator == task].set_index(['Gender_Encoded', 'Age (months)'])
        coefficients = table.loc[pd.MultiIndex.from_frame(original[['Gender_Encoded', 'Age (months)']])]
        independent = np.array([scalar_score(value, l, m, s, task == 'WAZ')
                                for value, l, m, s in zip(original[measurement], coefficients.L,
                                                         coefficients.M, coefficients.S)])
        np.testing.assert_allclose(independent, corrected[task], rtol=0, atol=1e-12)
        independent_errors[task] = float(np.max(np.abs(independent - corrected[task])))
        delta = corrected[task].to_numpy() - original[task].to_numpy()
        mask = np.abs(delta) > 1e-12
        if task == 'HAZ':
            assert mask.sum() == 3954
            np.testing.assert_array_equal(mask, original['Age (months)'].to_numpy() == 24)
        else:
            assert mask.sum() == 22063
            np.testing.assert_array_equal(mask, np.abs(original[task].to_numpy()) > 3)
        target_rows.append({'Target': task, 'Changed_records': int(mask.sum()),
                            'Changed_below_24': int((mask & (original['Age (months)'] < 24)).sum()),
                            'Changed_age_24': int((mask & (original['Age (months)'] == 24)).sum()),
                            'Maximum_absolute_change': float(np.abs(delta).max())})
    pd.DataFrame(target_rows).to_csv(OUT / 'target_changes.csv', index=False)
    print('Verified WHO tables and independent scalar recalculation for all 100,000 records.', flush=True)

    versions = {name: version(name) for name in [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost']}
    metric_rows, fresh_rows, overlap_rows = [], [], []
    aggregate = pd.read_csv(ROOT / 'results/revision_1/supervised/cv_metrics.csv')
    operational = {'n_jobs', 'thread_count', 'verbose', 'verbosity', 'allow_writing_files'}
    for reference, frame in [('legacy', original), ('corrected', corrected)]:
        for task in ['HAZ', 'WAZ']:
            X, y = frame[api.FEATURES[task]], frame[task].to_numpy()
            folds, diagnostic = api.make_folds(frame, y, task, 'random')
            original_folds = list(definitions['cv'].split(X, y))
            for (tr, va), (ot, ov) in zip(folds, original_folds):
                np.testing.assert_array_equal(tr, ot)
                np.testing.assert_array_equal(va, ov)
            protocol = read_json(CV / f'{reference}_random_{task}_protocol.json')
            assert protocol['versions'] == versions
            fold_hash = hashlib.sha256(b''.join(va.tobytes() for _, va in folds)).hexdigest()
            assert fold_hash == protocol['validation_folds_sha256']
            fold_csv = pd.read_csv(ROOT / f'results/revision_1/supervised/folds_{reference}_random_{task}.csv')
            np.testing.assert_array_equal(fold_csv.Source_Row, frame.Source_Row)
            np.testing.assert_array_equal(fold_csv.decision_group, api.group_ids(frame, task))
            for number, (_, va) in enumerate(folds):
                assert (fold_csv.fold.iloc[va] == number).all()
                overlap_rows.append({'Reference': reference, 'Target': task, 'Fold': number + 1,
                                     'Validation_input_overlap': diagnostic['validation_input_overlap_by_fold'][number]})
            for name in api.REG_NAMES:
                stored = read_json(CV / f'{reference}_random_{task}_{api.CODES[name]}.json')
                model = api.create_model(task, name)
                assert stored['fingerprint'] == api.fingerprint(X, y, model)
                assert stored['parameters'] == api.serializable_parameters(model.get_params())
                assert stored['diagnostics'] == diagnostic
                original_params = definitions['models_reg'][name]().get_params()
                assert {k: v for k, v in original_params.items() if k not in operational} == {
                    k: v for k, v in model.get_params().items() if k not in operational}
                row = aggregate.loc[(aggregate.reference == reference) & (aggregate.protocol == 'random')
                                    & (aggregate.task == task) & (aggregate.model == name)]
                assert len(row) == 1 and len(stored['folds']) == 5
                for metric in METRICS:
                    values = [f[metric] for f in stored['folds']]
                    assert float(np.mean(values)) == stored['mean'][metric]
                    assert float(np.std(values)) == stored['sd'][metric]
                    np.testing.assert_allclose(row.iloc[0][metric], stored['mean'][metric], rtol=0, atol=1e-14)
                    np.testing.assert_allclose(row.iloc[0][f'{metric}_sd'], stored['sd'][metric], rtol=0, atol=1e-14)
                    if reference == 'legacy':
                        assert f"{stored['mean'][metric]:.6f}" == submitted[(task, name)][metric]
                metric_rows.append({'Reference': reference, 'Target': task, 'Model': name,
                                    **stored['mean'], 'Cached_training_seconds': stored['training_seconds']})
                if name == 'Random Forest':
                    for number, (train, valid) in enumerate(folds):
                        fitted = api.create_model(task, name).fit(X.iloc[train], y[train])
                        actual = api.regression_metrics(y[valid], fitted.predict(X.iloc[valid]))
                        for metric in METRICS:
                            np.testing.assert_allclose(actual[metric], stored['folds'][number][metric], rtol=0, atol=1e-10)
                        fresh_rows.append({'Reference': reference, 'Target': task, 'Fold': number + 1, **actual})
                    print(f'Fresh RF reproduction passed: {reference} / {task}, five folds.', flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / 'regression_metrics.csv', index=False)
    pd.DataFrame(fresh_rows).to_csv(OUT / 'fresh_rf_verification.csv', index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / 'random_fold_overlap.csv', index=False)
    ranking = {task: metrics.loc[(metrics.Reference == 'corrected') & (metrics.Target == task)]
               .sort_values('rmse').Model.tolist() for task in ['HAZ', 'WAZ']}
    assert all(names[:2] == ['Random Forest', 'CatBoost'] for names in ranking.values())

    assert all(digest(ROOT / path) == sha for path, sha in source_hashes.items())
    summary = {
        'scope': 'Reviewer 2 phase 1 reference/cache verification only; no external evaluation or final model replacement',
        'original_sources_sha256': source_hashes, 'protected_sources_unchanged': True,
        'references': {'rows': len(rebuilt), 'coefficients': 3 * len(rebuilt), 'changed_legacy_rows': len(changed),
                       'changed_legacy_coefficients': int(changed_coefficients.sum()),
                       'sources': manifest['source_pages'], 'algorithm': manifest['algorithm_source'],
                       'convention': manifest['convention']},
        'target_changes': target_rows, 'classification_labels_unchanged': True,
        'independent_scalar_max_error': independent_errors,
        'cache_checks': {'verified_model_target_reference_caches': len(metric_rows),
                         'notebook_metrics_matched_at_six_decimals': 36,
                         'original_five_fold_assignments_matched': True,
                         'input_target_parameter_fingerprints_matched': True,
                         'saved_fold_means_sd_and_aggregate_csv_matched': True},
        'fresh_verification': {'algorithm': 'Random Forest', 'fits': len(fresh_rows),
                                'absolute_metric_tolerance': 1e-10,
                                'note': 'Other algorithms received cache/protocol/arithmetic verification, not fresh fitting'},
        'corrected_primary_rmse_ranking': ranking,
        'environment': {'python': platform.python_version(), 'packages': versions,
                        'original_notebook_declared_language': notebook_language,
                        'original_n_jobs': -1, 'verification_n_jobs': 4,
                        'timing_note': 'Historical training times are not directly comparable; do not combine old times with corrected metrics'},
        'limitations': [
            'Monthly references and assumed age-based position are not exact-day clinical gold standards',
            'Reference correction changes regression targets, not source classification labels',
            'Existing regression models, SHAP and prediction figures still use original targets',
            'Primary random-fold ranking is distinct from grouped selection; RF is not best in grouped WAZ',
            'No independence, clinical validity, or national generalizability claim follows from these checks',
        ],
    }
    summary['output_sha256'] = {str(p.relative_to(OUT)): digest(p) for p in OUT.glob('*.csv')}
    (OUT / 'verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('PASS: 244 reference rows, 24 cache records, 36 historical metrics and 20 fresh RF folds.', flush=True)


if __name__ == '__main__':
    main()
