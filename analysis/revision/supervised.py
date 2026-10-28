"""Reproduce, strengthen and externally evaluate the submitted supervised tasks.

Run from the repository root: venv/bin/python -m analysis.revision.supervised
External records never enter model fitting or model selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
THREADS = 4
CLASSES = ['Normal', 'Severely Stunted', 'Stunted', 'Tall']
FEATURES = {
    'classification': ['Gender_Encoded', 'Age (months)', 'Height (cm)', 'Weight (kg)'],
    'HAZ': ['Gender_Encoded', 'Age (months)', 'Height (cm)'],
    'WAZ': ['Gender_Encoded', 'Age (months)', 'Weight (kg)'],
}
CLF_NAMES = ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 'CatBoost']
REG_NAMES = ['Linear Regression', 'Ridge Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']
CODES = {'Random Forest': 'rf', 'Extra Trees': 'et', 'XGBoost': 'xgb', 'LightGBM': 'lgbm',
         'CatBoost': 'cb', 'Linear Regression': 'linear', 'Ridge Regression': 'ridge'}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')


def serializable_parameters(value):
    """Record a model's NaN missing-value sentinel as JSON null; metrics stay strict."""
    if isinstance(value, dict):
        return {k: serializable_parameters(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable_parameters(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def group_ids(frame, task):
    cols = FEATURES['HAZ'] if task == 'classification' else FEATURES[task]
    return pd.factorize(pd.MultiIndex.from_frame(frame[cols]), sort=False)[0]


def binary_predictions(predicted_classes, probabilities):
    # Preserve the original four-class decision, then collapse it; do not tune .5 on external data.
    return np.isin(predicted_classes, [1, 2]).astype(int), probabilities[:, 1] + probabilities[:, 2]


def make_folds(frame, y, task, protocol):
    groups = group_ids(frame, task)
    if protocol == 'grouped':
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED) if task == 'classification' else GroupKFold(5, shuffle=True, random_state=SEED)
        folds = list(splitter.split(frame, y, groups))
    else:
        splitter = StratifiedKFold(5, shuffle=True, random_state=SEED) if task == 'classification' else KFold(5, shuffle=True, random_state=SEED)
        folds = list(splitter.split(frame, y))
    overlaps = [float(np.isin(groups[v], groups[t]).mean()) for t, v in folds]
    if protocol == 'grouped' and any(x != 0 for x in overlaps):
        raise AssertionError('Decision input groups leaked across a grouped fold')
    return folds, {'n': len(frame), 'unique_decision_groups': len(np.unique(groups)),
        'validation_input_overlap_by_fold': overlaps, 'max_validation_input_overlap': max(overlaps)}


def create_model(task, name):
    if task == 'classification':
        if name == 'Random Forest': return RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=SEED, n_jobs=THREADS)
        if name == 'Extra Trees': return ExtraTreesClassifier(n_estimators=50, class_weight='balanced', random_state=SEED, n_jobs=THREADS)
        if name == 'XGBoost': return XGBClassifier(random_state=SEED, eval_metric='mlogloss', n_jobs=THREADS)
        if name == 'LightGBM': return LGBMClassifier(class_weight='balanced', random_state=SEED, verbosity=-1, n_jobs=THREADS)
        if name == 'CatBoost': return CatBoostClassifier(auto_class_weights='Balanced', random_seed=SEED, verbose=False, thread_count=THREADS, allow_writing_files=False)
    else:
        if name == 'Linear Regression': return LinearRegression()
        if name == 'Ridge Regression': return Ridge()
        if name == 'Random Forest': return RandomForestRegressor(n_estimators=30, random_state=SEED, n_jobs=THREADS)
        if name == 'XGBoost': return XGBRegressor(random_state=SEED, n_jobs=THREADS)
        if name == 'LightGBM': return LGBMRegressor(random_state=SEED, verbosity=-1, n_jobs=THREADS)
        if name == 'CatBoost': return CatBoostRegressor(random_seed=SEED, verbose=False, thread_count=THREADS, allow_writing_files=False)
    raise ValueError((task, name))


def fit_model(model, task, name, X, y):
    if task == 'classification' and name == 'XGBoost':
        model.fit(X, y, sample_weight=compute_sample_weight(class_weight='balanced', y=y))
    else:
        model.fit(X, y)
    return model


def classification_metrics(y, pred, probs=None, sample_weight=None, binary=False):
    labels = [0, 1] if binary else [0, 1, 2, 3]
    metrics = {
        'accuracy': float(accuracy_score(y, pred, sample_weight=sample_weight)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred, sample_weight=sample_weight)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=labels, zero_division=0, sample_weight=sample_weight)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', labels=labels, zero_division=0, sample_weight=sample_weight)),
        'weighted_precision': float(precision_score(y, pred, average='weighted', labels=labels, zero_division=0, sample_weight=sample_weight)),
        'weighted_recall': float(recall_score(y, pred, average='weighted', labels=labels, zero_division=0, sample_weight=sample_weight)),
    }
    rec = recall_score(y, pred, average=None, labels=labels, zero_division=0, sample_weight=sample_weight)
    for i, val in enumerate(rec): metrics[f'recall_{i}'] = float(val)
    if binary:
        metrics['sensitivity'] = float(rec[1])
        metrics['specificity'] = float(rec[0])
    if probs is not None and len(np.unique(y)) == len(labels):
        metrics['auc'] = float(roc_auc_score(y, probs if binary else probs, sample_weight=sample_weight,
            **({} if binary else {'multi_class': 'ovr', 'average': 'weighted'})))
    return metrics


def regression_metrics(y, pred, sample_weight=None):
    return {'mae': float(mean_absolute_error(y, pred, sample_weight=sample_weight)),
        'rmse': float(np.sqrt(mean_squared_error(y, pred, sample_weight=sample_weight))),
        'r2': float(r2_score(y, pred, sample_weight=sample_weight))}


def fingerprint(frame, y, model):
    h = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(pd.Series(y), index=True).values.tobytes())
    h.update(json.dumps(model.get_params(), sort_keys=True, default=str).encode())
    h.update(b'icdees1571345184-protocol-1')
    return h.hexdigest()


def evaluate(frame, task, names, protocol, reference, output):
    X = frame[FEATURES[task]]
    y = frame['Stunting'].map({x: i for i, x in enumerate(CLASSES)}).to_numpy() if task == 'classification' else frame[task].to_numpy()
    if not np.isfinite(y).all() or not np.isfinite(X.to_numpy()).all():
        raise ValueError('Invalid synthetic training inputs or targets')
    folds, diagnostic = make_folds(frame, y, task, protocol)
    cache_protocol = {'versions': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost']},
        'validation_folds_sha256': hashlib.sha256(b''.join(v.tobytes() for _, v in folds)).hexdigest()}
    cache_path = output / 'cv' / f'{reference}_{protocol}_{task}_protocol.json'
    if cache_path.exists() and json.loads(cache_path.read_text()) != cache_protocol:
        raise RuntimeError(f'CV cache protocol changed: remove the matching cached results before rerunning {cache_path}')
    write_json(cache_path, cache_protocol)
    fold_assignments = np.full(len(frame), -1, dtype=int)
    for i, (_, v) in enumerate(folds): fold_assignments[v] = i
    pd.DataFrame({'Source_Row': frame.Source_Row, 'fold': fold_assignments,
        'decision_group': group_ids(frame, task)}).to_csv(output / f'folds_{reference}_{protocol}_{task}.csv', index=False)
    results = []
    for name in names:
        path = output / 'cv' / f'{reference}_{protocol}_{task}_{CODES[name]}.json'
        signature = fingerprint(X, y, create_model(task, name))
        if path.exists():
            stored = json.loads(path.read_text())
            if stored.get('fingerprint') == signature:
                results.append(stored)
                print(f'RESUME {reference}/{protocol}/{task}/{name}', flush=True)
                continue
        metrics, times = [], []
        for fold, (train, valid) in enumerate(folds):
            model = create_model(task, name)
            start = time.perf_counter()
            fit_model(model, task, name, X.iloc[train], y[train])
            elapsed = time.perf_counter() - start
            pred = np.asarray(model.predict(X.iloc[valid])).ravel()
            metrics.append(classification_metrics(y[valid], pred, model.predict_proba(X.iloc[valid])) if task == 'classification' else regression_metrics(y[valid], pred))
            times.append(elapsed)
            print(f'CV {reference}/{protocol}/{task}/{name} fold {fold+1}/5 {elapsed:.2f}s', flush=True)
            del model
            gc.collect()
        row = {'protocol': protocol, 'reference': reference, 'task': task, 'model': name,
            'fingerprint': signature, 'mean': {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
            'sd': {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]},
            'folds': metrics, 'training_seconds': sum(times), 'diagnostics': diagnostic,
            'parameters': serializable_parameters(create_model(task, name).get_params())}
        write_json(path, row)
        results.append(row)
    return results


def cluster_bootstrap(y, pred, groups, binary=False, repetitions=2000):
    """Profile-cluster percentile intervals, not intervals for identified unique children."""
    ids, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    if binary:
        contributions = np.zeros((len(ids), 4))
        for c in range(4): contributions[:, c] = np.bincount(inverse, weights=((2*y + pred) == c), minlength=len(ids))
        keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'sensitivity', 'specificity']
    else:
        error = np.asarray(pred) - np.asarray(y)
        contributions = np.column_stack([np.bincount(inverse, weights=v, minlength=len(ids)) for v in
            [np.ones(len(y)), np.abs(error), error**2, y, np.asarray(y)**2]])
        keys = ['mae', 'rmse', 'r2']
    values = []
    for _ in range(repetitions):
        total = contributions[rng.integers(0, len(ids), len(ids))].sum(axis=0)
        if binary:
            tn, fp, fn, tp = total
            if (tp+fn) == 0 or (tn+fp) == 0: continue
            sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
            f_pos = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
            f_neg = 2*tn/(2*tn+fp+fn) if (2*tn+fp+fn) else 0
            values.append([(tp+tn)/total.sum(), (sensitivity+specificity)/2, (f_pos+f_neg)/2, sensitivity, specificity])
        else:
            n, absolute, square, sum_y, sum_y2 = total
            denominator = sum_y2 - sum_y**2/n
            if denominator <= 0: continue
            values.append([absolute/n, np.sqrt(square/n), 1-square/denominator])
    return {k: np.percentile(np.asarray(values)[:, i], [2.5, 97.5]).tolist() for i, k in enumerate(keys)}


def external_evaluation(frame, selected, models, output):
    result = {}
    for cohort, age_mask in [('primary', frame['Age (months)'] < 24), ('age24', frame['Age (months)'] == 24)]:
        result[cohort] = {}
        for task in FEATURES:
            mask = age_mask & frame['Valid_Classification' if task == 'classification' else f'Valid_{task}']
            data = frame.loc[mask].copy()
            if data.empty:
                result[cohort][task] = {'n': 0, 'not_evaluated': 'No eligible observations'}
                continue
            model = models[(task, selected[task]['model'])]
            X = data[FEATURES[task]]
            pred = np.asarray(model.predict(X)).ravel()
            groups = group_ids(data, task)
            _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
            profile_weight = 1 / counts[inv]
            if task == 'classification':
                probabilities = model.predict_proba(X)
                pred, probability = binary_predictions(pred, probabilities)
                y = data.Stunting_Binary.to_numpy(dtype=int)
                metrics = classification_metrics(y, pred, probability, binary=True)
                weighted = classification_metrics(y, pred, probability, sample_weight=profile_weight, binary=True)
                detail = {'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                    'class_counts': {str(i): int((y == i).sum()) for i in [0, 1]}}
                data['positive_probability'] = probability
            else:
                y = data[task].to_numpy()
                metrics = regression_metrics(y, pred)
                weighted = regression_metrics(y, pred, sample_weight=profile_weight)
                detail = {}
            data['prediction'] = pred
            data['evaluation_target'] = y
            data.to_csv(output / f'external_predictions_{cohort}_{task}.csv', index=False)
            intervals = cluster_bootstrap(y, pred, groups, binary=task == 'classification')
            result[cohort][task] = {'n': len(data), 'unique_decision_profiles': len(np.unique(groups)),
                'model': selected[task]['model'], 'metrics': metrics, 'ci95': intervals,
                'profile_balanced_metrics': weighted, 'bootstrap_repetitions': 2000, **detail}
            print(f'EXTERNAL {cohort}/{task}: {len(data)} observations {metrics}', flush=True)
    result['limitations'] = ['No tuning or fitting on external data', 'Source_Row is not a unique child identifier',
        'Profile bootstrap cannot identify repeated visits or facility/family dependence',
        'Source measurement position unavailable; age-conventional length/height assumed without undocumented corrections',
        'Monthly reference-derived Z-scores are not independent clinical outcomes', 'One regency is not nationally representative']
    return result


def export_model(model, task, name, path):
    """Export fitted artifacts, never independently retrain a browser model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    features = FEATURES[task]
    if name in ['Random Forest', 'Extra Trees']:
        def tree_node(tree, node=0):
            if tree.children_left[node] == tree.children_right[node]:
                values = tree.value[node].ravel()
                leaf = (values / values.sum()).tolist() if task == 'classification' else float(values[0])
                return {'leaf': leaf}
            return {'split': features[tree.feature[node]], 'split_condition': float(tree.threshold[node]),
                'left': tree_node(tree, tree.children_left[node]), 'right': tree_node(tree, tree.children_right[node])}
        payload = [tree_node(est.tree_) for est in model.estimators_]
    elif name == 'XGBoost':
        booster = model.get_booster()
        config = json.loads(booster.save_config())
        raw = config['learner']['learner_model_param']['base_score']
        base = json.loads(raw) if raw.startswith('[') else float(raw)
        payload = {'base_scores' if task == 'classification' else 'base_score': base,
            'trees': [json.loads(t) for t in booster.get_dump(dump_format='json')]}
    elif name == 'LightGBM':
        payload = model.booster_.dump_model()
    elif name == 'CatBoost':
        model.save_model(str(path), format='json')
        return
    else:
        payload = {'feature_names': features, 'coefficients': np.asarray(model.coef_).tolist(), 'intercept': float(model.intercept_)}
    # Compact JSON materially reduces browser download size without rounding thresholds.
    path.write_text(json.dumps(payload, default=json_default, separators=(',', ':'), allow_nan=False))


def make_shap_plots(frame, models, selected, cv_results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shap
    plots = {}
    panels = []
    for task in FEATURES:
        name = selected[task]['model']
        y = frame.Stunting.map({x: i for i, x in enumerate(CLASSES)}).to_numpy() if task == 'classification' else frame[task].to_numpy()
        folds, _ = make_folds(frame, y, task, 'grouped')
        train, valid = folds[0]
        X = frame[FEATURES[task]]
        # Explain a held-out fold model, rather than a full-data model on its training rows.
        model = fit_model(create_model(task, name), task, name, X.iloc[train], y[train])
        sample = X.iloc[valid].sample(min(500, len(valid)), random_state=SEED)
        if name in ['Linear Regression', 'Ridge Regression']:
            background = X.iloc[train].sample(min(100, len(train)), random_state=SEED)
            explainer = shap.LinearExplainer(model, background)
        else:
            explainer = shap.TreeExplainer(model)
        explanation = explainer(sample)
        vals = explanation.values[:, :, 2] if task == 'classification' else explanation.values
        plt.figure(figsize=(6, 4))
        shap.summary_plot(vals, sample, show=False, plot_size=None)
        plt.title(f'{task}: {name}' + (' (Stunted class)' if task == 'classification' else ''), fontsize=10)
        plt.tight_layout()
        file = output / f'shap_{task}.png'
        plt.savefig(file, dpi=220, bbox_inches='tight')
        plt.close('all')
        np.savez_compressed(output / f'shap_{task}.npz', values=vals, features=sample.to_numpy(), indices=sample.index.to_numpy())
        plots[task] = str(file.relative_to(output.parents[2]))
        panels.append((task, name, sample, vals))
        del model, explainer
        gc.collect()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (task, name, sample, vals) in zip(axes, panels):
        values = np.abs(vals).mean(axis=0)
        order = np.argsort(values)
        ax.barh(np.array(sample.columns)[order], values[order], color='#466777')
        ax.set_title(f'{task} — {name}', fontsize=9)
        ax.set_xlabel('Mean |SHAP| (model output units)', fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output / 'shap_selected_combined.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    plots['combined'] = str((output / 'shap_selected_combined.png').relative_to(output.parents[2]))
    plots['interpretation'] = 'Global explanations of selected model configurations fitted on grouped fold 1 training; sample is fold 1 validation. Classification explains Stunted only; magnitudes across tasks have different output units.'
    return plots


def run(root=ROOT):
    from .data import load_synthetic, load_jeneponto
    output = root / 'results/revision_1/supervised'
    output.mkdir(parents=True, exist_ok=True)
    versions = {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'shap']}
    legacy, legacy_audit = load_synthetic(reference='legacy', root=root)
    corrected, corrected_audit = load_synthetic(reference='corrected', root=root)
    cv_results = []
    # Classification labels and four physical inputs are unchanged by reference correction.
    cv_results.extend(evaluate(legacy, 'classification', CLF_NAMES, 'random', 'legacy', output))
    for reference, frame in [('legacy', legacy), ('corrected', corrected)]:
        for task in ['HAZ', 'WAZ']:
            cv_results.extend(evaluate(frame, task, REG_NAMES, 'random', reference, output))
    for task in FEATURES:
        cv_results.extend(evaluate(corrected, task, CLF_NAMES if task == 'classification' else REG_NAMES, 'grouped', 'corrected', output))
    selected = {}
    for task in FEATURES:
        choices = [r for r in cv_results if r['task'] == task and r['protocol'] == 'grouped']
        if task == 'classification': choices.sort(key=lambda r: (-r['mean']['macro_f1'], -r['mean']['balanced_accuracy'], CLF_NAMES.index(r['model'])))
        else: choices.sort(key=lambda r: (r['mean']['rmse'], r['mean']['mae'], REG_NAMES.index(r['model'])))
        winner = choices[0]
        selected[task] = {'model': winner['model'], 'code': CODES[winner['model']],
            'features': FEATURES[task], 'cv_mean': winner['mean'], 'cv_sd': winner['sd']}
    models = {}
    exported = {}
    # Only these frozen configurations are tested externally; all comparative choices came from synthetic CV.
    for task in FEATURES:
        names = CLF_NAMES if task == 'classification' else list(dict.fromkeys(['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', selected[task]['model']]))
        for name in names:
            X = corrected[FEATURES[task]]
            y = corrected.Stunting.map({x: i for i, x in enumerate(CLASSES)}).to_numpy() if task == 'classification' else corrected[task].to_numpy()
            model = fit_model(create_model(task, name), task, name, X, y)
            model_file = output / 'models' / f'{task}_{CODES[name]}.joblib'
            model_file.parent.mkdir(exist_ok=True)
            joblib.dump(model, model_file, compress=3)
            json_file = root / 'docs/models/revision_1' / (f'{CODES[name]}.json' if task == 'classification' else f'{CODES[name]}_{task.lower()}.json')
            export_model(model, task, name, json_file)
            key = f'{task}_{CODES[name]}'
            exported[key] = {'model': name, 'features': FEATURES[task],
                'joblib': str(model_file.relative_to(root)), 'json': str(json_file.relative_to(root)),
                'sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(), 'json_sha256': hashlib.sha256(json_file.read_bytes()).hexdigest(),
                'parameters': serializable_parameters(model.get_params())}
            if name == selected[task]['model']: models[(task, name)] = model
            else: del model
            gc.collect()
    freeze = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selection_data': 'Synthetic only; corrected grouped 5-fold CV',
        'classification_selection': 'Highest macro F1, then balanced accuracy, then submitted algorithm order',
        'regression_selection': 'Lowest RMSE, then MAE, then submitted algorithm order',
        'selected': selected, 'artifacts': exported, 'versions': versions}
    write_json(output / 'model_freeze_manifest.json', freeze)
    print('MODEL CONFIGURATIONS AND ARTIFACTS FROZEN BEFORE EXTERNAL PERFORMANCE', flush=True)
    external_frame, external_audit = load_jeneponto(root=root)
    external = external_evaluation(external_frame, selected, models, output)
    shap = make_shap_plots(corrected, models, selected, cv_results, output)
    # Native Python reference samples for actual browser artifact validation.
    parity = []
    examples = pd.concat([corrected.sample(30, random_state=SEED), external_frame.loc[external_frame.Valid_Classification & (external_frame['Age (months)'] < 24)].sample(30, random_state=SEED)], ignore_index=True)
    for key, artifact in exported.items():
        task, code = key.split('_', 1)
        model = joblib.load(root / artifact['joblib'])
        X = examples[FEATURES[task]]
        entry = {'task': task, 'code': code, 'features': X.to_dict('records'), 'prediction': np.asarray(model.predict(X)).ravel().tolist()}
        if task == 'classification': entry['probabilities'] = model.predict_proba(X).tolist()
        parity.append(entry)
        del model
    write_json(root / 'docs/models/revision_1/parity_samples.json', parity)
    summary = {'paper_id': '1571345184', 'versions': versions, 'seed': SEED, 'folds': 5,
        'features': FEATURES, 'classes': CLASSES, 'gender_encoding': {'Male': 1, 'Female': 0},
        'synthetic_audit': {'legacy': legacy_audit, 'corrected': corrected_audit},
        'cv': cv_results, 'selected': selected, 'freeze': freeze, 'external_audit': external_audit,
        'external': external, 'shap': shap,
        'scope': 'Current anthropometric status and reference-function approximation; interface-level integration; no prospective prediction or national generalizability claim',
        'preprocessing': 'Gender encoding only for supervised models. Numeric inputs retain original units, including submitted unscaled Linear/Ridge baselines. Regional standardization is separate.'}
    write_json(output / 'summary.json', summary)
    pd.DataFrame([{'protocol': r['protocol'], 'reference': r['reference'], 'task': r['task'], 'model': r['model'],
        **r['mean'], **{f'{k}_sd': v for k, v in r['sd'].items()}} for r in cv_results]).to_csv(output / 'cv_metrics.csv', index=False)
    dashboard_manifest = {'paper_id': '1571345184', 'model_directory': 'models/revision_1', 'selected': selected,
        'classes': CLASSES, 'gender_encoding': {'Male': 1, 'Female': 0}, 'external': external,
        'scope': summary['scope'], 'versions': versions}
    write_json(root / 'docs/revision_manifest.json', dashboard_manifest)
    print('SUPERVISED REVISION COMPLETE: ' + str(output / 'summary.json'), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
