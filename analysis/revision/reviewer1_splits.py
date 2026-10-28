"""Controlled RF split comparison for discussion of reviewer 1, comment 1.

Run: python -m analysis.revision.reviewer1_splits
Only the split changes. No manuscript, prior results, or dashboard is updated.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ['Gender_Encoded', 'Age (months)', 'Height (cm)', 'Weight (kg)']
PROTOCOLS = ['random', 'four_inputs', 'sex_age_height']
SPLIT_SEEDS = [42, 43, 44]


def split_folds(frame, y, protocol, seed):
    if protocol == 'random':
        return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, y))
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    cols = FEATURES if protocol == 'four_inputs' else FEATURES[:3]
    groups = pd.factorize(pd.MultiIndex.from_frame(frame[cols]), sort=False)[0]
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(frame, y, groups))


def metrics(y, pred):
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', labels=[0, 1, 2, 3], zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', labels=[0, 1, 2, 3], zero_division=0)),
        **{f'recall_{i}': float(v) for i, v in enumerate(recall_score(y, pred,
            average=None, labels=[0, 1, 2, 3], zero_division=0))},
    }


def main():
    source = ROOT/'dataset/Stunting Wasting Dataset (Synthetic)/stunting_wasting_dataset.csv'
    notebook = ROOT/'Stunting_Classification_Analysis.ipynb'
    checksum = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p.relative_to(ROOT)): checksum(p) for p in [source, notebook]}
    df = pd.read_csv(source).rename(columns={'Jenis Kelamin': 'Gender',
        'Umur (bulan)': 'Age (months)', 'Tinggi Badan (cm)': 'Height (cm)',
        'Berat Badan (kg)': 'Weight (kg)'})
    df['Gender_Encoded'] = df.Gender.map({'Laki-laki': 1, 'Perempuan': 0})
    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df.Stunting)
    assert encoder.classes_.tolist() == ['Normal', 'Severely Stunted', 'Stunted', 'Tall']
    assert np.isfinite(X.to_numpy()).all()
    groups = {name: pd.factorize(pd.MultiIndex.from_frame(X[cols]), sort=False)[0]
        for name, cols in [('four_inputs', FEATURES), ('sex_age_height', FEATURES[:3])]}
    params = dict(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=4)
    output = ROOT/'results/reviewer1_question1'
    output.mkdir(parents=True, exist_ok=True)
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Synthetic-only controlled RF experiment; split seeds fixed before running, no model selection.',
        'source_sha256': before, 'python': sys.version,
        'versions': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn']},
        'features': FEATURES, 'classes': encoder.classes_.tolist(),
        'class_counts': df.Stunting.value_counts().to_dict(),
        'parameters': RandomForestClassifier(**params).get_params(),
        'split_seeds': SPLIT_SEEDS, 'model_seed_fixed': 42,
        'group_counts': {k: int(len(np.unique(v))) for k, v in groups.items()},
        'fold_mean_note': 'Unweighted means over five folds, matching original notebook; also report pooled OOF metrics.',
        'trials': []}
    predictions = pd.DataFrame({'Source_Row': np.arange(1, len(df)+1), 'target': y})
    for protocol in PROTOCOLS:
        for seed in SPLIT_SEEDS:
            pred = np.full(len(df), -1, dtype=int)
            assigned = np.full(len(df), -1, dtype=int)
            folds = []
            for fold, (train, valid) in enumerate(split_folds(X, y, protocol, seed), 1):
                assert not np.intersect1d(train, valid).size
                assert np.all(assigned[valid] == -1)
                assert set(y[train]) == set(y[valid]) == {0, 1, 2, 3}
                overlap = {k: float(np.isin(v[valid], v[train]).mean()) for k, v in groups.items()}
                if protocol != 'random':
                    assert overlap[protocol] == 0
                model = RandomForestClassifier(**params)
                model.fit(X.iloc[train], y[train])
                pred[valid] = model.predict(X.iloc[valid])
                assigned[valid] = fold
                folds.append({'fold': fold, 'n_train': len(train), 'n_valid': len(valid),
                    'validation_class_counts': np.bincount(y[valid], minlength=4).tolist(),
                    'validation_profile_overlap': overlap, 'metrics': metrics(y[valid], pred[valid])})
                print(protocol, seed, 'fold', fold, 'accuracy', folds[-1]['metrics']['accuracy'], flush=True)
            assert np.all(assigned > 0) and np.all(pred >= 0)
            keys = folds[0]['metrics'].keys()
            result = {'protocol': protocol, 'split_seed': seed, 'folds': folds,
                'fold_mean': {k: float(np.mean([f['metrics'][k] for f in folds])) for k in keys},
                'fold_sd': {k: float(np.std([f['metrics'][k] for f in folds])) for k in keys},
                'pooled_oof': metrics(y, pred), 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist()}
            report['trials'].append(result)
            predictions[f'{protocol}_{seed}_fold'] = assigned
            predictions[f'{protocol}_{seed}_prediction'] = pred
            (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    report['across_split_seeds'] = {}
    for protocol in PROTOCOLS:
        trials = [r for r in report['trials'] if r['protocol'] == protocol]
        report['across_split_seeds'][protocol] = {k: {
            'mean': float(np.mean([t['fold_mean'][k] for t in trials])),
            'min': float(min(t['fold_mean'][k] for t in trials)),
            'max': float(max(t['fold_mean'][k] for t in trials))} for k in trials[0]['fold_mean']}
    majority = np.full(len(y), np.bincount(y).argmax())
    report['majority_baseline'] = metrics(y, majority)
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['original_sources_unchanged'] = all(checksum(ROOT/p) == h for p, h in before.items())
    assert report['original_sources_unchanged']
    predictions.to_csv(output/'oof_predictions_and_folds.csv', index=False)
    report['oof_sha256'] = checksum(output/'oof_predictions_and_folds.csv')
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    pd.DataFrame([{'protocol': r['protocol'], 'split_seed': r['split_seed'],
        **r['fold_mean']} for r in report['trials']]).to_csv(output/'comparison.csv', index=False)
    print(json.dumps(report['across_split_seeds'], indent=2), flush=True)


if __name__ == '__main__':
    main()
