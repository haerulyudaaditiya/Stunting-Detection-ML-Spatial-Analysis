"""Test frozen primary-comparison models on cached Jeneponto observations.

This module never fits, selects, relabels, deduplicates, downloads, or modifies
source data. Monthly reference targets measure approximation of the reference
calculation; they are not independently observed clinical outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import JENEPONTO_SHA256, monthly_zscore, parse_jeneponto_workbook


ROOT = Path(__file__).resolve().parents[2]
FEATURES = {
    'classification': ['Gender_Encoded', 'Age (months)', 'Height (cm)', 'Weight (kg)'],
    'HAZ': ['Gender_Encoded', 'Age (months)', 'Height (cm)'],
    'WAZ': ['Gender_Encoded', 'Age (months)', 'Weight (kg)'],
}
CLASS_MAPPING = {'0': 'Normal', '1': 'Severely Stunted', '2': 'Stunted', '3': 'Tall'}
COHORTS = {'primary': 'In_Primary_External', 'age24': 'In_Age24_Sensitivity'}
LIMITATIONS = [
    'No unique child identifier: source row ordinals do not identify children or repeat visits.',
    'Repeated input profiles are retained; they are not proven duplicate children.',
    'Measurement position and exact-day ages are unavailable; monthly length/height conventions are assumed without a 0.7 cm conversion.',
    'Regression targets are corrected monthly WHO calculations from the same anthropometric inputs, not independently measured clinical outcomes.',
    'Supplied binary labels are retained even where they disagree with recalculated monthly HAZ.',
    'Profile-bootstrap intervals resample sex-age-height profiles for classification and the three input profiles for each regressor, not identified children; facility and family dependence are unknown.',
    'A single regency does not establish national representativeness or prospective clinical validity.',
    'Earlier exploratory evaluations used this cohort with different model artifacts; this is not a previously unseen prospective cohort.',
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collapse_four_class(probabilities, classes):
    """Keep the original four-class argmax, then collapse its predicted class."""
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 4 or len(classes) != 4
            or set(classes.tolist()) != {0, 1, 2, 3}):
        raise ValueError('Expected all four original classes with matching probability columns')
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-8, rtol=0)):
        raise ValueError('Invalid class probabilities')
    predicted = classes[probabilities.argmax(axis=1)].astype(int)
    binary = np.isin(predicted, [1, 2]).astype(int)
    positive = probabilities[:, np.isin(classes, [1, 2])].sum(axis=1)
    return predicted, binary, positive


def _expected_masks(frame):
    age = frame['Age (months)']
    flags = {
        'Flag_Age': ~np.isfinite(age) | ~age.between(0, 24) | age.mod(1).ne(0),
        'Flag_Gender': ~frame.Gender_Encoded.isin([0, 1]),
        'Flag_Height': ~np.isfinite(frame['Height (cm)']) | frame['Height (cm)'].le(0),
        'Flag_Weight': ~np.isfinite(frame['Weight (kg)']) | frame['Weight (kg)'].le(0),
        'Flag_Source_Label': ~frame.Stunting_Binary.isin([0, 1]),
        'Flag_HAZ': ~np.isfinite(frame.HAZ) | ~frame.HAZ.between(-6, 6),
        'Flag_WAZ': ~np.isfinite(frame.WAZ) | ~frame.WAZ.between(-6, 5),
    }
    basic = ~flags['Flag_Age'] & ~flags['Flag_Gender']
    flags['Valid_HAZ'] = basic & ~flags['Flag_Height'] & ~flags['Flag_HAZ']
    flags['Valid_WAZ'] = basic & ~flags['Flag_Weight'] & ~flags['Flag_WAZ']
    flags['Valid_Classification'] = flags['Valid_HAZ'] & flags['Valid_WAZ'] & ~flags['Flag_Source_Label']
    flags['In_Primary_External'] = age.ge(0) & age.lt(24)
    flags['In_Age24_Sensitivity'] = age.eq(24)
    return flags


def audit_harmonized(frame, reference):
    """Recheck cached targets and eligibility without changing observations."""
    if frame.Source_Excel_Row.isna().any() or frame.Source_Excel_Row.duplicated().any():
        raise ValueError('Source_Excel_Row must trace each retained source observation uniquely')
    keys = pd.MultiIndex.from_frame(frame[['Gender_Encoded', 'Age (months)']])
    for task, measure in [('HAZ', 'Height (cm)'), ('WAZ', 'Weight (kg)')]:
        coefficients = reference[reference.Indicator.eq(task)].set_index(['Gender_Encoded', 'Age (months)']).reindex(keys)
        calculated = monthly_zscore(frame[measure], coefficients.L, coefficients.M,
                                    coefficients.S, restrict_tails=task == 'WAZ')
        if not np.allclose(calculated, frame[task], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f'Cached {task} differs from the corrected monthly reference')
    masks = _expected_masks(frame)
    for name, expected in masks.items():
        if name not in frame and name.startswith('Flag_'):
            continue
        if (name not in frame or not frame[name].isin([True, False]).all()
                or not np.array_equal(frame[name].to_numpy(), expected.to_numpy())):
            raise ValueError(f'Cached eligibility flag differs: {name}')
    audit = {'rows': int(len(frame)), 'rows_removed_as_duplicates': 0,
             'rows_age_0_to_24': int((~masks['Flag_Age']).sum()),
             'rows_age_24': int(masks['In_Age24_Sensitivity'].sum()),
             'flags': {name: int(mask.sum()) for name, mask in masks.items() if name.startswith('Flag_')}}
    for cohort, column in COHORTS.items():
        audit[cohort] = {}
        for task, features in FEATURES.items():
            data = frame.loc[masks[column] & masks[f'Valid_{"Classification" if task == "classification" else task}']]
            audit[cohort][task] = {'n': int(len(data)),
                'unique_input_profiles': int(len(data[features].drop_duplicates())),
                'repeat_profile_excess': int(data[features].duplicated().sum())}
        comparable = frame.loc[masks[column] & masks['Valid_HAZ'] & ~masks['Flag_Source_Label']]
        audit[cohort]['source_label_vs_monthly_haz_mismatches'] = int(comparable.Stunting_Binary.ne(comparable.HAZ.lt(-2).astype(int)).sum())
        audit[cohort]['source_label_comparison_denominator'] = int(len(comparable))
    return audit


def verify_frozen_models(manifest, root=ROOT):
    paths = {}
    if set(manifest['models']) != set(FEATURES):
        raise ValueError('Freeze manifest must identify exactly classification, HAZ and WAZ models')
    for task, features in FEATURES.items():
        entry = manifest['models'][task]
        path = Path(root) / entry['path']
        if entry['features'] != features:
            raise ValueError(f'Frozen feature order changed for {task}')
        if entry.get('training_rows') != 80000:
            raise ValueError(f'Frozen {task} must declare the original 80000-row training split')
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Frozen model checksum differs for {task}')
        paths[task] = path
    return paths


def frozen_training_indices(manifest, reference_path, synthetic_path):
    if sha256(reference_path) != manifest['reference']['sha256']:
        raise ValueError('Frozen target reference checksum differs')
    if sha256(synthetic_path) != manifest['training_data']['sha256']:
        raise ValueError('Frozen synthetic training data checksum differs')
    split = manifest['training_split']
    if (split.get('rows', 100000) != 100000 or split['test_size'] != .2
            or split['random_state'] != 42 or split.get('shuffle', True) is not True
            or split['stratify'] is not None):
        raise ValueError('Expected the original random 80:20 synthetic training split')
    train, test = train_test_split(np.arange(100000, dtype=np.int64),
                                 test_size=.2, random_state=42, shuffle=True)
    for name, indices in [('train', train), ('test', test)]:
        if hashlib.sha256(indices.tobytes()).hexdigest() != split[f'{name}_indices_sha256']:
            raise ValueError(f'Frozen {name} indices checksum differs')
    return train


def _classification_metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    return {
        'accuracy': float(accuracy_score(y, prediction)),
        'balanced_accuracy': (sensitivity + specificity) / 2 if None not in [sensitivity, specificity] else None,
        'macro_f1': float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)),
        'sensitivity': sensitivity, 'specificity': specificity,
        'auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def evaluate_models(frame, models, *, confidence_intervals=False, synthetic=None):
    """Predict eligible rows only; return metrics and traceable per-record outputs."""
    rows, predictions = [], {}
    for cohort, column in COHORTS.items():
        for task, features in FEATURES.items():
            valid = f'Valid_{"Classification" if task == "classification" else task}'
            data = frame.loc[frame[column] & frame[valid]].copy()
            row = {'cohort': cohort, 'task': task, 'n': int(len(data)), 'model': 'Random Forest'}
            if data.empty:
                rows.append(row)
                continue
            model = models[task]
            if list(model.feature_names_in_) != features:
                raise ValueError(f'Loaded feature order differs for {task}')
            X = data[features]
            if task == 'classification':
                four, predicted, probability = collapse_four_class(model.predict_proba(X), model.classes_)
                y = data.Stunting_Binary.to_numpy(dtype=int)
                row.update(_classification_metrics(y, predicted, probability))
                data['predicted_class'] = four
                data['predicted_class_name'] = [CLASS_MAPPING[str(value)] for value in four]
                data['positive_probability'] = probability
            else:
                predicted = np.asarray(model.predict(X)).ravel()
                y = data[task].to_numpy(dtype=float)
                row.update({'mae': float(mean_absolute_error(y, predicted)),
                    'rmse': float(np.sqrt(mean_squared_error(y, predicted))),
                    'r2': float(r2_score(y, predicted)) if len(y) > 1 and np.var(y) > 0 else None})
            if not np.isfinite(predicted).all():
                raise ValueError(f'Nonfinite predictions for {cohort}/{task}')
            groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=False)[0]
            row['unique_input_profiles'] = int(len(np.unique(groups)))
            if synthetic is not None:
                overlap = pd.MultiIndex.from_frame(X).isin(pd.MultiIndex.from_frame(synthetic[features]))
                row['synthetic_training_input_overlap_n'] = int(overlap.sum())
                row['synthetic_training_input_overlap_fraction'] = float(overlap.mean())
                if task == 'classification':
                    decision_features = FEATURES['HAZ']
                    decision_overlap = pd.MultiIndex.from_frame(data[decision_features]).isin(pd.MultiIndex.from_frame(synthetic[decision_features]))
                    row['synthetic_training_sex_age_height_overlap_n'] = int(decision_overlap.sum())
                    row['synthetic_training_sex_age_height_overlap_fraction'] = float(decision_overlap.mean())
            if confidence_intervals:
                from .supervised import cluster_bootstrap
                bootstrap_features = FEATURES['HAZ'] if task == 'classification' else features
                bootstrap_groups = pd.factorize(pd.MultiIndex.from_frame(data[bootstrap_features]), sort=False)[0]
                intervals = cluster_bootstrap(y, predicted, bootstrap_groups, binary=task == 'classification', repetitions=2000)
                for metric, (low, high) in intervals.items():
                    row[f'{metric}_ci95_low'], row[f'{metric}_ci95_high'] = low, high
                row['bootstrap_repetitions'] = 2000
                row['bootstrap_unique_profiles'] = int(len(np.unique(bootstrap_groups)))
            data['evaluation_target'] = y
            data['prediction'] = predicted
            predictions[f'{cohort}_{task}'] = data
            rows.append(row)
    return pd.DataFrame(rows), predictions


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def evaluate_frozen(root, manifest_path, output):
    root, manifest_path, output = Path(root), Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    model_paths = verify_frozen_models(manifest, root)
    if manifest.get('classification_class_mapping') != CLASS_MAPPING:
        raise ValueError('Frozen classification label encoding differs')
    source = root / 'dataset' / 'revision'
    raw = source / 'raw' / 'jeneponto_2024.xlsx'
    harmonized = source / 'derived' / 'jeneponto_2024_harmonized.csv'
    reference_path = source / 'references' / 'who_monthly_corrected.csv'
    reference_manifest = source / 'references' / 'manifest.json'
    synthetic_path = root / 'dataset' / 'Stunting Wasting Dataset (Synthetic)' / 'stunting_wasting_dataset.csv'
    protected = [manifest_path, raw, harmonized, reference_path, reference_manifest, synthetic_path, *model_paths.values()]
    before = {str(path): sha256(path) for path in protected}
    if before[str(raw)] != JENEPONTO_SHA256:
        raise ValueError('Jeneponto raw source checksum differs')
    if before[str(reference_path)] != json.loads(reference_manifest.read_text())['corrected_sha256']:
        raise ValueError('Corrected reference checksum differs')
    train_indices = frozen_training_indices(manifest, reference_path, synthetic_path)
    reference = pd.read_csv(reference_path)
    frame = pd.read_csv(harmonized)
    audit = audit_harmonized(frame, reference)
    reparsed, raw_audit = parse_jeneponto_workbook(raw, reference)
    cached_for_comparison = frame.copy()
    cached_for_comparison['Weight_Parse_Error'] = cached_for_comparison.Weight_Parse_Error.fillna('')
    pd.testing.assert_frame_equal(cached_for_comparison, reparsed, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    audit['raw_source_reparse_matches_cached_harmonized'] = True
    audit['date_weights_restored_from_explicit_decimal_format'] = raw_audit['date_weights_restored']
    synthetic = pd.read_csv(synthetic_path).rename(columns={
        'Umur (bulan)': 'Age (months)', 'Tinggi Badan (cm)': 'Height (cm)', 'Berat Badan (kg)': 'Weight (kg)'})
    synthetic['Gender_Encoded'] = synthetic['Jenis Kelamin'].map({'Laki-laki': 1, 'Perempuan': 0, 'Male': 1, 'Female': 0})
    if len(synthetic) != 100000:
        raise ValueError('Frozen synthetic source must contain exactly 100000 observations')
    synthetic = synthetic.iloc[train_indices]
    models = {}
    for task, path in model_paths.items():
        model = joblib.load(path)
        expected = RandomForestClassifier if task == 'classification' else RandomForestRegressor
        if not isinstance(model, expected):
            raise ValueError(f'Frozen model is not the expected Random Forest class for {task}')
        models[task] = model
    metrics, predictions = evaluate_models(frame, models, confidence_intervals=True, synthetic=synthetic)
    if before != {str(path): sha256(path) for path in protected}:
        raise ValueError('A frozen model or protected source changed during external evaluation')
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / 'external_metrics.csv', index=False)
    for name, detail in predictions.items():
        detail.to_csv(output / f'external_predictions_{name}.csv', index=False)
    summary = {
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'freeze_manifest': str(manifest_path), 'freeze_manifest_sha256': before[str(manifest_path)],
        'frozen_at_utc': manifest['frozen_at_utc'],
        'training_on_external_data': False, 'model_selection_on_external_data': False,
        'decision_rule': 'Four-class argmax; classes 1/2 positive and 0/3 negative. Probability sum is used only for AUC.',
        'regression_target': 'Corrected monthly HAZ/WAZ, not source-provided Z-score columns.',
        'primary_cohort': 'Completed months 0–23; age 24 is a separate sensitivity analysis.',
        'bootstrap': '2000 profile cluster resamples, seed 42; classification uses sex-age-height groups (as in the grouped sensitivity test), regressors use their three input features. Not unique-child intervals.',
        'input_overlap_reference': 'Actual frozen 80000-row synthetic training split, reconstructed and checked against train/test index SHA256 values.',
        'audit': audit, 'metrics': metrics.astype(object).where(pd.notna(metrics), None).to_dict('records'),
        'limitations': LIMITATIONS, 'protected_files_sha256': before, 'protected_files_unchanged': True,
    }
    _write_json(output / 'external_validation.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'reviewer2_validation')
    args = parser.parse_args()
    summary = evaluate_frozen(ROOT, args.manifest, args.output)
    print(json.dumps(summary['metrics'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
