"""Export verified, already-fitted paper models for the static dashboard."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .supervised import CLASSES, CODES, FEATURES, export_model

ROOT = Path(__file__).resolve().parents[2]
FINAL = Path('results/regression_corrected')
MODEL_DIRECTORY = Path('docs/models/paper_final')
CLASSIFIERS = ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 'CatBoost']
REGRESSORS = ['Linear Regression', 'Ridge Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']


def read_json(path):
    return json.loads(path.read_text())


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify_hash(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Source path escapes repository: {relative}')
    if checksum(path) != expected:
        raise ValueError(f'Source checksum mismatch: {relative}')


def source_catalog(root=ROOT):
    verification = read_json(root / FINAL / 'verification.json')
    catalog = []
    for task in FEATURES:
        for name in CLASSIFIERS if task == 'classification' else REGRESSORS:
            stem = name.lower().replace(' ', '_')
            if task == 'classification':
                source = f'results/classification/stunting_{stem}_classifier.joblib'
                digest = verification['source_sha256'][source]
            else:
                relative = f'models/stunting_{stem}_reg_{task.lower()}.joblib'
                source = str(FINAL / relative)
                digest = verification['output_sha256'][relative]
            catalog.append({'task': task, 'model': name, 'code': CODES[name],
                            'source': source, 'source_sha256': digest, 'features': FEATURES[task]})
    return catalog


def verify_sources(root=ROOT):
    catalog = source_catalog(root)
    for item in catalog:
        verify_hash(root, item['source'], item['source_sha256'])
    freeze = read_json(root / FINAL / 'model_freeze_manifest.json')
    for item in [freeze['training_data'], freeze['reference'], *freeze['models'].values()]:
        verify_hash(root, item['path'], item['sha256'])
    verification = read_json(root / FINAL / 'verification.json')
    for name in ['Stunting_Classification_Analysis.ipynb',
                 'results/classification/stunting_gender_encoder.joblib',
                 'results/classification/stunting_target_encoder.joblib']:
        verify_hash(root, name, verification['source_sha256'][name])
    return catalog


class NotebookTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.rows.append([])
        if tag in ('th', 'td'):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.rows[-1].append(''.join(self.cell).strip())
            self.cell = None


def classification_metrics(root):
    notebook = read_json(root / 'Stunting_Classification_Analysis.ipynb')
    for index, cell in enumerate(notebook['cells']):
        for output in cell.get('outputs', []):
            html = ''.join(output.get('data', {}).get('text/html', []))
            if 'Classification Model' not in html or 'Weighted F1-Score' not in html:
                continue
            parser = NotebookTable()
            parser.feed(html)
            records = [dict(zip(parser.rows[0], row)) for row in parser.rows[1:]]
            return {'source': 'Stunting_Classification_Analysis.ipynb', 'cell_index': index,
                    'protocol': 'Primary random stratified five-fold CV on synthetic data',
                    'recorded_precision': 'Rounded notebook display values, four decimals',
                    'rows': [{'model': row['Classification Model'], 'accuracy': float(row['Accuracy']),
                              'weighted_f1': float(row['Weighted F1-Score']),
                              'weighted_auc': float(row['Weighted AUC-ROC'])} for row in records]}
    raise ValueError('The recorded primary classification table was not found')


def evaluation_metadata(root):
    external = {'source': 'results/reviewer2_validation/external_metrics.csv'}
    table = pd.read_csv(root / external['source'])
    keys = ['accuracy', 'sensitivity', 'specificity', 'auc', 'macro_f1', 'balanced_accuracy', 'mae', 'rmse', 'r2']
    for row in table.to_dict('records'):
        external.setdefault(row['cohort'], {})[row['task']] = {
            'model': row['model'], 'code': 'rf', 'n': int(row['n']),
            'metrics': {key: float(row[key]) for key in keys if pd.notna(row[key])}}
    regression = read_json(root / FINAL / 'summary.json')
    grouped = pd.read_csv(root / 'results/revision_1/supervised/cv_metrics.csv')
    grouped = grouped[(grouped.task == 'classification') & (grouped.protocol == 'grouped')]
    return {'classification_cv': classification_metrics(root),
            'classification_grouped_cv': {'source': 'results/revision_1/supervised/cv_metrics.csv',
                'protocol': 'Supplementary grouped five-fold CV on synthetic data',
                'rows': grouped[['model', 'accuracy', 'macro_f1', 'balanced_accuracy']].to_dict('records')},
            'regression_cv': {'source': 'results/regression_corrected/summary.json', 'rows': regression['cv_metrics']},
            'external': external}


def parity_features(root):
    frame = pd.read_csv(root / 'dataset/Stunting Wasting Dataset (Synthetic)/stunting_wasting_dataset.csv')
    sample = frame.sample(40, random_state=42).rename(columns={
        'Umur (bulan)': 'Age (months)', 'Tinggi Badan (cm)': 'Height (cm)', 'Berat Badan (kg)': 'Weight (kg)'})
    encoder = joblib.load(root / 'results/classification/stunting_gender_encoder.joblib')
    if list(encoder.classes_) != ['Female', 'Male']:
        raise ValueError('Unexpected gender encoding')
    gender = sample['Jenis Kelamin'].replace({'Laki-laki': 'Male', 'Perempuan': 'Female'})
    sample['Gender_Encoded'] = encoder.transform(gender)
    records = sample[FEATURES['classification']].to_dict('records')
    for gender in (0, 1):
        for age, height, weight in [(0, 49.5, 3.1), (1, 50.5000001, 4.2), (12, 74.2, 8.5),
                                    (23, 84.5, 11.2), (24, 85.1, 11.5)]:
            records.append(dict(zip(FEATURES['classification'], [gender, age, height, weight])))
    return pd.DataFrame(records)


def regional_payload(root=ROOT):
    from .clustering import PROVINCE_ALIASES, SUMMARY_METHOD, browser_records

    directory = Path('results/clustering_revised')
    verification = read_json(root / directory / 'verification.json')
    names = ['province_profiles.csv', 'coverage.csv', 'group_profiles.csv', 'metrics.csv', 'summary.json']
    for name in names:
        verify_hash(root, str(directory / name), verification['output_sha256'][name])
    profiles = pd.read_csv(root / directory / 'province_profiles.csv')
    coverage = pd.read_csv(root / directory / 'coverage.csv')
    assignments = {name: profiles[column].to_numpy(dtype=int) - 1 for name, column in
                   [('K-Means', 'Group_KMeans'), ('GMM', 'Group_GMM'), ('Ward', 'Group_Hierarchical')]}
    records = browser_records(profiles, coverage, assignments)
    groups = pd.read_csv(root / directory / 'group_profiles.csv')
    clusters = {}
    for algorithm, key in [('K-Means', 'Klaster_KMeans'), ('GMM', 'Klaster_GMM'), ('Hierarchical', 'Klaster_HC')]:
        clusters[key] = []
        for row in groups[groups.Algorithm == algorithm].to_dict('records'):
            group = int(row['Group'].split()[-1])
            means = {name: float(row[name]) for name in ['Stunting_2019', 'KEK', 'Immunization', 'BBLR']}
            count = int(row['Provinces'])
            text = (f'stunting {means["Stunting_2019"]:.2f}%, KEK {means["KEK"]:.2f}%, '
                    f'immunization {means["Immunization"]:.2f}%, BBLR {means["BBLR"]:.2f}%.')
            clusters[key].append({'id': group - 1, 'title': {'id': f'Kelompok {group}', 'en': f'Group {group}'},
                'desc': {'id': f'{count} provinsi. Rerata indikator kabupaten/kota: ' + text.replace('immunization', 'imunisasi'),
                         'en': f'{count} provinces. Mean district indicators: ' + text},
                'province_count': count, 'indicator_means_pct': means, 'provinces': row['Members'].split(' | ')})
    reference = {'Stunting_2019': float(profiles.Stunting_2019.mean()),
                 'KEK_Ibu_Hamil': float(profiles.KEK.mean() / 100),
                 'Imunisasi_Dasar_Lengkap': float(profiles.Immunization.mean() / 100),
                 'BBLR': float(profiles.BBLR.mean() / 100)}
    geography = read_json(root / 'dataset/indonesia-province-simple.json')
    canonical = {name.upper(): name for name in profiles.Province}
    matched = {PROVINCE_ALIASES.get(feature['properties']['Propinsi'],
               canonical.get(feature['properties']['Propinsi'], feature['properties']['Propinsi']))
               for feature in geography['features']}
    missing = sorted(set(profiles.Province) - matched)
    metadata = {'selected_k': 2, 'summary_method': SUMMARY_METHOD, 'historical_province_count': 34,
                'results_source': str(directory / 'province_profiles.csv'),
                'results_sha256': verification['output_sha256']['province_profiles.csv'],
                'units': {'Stunting_2019': 'percent', 'KEK_Ibu_Hamil': 'proportion',
                          'Imunisasi_Dasar_Lengkap': 'proportion', 'BBLR': 'proportion'},
                'map_geometry': {'status': 'legacy_unverified', 'matched_provinces': 34-len(missing),
                                 'feature_count': len(geography['features']), 'missing_provinces': missing},
                'algorithm_metrics': pd.read_csv(root / directory / 'metrics.csv').to_dict('records')}
    return records, clusters, reference, metadata


def export_regions(root=ROOT):
    records, clusters, reference, metadata = regional_payload(root)
    def js(value):
        return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    (root / 'docs/province_data.js').write_text('const PROVINCE_DATA = ' + js(records)
        + ';\nconst PROVINCE_METADATA = ' + js(metadata) + ';\n')
    (root / 'docs/cluster_profiles.js').write_text('const CLUSTER_METRICS = ' + js(clusters)
        + ';\nconst REFERENCE_AVERAGES = ' + js(reference) + ';\n')
    metadata['browser_sha256'] = {path: checksum(root / path) for path in ['docs/province_data.js', 'docs/cluster_profiles.js']}
    return metadata


def export(root=ROOT):
    catalog = verify_sources(root)
    frame = parity_features(root)
    target_encoder = joblib.load(root / 'results/classification/stunting_target_encoder.joblib')
    if list(target_encoder.classes_) != CLASSES:
        raise ValueError('Unexpected classification class order')
    manifest = {'version': 'paper-final', 'model_directory': 'models/paper_final', 'classes': CLASSES,
                'gender_encoding': {'Female': 0, 'Male': 1},
                'selection_basis': read_json(root / FINAL / 'model_freeze_manifest.json')['selection_basis'],
                'selected': {task: {'code': 'rf', 'model': 'Random Forest', 'features': names} for task, names in FEATURES.items()},
                'available': {task.lower(): [CODES[name] for name in (CLASSIFIERS if task == 'classification' else REGRESSORS)] for task in FEATURES},
                'artifacts': {}, **evaluation_metadata(root)}
    fixtures = []
    (root / 'docs/models').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / 'docs/models', prefix='.paper-final-') as directory:
        staging = Path(directory)
        for item in catalog:
            task, code = item['task'], item['code']
            model = joblib.load(root / item['source'])
            if hasattr(model, 'n_jobs'):
                # Serial prediction makes reference fixtures byte-reproducible.
                model.set_params(n_jobs=1)
            names = list(getattr(model, 'feature_names_in_', getattr(model, 'feature_names_', FEATURES[task])))
            normalize = lambda name: name if name == 'Gender_Encoded' else name.replace('_', ' ')
            if [normalize(name) for name in names] != FEATURES[task]:
                raise ValueError(f'Unexpected feature order for {task}/{code}: {names}')
            filename = f'{code}.json' if task == 'classification' else f'{code}_{task.lower()}.json'
            export_model(model, task, item['model'], staging / filename)
            native = frame[FEATURES[task]].copy()
            native.columns = names
            prediction = np.asarray(model.predict(native)).ravel()
            fixture = {'task': task, 'code': code, 'features': frame.to_dict('records'),
                       'prediction': prediction.tolist()}
            if task == 'classification':
                if list(model.classes_) != [0, 1, 2, 3]:
                    raise ValueError(f'Unexpected class order for {code}')
                fixture['probabilities'] = model.predict_proba(native).tolist()
            fixtures.append(fixture)
            manifest['artifacts'][f'{task}_{code}'] = {**item,
                'browser_path': f'models/paper_final/{filename}',
                'json_sha256': checksum(staging / filename),
                'classes': CLASSES if task == 'classification' else None}
            print(f'Exported {task}/{item["model"]}', flush=True)
            del model
            gc.collect()
        write_json(staging / 'parity_samples.json', fixtures)
        destination = root / MODEL_DIRECTORY
        destination.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            os.replace(path, destination / path.name)
    manifest['parity_sha256'] = checksum(root / MODEL_DIRECTORY / 'parity_samples.json')
    manifest['regional'] = export_regions(root)
    old = root / 'docs/revision_manifest.json'
    history = root / 'docs/revision_1_manifest.json'
    if not history.exists() and old.exists():
        if read_json(old).get('version') == 'paper-final':
            raise ValueError('Historical manifest is missing; refusing to overwrite it')
        shutil.copyfile(old, history)
    write_json(old, manifest)
    return verify(root)


def verify(root=ROOT):
    catalog = verify_sources(root)
    manifest = read_json(root / 'docs/revision_manifest.json')
    expected = {f'{item["task"]}_{item["code"]}' for item in catalog}
    if set(manifest['artifacts']) != expected or manifest['version'] != 'paper-final':
        raise ValueError('The dashboard manifest does not contain all 17 final artifacts')
    available = {task.lower(): [CODES[name] for name in (CLASSIFIERS if task == 'classification' else REGRESSORS)]
                 for task in FEATURES}
    selected = {task: {'code': 'rf', 'model': 'Random Forest', 'features': names}
                for task, names in FEATURES.items()}
    if (manifest.get('classes') != CLASSES or manifest.get('gender_encoding') != {'Female': 0, 'Male': 1}
            or manifest.get('available') != available or manifest.get('selected') != selected
            or manifest.get('model_directory') != 'models/paper_final'):
        raise ValueError('Manifest class order, gender encoding, availability or defaults mismatch')
    for item in catalog:
        artifact = manifest['artifacts'][f'{item["task"]}_{item["code"]}']
        task, code = item['task'], item['code']
        filename = f'{code}.json' if task == 'classification' else f'{code}_{task.lower()}.json'
        if (any(artifact.get(key) != value for key, value in item.items())
                or artifact.get('classes') != (CLASSES if task == 'classification' else None)
                or artifact.get('browser_path') != f'models/paper_final/{filename}'):
            raise ValueError('Manifest source metadata mismatch')
        verify_hash(root, str(Path('docs') / artifact['browser_path']), artifact['json_sha256'])
    verify_hash(root, str(MODEL_DIRECTORY / 'parity_samples.json'), manifest['parity_sha256'])
    regional_payload(root)
    for path, digest in manifest['regional']['browser_sha256'].items():
        verify_hash(root, path, digest)
    print('Verified all 17 source models, browser exports, and parity fixtures.', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'verify'])
    arguments = parser.parse_args()
    export() if arguments.action == 'export' else verify()
