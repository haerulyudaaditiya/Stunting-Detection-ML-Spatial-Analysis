"""Read-only clustering audit; writes evidence only beside this script.

Run from repository root:
MPLCONFIGDIR=/tmp/icdees-mpl python results/reviewer1_question3/reproduce_analysis.py
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.revision.clustering import (
    FEATURES, PROVINCE_ALIASES, read_district_indicators, aggregate_profiles,
)

OUT = Path(__file__).resolve().parent
notebook = ROOT/'Stunting_Clustering_Analysis.ipynb'
source = ROOT/'dataset/Data-Status-Gizi-2018-2020-Kabupaten-Kota.xlsx'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p.relative_to(ROOT)): sha(p) for p in [notebook, source]}
nb = json.loads(notebook.read_text())
namespace = dict(pd=pd, np=np, KNNImputer=KNNImputer, StandardScaler=StandardScaler)
log = io.StringIO()
with redirect_stdout(log):
    exec(''.join(nb['cells'][3]['source']), namespace)
    exec(''.join(nb['cells'][9]['source']), namespace)
    original_missing = namespace['df_prov'].isna().sum().to_dict()
    before_imputation = namespace['df_prov'].copy(deep=True)
    exec(''.join(nb['cells'][11]['source']), namespace)
    exec(''.join(nb['cells'][15]['source']), namespace)
original = namespace['df_prov'].copy()
original['Province'] = original.Province.replace(PROVINCE_ALIASES)
original = original.sort_values('Province').reset_index(drop=True)
old_features = ['Total_Births', 'Stunting_2019', 'Maternal_Wasting',
                'Complete_Basic_Immunization', 'Low_Birth_Weight']
districts = read_district_indicators(source)
profiles, coverage = aggregate_profiles(districts)
complete, _ = aggregate_profiles(districts, complete_case=True)
saved_dir = ROOT/'results/revision_1/clustering'
saved_summary = json.loads((saved_dir/'summary.json').read_text())
assert sha(source) == saved_summary['source_sha256']
cached = pd.read_csv(saved_dir/'profiles.csv')
assert profiles.Province.tolist() == cached.Province.tolist()
assert np.allclose(profiles[FEATURES], cached[FEATURES], rtol=0, atol=1e-12)
assert len(districts) == 514 and len(profiles) == 34
assert original.Province.tolist() == profiles.Province.tolist()

rows = []
with threadpool_limits(limits=1):
    # These three cases keep the original constructor settings. Correcting
    # preprocessing and changing k can therefore be distinguished explicitly.
    for scenario, frame, features, k in [
        ('original_reconstructed_k3', original, old_features, 3),
        ('corrected_preprocessing_k3_original_model_settings', profiles, FEATURES, 3),
        ('corrected_preprocessing_k2_original_model_settings', profiles, FEATURES, 2),
    ]:
        x = StandardScaler().fit_transform(frame[features])
        models = {'K-Means': KMeans(n_clusters=k, random_state=42),
                  'GMM': GaussianMixture(n_components=k, random_state=42),
                  'Ward': AgglomerativeClustering(n_clusters=k, linkage='ward')}
        for name, model in models.items():
            labels = model.fit_predict(x)
            rows.append({'scenario': scenario, 'algorithm': name, 'k': k,
                'silhouette': float(silhouette_score(x, labels)),
                'davies_bouldin': float(davies_bouldin_score(x, labels)),
                'calinski_harabasz': float(calinski_harabasz_score(x, labels)),
                'group_sizes': np.bincount(labels).tolist(), 'parameters': model.get_params()})

    # Independently refit the existing corrected configurations, without
    # calling run_clustering (which overwrites its old outputs).
    x = StandardScaler().fit_transform(profiles[FEATURES])
    saved_metrics = pd.read_csv(saved_dir/'metrics.csv')
    for k, scenario in [(2, 'primary'), (3, 'submitted_k3_comparison')]:
        models = {'K-Means': KMeans(n_clusters=k, n_init=50, random_state=42),
            'GMM': GaussianMixture(n_components=k, n_init=20, max_iter=1000,
                covariance_type='full', reg_covar=1e-6, random_state=42),
            'Ward': AgglomerativeClustering(n_clusters=k, linkage='ward')}
        for name, model in models.items():
            labels = model.fit_predict(x)
            actual = [silhouette_score(x, labels), davies_bouldin_score(x, labels), calinski_harabasz_score(x, labels)]
            expected = saved_metrics[(saved_metrics.scenario == scenario) & (saved_metrics.algorithm == name)].iloc[0]
            assert np.allclose(actual, expected[['silhouette','davies_bouldin','calinski_harabasz']].to_numpy(dtype=float), rtol=0, atol=1e-10)
    selections = []
    for k in range(2, 9):
        m = KMeans(n_clusters=k, n_init=50, random_state=42).fit(x)
        selections.append({'k': k, 'wss': float(m.inertia_), 'silhouette': float(silhouette_score(x, m.labels_))})
    expected_selection = pd.read_csv(saved_dir/'k_selection.csv')
    assert np.allclose(pd.DataFrame(selections)[['wss','silhouette']], expected_selection[['wss','silhouette']], rtol=0, atol=1e-10)
    original_setting_selections = []
    for k in range(2, 9):
        m = KMeans(n_clusters=k, random_state=42).fit(x)
        original_setting_selections.append({'k': k, 'wss': float(m.inertia_),
            'silhouette': float(silhouette_score(x, m.labels_))})

old_rates = original.rename(columns={'Maternal_Wasting':'KEK',
    'Complete_Basic_Immunization':'Immunization', 'Low_Birth_Weight':'BBLR'})[['Province',*FEATURES]].copy()
old_rates[['KEK','Immunization','BBLR']] *= 100
comparison = old_rates.merge(profiles, on='Province', suffixes=('_original_weighted','_corrected_equal_district'))
for feature in FEATURES:
    comparison[feature+'_difference_pp'] = comparison[feature+'_corrected_equal_district'] - comparison[feature+'_original_weighted']
comparison.to_csv(OUT/'province_summary_comparison.csv', index=False)
pd.DataFrame([{k:v for k,v in r.items() if k != 'parameters'} for r in rows]).to_csv(OUT/'controlled_comparison.csv', index=False)

report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': before,
    'scope': 'Analysis only. No notebook, paper, dashboard, or previous result is overwritten.',
    'versions': {p: version(p) for p in ['numpy','pandas','scikit-learn']},
    'original_missing_before_province_imputation': original_missing,
    'original_imputation_changed_values': not before_imputation.equals(namespace['df_prov']),
    'district_missing_by_indicator': districts[FEATURES].isna().sum().to_dict(),
    'districts': len(districts), 'provinces': len(profiles),
    'complete_case_districts': int(districts[FEATURES].notna().all(axis=1).sum()),
    'corrected_has_missing_province_indicators': bool(profiles[FEATURES].isna().any().any()),
    'controlled_comparison': rows, 'corrected_k_selection_recomputed': selections,
    'corrected_k_selection_original_constructor_settings': original_setting_selections,
    'existing_corrected_profiles_and_six_metric_rows_reproduced': True,
    'important_choice': 'Corrected descriptive profiles use unweighted observed-district means. This is an explicit estimand choice, not the uniquely correct replacement for population weights.',
    'changes_beyond_input_fix': ['Four rate-only features', 'Available-case equal-district means',
        'K-Means n_init=50', 'GMM n_init=20 and max_iter=1000', 'k=2 by maximum K-Means silhouette over 2..8'],
    'legacy_execution_log': log.getvalue(),
    'source_files_unchanged': all(sha(ROOT/p)==h for p,h in before.items()),
}
assert report['source_files_unchanged']
(OUT/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(pd.DataFrame(rows)[['scenario','algorithm','k','silhouette','davies_bouldin','calinski_harabasz']].to_string(index=False))
print('Existing corrected profiles, six metric rows, and k-selection results independently reproduced.')
print('Original province imputation changed values:',report['original_imputation_changed_values'])
print('Source files unchanged:',report['source_files_unchanged'])
