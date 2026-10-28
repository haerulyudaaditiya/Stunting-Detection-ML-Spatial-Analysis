"""Post-hoc explanation only: training-profile lookup on original random folds.

Run from the repository root: python results/reviewer1_question1/reproduce_lookup.py
This diagnostic neither changes labels nor trains/selects a deployment model.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

root = Path(__file__).resolve().parents[2]
source = root/'dataset/Stunting Wasting Dataset (Synthetic)/stunting_wasting_dataset.csv'
data = pd.read_csv(source)
keys = ['Jenis Kelamin', 'Umur (bulan)', 'Tinggi Badan (cm)']
groups = pd.Series(pd.factorize(pd.MultiIndex.from_frame(data[keys]), sort=False)[0])
target = data.Stunting
predictions = pd.Series(index=data.index, dtype='object')
folds = []
for fold, (train, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(data, target), 1):
    training = pd.DataFrame({'group': groups.iloc[train].to_numpy(), 'label': target.iloc[train].to_numpy()})
    assert training.groupby('group').label.nunique().max() == 1
    lookup = training.groupby('group').label.first()
    guessed = groups.iloc[valid].map(lookup)
    seen = guessed.notna()
    fallback = target.iloc[train].value_counts().idxmax()
    guessed = guessed.fillna(fallback)
    predictions.iloc[valid] = guessed.to_numpy()
    folds.append({'fold': fold, 'n_valid': len(valid), 'seen_profile_rows': int(seen.sum()),
        'unseen_profile_rows': int((~seen).sum()), 'fallback_from_training': fallback,
        'accuracy': float(accuracy_score(target.iloc[valid], guessed))})
report = {
    'purpose': 'Post-hoc explanatory diagnostic, not model selection or clinical validation.',
    'rule': 'Look up the training label for an identical sex-age-height profile; use training majority class only for unseen profiles.',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'split': 'Original stratified 5-fold, shuffle=True, seed=42', 'folds': folds,
    'pooled_accuracy': float(accuracy_score(target, predictions)),
    'pooled_macro_f1': float(f1_score(target, predictions, average='macro')),
    'interpretation_limit': 'Shows that a lookup can achieve high accuracy on this random split; does not establish that RF memorized or reveal the original label generator.',
}
Path(__file__).with_name('lookup_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
