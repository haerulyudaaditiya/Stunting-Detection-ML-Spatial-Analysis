import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class FixedRegressor:
    def __init__(self, features):
        self.feature_names_in_ = np.array(features)

    def predict(self, features):
        return np.zeros(len(features))

    def fit(self, *args):
        raise AssertionError('External observations must never enter fitting')


class FixedClassifier(FixedRegressor):
    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, features):
        return np.tile([.4, .3, .3, 0.], (len(features), 1))


class ExternalValidationTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('analysis.revision.reviewer2_external_validation')
        except ModuleNotFoundError:
            self.fail('The frozen-model external evaluator is not implemented')

    def frame(self):
        frame = pd.DataFrame({
            'Source_Row': [1, 2, 3, 4], 'Source_Excel_Row': [2, 3, 4, 5],
            'Gender_Encoded': [1, 1, 0, 1], 'Age (months)': [12., 12., 24., 25.],
            'Height (cm)': [75., 75., 75., 75.], 'Weight (kg)': [10., 10., 10., 10.],
            'Stunting_Binary': [1, 0, 1, 0],
            'HAZ': [0., 0., 0., 0.], 'WAZ': [0., 0., 0., 0.],
        })
        valid = [True, True, True, False]
        for task in ['HAZ', 'WAZ', 'Classification']:
            frame[f'Valid_{task}'] = valid
        frame['In_Primary_External'] = [True, True, False, False]
        frame['In_Age24_Sensitivity'] = [False, False, True, False]
        return frame

    def reference(self):
        return pd.DataFrame([
            [indicator, sex, age, 1., 75. if indicator == 'HAZ' else 10., .1]
            for indicator in ['HAZ', 'WAZ'] for sex in [0, 1] for age in [12, 24, 25]
        ], columns=['Indicator', 'Gender_Encoded', 'Age (months)', 'L', 'M', 'S'])

    def test_four_class_argmax_is_collapsed_without_probability_threshold(self):
        api = self.api()
        probabilities = np.array([[.3, 0., .4, .3], [.6, .1, .1, .2]])
        four, binary, positive = api.collapse_four_class(probabilities, [2, 3, 0, 1])
        np.testing.assert_array_equal(four, [0, 2])
        np.testing.assert_array_equal(binary, [0, 1])
        np.testing.assert_allclose(positive, [.6, .8])
        with self.assertRaisesRegex(ValueError, 'four'):
            api.collapse_four_class(np.array([[.5, .5]]), [0, 1])

    def test_duplicate_profiles_and_source_label_disagreements_are_retained(self):
        api = self.api()
        frame = self.frame()
        before = frame.copy(deep=True)
        audit = api.audit_harmonized(frame, self.reference())
        self.assertEqual(audit['rows'], 4)
        self.assertEqual(audit['primary']['classification']['n'], 2)
        self.assertEqual(audit['primary']['classification']['unique_input_profiles'], 1)
        self.assertEqual(audit['primary']['source_label_vs_monthly_haz_mismatches'], 1)
        self.assertEqual(audit['age24']['classification']['n'], 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_stale_validity_masks_and_stale_targets_are_rejected(self):
        api = self.api()
        frame = self.frame()
        frame.loc[3, 'Valid_HAZ'] = True
        with self.assertRaisesRegex(ValueError, 'Valid_HAZ'):
            api.audit_harmonized(frame, self.reference())
        frame = self.frame()
        frame.loc[0, 'WAZ'] = .01
        with self.assertRaisesRegex(ValueError, 'WAZ'):
            api.audit_harmonized(frame, self.reference())

    def test_evaluation_uses_task_masks_preserves_labels_and_never_fits(self):
        api = self.api()
        frame = self.frame()
        frame.loc[1, 'Valid_WAZ'] = False
        frame.loc[1, 'Valid_Classification'] = False
        models = {task: FixedRegressor(features) for task, features in api.FEATURES.items()}
        models['classification'] = FixedClassifier(api.FEATURES['classification'])
        metrics, predictions = api.evaluate_models(frame, models)
        row = metrics.query("cohort == 'primary' and task == 'classification'").iloc[0]
        self.assertEqual(row['n'], 1)
        self.assertEqual(row['accuracy'], 0.)
        detail = predictions['primary_classification']
        self.assertEqual(detail['evaluation_target'].tolist(), [1])
        self.assertEqual(detail['prediction'].tolist(), [0])
        self.assertEqual(detail['predicted_class'].tolist(), [0])
        self.assertAlmostEqual(detail['positive_probability'].iloc[0], .6)
        self.assertEqual(metrics.query("cohort == 'primary' and task == 'HAZ'").iloc[0]['n'], 2)

    def test_freeze_integrity_rejects_changed_model_before_loading(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'frozen.joblib'
            path.write_bytes(b'frozen model')
            entry = {'path': 'frozen.joblib', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'training_rows': 80000}
            manifest = {'models': {task: {**entry, 'features': features}
                                    for task, features in api.FEATURES.items()}}
            api.verify_frozen_models(manifest, root)
            path.write_bytes(b'modified model')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                api.verify_frozen_models(manifest, root)

    def test_overlap_reports_all_four_classifier_inputs_and_separate_three_input_groups(self):
        api = self.api()
        frame = self.frame()
        synthetic = frame.iloc[[0]].copy()
        synthetic['Weight (kg)'] = 9.
        models = {task: FixedRegressor(features) for task, features in api.FEATURES.items()}
        models['classification'] = FixedClassifier(api.FEATURES['classification'])
        metrics, _ = api.evaluate_models(frame, models, synthetic=synthetic)
        row = metrics.query("cohort == 'primary' and task == 'classification'").iloc[0]
        self.assertEqual(row['synthetic_training_input_overlap_n'], 0)
        self.assertEqual(row['synthetic_training_sex_age_height_overlap_n'], 2)

    def test_declared_full_data_models_are_rejected_for_the_frozen_holdout_protocol(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'frozen.joblib'
            path.write_bytes(b'frozen model')
            manifest = {'models': {task: {'path': 'frozen.joblib',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'features': features, 'training_rows': 100000}
                for task, features in api.FEATURES.items()}}
            with self.assertRaisesRegex(ValueError, '80000'):
                api.verify_frozen_models(manifest, root)

    def test_frozen_source_reference_and_training_indices_are_bound_before_evaluation(self):
        api = self.api()
        train, test = train_test_split(np.arange(100000, dtype=np.int64), test_size=.2, random_state=42)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'synthetic.csv'
            reference = Path(directory) / 'reference.csv'
            source.write_bytes(b'original synthetic observations')
            reference.write_bytes(b'corrected WHO reference')
            manifest = {'training_data': {'sha256': api.sha256(source)},
                        'reference': {'sha256': api.sha256(reference)},
                        'training_split': {'test_size': .2, 'random_state': 42,
                            'shuffle': True, 'stratify': None,
                            'train_indices_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
                            'test_indices_sha256': hashlib.sha256(test.tobytes()).hexdigest()}}
            actual = api.frozen_training_indices(manifest, reference, source)
            np.testing.assert_array_equal(actual, train)
            reference.write_bytes(b'changed WHO reference')
            with self.assertRaisesRegex(ValueError, 'reference'):
                api.frozen_training_indices(manifest, reference, source)
            reference.write_bytes(b'corrected WHO reference')
            source.write_bytes(b'changed synthetic observations')
            with self.assertRaisesRegex(ValueError, 'training data'):
                api.frozen_training_indices(manifest, reference, source)
            source.write_bytes(b'original synthetic observations')
            manifest['training_split']['train_indices_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'indices'):
                api.frozen_training_indices(manifest, reference, source)


if __name__ == '__main__':
    unittest.main()
