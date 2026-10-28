import importlib
import unittest

import numpy as np
import pandas as pd


class ReviewerOneSplitTests(unittest.TestCase):
    def setUp(self):
        try:
            self.study = importlib.import_module('analysis.revision.reviewer1_splits')
        except ModuleNotFoundError:
            self.fail('The separate reviewer-one split experiment is not implemented')
        self.frame = pd.DataFrame([
            {'Gender_Encoded': c % 2, 'Age (months)': p,
             'Height (cm)': 50 + c, 'Weight (kg)': weight, 'target': c}
            for c in range(4) for p in range(12)
            for weight in [4., 5.] for repeat in range(2)
        ])

    def test_four_input_grouping_keeps_exact_duplicates_together(self):
        folds = self.study.split_folds(self.frame, self.frame.target, 'four_inputs', 42)
        groups = pd.MultiIndex.from_frame(self.frame[self.study.FEATURES])
        covered = []
        for train, valid in folds:
            self.assertFalse(groups[valid].isin(groups[train]).any())
            self.assertFalse(np.intersect1d(train, valid).size)
            covered.extend(valid)
        self.assertEqual(sorted(covered), list(range(len(self.frame))))

    def test_three_input_grouping_keeps_weight_variants_together(self):
        folds = self.study.split_folds(self.frame, self.frame.target, 'sex_age_height', 42)
        groups = pd.MultiIndex.from_frame(self.frame[self.study.FEATURES[:3]])
        for train, valid in folds:
            self.assertFalse(groups[valid].isin(groups[train]).any())

    def test_random_baseline_reproduces_original_splitter(self):
        from sklearn.model_selection import StratifiedKFold
        actual = self.study.split_folds(self.frame, self.frame.target, 'random', 42)
        expected = StratifiedKFold(5, shuffle=True, random_state=42).split(self.frame, self.frame.target)
        for (train, valid), (old_train, old_valid) in zip(actual, expected):
            np.testing.assert_array_equal(train, old_train)
            np.testing.assert_array_equal(valid, old_valid)
