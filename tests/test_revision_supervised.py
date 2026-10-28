import unittest
import numpy as np
import pandas as pd

try:
    from analysis.revision import supervised
except ImportError:
    supervised = None


class SupervisedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(supervised, 'The reproducible evaluation module must exist')

    def test_identical_decision_inputs_stay_in_one_group_despite_weight_changes(self):
        frame = pd.DataFrame({
            'Gender_Encoded': [1, 1, 0], 'Age (months)': [12, 12, 12],
            'Height (cm)': [70., 70., 70.], 'Weight (kg)': [8., 11., 8.],
        })
        groups = supervised.group_ids(frame, 'classification')
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_binary_status_collapses_original_argmax_without_retuning_threshold(self):
        # Normal remains negative even when the TWO positive probabilities total > .5.
        probs = np.array([[.4, .3, .3, 0.], [.1, .6, .2, .1], [.1, .1, .6, .2]])
        predicted, positive_probability = supervised.binary_predictions(np.array([0, 1, 2]), probs)
        np.testing.assert_array_equal(predicted, [0, 1, 1])
        np.testing.assert_allclose(positive_probability, [.6, .8, .7])

    def test_grouped_folds_have_no_duplicate_decision_inputs_across_split(self):
        frame = pd.DataFrame({
            'Gender_Encoded': [i % 2 for i in range(40)],
            'Age (months)': [i // 4 for i in range(40)],
            'Height (cm)': [60. + i // 4 for i in range(40)],
            'Weight (kg)': [6. + i % 4 for i in range(40)],
        })
        y = np.array([i // 4 % 4 for i in range(40)])
        folds, diagnostics = supervised.make_folds(frame, y, 'classification', 'grouped')
        groups = supervised.group_ids(frame, 'classification')
        for train, valid in folds:
            self.assertFalse(set(groups[train]) & set(groups[valid]))
        self.assertEqual(diagnostics['max_validation_input_overlap'], 0.)

    def test_nonfinite_default_parameters_are_serializable_without_hiding_metric_errors(self):
        import json
        params = supervised.serializable_parameters({'missing': float('nan'), 'depth': 6})
        self.assertIsNone(params['missing'])
        self.assertEqual(json.loads(json.dumps(params, allow_nan=False))['depth'], 6)


if __name__ == '__main__':
    unittest.main()
