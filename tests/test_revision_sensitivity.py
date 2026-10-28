"""Check that supplemental diagnostics preserve frozen data and model decisions."""

import importlib
import unittest

import numpy as np
import pandas as pd


class RevisionSensitivityTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("analysis.revision.sensitivity")
        except ModuleNotFoundError:
            self.module = None
        self.assertIsNotNone(self.module, "Supplemental sensitivity module is required")

    def test_majority_baseline_is_fitted_on_synthetic_labels_only(self):
        synthetic = pd.DataFrame({"Stunting": ["Normal"] * 8 + ["Stunted"] * 2})
        external = pd.DataFrame({"Stunting_Binary": [1] * 9 + [0]})
        rows = self.module.majority_baselines(synthetic, external)
        indexed = {row["cohort"]: row for row in rows}
        self.assertEqual(indexed["synthetic_four_class"]["constant_class"], "Normal")
        self.assertEqual(indexed["synthetic_four_class"]["accuracy"], 0.8)
        self.assertEqual(indexed["external_primary_binary"]["accuracy"], 0.1)
        self.assertEqual(indexed["external_primary_binary"]["sensitivity"], 0.0)
        self.assertEqual(indexed["external_primary_binary"]["specificity"], 1.0)

    def test_standing_assumption_adds_length_only_below_24_months(self):
        frame = pd.DataFrame({"Gender_Encoded": [1, 1], "Age (months)": [12, 24], "Height (cm)": [70.0, 85.0]})
        reference = pd.DataFrame({"Indicator": ["HAZ", "HAZ"], "Gender_Encoded": [1, 1], "Age (months)": [12, 24], "L": [1.0, 1.0], "M": [70.0, 85.0], "S": [0.1, 0.1]})
        original = frame.copy(deep=True)
        base, converted = self.module.measurement_assumptions(frame, reference)
        self.assertEqual(converted["Height (cm)"].tolist(), [70.7, 85.0])
        self.assertAlmostEqual(base.HAZ.iloc[0], 0.0)
        self.assertAlmostEqual(converted.HAZ.iloc[0], 0.1)
        self.assertAlmostEqual(converted.HAZ.iloc[1], 0.0)
        pd.testing.assert_frame_equal(frame, original)

    def test_measurement_sensitivity_keeps_primary_cohort_and_source_labels(self):
        frame = pd.DataFrame({"Source_Row": [1, 2, 3], "Gender_Encoded": [1, 1, 1], "Age (months)": [12, 12, 24], "Height (cm)": [55.5, 70.0, 85.0], "Stunting_Binary": [1, 0, 1], "Valid_HAZ": [True, True, True], "Flag_Source_Label": [False, False, False]})
        reference = pd.DataFrame({"Indicator": ["HAZ", "HAZ"], "Gender_Encoded": [1, 1], "Age (months)": [12, 24], "L": [1.0, 1.0], "M": [70.0, 85.0], "S": [0.1, 0.1]})
        summary, records = self.module.measurement_diagnostics(frame, reference)
        self.assertEqual(summary["fixed_primary_haz_cohort_n"], 2)
        self.assertEqual(summary["source_label_comparison_n"], 2)
        self.assertEqual(summary["as_recorded_source_label_disagreements"], 0)
        self.assertEqual(summary["standing_assumption_source_label_disagreements"], 1)
        self.assertEqual(records.Source_Row.tolist(), [1, 2])
        self.assertEqual(records.Stunting_Binary.tolist(), [1, 0])

    def test_label_conflicts_are_audited_with_the_same_decision_input_group(self):
        frame = pd.DataFrame({"Gender_Encoded": [1, 1, 1], "Age (months)": [12, 12, 13], "Height (cm)": [70.0, 70.0, 72.0], "Stunting": ["Normal", "Stunted", "Normal"], "Source_Row": [1, 2, 3]})
        summary, conflicts = self.module.label_diagnostics(frame, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(summary["conflicting_original_label_groups"], 1)
        self.assertEqual(summary["rows_in_conflicting_original_label_groups"], 2)
        self.assertEqual(summary["corrected_haz_category_disagreements"], 1)
        self.assertEqual(len(conflicts), 1)


if __name__ == "__main__":
    unittest.main()
