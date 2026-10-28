"""Numerical and source-integrity contracts for the revision data pipeline."""
import importlib
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook


class RevisionDataTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('analysis.revision.data')
        except ModuleNotFoundError:
            self.fail('The canonical revision data API has not been implemented')

    def test_restores_only_explicit_decimal_display(self):
        value, restored, error = self.api().parse_weight(datetime(2025, 5, 13), r'd\.m')
        self.assertEqual(value, 13.5)
        self.assertTrue(restored)
        self.assertIsNone(error)
        value, restored, error = self.api().parse_weight(datetime(2025, 5, 13), 'yyyy-mm-dd')
        self.assertTrue(math.isnan(value))
        self.assertFalse(restored)
        self.assertEqual(error, 'unsupported_date_format')

    def test_missing_and_malformed_weight_is_never_estimated(self):
        for value in [None, '', 'unknown', True]:
            parsed, restored, error = self.api().parse_weight(value, 'General')
            self.assertTrue(math.isnan(parsed))
            self.assertIsNotNone(error)
        self.assertEqual(self.api().parse_weight('7.25')[0], 7.25)
        self.assertEqual(self.api().parse_weight(3100)[0], 3100)

    def test_lms_logarithmic_zero_power(self):
        actual = self.api().monthly_zscore([10, 10 * math.e], [0, 0], [10, 10], [1, 1])
        np.testing.assert_allclose(actual, [0, 1], atol=1e-12)

    def test_manifest_rejects_tampered_corrected_reference_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / 'dataset' / 'revision' / 'references'
            references.mkdir(parents=True)
            (references / 'who_monthly_corrected.csv').write_text('Indicator,Gender_Encoded,Age (months),L,M,S\nHAZ,1,24,1,99,.1\n')
            (references / 'manifest.json').write_text(json.dumps({'corrected_sha256': '0' * 64}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.api().prepare_references(root)

    def test_waz_tail_uses_fixed_distance_beyond_three_sd(self):
        # Hand-calculated from L=1, M=10, S=.1: SD2=12 and SD3=13.
        actual = self.api().monthly_zscore([6, 10, 14], 1, 10, .1, restrict_tails=True)
        np.testing.assert_allclose(actual, [-4, 0, 4], atol=1e-12)
        # Nonlinear positive tail: L=.5 gives SD2=12.1, SD3=13.225.
        actual = self.api().monthly_zscore([14.4], .5, 10, .1, restrict_tails=True)
        self.assertAlmostEqual(float(actual[0]), 4.044444444444444, places=10)
        legacy = self.api().monthly_zscore([14.4], .5, 10, .1)
        self.assertAlmostEqual(float(legacy[0]), 4, places=10)

    def test_source_labels_and_invalid_rows_survive_harmonization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.xlsx'
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(['No.', 'Gender', 'Age (Month)', 'Weight', 'Height', 'Height for Age'])
            sheet.append([1, 'M', 12, 10, 75, 'Stunted'])
            sheet.append([2, 'F', 12, 3100, 74, 'Not Stunted'])
            sheet.append([3, 'M', None, 10, 75, 'Not Stunted'])
            sheet.append([4, 'M', 24, datetime(2025, 5, 13), 87.1161, 'Not Stunted'])
            sheet['D5'].number_format = r'd\.m'
            workbook.save(path)
            reference = pd.DataFrame([
                ['HAZ', 1, 12, 1, 75, .04], ['HAZ', 0, 12, 1, 74, .04],
                ['WAZ', 1, 12, 1, 10, .1], ['WAZ', 0, 12, 1, 10, .1],
                ['HAZ', 1, 24, 1, 87.1161, .03507], ['WAZ', 1, 24, 1, 12, .1],
            ], columns=['Indicator', 'Gender_Encoded', 'Age (months)', 'L', 'M', 'S'])
            frame, audit = self.api().parse_jeneponto_workbook(path, reference)
            self.assertEqual(frame['Source_Row'].tolist(), [1, 2, 3, 4])
            self.assertEqual(frame['Stunting_Binary'].tolist(), [1, 0, 0, 0])
            self.assertTrue(frame.loc[0, 'Valid_Classification'])
            self.assertFalse(frame.loc[1, 'Valid_WAZ'])
            self.assertTrue(frame.loc[1, 'Valid_HAZ'])
            self.assertFalse(frame.loc[2, 'Valid_Classification'])
            self.assertTrue(math.isnan(frame.loc[2, 'HAZ']))
            self.assertTrue(frame.loc[3, 'In_Age24_Sensitivity'])
            self.assertFalse(frame.loc[3, 'In_Primary_External'])
            self.assertEqual(frame.loc[3, 'Weight (kg)'], 13.5)
            self.assertEqual(audit['source_label_mismatches'], 1)
            self.assertEqual(audit.get('primary_source_label_mismatches'), 1)
            self.assertEqual(audit.get('primary_source_label_comparison_denominator'), 2)

    def test_corrected_reference_preserves_original_classification_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'dataset' / 'Stunting Wasting Dataset (Synthetic)'
            refs = original / 'z-scores who'
            refs.mkdir(parents=True)
            pd.DataFrame([
                ['Laki-laki', 24, 87.1161, 12.1515, 'Normal', 'Normal'],
                ['Perempuan', 24, 85.7153, 11.4775, 'Stunted', 'Normal'],
            ], columns=['Jenis Kelamin', 'Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)', 'Stunting', 'Wasting']).to_csv(original / 'stunting_wasting_dataset.csv', index=False)
            for sex, median, spread, weight in [('boys', 87.4661, .03493, 12.1515), ('girls', 86.0653, .03749, 11.4775)]:
                pd.DataFrame([[24, 1, median, spread]], columns=['Month', 'L', 'M', 'S']).to_csv(refs / f'lhfa_{sex}_0-to-5-years_zscores.csv', sep=';', decimal=',', index=False)
                pd.DataFrame([[24, 1, weight, .1]], columns=['Month', 'L', 'M', 'S']).to_csv(refs / f'wfa_{sex}_0-to-5-years_zscores.csv', sep=';', decimal=',', index=False)
            target = root / 'dataset' / 'revision' / 'references'
            target.mkdir(parents=True)
            pd.DataFrame([
                ['HAZ', 1, 24, 1, 87.1161, .03507], ['HAZ', 0, 24, 1, 85.7153, .03764],
                ['WAZ', 1, 24, 1, 12.1515, .1], ['WAZ', 0, 24, 1, 11.4775, .1],
            ], columns=['Indicator', 'Gender_Encoded', 'Age (months)', 'L', 'M', 'S']).to_csv(target / 'who_monthly_corrected.csv', index=False)
            legacy, _ = self.api().load_synthetic('legacy', root)
            corrected, _ = self.api().load_synthetic('corrected', root)
            self.assertEqual(legacy['Stunting'].tolist(), ['Normal', 'Stunted'])
            self.assertEqual(corrected['Stunting'].tolist(), ['Normal', 'Stunted'])
            self.assertEqual(corrected['Gender_Encoded'].tolist(), [1, 0])
            np.testing.assert_allclose(corrected['HAZ'], [0, 0], atol=1e-12)
            self.assertTrue((legacy['HAZ'] < 0).all())


if __name__ == '__main__':
    unittest.main()
