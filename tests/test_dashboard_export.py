import importlib
import importlib.util
import tempfile
import unittest
import copy
from unittest.mock import patch
from pathlib import Path


class DashboardExportTests(unittest.TestCase):
    def test_invalid_manifest_semantics_are_rejected(self):
        module = self.exporter()
        original = module.read_json
        valid = original(module.ROOT / 'docs/revision_manifest.json')
        changes = {
            'classes': lambda m: m['classes'].reverse(),
            'gender': lambda m: m['gender_encoding'].update(Male=0),
            'availability': lambda m: m['available'].update(haz=['unknown']),
            'default': lambda m: m['selected']['classification'].update(code='unknown'),
            'path': lambda m: m['artifacts']['classification_rf'].update(browser_path='models/old/rf.json'),
            'artifact_class': lambda m: m['artifacts']['classification_rf']['classes'].reverse(),
            'binding': lambda m: m['artifacts']['classification_rf'].update(source='wrong.joblib'),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                data = copy.deepcopy(valid)
                change(data)
                def read(path):
                    return data if path == module.ROOT / 'docs/revision_manifest.json' else original(path)
                with patch.object(module, 'read_json', side_effect=read), self.assertRaises(ValueError):
                    module.verify()

    def exporter(self):
        name = 'analysis.revision.dashboard_export'
        self.assertIsNotNone(importlib.util.find_spec(name), 'The final-model exporter is missing')
        return importlib.import_module(name)

    def test_catalog_includes_all_seventeen_final_artifacts(self):
        module = self.exporter()
        catalog = module.source_catalog()
        self.assertEqual(len(catalog), 17)
        self.assertEqual({item['code'] for item in catalog if item['task'] == 'classification'},
                         {'rf', 'et', 'xgb', 'lgbm', 'cb'})
        for task in ('HAZ', 'WAZ'):
            self.assertEqual({item['code'] for item in catalog if item['task'] == task},
                             {'linear', 'ridge', 'rf', 'xgb', 'lgbm', 'cb'})
        xgb = next(item for item in catalog if item['task'] == 'classification' and item['code'] == 'xgb')
        self.assertTrue(xgb['source'].endswith('stunting_xgboost_classifier.joblib'))
        self.assertTrue(all(len(item['source_sha256']) == 64 for item in catalog))

    def test_changed_source_is_rejected_before_loading(self):
        module = self.exporter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.joblib').write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError, 'hash|checksum'):
                module.verify_hash(root, 'model.joblib', '0' * 64)

    def test_missing_source_is_rejected(self):
        module = self.exporter()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                module.verify_hash(Path(directory), 'missing.joblib', '0' * 64)

    def test_source_paths_cannot_escape_repository(self):
        module = self.exporter()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'outside|escape'):
                module.verify_hash(Path(directory), '../model.joblib', '0' * 64)

    def test_parity_samples_preserve_the_source_gender_mapping(self):
        module = self.exporter()
        frame = module.parity_features(module.ROOT)
        self.assertEqual(set(frame.Gender_Encoded), {0, 1})
        self.assertTrue(frame['Age (months)'].between(0, 24).all())

    def test_regional_exports_use_final_group_assignments_and_profile_sizes(self):
        module = self.exporter()
        self.assertTrue(callable(getattr(module, 'regional_payload', None)), 'The final regional exporter is missing')
        records, clusters, reference, metadata = module.regional_payload()
        provinces = {row['Provinsi']: row for row in records}
        self.assertEqual(len(provinces), 34)
        self.assertEqual(provinces['Banten']['Klaster_KMeans'], 1)
        self.assertEqual(provinces['Sumatera Utara']['Klaster_KMeans'], 1)
        self.assertEqual([row['province_count'] for row in clusters['Klaster_KMeans']], [15, 19])
        self.assertAlmostEqual(reference['Stunting_2019'], 27.87973095314663, places=6)
        self.assertEqual(metadata['map_geometry']['matched_provinces'], 31)


if __name__ == '__main__':
    unittest.main()
