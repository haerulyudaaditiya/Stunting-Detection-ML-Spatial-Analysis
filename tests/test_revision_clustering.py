"""Regression checks for the revised regional estimand and dashboard export."""

import importlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class RevisionClusteringTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("analysis.revision.clustering")
        except ModuleNotFoundError:
            self.module = None
        self.assertIsNotNone(self.module, "The revised clustering module is required")

    @staticmethod
    def districts():
        return pd.DataFrame(
            {
                "Province": ["A", "A", "B", "B"],
                "District": ["a1", "a2", "b1", "b2"],
                "Stunting_2019": [10.0, 30.0, 20.0, 40.0],
                "KEK": [10.0, np.nan, 20.0, 40.0],
                "Immunization": [40.0, 60.0, 70.0, 90.0],
                "BBLR": [2.0, 4.0, 6.0, 8.0],
                "Misidentified_Count": [1.0, 1_000_000.0, 1.0, 1.0],
            }
        )

    def test_available_case_means_ignore_count_and_record_actual_coverage(self):
        profiles, coverage = self.module.aggregate_profiles(self.districts())
        a = profiles.set_index("Province").loc["A"]
        self.assertEqual(a["Stunting_2019"], 20.0)
        self.assertEqual(a["KEK"], 10.0)
        self.assertEqual(a["Immunization"], 50.0)
        self.assertEqual(a["BBLR"], 3.0)
        self.assertNotIn("Misidentified_Count", profiles.columns)
        observed = coverage.set_index(["Province", "indicator"]).loc[("A", "KEK")]
        self.assertEqual(observed["observed_districts"], 1)
        self.assertEqual(observed["total_districts"], 2)
        self.assertEqual(observed["coverage_pct"], 50.0)

    def test_complete_case_sensitivity_uses_one_shared_district_subset(self):
        profiles, _ = self.module.aggregate_profiles(self.districts(), complete_case=True)
        a = profiles.set_index("Province").loc["A"]
        self.assertEqual(a["Stunting_2019"], 10.0)
        self.assertEqual(a["Immunization"], 40.0)
        self.assertEqual(a["BBLR"], 2.0)

    def test_empty_province_indicator_is_rejected_instead_of_imputed(self):
        frame = self.districts()
        frame.loc[frame.Province.eq("A"), "KEK"] = np.nan
        with self.assertRaisesRegex(ValueError, "A.*KEK|KEK.*A"):
            self.module.aggregate_profiles(frame)

    def test_excel_reader_uses_rate_columns_and_converts_only_proportions(self):
        raw = pd.DataFrame(np.full((2, 40), np.nan), columns=[f"col{i}" for i in range(40)], dtype=object)
        raw.iloc[:, 1] = pd.Series(["Aceh", "Aceh"], dtype=object)
        raw.iloc[:, 2] = pd.Series(["D1", "D1"], dtype=object)
        raw.iloc[:, 4] = [10000.0, 900000.0]
        raw.iloc[:, 35] = [28.8, 37.4]
        raw.iloc[:, 37] = [0.136, 0.108]
        raw.iloc[:, 38] = [0.54, 0.56]
        raw.iloc[:, 39] = [0.04, 0.06]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.xlsx"
            raw.to_excel(path, sheet_name="Data", startrow=4, index=False)
            frame = self.module.read_district_indicators(path)
        self.assertEqual(frame.columns.tolist(), ["Province", "District", "Stunting_2019", "KEK", "Immunization", "BBLR"])
        self.assertEqual(frame.iloc[0].Stunting_2019, 28.8)
        self.assertAlmostEqual(frame.iloc[0].KEK, 13.6)
        self.assertEqual(frame.iloc[0].Immunization, 54.0)
        self.assertEqual(frame.iloc[0].BBLR, 4.0)
        self.assertEqual(len(frame), 2, "Regency/city names may repeat; source rows must be retained")

    def test_k_selection_uses_silhouette_and_lower_k_breaks_exact_ties(self):
        frame = pd.DataFrame({"k": [4, 3, 2], "silhouette": [0.51, 0.6, 0.6]})
        self.assertEqual(self.module.select_k(frame), 2)

    def test_browser_export_retains_fraction_units_without_fabricated_births(self):
        profiles, coverage = self.module.aggregate_profiles(self.districts())
        assignments = {"K-Means": [0, 1], "GMM": [1, 0], "Ward": [0, 1]}
        records = self.module.browser_records(profiles, coverage, assignments)
        self.assertEqual(records[0]["Stunting_2019"], 20.0)
        self.assertEqual(records[0]["KEK_Ibu_Hamil"], 0.1)
        self.assertEqual(records[0]["Imunisasi_Dasar_Lengkap"], 0.5)
        self.assertEqual(records[0]["BBLR"], 0.03)
        self.assertNotIn("Jumlah_Lahir", records[0])
        self.assertEqual(records[0]["coverage"]["KEK"]["coverage_pct"], 50.0)
        self.assertEqual(records[0]["Klaster_GMM"], 1)
        self.assertIn("district", records[0]["summary_method"])

    def test_neutral_groups_are_stably_ordered_by_stunting_profile(self):
        profiles = pd.DataFrame({"Stunting_2019": [30.0, 10.0, 40.0, 20.0]})
        labels = self.module.neutral_labels(profiles, np.array([5, 3, 5, 3]))
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])

    def test_browser_province_alias_matches_existing_map_lookup(self):
        profiles, coverage = self.module.aggregate_profiles(self.districts())
        profiles.loc[0, "Province"] = "DI Yogyakarta"
        coverage.loc[coverage.Province.eq("A"), "Province"] = "DI Yogyakarta"
        records = self.module.browser_records(profiles, coverage, {"Ward": [0, 1]})
        self.assertEqual(records[0]["Provinsi"], "DI Yogyakarta")
        self.assertEqual(records[0]["Provinsi_Map"], "DIY")

    def test_real_workbook_pipeline_preserves_all_provinces_and_emits_auditable_outputs(self):
        repository = Path(__file__).resolve().parents[1]
        source_name = "Data-Status-Gizi-2018-2020-Kabupaten-Kota.xlsx"
        source_bytes = (repository / "dataset" / source_name).read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "dataset").mkdir()
            (root / "dataset" / source_name).write_bytes(source_bytes)
            summary = self.module.run_clustering(root)
            output = root / "results" / "revision_1" / "clustering"
            self.assertEqual(summary["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(summary["districts"], 514)
            self.assertEqual(summary["provinces"], 34)
            self.assertEqual(summary["sensitivity_complete_districts"], 430)
            self.assertEqual(summary["sensitivity_provinces"], 34)
            self.assertEqual(summary["province_imputation"], "none")
            profiles = pd.read_csv(output / "profiles.csv")
            self.assertEqual(len(profiles), 34)
            self.assertNotIn("Total_Births", profiles.columns)
            aceh = profiles.set_index("Province").loc["Aceh"]
            self.assertAlmostEqual(aceh.Stunting_2019, 33.504347826086956)
            self.assertAlmostEqual(aceh.KEK, 12.306009138472334)
            metrics = pd.read_csv(output / "metrics.csv")
            primary = metrics[metrics.scenario.eq("primary")]
            self.assertEqual(set(primary.algorithm), {"K-Means", "GMM", "Ward"})
            selection = pd.read_csv(output / "k_selection.csv")
            self.assertEqual(selection.k.tolist(), list(range(2, 9)))
            self.assertEqual(summary["selected_k"], int(selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k))
            self.assertEqual(len(pd.read_csv(output / "sensitivity_membership.csv")), 34)
            for filename in ["summary.json", "coverage.csv", "sensitivity_profiles.csv", "sensitivity_comparison.csv", "cluster_profiles.csv", "province_data.js", "cluster_profiles.js", "k_selection.png", "regional_profiles.png"]:
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
