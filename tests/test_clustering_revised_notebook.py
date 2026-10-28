import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import nbformat
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class CorrectedClusteringNotebookTests(unittest.TestCase):
    def test_revised_notebook_corrects_inputs_without_changing_original_initialization(self):
        root = Path(__file__).resolve().parents[1]
        notebook = root/'Stunting_Clustering_Analysis_Revised.ipynb'
        self.assertTrue(notebook.exists(), 'The reviewed two-group notebook has not been created')
        cells = nbformat.read(notebook, as_version=4).cells
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder)
            (temporary/'dataset').mkdir()
            shutil.copyfile(root/'dataset/Data-Status-Gizi-2018-2020-Kabupaten-Kota.xlsx',
                            temporary/'dataset/Data-Status-Gizi-2018-2020-Kabupaten-Kota.xlsx')
            shutil.copyfile(root/'Stunting_Clustering_Analysis.ipynb',
                            temporary/'Stunting_Clustering_Analysis.ipynb')
            old_cwd = Path.cwd()
            namespace = {'__name__': '__main__'}
            try:
                os.chdir(temporary)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    for cell in cells:
                        if cell.cell_type == 'code':
                            exec(compile(cell.source, str(notebook), 'exec'), namespace)
            finally:
                os.chdir(old_cwd)
            out = temporary/'results/clustering_revised'
            summary = json.loads((out/'summary.json').read_text())
            profiles = pd.read_csv(out/'province_profiles.csv')
            metrics = pd.read_csv(out/'metrics.csv').set_index('Algorithm')
            self.assertEqual(summary['districts'], 514)
            self.assertEqual(summary['provinces'], 34)
            self.assertEqual(summary['selected_k'], 2)
            self.assertNotIn('Total_Births', profiles.columns)
            aceh = profiles.set_index('Province').loc['Aceh']
            self.assertAlmostEqual(aceh.Stunting_2019, 33.504347826086956)
            self.assertAlmostEqual(aceh.KEK, 12.306009138472334)
            self.assertEqual(summary['models']['K-Means']['parameters']['n_init'], KMeans().get_params()['n_init'])
            self.assertEqual(summary['models']['GMM']['parameters']['n_init'], GaussianMixture().get_params()['n_init'])
            self.assertAlmostEqual(metrics.loc['Hierarchical', 'Silhouette'], 0.3851651216847638)
            self.assertEqual(metrics.CHI.idxmax(), 'GMM')
            for col in ['Group_KMeans', 'Group_GMM', 'Group_Hierarchical']:
                self.assertEqual(set(profiles[col]), {1, 2})
            groups = pd.read_csv(out/'group_profiles.csv')
            self.assertEqual(len(groups), 6)
            for _, rows in groups.groupby('Algorithm'):
                self.assertEqual(rows.Provinces.sum(), 34)
                self.assertEqual(set(rows.Group), {'Group 1', 'Group 2'})
            for name in ['k_selection.png', 'k_selection.pdf', 'regional_profiles.png', 'regional_profiles.pdf']:
                self.assertGreater((out/name).stat().st_size, 1000)
