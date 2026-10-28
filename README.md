# Interpretable Machine Learning for Child Stunting Assessment

Research code and results for ICDEES 2026 paper **1571345184**. The study covers provincial clustering, four-class stunting classification, and HAZ/WAZ regression.

## Notebooks

| Notebook | Purpose |
|---|---|
| `Stunting_Classification_Analysis.ipynb` | Five-model classification experiment |
| `Stunting_Clustering_Analysis.ipynb` | Original clustering analysis |
| `Stunting_Regression_Analysis.ipynb` | Original regression analysis with the earlier target calculations |
| `Stunting_Clustering_Analysis_Revised.ipynb` | Revised clustering with corrected indicators and provincial aggregation |
| `Stunting_Regression_Analysis_Revised.ipynb` | Revised regression with corrected WHO targets and six models |
| `Stunting_SHAP_Analysis_Revised.ipynb` | Classification SHAP analysis; final regression SHAP is in the revised regression notebook |

The original notebooks are retained because revision analyses and checks still depend on some of their contents or outputs. Rerunning notebooks can train models and overwrite saved results.

## Main directories

- `dataset/`: research data, reference sources, and derived datasets.
- `analysis/revision/`: audit scripts and supplementary evaluations.
- `results/`: saved models, metrics, predictions, and figures.
- `docs/`: dashboard and browser model exports.
- `tests/`: checks for analysis code and outputs.

## Running the dashboard

The dashboard provides five classifiers, six regressors for each HAZ/WAZ target, and three clustering algorithms. Random Forest is the default supervised model. The map uses local GeoJSON; 31 of 34 province profiles have matching polygons.

Run from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000> in your browser.

To install the analysis dependencies:

```sh
python -m pip install -r requirements.txt
```

## Dashboard exports and checks

Browser models in `docs/models/paper_final/` are exported from verified fitted artifacts. `docs/revision_manifest.json` records their sources, hashes, feature order, and evaluation protocols. Re-exporting requires all 17 local source models, including alternatives excluded from Git, and does not retrain them. Regional browser data are exported from the verified final clustering CSVs without rerunning clustering.

```sh
python -m analysis.revision.dashboard_export export
python -m analysis.revision.dashboard_export verify
node analysis/revision/validate_browser.js
python analysis/revision/verify_dashboard.py
```

To check a deployed copy, run the browser verifier with `--url <dashboard-url>`.

Browser checks use Playwright and system Chrome, or a Playwright-managed Chromium installed with `python -m playwright install chromium`. Reports and screenshots are saved locally in `scratch/dashboard_validation/` and excluded from Git. The site still uses external CDNs for its interface libraries, fonts, and icons; the map does not use a tile API or require an API key.

## Research notes

- The paper selects Random Forest for classification and both regression targets. Jeneponto data are used for external evaluation after model freezing; ages 0–23 months form the primary cohort, while age 24 months is reported separately.
- HAZ/WAZ regression approximates WHO reference calculations using anthropometric variables that also serve as model inputs. The results do not establish a diagnosis or predict future growth.
- Clustering summarizes district/city indicators across 34 historical provinces. The groups are descriptive, and provincial means are not population-weighted prevalence estimates.

To run the Python checks:

```sh
python -m unittest discover -s tests -v
```
