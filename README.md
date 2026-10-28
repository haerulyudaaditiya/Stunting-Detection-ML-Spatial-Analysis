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

Run from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000> in your browser.

To install the analysis dependencies:

```sh
python -m pip install -r requirements.txt
```

## Research notes

- The paper selects Random Forest for classification and both regression targets. Jeneponto data are used for external evaluation after model freezing; ages 0–23 months form the primary cohort, while age 24 months is reported separately.
- HAZ/WAZ regression approximates WHO reference calculations using anthropometric variables that also serve as model inputs. The results do not establish a diagnosis or predict future growth.
- Clustering summarizes district/city indicators across 34 historical provinces. The groups are descriptive, and provincial means are not population-weighted prevalence estimates.

To run the Python checks:

```sh
python -m unittest discover -s tests -v
```
