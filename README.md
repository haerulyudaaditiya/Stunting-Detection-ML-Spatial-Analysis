# Interpretable Machine Learning for Child Stunting Assessment

Research code and supporting materials for ICDEES 2026, paper **1571345184**. The study covers provincial clustering, HAZ/WAZ regression, and four-class stunting classification. The submitted conference documents are kept locally under `paper/rev_2/` and excluded from Git.

The repository keeps the original experiments alongside the analyses added during revision. All six notebooks retain their original filenames and locations because supporting code reads some of them directly. Their roles are listed below.

## Notebooks

| Notebook | Role | Results |
|---|---|---|
| [Stunting_Classification_Analysis.ipynb](Stunting_Classification_Analysis.ipynb) | Primary five-model classification experiment and saved Random Forest classifier. Its XGBoost SHAP section belongs to the original analysis. | [results/classification/](results/classification/) |
| [Stunting_Clustering_Analysis.ipynb](Stunting_Clustering_Analysis.ipynb) | Original regional analysis, preserved as research history. The revised notebook below supplies the final clustering results. | [results/clustering/](results/clustering/) |
| [Stunting_Regression_Analysis.ipynb](Stunting_Regression_Analysis.ipynb) | Original regression experiment using the earlier WHO target calculations. | [results/regression/](results/regression/) |
| [Stunting_Clustering_Analysis_Revised.ipynb](Stunting_Clustering_Analysis_Revised.ipynb) | Final clustering analysis: corrected indicators, province aggregation, and two descriptive groups. | [results/clustering_revised/](results/clustering_revised/) |
| [Stunting_Regression_Analysis_Revised.ipynb](Stunting_Regression_Analysis_Revised.ipynb) | Final regression analysis: corrected WHO targets, six-model CV, holdout models, and regression SHAP. | [results/regression_corrected/](results/regression_corrected/) |
| [Stunting_SHAP_Analysis_Revised.ipynb](Stunting_SHAP_Analysis_Revised.ipynb) | Random Forest explanations. The classification section supplies the final classification SHAP; its regression section explains the earlier regression models. | [results/shap_revised/](results/shap_revised/) |

The final paper selects Random Forest for classification and both regression targets using primary synthetic-data CV. Grouped CV provides supplementary checks. Jeneponto is used for external evaluation after the selected models are frozen. The final regression SHAP comes from the corrected regression notebook.

## Repository contents

| Location | Contents |
|---|---|
| [dataset/](dataset/) | Synthetic data, regional indicators, geographic boundaries, and WHO/Jeneponto source and derived data |
| [analysis/revision/](analysis/revision/) | Data preparation, reference audits, supplementary evaluations, and supporting utilities |
| [results/](results/) | Saved models, metrics, fold assignments, predictions, plots, and verification records from successive research stages |
| [tests/](tests/) | Checks for data processing, evaluation, revised clustering, and browser prediction code |
| [docs/](docs/) | Dashboard source, figures, and browser model exports |
| `paper/rev_1/` (local) | Original submission, conference instructions, and supporting documents |
| `paper/rev_2/Revisi/` (local) | Revision package, camera-ready manuscript, reports, and declarations |
| `paper/rev_2/Submit Final/` (local) | Final submission materials, presentation, and recording |
| `scratch/` (local) | Working files, references, earlier manuscripts, diagrams, and debugging/export scripts |

Keep the saved outputs when working with these notebooks. Some revision checks compare against earlier CV records, models, and notebook contents. The [browser export manifest](docs/revision_manifest.json) records the dashboard's exported configurations separately from the final paper's model selection.

## What belongs in Git

The repository includes the analysis code, six notebooks, tests, source/reference data, CV records, metrics, final figures, selected frozen models, and active dashboard assets. The earlier selected models are retained where needed to reproduce the supporting analyses.

[.gitignore](.gitignore) keeps the conference submission package, working snapshots, earlier figures, additional fitted models, and detailed prediction/SHAP arrays local. These files remain on the author's machine. A fresh clone contains the retained summaries and required inputs; rerun the corresponding notebooks or analysis commands to regenerate omitted outputs.

## Working with the notebooks

Use the repository root as the working directory. Open notebooks in Jupyter or another notebook editor with the analysis environment selected. [requirements.txt](requirements.txt) contains the recorded package versions for the original and revised analyses. The project has recorded analysis environments using Python 3.12.3 and Python 3.14.x.

```sh
python -m pip install -r requirements.txt
```

Read the notebook's role and dependencies before rerunning it. Existing outputs already contain the completed analyses; rerunning cells can train models and replace their result files.

To serve the dashboard locally:

```sh
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>.

## Supporting revision analyses

| Analysis | Code | Saved evidence |
|---|---|---|
| Additional Random Forest split checks | [reviewer1_splits.py](analysis/revision/reviewer1_splits.py) | [results/reviewer1_question1/](results/reviewer1_question1/) |
| Supplementary grouped CV for all five classifiers | Retained classification CV records | [cv_metrics.csv](results/revision_1/supervised/cv_metrics.csv) |
| WHO reference and target audit | [reviewer2_reference_audit.py](analysis/revision/reviewer2_reference_audit.py) | [results/reviewer2_reference_audit/](results/reviewer2_reference_audit/) |
| Jeneponto evaluation of frozen models | [reviewer2_external_validation.py](analysis/revision/reviewer2_external_validation.py) | [results/reviewer2_validation/](results/reviewer2_validation/) |

The external evaluation uses the final [model freeze manifest](results/regression_corrected/model_freeze_manifest.json):

```sh
python -m analysis.revision.reviewer2_external_validation --manifest results/regression_corrected/model_freeze_manifest.json
```

This verifies the frozen artifacts and evaluates them without fitting on Jeneponto. Its default output directory is `results/reviewer2_validation/`; `--output` selects a separate destination.

Keep the original notebooks and saved outputs in place:

- Revised clustering records the original clustering notebook's checksum.
- The WHO reference audit reads the original regression notebook directly.
- Corrected regression compares CV results with records in `results/revision_1/supervised/cv/` and uses saved classification artifacts.
- The revised SHAP notebook loads models from `results/regression/` and `results/classification/`. Final regression SHAP is in [results/regression_corrected/plots/](results/regression_corrected/plots/), including `shap_summary_komparasi.png`.

## Earlier revision pipeline

[results/revision_1/](results/revision_1/) preserves an earlier revision stage, including CV records and supplementary analyses used by later checks.

The `analysis.revision.supervised` command reproduces that stage's grouped-CV selection: Extra Trees for classification, Random Forest for HAZ, and CatBoost for WAZ. It also writes dashboard exports. The `analysis.revision.clustering` command uses increased initialization counts from that stage. The notebook table above identifies the final paper's analyses.

The submitted documents were prepared separately and are stored under `paper/rev_2/`.

## Research scope

The supervised experiments use 100,000 synthetic records aged 0-24 months. Classification retains the source's four classes; HAZ/WAZ regression approximates monthly WHO calculations from anthropometric inputs. External evaluation uses Jeneponto 2024, with ages 0-23 months as the primary cohort and age 24 months reported separately. Regional clustering describes available district indicators aggregated within 34 historical provinces.

The results concern anthropometry at measurement. Their applicability is limited by synthetic training data, monthly reference calculations, and external observations from one regency.

Classification uses sex, age, height, and weight, with Female encoded as 0 and Male as 1. HAZ/WAZ are excluded from classification inputs. Regression targets are calculated from variables also used as model inputs. Grouped CV separates identical decision-input combinations across folds within the synthetic data.

The corrected monthly WHO reference uses length-for-age below 24 months and height-for-age from 24 months. WAZ applies the WHO extension beyond +/-3 SD. Source workbooks and recorded checksums are in [dataset/revision/raw/](dataset/revision/raw/) and [dataset/revision/references/](dataset/revision/references/).

The external source is *Dataset Stunting and Nutritional Status of Toddler from Jeneponto Regency, South Sulawesi, Indonesia*, Mendeley Data version 4, DOI [10.17632/wzwpc9j5bx.4](https://data.mendeley.com/datasets/wzwpc9j5bx/4). The 2024 workbook contains 31,740 source observations. The [harmonized data](dataset/revision/derived/) retain source rows, recovered measurements, source binary labels, and task-specific eligibility flags.

External classification maps the four-class argmax prediction to stunted/severely stunted versus normal/tall and sums the two stunting-class probabilities for AUC. HAZ/WAZ eligibility uses their respective reference bounds and valid inputs. Exact age in days, measurement position, and unique child identifiers are unavailable.

Corrected regional clustering uses unweighted means of available district rates. The count previously described as births is excluded. K-Means silhouette over k=2 through 8 selects two groups for the three algorithms. Group labels are descriptive; provincial means summarize district rates rather than population-weighted prevalence.

## Checking retained evidence

Analysis directories contain metrics and verification records. Freeze manifests identify fitted models and their checksums. When recalculating AUC from prediction CSV files, use `pd.read_csv(path, float_precision="round_trip")` to preserve probability ties.

For code checks:

```sh
python -m unittest discover -s tests -v
```

The revised clustering integration test fits clustering models in a temporary directory. Completed notebook training and visual inspection of submitted documents are separate verification steps.
