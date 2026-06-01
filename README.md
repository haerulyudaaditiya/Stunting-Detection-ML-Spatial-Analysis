# Stunting Detection: Machine Learning and Spatial Analysis

## Overview

This repository contains a comprehensive analysis framework for childhood stunting detection using machine learning and spatial analysis techniques. The project develops predictive models and an interactive dashboard for assessing stunting prevalence across Indonesian provinces.

## Research Objectives

1. Develop classification models to predict stunting risk based on anthropometric measurements
2. Identify regional clustering patterns of stunting prevalence across Indonesia
3. Establish regression models for continuous Z-score prediction (HAZ and WAZ)
4. Create an interactive spatial visualization system for public health decision support

## Methodology

### Machine Learning Approaches

- **Classification**: XGBoost for binary stunting status prediction using physical features
- **Clustering**: K-Means and hierarchical clustering for regional health profile identification
- **Regression**: LightGBM for continuous Z-score estimation (Height-for-Age and Weight-for-Age)

### Data Basis

- Synthetic stunting and wasting dataset with 100,000 child records
- WHO anthropometric reference standards (LMS parameters) for z-score calculation
- Indonesia provincial administrative boundaries (GeoJSON)
- Regional health indicators at district level

## Project Structure

```
Stunting-Detection-ML-Spatial-Analysis/
├── Stunting_Classification_Analysis.ipynb    # Classification model development
├── Stunting_Clustering_Analysis.ipynb         # Regional clustering analysis
├── Stunting_Regression_Analysis.ipynb         # Continuous z-score regression
├── dataset/                                   # Data sources
│   ├── Stunting Wasting Dataset (Synthetic)/
│   └── z-scores who/
├── results/                                   # Output artifacts
│   ├── classification/                       # Classification model & visual outputs
│   │   ├── stunting_xgb_classifier.joblib
│   │   ├── stunting_target_encoder.joblib
│   │   ├── stunting_gender_encoder.joblib
│   │   └── [plots: confusion matrix, distributions, SHAP summary/waterfall]
│   ├── clustering/                           # Clustering maps & visual outputs
│   │   ├── indonesia_stunting_map.html
│   │   └── [plots: evaluations, correlation, scatter, dendrogram]
│   ├── regression/                           # Regression models & visual outputs
│   │   ├── stunting_lgbm_reg_haz.joblib
│   │   ├── stunting_lgbm_reg_waz.joblib
│   │   └── [plots: residual analysis, jointplot, SHAP summary]
│   └── dashboard/                            # Interactive web dashboard & JS files
│       └── stunting_dashboard.html
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
└── .gitignore
```

## Installation

### Requirements

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/haerulyudaaditiya/Stunting-Detection-ML-Spatial-Analysis.git
cd Stunting-Detection-ML-Spatial-Analysis
```

2. Create virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Analyses

Execute Jupyter notebooks in the following order:

```bash
jupyter notebook Stunting_Classification_Analysis.ipynb
jupyter notebook Stunting_Clustering_Analysis.ipynb
jupyter notebook Stunting_Regression_Analysis.ipynb
```

### Accessing the Dashboard

After model generation, open the interactive dashboard:

```bash
open results/dashboard/stunting_dashboard.html
# or serve via local web server
python -m http.server 8000
# Navigate to http://localhost:8000/results/dashboard/stunting_dashboard.html
```

## Key Results

### Classification Performance

- Model: XGBoost
- Primary metrics: Accuracy, F1-score, confusion matrix analysis

### Regional Clustering

- Optimal cluster analysis using silhouette score and Davies-Bouldin index
- Hierarchical clustering of provincial health profiles
- Regional pattern visualization

### Regression Accuracy

- HAZ (Height-for-Age Z-score) prediction model
- WAZ (Weight-for-Age Z-score) prediction model
- Residual analysis and model diagnostics

## Dashboard Features

The interactive dashboard provides:

- Geospatial mapping of stunting prevalence by province
- Individual child stunting risk assessment via ML model inference
- Regional comparative analysis tools
- Multi-language interface (Indonesian/English)
- WCAG AA/AAA accessibility compliance

## Dataset Information

- **Source**: Synthetic dataset for research demonstration
- **Records**: 100,000 child observations
- **Features**: Age, gender, anthropometric measurements
- **Reference**: WHO growth standards (LMS parameters)

## Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: matplotlib, seaborn, Folium, Chart.js
- **Frontend**: HTML5, CSS3, JavaScript (Leaflet.js)
- **Model Interpretation**: SHAP

## Notes

This project is developed for research and educational purposes. The analysis framework uses synthetic data and WHO anthropometric standards to demonstrate stunting detection methodologies across Indonesian provinces.
