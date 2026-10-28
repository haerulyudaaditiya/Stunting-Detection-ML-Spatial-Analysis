# Corrected clustering — two descriptive groups

Source of calculations: [Stunting_Clustering_Analysis_Revised.ipynb](../../Stunting_Clustering_Analysis_Revised.ipynb). Open from the repository root and run all cells. The notebook includes actual executed tables and figures.

This is the clustering correction approved during the manual paper revision on 18 October 2026. The original notebook and workbook are unchanged. No manuscript or live dashboard was edited in this step.

## Method

- Preserve all 514 district/city observations and 34 historical provinces.
- Use stunting, maternal KEK, complete basic immunization, and BBLR rates. The immunization source header refers to ages 12–23 months.
- Exclude the stunted-child count previously misidentified as `Total_Births`, including its use as a common weight.
- Calculate each province's mean from its available district observations, recording indicator-specific coverage. These descriptive means are not population-weighted provincial prevalence estimates.
- Convert KEK/immunization/BBLR source fractions to percent, then standardize the four provincial indicators.
- Select the common k by maximum K-Means silhouette over 2–8. For this source and configuration, k=2 is selected. This does not prove that two groups are optimal for all algorithms or populations.
- Retain the original constructors: KMeans(random_state=42), GaussianMixture(random_state=42), and AgglomerativeClustering(linkage='ward'), setting the group count to the selected k. Defaults are resolved by the recorded scikit-learn version; KMeans n_init is `auto`, GMM n_init=1 and max_iter=100.
- Use descriptive Group 1 and Group 2, ordered separately for each algorithm by group mean stunting. They are not validated clinical-risk categories.

## Results

| Algorithm | Silhouette | DBI | CHI |
|---|---:|---:|---:|
| K-Means | 0.2800 | 1.2066 | 17.0252 |
| GMM | 0.3464 | 1.1307 | 18.9593 |
| Hierarchical (Ward) | 0.3852 | 0.9358 | 17.2035 |

Hierarchical has the highest silhouette and lowest DBI. GMM has the highest CHI. Absolute separation remains limited; these are exploratory descriptions, not clinical validation.

The earlier `results/revision_1/clustering/` uses KMeans n_init=50 and GMM n_init=20/max_iter=1000. Do not mix its KMeans metrics, group memberships, or CHI ranking with this original-configuration correction. The Ward result is the same because its configuration did not change.

## Files

- `district_indicators.csv`: all source rows and four verified rate indicators, in percent.
- `coverage.csv`: observed/total districts per province and indicator.
- `province_profiles.csv`: 34 province means and separate group assignments for all three algorithms.
- `k_selection.csv`: the seven K-Means fits used to select k.
- `metrics.csv`: three algorithm metric rows.
- `group_profiles.csv`: six descriptive group profiles and province memberships. Group means weight each province profile equally.
- `algorithm_agreement.csv`: pairwise ARI between the three partitions.
- `k_selection.png` / `.pdf`: corrected selection figure using the original 10 × 5 inch layout, red WSS line, blue silhouette line, and matching fonts, markers, grid, and title.
- `regional_profiles.png` / `.pdf`: corrected three-panel scatterplot using the original 18 × 5 inch layout, Set1 palette, marker size 100, and matching fonts, grid, panel titles, and framed legends. Labels identify district means and descriptive groups. The horizontal and vertical axes alone do not display the full four-dimensional separation.
- `summary.json`: source hashes, parameters, versions, group counts, metric winners, and limitations.
- `verification.json`: independent rechecks of source values, profile aggregation, saved metrics, output completeness, and source integrity.

Use these outputs to revise the paper manually. Relevant passages, tables and figures in the old manuscript must be updated consistently before submission. In particular, remove the old live-birth feature description, three clinical-risk labels, and claims that one algorithm leads every metric.

Both PNG figures retain 500 DPI, as in the original notebook. Adjust their displayed size when inserting them into Word. Styling changes do not change the corrected numerical results.
