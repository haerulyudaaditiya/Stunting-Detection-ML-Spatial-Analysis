"""Revised, reproducible clustering of province summaries of district indicators.

These summaries estimate average observed district indicators, not population
prevalence. No births/count variable, invented denominator, or provincial
imputation is used. Run with ``python -m analysis.revision.clustering``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.patches import Polygon
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FEATURES = ["Stunting_2019", "KEK", "Immunization", "BBLR"]
ALGORITHMS = ["K-Means", "GMM", "Ward"]
JS_CLUSTERS = {"K-Means": "Klaster_KMeans", "GMM": "Klaster_GMM", "Ward": "Klaster_HC"}
SOURCE_NAME = "Data-Status-Gizi-2018-2020-Kabupaten-Kota.xlsx"
SUMMARY_METHOD = "available-case unweighted mean of district indicators within each historical province"
COLORS = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb", "#332288"]
PROVINCE_ALIASES = {
    "DIY": "DI Yogyakarta", "Dki Jakarta": "DKI Jakarta",
    "DAERAH ISTIMEWA YOGYAKARTA": "DI Yogyakarta", "DKI JAKARTA": "DKI Jakarta",
    "DI. ACEH": "Aceh", "IRIAN JAYA TIMUR": "Papua", "IRIAN JAYA TENGAH": "Papua",
    "IRIAN JAYA BARAT": "Papua Barat", "PROBANTEN": "Banten",
    "NUSATENGGARA BARAT": "Nusa Tenggara Barat",
}


def read_district_indicators(path: Path | str) -> pd.DataFrame:
    """Select verified rate columns; the stunted-count column is never selected."""
    raw = pd.read_excel(path, sheet_name="Data", skiprows=4)
    if raw.shape[1] < 40:
        raise ValueError("The source workbook must contain the verified 40-column structure")
    frame = raw.iloc[:, [1, 2, 35, 37, 38, 39]].copy()
    frame.columns = ["Province", "District", *FEATURES]
    frame = frame.dropna(subset=["Province", "District"])
    for name in ["Province", "District"]:
        frame[name] = frame[name].astype(str).str.strip()
        if frame[name].eq("").any():
            raise ValueError(f"Empty {name} identifier in the source workbook")
    # Regency/city pairs can share a displayed name (e.g. Bogor). Preserve
    # workbook rows; district labels are not unique administrative identifiers.
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        upper = 100 if feature == "Stunting_2019" else 1
        invalid = frame[feature].notna() & ~frame[feature].between(0, upper)
        if invalid.any():
            raise ValueError(f"{feature} contains values outside the source unit range 0–{upper}")
        if feature != "Stunting_2019":
            frame[feature] = frame[feature] * 100
    frame["Province"] = frame["Province"].replace(PROVINCE_ALIASES)
    return frame.reset_index(drop=True)


def aggregate_profiles(districts: pd.DataFrame, complete_case: bool = False):
    """Return equal-district means and explicit, indicator-specific coverage."""
    source = districts.dropna(subset=FEATURES) if complete_case else districts
    profiles = source.groupby("Province", sort=True)[FEATURES].mean()
    provinces = sorted(districts.Province.unique())
    profiles = profiles.reindex(provinces)
    missing = profiles.isna()
    if missing.any().any():
        unavailable = [f"{p}: {f}" for p in provinces for f in FEATURES if missing.loc[p, f]]
        raise ValueError("No observed district data; provincial imputation is prohibited: " + "; ".join(unavailable))
    counts = source.groupby("Province")[FEATURES].count().reindex(provinces, fill_value=0)
    totals = districts.groupby("Province").size()
    rows = [
        {"Province": province, "indicator": feature,
         "total_districts": int(totals.loc[province]),
         "observed_districts": int(counts.loc[province, feature]),
         "coverage_pct": float(100 * counts.loc[province, feature] / totals.loc[province])}
        for province in provinces for feature in FEATURES
    ]
    return profiles.reset_index(), pd.DataFrame(rows)


def select_k(metrics: pd.DataFrame) -> int:
    if metrics.empty or not np.isfinite(metrics.silhouette).all():
        raise ValueError("Finite silhouette scores are required to select k")
    return int(metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0].k)


def neutral_labels(profiles: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Make group IDs stable by increasing mean district stunting, without risk labels."""
    order = sorted(np.unique(labels), key=lambda label: (float(profiles.loc[labels == label, "Stunting_2019"].mean()), int(label)))
    mapping = {label: group for group, label in enumerate(order)}
    return np.array([mapping[label] for label in labels], dtype=int)


def browser_records(profiles, coverage, assignments):
    coverage_index = coverage.set_index(["Province", "indicator"])
    records = []
    for index, row in profiles.iterrows():
        province = row.Province
        record = {
            "Provinsi": province,
            "Provinsi_Map": "DIY" if province == "DI Yogyakarta" else province,
            "Stunting_2019": float(row.Stunting_2019),
            "KEK_Ibu_Hamil": float(row.KEK / 100),
            "Imunisasi_Dasar_Lengkap": float(row.Immunization / 100),
            "BBLR": float(row.BBLR / 100),
            "summary_method": SUMMARY_METHOD,
            "coverage": {
                feature: {
                    "total_districts": int(coverage_index.loc[(province, feature), "total_districts"]),
                    "observed_districts": int(coverage_index.loc[(province, feature), "observed_districts"]),
                    "coverage_pct": float(coverage_index.loc[(province, feature), "coverage_pct"]),
                } for feature in FEATURES
            },
        }
        record.update({JS_CLUSTERS[name]: int(labels[index]) for name, labels in assignments.items()})
        records.append(record)
    return records


def _quality(features, labels):
    if not 1 < len(np.unique(labels)) < len(labels):
        raise ValueError("Clustering produced an invalid number of distinct groups")
    return {"silhouette": float(silhouette_score(features, labels)),
            "davies_bouldin": float(davies_bouldin_score(features, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(features, labels))}


def _fit_algorithms(profiles, features, k, scenario):
    models = {
        "K-Means": KMeans(n_clusters=k, n_init=50, random_state=42),
        "GMM": GaussianMixture(n_components=k, n_init=20, covariance_type="full", reg_covar=1e-6, max_iter=1000, random_state=42),
        "Ward": AgglomerativeClustering(n_clusters=k, linkage="ward"),
    }
    assignments, metrics = {}, []
    for name, model in models.items():
        labels = model.fit_predict(features)
        if name == "GMM" and not model.converged_:
            raise RuntimeError(f"GMM failed to converge for {scenario}, k={k}")
        assignments[name] = neutral_labels(profiles, labels)
        metrics.append({"scenario": scenario, "algorithm": name, "k": k, **_quality(features, labels),
                        "group_sizes": json.dumps(np.bincount(assignments[name]).tolist())})
    return assignments, metrics


def _cluster_profiles(profiles, assignments, scenario):
    rows = []
    for algorithm, labels in assignments.items():
        for group in np.unique(labels):
            subset = profiles.loc[labels == group]
            rows.append({"scenario": scenario, "algorithm": algorithm,
                         "group": int(group), "group_name": f"Group {group + 1}",
                         "province_count": len(subset),
                         **{feature: float(subset[feature].mean()) for feature in FEATURES},
                         "provinces": " | ".join(subset.Province)})
    return rows


def _plot_outputs(output, profiles, assignments, selection):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].plot(selection.k, selection.wss, marker="o", color=COLORS[0])
    axes[0].set(xlabel="Number of groups (k)", ylabel="Within-cluster sum of squares")
    axes[1].plot(selection.k, selection.silhouette, marker="o", color=COLORS[2])
    axes[1].set(xlabel="Number of groups (k)", ylabel="Silhouette coefficient")
    for ax in axes:
        ax.set_xticks(selection.k)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "k_selection.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "k_selection.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharex=True, sharey=True)
    for ax, algorithm in zip(axes, ALGORITHMS):
        labels = assignments[algorithm]
        for group in np.unique(labels):
            subset = profiles.loc[labels == group]
            ax.scatter(subset.Stunting_2019, subset.KEK, s=28, color=COLORS[group], label=f"Group {group + 1}")
        ax.set_title(algorithm)
        ax.set_xlabel("Mean district stunting (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Mean district maternal KEK (%)")
    fig.tight_layout()
    fig.savefig(output / "regional_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "regional_profiles.pdf", bbox_inches="tight")
    plt.close(fig)

    matrix = StandardScaler().fit_transform(profiles[FEATURES])
    fig, ax = plt.subplots(figsize=(5.5, 9.5))
    limit = max(2, float(np.abs(matrix).max()))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(profiles)), profiles.Province, fontsize=8)
    ax.set_xticks(range(4), ["Stunting", "Maternal KEK", "Immunization", "BBLR"], rotation=20, ha="right")
    fig.colorbar(image, ax=ax, label="Standardized mean district indicator")
    fig.tight_layout()
    fig.savefig(output / "province_profiles_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _legacy_map(root, output, profiles, labels):
    """Provide an explicitly qualified geometry preview, never claim 34 valid boundaries."""
    path = root / "dataset" / "indonesia-province-simple.json"
    if not path.exists():
        return {"status": "unavailable", "reason": "No local polygon source; no boundaries fabricated"}
    data = json.loads(path.read_text())
    by_name = profiles.set_index("Province")
    mapping = {p: int(label) for p, label in zip(profiles.Province, labels)}
    normalized = {p.upper(): p for p in profiles.Province}
    matched = set()
    fig, ax = plt.subplots(figsize=(10, 4))
    for feature in data["features"]:
        name = feature["properties"].get("Propinsi", "")
        canonical = PROVINCE_ALIASES.get(name, normalized.get(name, name))
        label = mapping.get(canonical)
        if label is not None:
            matched.add(canonical)
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            ax.add_patch(Polygon(np.asarray(polygon[0]), closed=True, facecolor=COLORS[label] if label is not None else "#dddddd", edgecolor="white", linewidth=0.2))
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.deg2rad(-2)))
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Illustrative grouping on legacy local polygons; not verified historical boundaries")
    missing = sorted(set(by_name.index) - matched)
    fig.text(0.01, 0.01, "Polygon source is outdated; unmatched provinces: " + ", ".join(missing), fontsize=7)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(output / "regional_map_legacy_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"status": "legacy_unverified", "feature_count": len(data["features"]), "matched_provinces": len(matched),
            "missing_provinces": missing, "warning": "Not a verified 34-province boundary layer; Irian polygons and absent provinces prevent complete cartographic validation"}


def run_clustering(root: Path | str | None = None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path = root / "dataset" / SOURCE_NAME
    output = root / "results" / "revision_1" / "clustering"
    output.mkdir(parents=True, exist_ok=True)
    districts = read_district_indicators(path)
    profiles, coverage = aggregate_profiles(districts)
    sensitivity, sensitivity_coverage = aggregate_profiles(districts, complete_case=True)
    scaled = StandardScaler().fit_transform(profiles[FEATURES])
    scaled_sensitivity = StandardScaler().fit_transform(sensitivity[FEATURES])
    with threadpool_limits(limits=1):
        selection = []
        for k in range(2, 9):
            model = KMeans(n_clusters=k, n_init=50, random_state=42)
            labels = model.fit_predict(scaled)
            selection.append({"k": k, "wss": float(model.inertia_), **_quality(scaled, labels)})
        selection = pd.DataFrame(selection)
        selected_k = select_k(selection)
        assignments, metrics = _fit_algorithms(profiles, scaled, selected_k, "primary")
        sensitivity_assignments, sensitivity_metrics = _fit_algorithms(sensitivity, scaled_sensitivity, selected_k, "complete_case_sensitivity")
        metrics.extend(sensitivity_metrics)
        cluster_profiles = _cluster_profiles(profiles, assignments, "primary")
        cluster_profiles.extend(_cluster_profiles(sensitivity, sensitivity_assignments, "complete_case_sensitivity"))
        if selected_k != 3:
            comparison_assignments, comparison_metrics = _fit_algorithms(profiles, scaled, 3, "submitted_k3_comparison")
            metrics.extend(comparison_metrics)
            cluster_profiles.extend(_cluster_profiles(profiles, comparison_assignments, "submitted_k3_comparison"))

    membership = pd.DataFrame({"Province": profiles.Province})
    comparison_rows, ari = [], {}
    for algorithm in ALGORITHMS:
        primary_labels = assignments[algorithm]
        alternate_labels = sensitivity_assignments[algorithm]
        score = float(adjusted_rand_score(primary_labels, alternate_labels))
        ari[algorithm] = score
        contingency = np.zeros((selected_k, selected_k), dtype=int)
        for a, b in zip(primary_labels, alternate_labels):
            contingency[a, b] += 1
        rows, columns = linear_sum_assignment(-contingency)
        relabel = dict(zip(columns, rows))
        aligned = np.array([relabel[label] for label in alternate_labels])
        membership[f"{algorithm}_primary"] = primary_labels
        membership[f"{algorithm}_sensitivity"] = alternate_labels
        membership[f"{algorithm}_sensitivity_aligned"] = aligned
        membership[f"{algorithm}_changed_after_alignment"] = primary_labels != aligned
        comparison_rows.append({"algorithm": algorithm, "k": selected_k, "adjusted_rand_index": score,
                                "changed_provinces_after_label_alignment": int((primary_labels != aligned).sum())})

    for algorithm, labels in assignments.items():
        profiles[JS_CLUSTERS[algorithm]] = labels
        sensitivity[JS_CLUSTERS[algorithm]] = sensitivity_assignments[algorithm]
    for name, frame in {
        "profiles.csv": profiles, "coverage.csv": coverage, "k_selection.csv": selection,
        "metrics.csv": pd.DataFrame(metrics), "cluster_profiles.csv": pd.DataFrame(cluster_profiles),
        "sensitivity_profiles.csv": sensitivity, "sensitivity_coverage.csv": sensitivity_coverage,
        "sensitivity_membership.csv": membership, "sensitivity_comparison.csv": pd.DataFrame(comparison_rows),
        "sensitivity_indicator_changes.csv": profiles[["Province", *FEATURES]].merge(
            sensitivity[["Province", *FEATURES]], on="Province", suffixes=("_primary", "_complete_case")),
    }.items():
        frame.to_csv(output / name, index=False)
    _plot_outputs(output, profiles, assignments, selection)
    primary_metrics = pd.DataFrame(metrics).query("scenario == 'primary'")
    best_algorithm = primary_metrics.sort_values(["silhouette", "davies_bouldin", "algorithm"], ascending=[False, True, True]).iloc[0].algorithm
    map_status = _legacy_map(root, output, profiles, assignments[best_algorithm])
    units = {"Stunting_2019": "percent", "KEK_Ibu_Hamil": "proportion", "Imunisasi_Dasar_Lengkap": "proportion", "BBLR": "proportion"}
    summary = {
        "source": str(path.relative_to(root)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://dashboard.stunting.go.id/wp-content/uploads/2021/04/" + SOURCE_NAME,
        "districts": len(districts), "provinces": len(profiles), "historical_province_count": 34,
        "features": FEATURES, "feature_units": "percent", "selected_k": selected_k,
        "selection_rule": "K-Means maximum silhouette across k=2..8; exact ties prefer lower k",
        "summary_method": SUMMARY_METHOD, "estimand": "average observed district indicator, not provincial population prevalence",
        "province_imputation": "none", "removed_feature": "column 4: estimated stunted under-five count incorrectly called live births",
        "sensitivity_method": "unweighted means on shared complete-case district subset; scaler refitted, same primary k",
        "sensitivity_complete_districts": int(districts[FEATURES].notna().all(axis=1).sum()),
        "sensitivity_provinces": len(sensitivity), "sensitivity_ari": ari,
        "best_primary_algorithm_by_silhouette": best_algorithm,
        "primary_metrics": primary_metrics.to_dict(orient="records"),
        "kmeans": {"n_init": 50, "random_state": 42},
        "gmm": {"n_init": 20, "covariance_type": "full", "reg_covar": 1e-6, "max_iter": 1000, "random_state": 42},
        "ward": {"linkage": "ward", "distance": "euclidean"},
        "neutral_group_order": "ascending within-group mean district stunting; IDs are not clinical risk levels",
        "browser_units": units, "map_geometry": map_status,
        "limitations": ["District means are not population-weighted provincial prevalences",
                        "Missing indicators produce unequal available-district coverage",
                        "The compiled indicators can reflect different source years and denominator populations",
                        "Internal clustering scores and complete-case sensitivity do not establish intervention effectiveness",
                        "Historical 34-province profiles are not current 38-province data"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    records = browser_records(profiles, coverage, assignments)
    metadata = {"selected_k": selected_k, "summary_method": SUMMARY_METHOD, "estimand": summary["estimand"],
                "source_sha256": summary["source_sha256"], "units": units, "map_geometry": map_status,
                "historical_province_count": 34, "province_imputation": "none"}
    (output / "province_data.js").write_text("const PROVINCE_DATA = " + json.dumps(records, indent=2, ensure_ascii=False) + ";\nconst PROVINCE_METADATA = " + json.dumps(metadata, indent=2, ensure_ascii=False) + ";\n")
    browser_profiles = {}
    for algorithm in ALGORITHMS:
        algorithm_rows = [row for row in cluster_profiles if row["scenario"] == "primary" and row["algorithm"] == algorithm]
        browser_profiles[JS_CLUSTERS[algorithm]] = [{
            "id": row["group"], "title": {"id": f"Kelompok {row['group'] + 1}", "en": row["group_name"]},
            "desc": {"id": f"{row['province_count']} provinsi. Rerata indikator kabupaten/kota: stunting {row['Stunting_2019']:.2f}%, KEK {row['KEK']:.2f}%, imunisasi {row['Immunization']:.2f}%, BBLR {row['BBLR']:.2f}%.",
                     "en": f"{row['province_count']} provinces. Mean district indicators: stunting {row['Stunting_2019']:.2f}%, maternal KEK {row['KEK']:.2f}%, immunization {row['Immunization']:.2f}%, low birth weight {row['BBLR']:.2f}%."},
            "province_count": row["province_count"], "indicator_means_pct": {feature: row[feature] for feature in FEATURES},
            "provinces": row["provinces"].split(" | "),
        } for row in algorithm_rows]
    reference = {"Stunting_2019": float(profiles.Stunting_2019.mean()), "KEK_Ibu_Hamil": float(profiles.KEK.mean() / 100),
                 "Imunisasi_Dasar_Lengkap": float(profiles.Immunization.mean() / 100), "BBLR": float(profiles.BBLR.mean() / 100)}
    (output / "cluster_profiles.js").write_text("const CLUSTER_METRICS = " + json.dumps(browser_profiles, indent=2, ensure_ascii=False) + ";\nconst REFERENCE_AVERAGES = " + json.dumps(reference, indent=2) + ";\n// Reference is an equal-province mean of district summaries, not national prevalence.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    summary = run_clustering(args.root)
    print(json.dumps({key: summary[key] for key in ["districts", "provinces", "selected_k", "sensitivity_complete_districts", "sensitivity_ari", "source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
