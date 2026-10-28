"""Supplemental majority, label-consistency and measurement-assumption checks.

Reads the existing audited caches and frozen artifacts. Does not fit a clinical
model, select an algorithm, relabel source outcomes, or alter the primary cohort.
Run after the supervised freeze: ``python -m analysis.revision.sensitivity``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score

from .data import _legacy_references, monthly_zscore


CLASSES = ["Normal", "Severely Stunted", "Stunted", "Tall"]
HAZ_FEATURES = ["Gender_Encoded", "Age (months)", "Height (cm)"]
WHO_POSITION_SOURCE = "https://github.com/WorldHealthOrganization/anthro/blob/master/R/z-score-helper.R"


def _classification_metrics(y, prediction, binary=False):
    labels = [0, 1] if binary else [0, 1, 2, 3]
    result = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", labels=labels, zero_division=0)),
        "confusion_matrix": json.dumps(confusion_matrix(y, prediction, labels=labels).tolist()),
    }
    if binary:
        recalls = recall_score(y, prediction, labels=labels, average=None, zero_division=0)
        result.update(sensitivity=float(recalls[1]), specificity=float(recalls[0]))
    return result


def majority_baselines(synthetic, external):
    """Fit one majority decision on synthetic labels; never refit on external labels."""
    labels = synthetic.Stunting.map({name: index for index, name in enumerate(CLASSES)}).to_numpy()
    if not np.isfinite(labels).all():
        raise ValueError("Unexpected original synthetic classification label")
    classifier = DummyClassifier(strategy="most_frequent", random_state=42)
    classifier.fit(np.zeros((len(labels), 1)), labels)
    prediction = classifier.predict(np.zeros((len(labels), 1)))
    majority = int(prediction[0])
    synthetic_binary = np.isin(labels, [1, 2]).astype(int)
    prediction_binary = np.isin(prediction, [1, 2]).astype(int)
    external_prediction = np.isin(classifier.predict(np.zeros((len(external), 1))), [1, 2]).astype(int)
    rows = [
        {"cohort": "synthetic_four_class", "n": len(labels), "constant_class": CLASSES[majority],
         "training_source": "synthetic labels only", **_classification_metrics(labels, prediction)},
        {"cohort": "synthetic_binary", "n": len(labels), "constant_class": CLASSES[majority],
         "training_source": "same frozen synthetic four-class majority, collapsed", **_classification_metrics(synthetic_binary, prediction_binary, binary=True)},
        {"cohort": "external_primary_binary", "n": len(external), "constant_class": CLASSES[majority],
         "training_source": "same frozen synthetic four-class majority, collapsed", **_classification_metrics(external.Stunting_Binary.to_numpy(dtype=int), external_prediction, binary=True)},
    ]
    return rows


def _haz(frame, reference):
    table = reference.loc[reference.Indicator.eq("HAZ")].set_index(["Gender_Encoded", "Age (months)"])
    if not table.index.is_unique:
        raise ValueError("HAZ references must have one coefficient row per sex/month")
    coefficients = table[["L", "M", "S"]].reindex(pd.MultiIndex.from_frame(frame[["Gender_Encoded", "Age (months)"]]))
    return monthly_zscore(frame["Height (cm)"].to_numpy(), coefficients.L.to_numpy(), coefficients.M.to_numpy(), coefficients.S.to_numpy())


def measurement_assumptions(frame, reference):
    as_recorded = frame.copy(deep=True)
    standing = frame.copy(deep=True)
    under24 = standing["Age (months)"].ge(0) & standing["Age (months)"].lt(24)
    standing.loc[under24, "Height (cm)"] += 0.7
    as_recorded["HAZ"] = _haz(as_recorded, reference)
    standing["HAZ"] = _haz(standing, reference)
    return as_recorded, standing


def measurement_diagnostics(frame, reference):
    """Keep the original HAZ eligibility mask; alternative position is an assumption."""
    mask = frame.Valid_HAZ & frame["Age (months)"].ge(0) & frame["Age (months)"].lt(24)
    cohort = frame.loc[mask].copy()
    as_recorded, standing = measurement_assumptions(cohort, reference)
    known_labels = ~cohort.Flag_Source_Label
    source = cohort.loc[known_labels, "Stunting_Binary"].to_numpy(dtype=int)
    base_category = (as_recorded.loc[known_labels, "HAZ"].to_numpy() < -2).astype(int)
    standing_category = (standing.loc[known_labels, "HAZ"].to_numpy() < -2).astype(int)
    records = cohort.copy()
    records["Height_as_recorded_cm"] = as_recorded["Height (cm)"]
    records["Length_if_standing_cm"] = standing["Height (cm)"]
    records["HAZ_as_recorded"] = as_recorded.HAZ
    records["HAZ_if_standing"] = standing.HAZ
    records["HAZ_assumption_difference"] = standing.HAZ - as_recorded.HAZ
    records["Source_disagreement_as_recorded"] = known_labels & records.Stunting_Binary.ne((records.HAZ_as_recorded < -2).astype(int))
    records["Source_disagreement_if_standing"] = known_labels & records.Stunting_Binary.ne((records.HAZ_if_standing < -2).astype(int))
    summary = {
        "fixed_primary_haz_cohort_n": len(cohort), "source_label_comparison_n": int(known_labels.sum()),
        "as_recorded_source_label_disagreements": int((source != base_category).sum()),
        "standing_assumption_source_label_disagreements": int((source != standing_category).sum()),
        "reference_binary_status_changes": int((base_category != standing_category).sum()),
        "mean_haz_assumption_difference": float(records.HAZ_assumption_difference.mean()),
        "min_haz_assumption_difference": float(records.HAZ_assumption_difference.min()),
        "max_haz_assumption_difference": float(records.HAZ_assumption_difference.max()),
        "alternative_haz_outside_primary_flag_range": int((~standing.HAZ.between(-6, 6)).sum()),
        "primary_cohort_refiltered": False, "source_labels_changed": False,
        "interpretation": "Alternative assumption that all recorded measurements below 24 months were standing heights; add 0.7 cm to represent length. Actual measurement position is unavailable. Primary analysis stays as recorded.",
        "who_adjustment_source": WHO_POSITION_SOURCE,
    }
    return summary, records


def _categories(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Synthetic HAZ must be finite before comparing categories")
    return np.select([values < -3, values < -2, values > 3], ["Severely Stunted", "Stunted", "Tall"], default="Normal")


def label_diagnostics(frame, legacy_haz, corrected_haz):
    keys = HAZ_FEATURES
    groups = frame.groupby(keys, dropna=False).Stunting.agg(label_count="nunique", rows="size", labels=lambda x: " | ".join(sorted(x.unique())))
    conflicts = groups.loc[groups.label_count.gt(1)].reset_index()
    legacy_category, corrected_category = _categories(legacy_haz), _categories(corrected_haz)
    summary = {
        "rows": len(frame), "unique_decision_profiles": len(groups),
        "conflicting_original_label_groups": len(conflicts),
        "rows_in_conflicting_original_label_groups": int(conflicts.rows.sum()),
        "legacy_haz_category_disagreements": int((legacy_category != frame.Stunting).sum()),
        "corrected_haz_category_disagreements": int((corrected_category != frame.Stunting).sum()),
        "haz_reference_category_changes": int((legacy_category != corrected_category).sum()),
        "original_labels_preserved": True,
        "definition": "HAZ<-3 severe; -3<=HAZ<-2 stunted; HAZ>3 tall; otherwise normal. Differences are diagnostics, not automatic source relabeling.",
    }
    return summary, conflicts


def _regression_metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)), "rmse": float(np.sqrt(mean_squared_error(y, prediction))), "r2": float(r2_score(y, prediction))}


def run(root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    output = root / "results" / "revision_1" / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    synthetic_path = root / "dataset" / "Stunting Wasting Dataset (Synthetic)" / "stunting_wasting_dataset.csv"
    external_path = root / "dataset" / "revision" / "derived" / "jeneponto_2024_harmonized.csv"
    reference_path = root / "dataset" / "revision" / "references" / "who_monthly_corrected.csv"
    reference_manifest = json.loads(reference_path.with_name("manifest.json").read_text())
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_manifest["corrected_sha256"]:
        raise ValueError("Corrected WHO reference cache checksum differs from manifest")
    reference = pd.read_csv(reference_path)
    raw = pd.read_csv(synthetic_path)
    synthetic = raw.rename(columns={"Jenis Kelamin": "Gender", "Umur (bulan)": "Age (months)", "Tinggi Badan (cm)": "Height (cm)", "Berat Badan (kg)": "Weight (kg)"}).copy()
    synthetic["Gender_Encoded"] = synthetic.Gender.map({"Laki-laki": 1, "Perempuan": 0, "Male": 1, "Female": 0})
    synthetic["Source_Row"] = np.arange(1, len(synthetic) + 1)
    external = pd.read_csv(external_path)
    external_primary = external.loc[external.Valid_Classification & external["Age (months)"].ge(0) & external["Age (months)"].lt(24)].copy()
    baseline_rows = majority_baselines(synthetic, external_primary)
    fold_path = root / "results" / "revision_1" / "supervised" / "folds_corrected_grouped_classification.csv"
    folds = pd.read_csv(fold_path)
    joined = synthetic.merge(folds, on="Source_Row", how="left", validate="one_to_one")
    if joined.fold.isna().any() or len(joined) != len(synthetic):
        raise ValueError("Frozen grouped folds must cover every synthetic source row once")
    fold_counts = joined.groupby(HAZ_FEATURES).fold.nunique()
    leaking_groups = int(fold_counts.gt(1).sum())
    if leaking_groups:
        raise ValueError(f"{leaking_groups} identical decision profiles occur in multiple grouped folds")
    fold_baselines = []
    mapping = {name: index for index, name in enumerate(CLASSES)}
    for fold in sorted(joined.fold.unique()):
        train = joined.loc[joined.fold.ne(fold)]
        valid = joined.loc[joined.fold.eq(fold)]
        dummy = DummyClassifier(strategy="most_frequent", random_state=42).fit(np.zeros((len(train), 1)), train.Stunting.map(mapping))
        prediction = dummy.predict(np.zeros((len(valid), 1)))
        fold_baselines.append({"fold": int(fold), "n": len(valid), "constant_class": CLASSES[int(prediction[0])], **_classification_metrics(valid.Stunting.map(mapping), prediction)})
    legacy_haz, corrected_haz = _haz(synthetic, _legacy_references(root)), _haz(synthetic, reference)
    label_summary, conflicts = label_diagnostics(synthetic, legacy_haz, corrected_haz)
    label_summary["decision_profiles_across_multiple_grouped_folds"] = leaking_groups
    details = synthetic[["Source_Row", *HAZ_FEATURES, "Stunting"]].copy()
    details["Legacy_HAZ"] = legacy_haz
    details["Corrected_HAZ"] = corrected_haz
    details["Legacy_Category"] = _categories(legacy_haz)
    details["Corrected_Category"] = _categories(corrected_haz)
    details.loc[details.Legacy_Category.ne(details.Stunting) | details.Corrected_Category.ne(details.Stunting)].to_csv(output / "synthetic_label_disagreements.csv", index=False)
    conflicts.to_csv(output / "synthetic_label_conflicts.csv", index=False)
    confusion_rows = []
    for name, category in [("legacy", details.Legacy_Category), ("corrected", details.Corrected_Category)]:
        for source_class in CLASSES:
            for reference_class in CLASSES:
                confusion_rows.append({"reference": name, "original_label": source_class, "haz_category": reference_class, "count": int((details.Stunting.eq(source_class) & category.eq(reference_class)).sum())})
    pd.DataFrame(confusion_rows).to_csv(output / "synthetic_label_vs_reference.csv", index=False)
    measurement_summary, records = measurement_diagnostics(external, reference)
    classification_cohort = records.loc[records.Valid_Classification]
    measurement_summary["fixed_primary_classification_cohort_n"] = len(classification_cohort)
    measurement_summary["classification_cohort_source_disagreements_as_recorded"] = int(classification_cohort.Source_disagreement_as_recorded.sum())
    measurement_summary["classification_cohort_source_disagreements_if_standing"] = int(classification_cohort.Source_disagreement_if_standing.sum())

    freeze_path = root / "results" / "revision_1" / "supervised" / "model_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    artifact = freeze["artifacts"]["HAZ_rf"]
    model_path = root / artifact["joblib"]
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError("Frozen RF HAZ artifact hash changed; sensitivity cannot use an unfrozen replacement")
    model = joblib.load(model_path)
    original_inputs = records[HAZ_FEATURES].copy()
    alternate_inputs = original_inputs.copy()
    alternate_inputs["Height (cm)"] = records.Length_if_standing_cm
    original_prediction = model.predict(original_inputs)
    alternate_prediction = model.predict(alternate_inputs)
    records["Frozen_RF_HAZ_prediction_as_recorded"] = original_prediction
    records["Frozen_RF_HAZ_prediction_if_standing"] = alternate_prediction
    records["RF_residual_as_recorded"] = original_prediction - records.HAZ_as_recorded
    records["RF_residual_if_standing"] = alternate_prediction - records.HAZ_if_standing
    original_metrics = _regression_metrics(records.HAZ_as_recorded, original_prediction)
    converted_metrics = _regression_metrics(records.HAZ_if_standing, alternate_prediction)
    target_only_metrics = _regression_metrics(records.HAZ_if_standing, original_prediction)
    measurement_summary["frozen_rf"] = {
        "model_artifact": artifact["joblib"], "sha256": digest,
        "selected_primary_haz_model": freeze["selected"]["HAZ"]["model"],
        "as_recorded": original_metrics,
        "standing_assumption_inputs_and_reference": converted_metrics,
        "standing_reference_only_original_inputs": target_only_metrics,
        "rmse_difference_inputs_and_reference": converted_metrics["rmse"] - original_metrics["rmse"],
        "mean_prediction_shift": float(np.mean(alternate_prediction - original_prediction)),
        "refitted": False, "primary_model_selection_changed": False,
    }
    records.to_csv(output / "external_measurement_assumptions.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output / "majority_baselines.csv", index=False)
    pd.DataFrame(fold_baselines).to_csv(output / "majority_grouped_cv.csv", index=False)
    grouped_baseline = pd.DataFrame(fold_baselines)
    summary = {
        "majority_baselines": baseline_rows,
        "majority_grouped_cv": {"mean": {key: float(grouped_baseline[key].mean()) for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]},
                                "sd": {key: float(grouped_baseline[key].std(ddof=0)) for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}},
        "label_diagnostics": label_summary, "measurement_position": measurement_summary,
        "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [synthetic_path, external_path, reference_path, freeze_path]},
        "limitations": ["Actual measurement position is absent; the standing scenario is a sensitivity assumption, not recovered metadata",
                        "Completed-month references remain approximate; exact-day clinical outcomes are not supplied",
                        "Comparing adjusted inputs with adjusted references checks function approximation under an assumption, not external clinical validation",
                        "Source classifications and primary QA cohorts are retained; the sensitivity does not tune or fit a model",
                        "Synthetic four-class and external binary baseline accuracies have different label spaces"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({"majority_baselines": result["majority_baselines"], "label_diagnostics": result["label_diagnostics"], "measurement_position": result["measurement_position"]}, indent=2))
