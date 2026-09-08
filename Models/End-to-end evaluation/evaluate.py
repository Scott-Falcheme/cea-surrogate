"""Evaluate the deployed four-stage demand cascade on a common held-out set."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CODES = HERE.parents[2]
EXPERIMENT = CODES / "Predict_timeseries" / "model" / "0816exp4"
ANNUAL = CODES / "Predict_annual"
MAPPING = CODES / "Predict_timeseries" / "data" / "adminer_timeseries_static_mapping.csv"


def building_hash(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    error = predicted - true
    nonzero = np.abs(true) > 1e-8
    denominator = max(float(np.abs(true).sum()), 1e-12)
    variance = max(float(np.square(true - true.mean()).sum()), 1e-12)
    return {
        "mae_kwh": float(np.abs(error).mean()),
        "rmse_kwh": float(np.sqrt(np.square(error).mean())),
        "nmae_percent": float(100.0 * np.abs(error).sum() / denominator),
        "mape_percent": float(100.0 * np.mean(np.abs(error[nonzero] / true[nonzero]))),
        "r2": float(1.0 - np.square(error).sum() / variance),
        "samples": int(true.size),
    }


def main() -> None:
    annual_shape_data = EXPERIMENT / "annual_shape_data"
    annual_shape_model = EXPERIMENT / "annual_shape_models" / "seed_42" / "gradient_boosting"
    daily_model = EXPERIMENT / "daily_shape_models" / "seed_42" / "lstm"

    manifest = json.loads(
        (ANNUAL / "annual_algorithm_gb_five_seed_experiment" / "experiment_manifest.json")
        .read_text(encoding="utf-8")
    )
    features = manifest["features"]
    mapping = pd.read_csv(MAPPING, usecols=["db_column", "building_uuid", *features])
    mapping["building_hash"] = mapping["db_column"].map(building_hash).astype("uint64")
    training_features = pd.read_csv(
        ANNUAL / "data" / "dataset.csv", usecols=["building_uuid", *features]
    ).set_index("building_uuid")
    gui_features = mapping.drop_duplicates("building_uuid").set_index("building_uuid")
    feature_comparison = training_features.join(
        gui_features[features], lsuffix="_training", rsuffix="_gui", how="inner"
    )
    training_values = feature_comparison[[f"{name}_training" for name in features]].to_numpy(float)
    gui_values = feature_comparison[[f"{name}_gui" for name in features]].to_numpy(float)
    changed = np.abs(training_values - gui_values) > 1e-6

    grid_frame = pd.read_csv(
        ANNUAL / "annual_algorithm_gb_five_seed_experiment" / "seed_001" /
        "gradient_boosting" / "predictions.csv"
    ).set_index("building_uuid")
    heating_frame = pd.read_csv(
        ANNUAL / "annual_algorithm_five_seed_experiment" / "seed_001" /
        "random_forest" / "predictions.csv"
    ).set_index("building_uuid")

    hashes = np.load(annual_shape_data / "building_hashes.npy")
    valid = np.load(annual_shape_data / "valid_buildings.npy")
    split = np.load(annual_shape_data / "split_seed_42.npy")
    annual_test_rows = np.flatnonzero(valid & (split == 2))
    annual_test_hashes = hashes[annual_test_rows]
    annual_shape_prediction = np.load(
        annual_shape_model / "test_predictions.npy", mmap_mode="r"
    )
    if len(annual_shape_prediction) != len(annual_test_rows):
        raise RuntimeError("Annual-shape prediction order does not match the seed-42 split")
    annual_shape_by_hash = {
        int(value): annual_shape_prediction[index]
        for index, value in enumerate(annual_test_hashes)
    }

    common = mapping[
        mapping["building_uuid"].isin(grid_frame.index)
        & mapping["building_hash"].isin(annual_test_hashes)
    ].drop_duplicates("building_hash")
    common_hashes = set(map(int, common["building_hash"]))
    uuid_by_hash = dict(zip(map(int, common["building_hash"]), common["building_uuid"]))
    deployed = CODES / "Final_bundle" / "Models" / "Annual"
    grid_model = joblib.load(deployed / "grid_gradient_boosting_seed_001.joblib")[0]
    heating_model = joblib.load(deployed / "heating_random_forest_seed_001.joblib")[1]
    deployed_input = common.set_index("building_hash").loc[sorted(common_hashes), features].to_numpy(float)
    deployed_grid = np.maximum(np.expm1(np.clip(grid_model.predict(deployed_input), 0.0, 16.1)), 0.0)
    deployed_heating = np.maximum(np.expm1(np.clip(heating_model.predict(deployed_input), 0.0, 16.1)), 0.0)
    grid_by_hash = dict(zip(sorted(common_hashes), deployed_grid))
    heating_by_hash = dict(zip(sorted(common_hashes), deployed_heating))

    shape_prediction = np.load(daily_model / "test_predictions.npy", mmap_mode="r")
    shape_target = np.load(daily_model / "test_targets.npy", mmap_mode="r")
    daily_hashes = np.load(daily_model / "test_building_hashes.npy", mmap_mode="r")
    source_days = np.load(daily_model / "test_source_days.npy", mmap_mode="r")
    daily_max = np.load(
        EXPERIMENT / "daily_shape_models" / "seed_42" / "test_daily_max_kwh.npy",
        mmap_mode="r",
    )

    rows_by_hash: dict[int, list[int]] = {value: [] for value in common_hashes}
    for index, value in enumerate(daily_hashes):
        integer = int(value)
        if integer in rows_by_hash:
            rows_by_hash[integer].append(index)

    annual_true = {"heating": [], "grid": []}
    annual_predicted = {"heating": [], "grid": []}
    hourly_true = {"heating": [], "grid": []}
    hourly_predicted = {"heating": [], "grid": []}
    conservation_error = {"heating": [], "grid": []}

    for value in sorted(common_hashes):
        indices = np.asarray(rows_by_hash[value], dtype=np.int64)
        calendar_days = np.where(source_days[indices] < 59, source_days[indices], source_days[indices] - 1)
        order = np.argsort(calendar_days)
        indices = indices[order]
        calendar_days = calendar_days[order]
        if len(indices) != 365 or not np.array_equal(calendar_days, np.arange(365)):
            raise RuntimeError(f"Building hash {value} does not have one row for each calendar day")

        target_shape = np.asarray(shape_target[indices], dtype=np.float64)
        predicted_shape = np.asarray(shape_prediction[indices], dtype=np.float64)
        scale = np.asarray(daily_max[indices], dtype=np.float64)
        truth_heating = target_shape[:, :24] * scale[:, 0, None]
        truth_grid = target_shape[:, 24:] * scale[:, 1, None]

        predicted_annual_heating = float(heating_by_hash[value])
        predicted_annual_grid = float(grid_by_hash[value])
        annual_pattern = np.asarray(annual_shape_by_hash[value], dtype=np.float64)
        heating_weight = annual_pattern[:, 0, None] * predicted_shape[:, :24]
        grid_weight = annual_pattern[:, 1, None] * predicted_shape[:, 24:]
        predicted_heating = predicted_annual_heating * heating_weight / heating_weight.sum()
        predicted_grid = predicted_annual_grid * grid_weight / grid_weight.sum()

        for name, truth, prediction, annual_prediction in (
            ("heating", truth_heating, predicted_heating, predicted_annual_heating),
            ("grid", truth_grid, predicted_grid, predicted_annual_grid),
        ):
            annual_true[name].append(float(truth.sum()))
            annual_predicted[name].append(annual_prediction)
            hourly_true[name].append(truth.reshape(-1))
            hourly_predicted[name].append(prediction.reshape(-1))
            conservation_error[name].append(abs(float(prediction.sum()) - annual_prediction))

    report = {
        "configuration": {
            "annual_grid": "Gradient Boosting, seed 1",
            "annual_heating": "Random Forest, seed 1",
            "annual_shape": "Gradient Boosting, seed 42",
            "daily_shape": "LSTM, seed 42",
            "final_profile": "annual-energy-conserving normalisation",
        },
        "evaluation": {
            "buildings": len(common_hashes),
            "selection": "intersection held out by annual seed 1 and both shape seed 42 splits",
            "calendar": "2024 with 29 February removed (365 days)",
            "annual_input": "current GUI-compatible 35-feature mapping",
            "training_to_gui_feature_drift": {
                "matched_buildings": len(feature_comparison),
                "buildings_with_any_changed_feature": int(changed.any(axis=1).sum()),
                "changed_cells": int(changed.sum()),
            },
        },
        "metrics": {},
        "annual_conservation_max_abs_kwh": {
            name: max(values) for name, values in conservation_error.items()
        },
    }
    for level in ("annual", "hourly"):
        report["metrics"][level] = {}
        for name in ("heating", "grid"):
            true = annual_true[name] if level == "annual" else np.concatenate(hourly_true[name])
            predicted = annual_predicted[name] if level == "annual" else np.concatenate(hourly_predicted[name])
            report["metrics"][level][name] = metrics(true, predicted)

    (HERE / "end_to_end_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (HERE / "end_to_end_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["level", "demand", *next(iter(report["metrics"]["annual"].values()))])
        writer.writeheader()
        for level, demands in report["metrics"].items():
            for demand, values in demands.items():
                writer.writerow({"level": level, "demand": demand, **values})

    labels = {"heating": "Heating", "grid": "Grid"}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
        r"\caption{End-to-end errors of the adopted demand-surrogate cascade on the common strictly held-out set using the current GUI-compatible 35-feature mapping. Annual rows evaluate building annual totals; hourly rows evaluate the final 8,760 hourly values after annual-energy-conserving normalisation. MAPE excludes zero targets.}",
        r"\label{tab:adopted_demand_end_to_end}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Level & Demand & MAE [kWh] & RMSE [kWh] & NMAE [\%] & MAPE [\%] & $R^2$ \\",
        r"\midrule",
    ]
    for level, level_label in (("annual", "Annual total"), ("hourly", "Hourly profile")):
        for index, demand in enumerate(("heating", "grid")):
            values = report["metrics"][level][demand]
            label = level_label if index == 0 else ""
            lines.append(
                f"{label} & {labels[demand]} & {values['mae_kwh']:.3f} & "
                f"{values['rmse_kwh']:.3f} & {values['nmae_percent']:.2f} & "
                f"{values['mape_percent']:.2f} & {values['r2']:.4f} \\\\"
            )
        if level == "annual":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "end_to_end_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
