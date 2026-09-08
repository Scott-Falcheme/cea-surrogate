"""Physics-lite radiation-to-35-solar-channel prototype and small-sample evaluation.

This intentionally does not fit any coefficient to CEA outputs.  Technology
parameters come from the CH CEA component database and CEA default inlet
temperatures.  It omits CEA's detailed incidence-angle optics, roof re-tilting,
hydraulic flow optimisation, thermal capacitance and pipe/pump losses so that
the remaining error is a useful target for a later residual model.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

try:
    from ..surrounding_agent import uuid_from_iri_or_uuid
    from ..weather_mapper import WEATHER_STATIONS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from surrounding_agent import uuid_from_iri_or_uuid
    from weather_mapper import WEATHER_STATIONS

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED = ROOT / "Predict_timeseries/modular_surrogates/data/processed"
SELECTION = ROOT / "Predict_timeseries/data/solar_m2_2700/selected_buildings.csv"
SURROGATE_DIR = ROOT / "Predict_timeseries/solar radiation data"
PV_DB = DATA_DIR / "PHOTOVOLTAIC_PANELS.csv"
SC_DB = DATA_DIR / "SOLAR_COLLECTORS.csv"
OUTPUT = ROOT / "codex/solar_physics_lite_prototype"

ORIENTATIONS = ("R", "N", "S", "E", "W")
TECHNOLOGIES = ("PV", "ET_Q", "ET_E", "FP_Q", "FP_E", "Th_ET", "Th_FP")
RAD_INDEX = {"R": 0, "E": 1, "N": 2, "S": 3, "W": 4}
STATIC_AREA_INDEX = {"R": 3, "E": 4, "N": 5, "S": 6, "W": 7}
SURROGATE_PREFIX = {"R": "roofs_top", "N": "walls_north", "S": "walls_south", "E": "walls_east", "W": "walls_west"}

# CEA defaults from cea/technologies/solar/constants.py
T_IN_PVT_C = 35.0
T_IN_SC_C = {"FP": 60.0, "ET": 75.0}

# A single transparent-cover optical factor is the deliberate simplification.
# CEA instead resolves beam/diffuse incidence-angle modifiers separately.
THERMAL_IAM_FACTOR = 0.95
# CEA flat-roof layout for this latitude/weather uses B=3.8 deg and 0.68 m
# row spacing.  Its own area formula gives 1/(spacing/2 + cos(B)) = 0.747.
ROOF_MODULE_COVERAGE = 0.747


@lru_cache(maxsize=1)
def load_parameters() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    pv_row = pd.read_csv(PV_DB).query("code == 'PV1'").iloc[0]
    pv = {name: float(pv_row[name]) for name in (
        "PV_n", "PV_noct", "PV_Bref", "misc_losses", "PV_th",
        "PV_a0", "PV_a1", "PV_a2", "PV_a3", "PV_a4",
    )}
    sc_frame = pd.read_csv(SC_DB)
    sc: dict[str, dict[str, float]] = {}
    for label, code in (("FP", "SC1"), ("ET", "SC2")):
        row = sc_frame.query("code == @code").iloc[0]
        sc[label] = {name: float(row[name]) for name in (
            "aperture_area_ratio", "n0", "c1", "c2", "mB0_r", "Cp_fluid"
        )}
    return pv, sc


def pv_electricity(absorbed: np.ndarray, area_m2: np.ndarray, ambient_c: np.ndarray,
                   pv: dict[str, float], module_temperature_c: np.ndarray | None = None) -> np.ndarray:
    if module_temperature_c is None:
        module_temperature_c = ambient_c + absorbed * (pv["PV_noct"] - 20.0) / 800.0
    temperature_factor = np.maximum(0.0, 1.0 - pv["PV_Bref"] * (module_temperature_c - 25.0))
    return pv["PV_n"] * area_m2 * absorbed * temperature_factor * (1.0 - pv["misc_losses"]) / 1000.0


def thermal_output(g_wm2: np.ndarray, absorbed_pv: np.ndarray, area_m2: np.ndarray, ambient_c: np.ndarray,
                   panel: dict[str, float], inlet_c: float, pvt: bool,
                   pv: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    absorbed_thermal = panel["n0"] * THERMAL_IAM_FACTOR * g_wm2
    if pvt:
        c1 = np.maximum(0.0, panel["c1"] - pv["PV_n"] * pv["PV_Bref"] * absorbed_pv)
    else:
        c1 = panel["c1"]
    delta_t = inlet_c - ambient_c
    useful_wm2_aperture = np.maximum(
        0.0, absorbed_thermal - c1 * delta_t - panel["c2"] * np.abs(delta_t) * delta_t
    )
    aperture_area = area_m2 * panel["aperture_area_ratio"]
    heat_kw = useful_wm2_aperture * aperture_area / 1000.0

    # Nominal-flow mean-fluid temperature: Tin + 0.5 * q/(m_dot*Cp).
    mass_flow_per_aperture = panel["mB0_r"] / 3600.0
    fluid_rise = np.divide(
        useful_wm2_aperture,
        mass_flow_per_aperture * panel["Cp_fluid"],
        out=np.zeros_like(useful_wm2_aperture),
        where=mass_flow_per_aperture > 0,
    )
    module_temperature = inlet_c + 0.5 * fluid_rise
    return heat_kw, module_temperature


def _load_weather_object(weather_IRI: str, weather_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Build an EPW-like pvlib weather object directly from a bundled weather CSV."""
    station_uuid = uuid_from_iri_or_uuid(weather_IRI)
    directory = Path(weather_dir) if weather_dir is not None else DATA_DIR
    weather_path = directory / f"weather_{station_uuid}.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather CSV not found for {weather_IRI}: {weather_path}")
    source = pd.read_csv(weather_path)
    required = {"time", "AirTemperature", "DiffuseHorizontalIrradiance", "DirectNormalIrradiance"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Weather CSV is missing columns {sorted(missing)}: {weather_path}")
    source["time"] = pd.to_datetime(source["time"], utc=True, errors="raise")
    source = source.drop_duplicates("time").sort_values("time")
    year = int(source["time"].dt.year.value_counts().idxmax())
    source = source[source["time"].dt.year == year]
    source = source[~((source["time"].dt.month == 2) & (source["time"].dt.day == 29))]
    if len(source) != 8760:
        raise ValueError(f"Weather station {station_uuid} has {len(source)} usable hours in {year}; expected 8760")
    station = next((item for item in WEATHER_STATIONS if uuid_from_iri_or_uuid(item[0]) == station_uuid), None)
    if station is None:
        raise KeyError(f"Weather-station coordinates are not registered for {weather_IRI}")
    weather = pd.DataFrame(
        {
            "temp_air": pd.to_numeric(source["AirTemperature"], errors="raise").to_numpy(float),
            "dhi": pd.to_numeric(source["DiffuseHorizontalIrradiance"], errors="raise").to_numpy(float),
            "dni": pd.to_numeric(source["DirectNormalIrradiance"], errors="raise").to_numpy(float),
        },
        index=pd.DatetimeIndex(source["time"]),
    )
    return weather, {
        "latitude": float(station[1]), "longitude": float(station[2]),
        "altitude": 0.0, "weather_IRI": weather_IRI,
        "weather_path": str(weather_path.resolve()), "weather_year": year,
    }


def pv_absorption_ratio(weather_IRI: str, weather_dir: str | Path | None = None) -> np.ndarray:
    """CEA PV optical model from bundled CSV weather; returns [hour, orientation]."""
    pv, _ = load_parameters()
    weather, meta = _load_weather_object(weather_IRI, weather_dir)
    position = pvlib.solarposition.get_solarposition(
        weather.index, float(meta["latitude"]), float(meta["longitude"]), altitude=float(meta["altitude"])
    )
    zenith_deg = position["apparent_zenith"].to_numpy(float)
    zenith = np.radians(np.clip(zenith_deg, 0.0, 89.999))
    solar_azimuth = position["azimuth"].to_numpy(float)
    dhi = np.clip(weather["dhi"].to_numpy(float), 0.0, None)
    dni = np.clip(weather["dni"].to_numpy(float), 0.0, None)
    ghi = np.clip(dhi + dni * np.clip(np.cos(np.radians(zenith_deg)), 0.0, None), 0.0, None)
    diffuse_ratio = np.divide(dhi, ghi, out=np.zeros(len(weather)), where=ghi > 0)
    diffuse_ratio = np.clip(diffuse_ratio, 0.0, 1.0)
    direct_fraction = 1.0 - diffuse_ratio
    n, extinction, thickness = 1.526, 4.0, pv["PV_th"]
    ta_normal = np.exp(-extinction * thickness) * (1.0 - ((n - 1.0) / (n + 1.0)) ** 2)
    altitude = float(meta["altitude"])
    air_mass = np.where(
        zenith <= np.radians(70.0), 1.0 / np.cos(zenith),
        np.exp(-0.0001184 * altitude) /
        (np.cos(zenith) + 0.5057 * np.maximum(96.080 - np.degrees(zenith), 1e-3) ** -1.634),
    )
    modifier = sum(pv[f"PV_a{i}"] * air_mass ** i for i in range(5))
    modifier = np.clip(modifier, 0.001, 1.1)

    def incidence_modifier(angle: np.ndarray) -> np.ndarray:
        angle = np.clip(angle, 1e-7, np.radians(89.999))
        refracted = np.arcsin(np.sin(angle) / n)
        plus, minus = refracted + angle, refracted - angle
        trans_abs = np.exp((-extinction * thickness) / np.cos(refracted)) * (
            1.0 - 0.5 * ((np.sin(minus) / np.sin(plus)) ** 2 + (np.tan(minus) / np.tan(plus)) ** 2)
        )
        return np.nan_to_num(trans_abs / ta_normal, nan=1.0, posinf=0.0, neginf=0.0)

    azimuth = {"R": 180.0, "N": 0.0, "S": 180.0, "E": 90.0, "W": 270.0}
    ratios = []
    for orientation in ORIENTATIONS:
        tilt_deg = 3.8 if orientation == "R" else 90.0
        tilt = np.radians(tilt_deg)
        aoi = np.radians(pvlib.irradiance.aoi(tilt_deg, azimuth[orientation], zenith_deg, solar_azimuth))
        aoi = np.clip(aoi, 0.0, np.radians(89.999))
        rb = np.where(zenith <= np.radians(85.0), np.cos(aoi) / np.cos(zenith), 0.0)
        theta_d = np.radians(59.7 - 0.1388 * tilt_deg + 0.001497 * tilt_deg ** 2)
        theta_g = np.radians(90.0 - 0.5788 * tilt_deg + 0.002693 * tilt_deg ** 2)
        ratio = modifier * ta_normal * (
            incidence_modifier(aoi) * direct_fraction * rb
            + incidence_modifier(np.full(len(weather), theta_d)) * diffuse_ratio * (1.0 + np.cos(tilt)) / 2.0
            + incidence_modifier(np.full(len(weather), theta_g)) * 0.2 * (1.0 - np.cos(tilt)) / 2.0
        )
        ratios.append(np.maximum(0.0, np.nan_to_num(ratio)))
    return np.stack(ratios, axis=1)


def convert(radiation_kw: np.ndarray, area_m2: np.ndarray, ambient_c: np.ndarray,
            absorption_ratio: np.ndarray,
            apply_aggregate_annual_filter: bool) -> np.ndarray:
    """Convert arrays shaped [building, hour, orientation] to [building, hour, 35]."""
    pv, panels = load_parameters()
    radiation_area = area_m2[:, None, :]
    g = np.divide(radiation_kw * 1000.0, radiation_area, out=np.zeros_like(radiation_kw), where=radiation_area > 0)
    if apply_aggregate_annual_filter:
        eligible = g.sum(axis=1) / 1000.0 >= 800.0
        g = np.where(eligible[:, None, :], np.where(g > 50.0, g, 0.0), 0.0)
    ambient = ambient_c[None, :, None]
    absorbed = g * absorption_ratio[None, :, :]
    coverage = np.ones((1, 1, 5), dtype=float)
    coverage[:, :, 0] = ROOF_MODULE_COVERAGE
    area = radiation_area * coverage

    pv_e = pv_electricity(absorbed, area, ambient, pv)
    fp_q, fp_temp = thermal_output(g, absorbed, area, ambient, panels["FP"], T_IN_PVT_C, True, pv)
    et_q, et_temp = thermal_output(g, absorbed, area, ambient, panels["ET"], T_IN_PVT_C, True, pv)
    fp_e = pv_electricity(absorbed, area, ambient, pv, fp_temp)
    et_e = pv_electricity(absorbed, area, ambient, pv, et_temp)
    th_fp, _ = thermal_output(g, absorbed, area, ambient, panels["FP"], T_IN_SC_C["FP"], False, pv)
    th_et, _ = thermal_output(g, absorbed, area, ambient, panels["ET"], T_IN_SC_C["ET"], False, pv)
    by_tech = {"PV": pv_e, "ET_Q": et_q, "ET_E": et_e, "FP_Q": fp_q,
               "FP_E": fp_e, "Th_ET": th_et, "Th_FP": th_fp}
    return np.stack([by_tech[tech][:, :, oi] for oi in range(5) for tech in TECHNOLOGIES], axis=2)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    active = (np.abs(pred) > 1e-9) | (np.abs(true) > 1e-9)
    error = pred - true
    true_sum = float(true.sum())
    pred_sum = float(pred.sum())
    if active.sum() >= 3 and np.std(pred[active]) > 0 and np.std(true[active]) > 0:
        corr = float(np.corrcoef(pred[active], true[active])[0, 1])
    else:
        corr = float("nan")
    return {
        "points": int(pred.size), "active_points": int(active.sum()),
        "true_kWh": true_sum, "predicted_kWh": pred_sum,
        "annual_bias_percent": 100.0 * (pred_sum - true_sum) / true_sum if true_sum else float("nan"),
        "mae_kWh": float(np.mean(np.abs(error))),
        "rmse_kWh": float(np.sqrt(np.mean(error ** 2))),
        "nmae_percent": 100.0 * float(np.abs(error).sum()) / true_sum if true_sum else float("nan"),
        "active_hour_correlation": corr,
    }


def load_surrogate(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    radiation = np.zeros((len(sample), 8760, 5), dtype=float)
    areas = np.zeros((len(sample), 5), dtype=float)
    for bi, uuid in enumerate(sample["uuid"]):
        frame = pd.read_csv(SURROGATE_DIR / f"{uuid}.csv")
        for oi, orientation in enumerate(ORIENTATIONS):
            prefix = SURROGATE_PREFIX[orientation]
            radiation[bi, :, oi] = frame[f"{prefix}_kW"].to_numpy(float)
            areas[bi, oi] = float(frame[f"{prefix}_m2"].iloc[0])
    return radiation, areas


def evaluate(label: str, prediction: np.ndarray, target: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    technology_rows = []
    orientation_rows = []
    channel_rows = []
    for ti, tech in enumerate(TECHNOLOGIES):
        indexes = [oi * 7 + ti for oi in range(5)]
        technology_rows.append({"input": label, "technology": tech, **metrics(prediction[:, :, indexes], target[:, :, indexes])})
    for oi, orientation in enumerate(ORIENTATIONS):
        indexes = list(range(oi * 7, oi * 7 + 7))
        orientation_rows.append({"input": label, "orientation": orientation, **metrics(prediction[:, :, indexes], target[:, :, indexes])})
        for ti, tech in enumerate(TECHNOLOGIES):
            ci = oi * 7 + ti
            channel_rows.append({"input": label, "orientation": orientation, "technology": tech,
                                 "channel": f"{orientation}_{tech}", **metrics(prediction[:, :, ci], target[:, :, ci])})
    return pd.DataFrame(technology_rows), pd.DataFrame(orientation_rows), pd.DataFrame(channel_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-iri", required=True)
    parser.add_argument("--weather-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, help="Optional .npy path for the [8760,5] absorption ratio")
    args = parser.parse_args()
    weather, metadata = _load_weather_object(args.weather_iri, args.weather_dir)
    absorption = pv_absorption_ratio(args.weather_iri, args.weather_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, absorption)
    print(json.dumps({
        **metadata, "hours": len(weather), "absorption_shape": list(absorption.shape),
        "pv_database": str(PV_DB.resolve()), "solar_collector_database": str(SC_DB.resolve()),
        "output": str(args.output.resolve()) if args.output is not None else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
