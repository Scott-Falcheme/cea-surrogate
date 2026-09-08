"""Cascade annual, annual-pattern, and daily-shape demand models."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from .demand_input_agent import demand_input
except ImportError:
    from demand_input_agent import demand_input


DEMAND_COLUMNS = [
    "GridConsumption",
    "ElectricityConsumption",
    "HeatingConsumption",
    "CoolingConsumption",
]
MODEL_ROOT = Path(__file__).resolve().parents[1] / "Models"


class DailyShapeCNN(nn.Module):
    """Architecture stored in Final_bundle/Models/Daily shape/cnn.pt."""

    def __init__(self, width: int = 128, outputs: int = 50) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(52, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(width * 24, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.transpose(1, 2))


class DailyShapeLSTM(nn.Module):
    """Architecture selected for the final 24-hour shape model."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.rnn = nn.LSTM(52, width, 2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 24, 256),
            nn.ReLU(),
            nn.Linear(256, 48),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(values)[0])


def _enable_sklearn_15_huber_compatibility() -> None:
    """Allow sklearn 1.8 to read the pattern bundles saved by sklearn 1.5."""
    try:
        import sklearn._loss._loss as cy_loss

        if not hasattr(cy_loss, "__pyx_unpickle_CyHuberLoss"):
            cy_loss.__pyx_unpickle_CyHuberLoss = (
                lambda loss_type, _checksum, state: loss_type(*state)
            )
        if not hasattr(cy_loss, "__pyx_unpickle_CyHalfSquaredError"):
            cy_loss.__pyx_unpickle_CyHalfSquaredError = (
                lambda loss_type, _checksum, state: loss_type(*state)
            )
    except ImportError:
        pass


@lru_cache(maxsize=None)
def _load_joblib(path: Path):
    """Load an immutable deployment artifact once per Python process."""
    _enable_sklearn_15_huber_compatibility()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        return joblib.load(path)


@lru_cache(maxsize=None)
def _load_pattern(path: Path) -> dict:
    _enable_sklearn_15_huber_compatibility()
    return _load_joblib(path)


def _pattern(bundle: dict, static_input: pd.DataFrame) -> np.ndarray:
    missing = [name for name in bundle["static_features"] if name not in static_input.index]
    if missing:
        raise ValueError(f"Static input missing pattern features: {missing}")
    values = static_input.loc[bundle["static_features"]].T.to_numpy(dtype=float)
    coefficients = bundle["regressor"].predict(bundle["input_scaler"].transform(values))
    return np.maximum(bundle["pca"].inverse_transform(coefficients), 0.0)


def _annual_shape(bundle: dict, hourly_input: np.ndarray) -> np.ndarray:
    """Predict the 365 Heating/Grid daily-peak ratios from hourly inputs."""
    daily_input = hourly_input.mean(axis=1).reshape(1, -1)
    coefficients = bundle["model"].predict(bundle["scaler"].transform(daily_input))
    prediction = bundle["pca"].inverse_transform(coefficients).reshape(365, 2)
    return np.maximum(prediction / 365.0, 0.0)


def _annual_shape_batch(bundle: dict, hourly_input: np.ndarray) -> np.ndarray:
    """Vectorised annual-pattern prediction for [building, day, hour, feature]."""
    daily_input = hourly_input.mean(axis=2).reshape(len(hourly_input), -1)
    coefficients = bundle["model"].predict(bundle["scaler"].transform(daily_input))
    prediction = bundle["pca"].inverse_transform(coefficients).reshape(-1, 365, 2)
    return np.maximum(prediction / 365.0, 0.0)


def _annual_conserving_profile(
    annual_kwh: float, daily_pattern: np.ndarray, hourly_shape: np.ndarray
) -> np.ndarray:
    """Distribute an annual prediction over 365 x 24 weights without changing its total."""
    weight = daily_pattern[:, None] * hourly_shape
    total = float(weight.sum())
    return annual_kwh * weight / total if total > 0.0 else np.zeros_like(weight)


def _load_cnn(model_root: Path) -> tuple[DailyShapeCNN, np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        model_root / "Daily shape" / "cnn.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    width = int(state["net.0.weight"].shape[0])
    outputs = int(state["net.7.weight"].shape[0])
    model = DailyShapeCNN(width=width, outputs=outputs)
    model.load_state_dict(state)
    model.eval()
    with np.load(model_root / "Daily shape" / "input_and_scale_scaler.npz") as scaler:
        x_mean = scaler["x_mean"].astype("float32")
        x_scale = scaler["x_scale"].astype("float32")
    return model, x_mean, x_scale


@lru_cache(maxsize=None)
def _load_daily_shape(model_root: Path):
    """Load the selected LSTM, retaining the former CNN as a fallback."""
    lstm_path = model_root / "Daily shape" / "lstm.pt"
    if lstm_path.exists():
        checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
        width = int(checkpoint["state_dict"]["rnn.weight_ih_l0"].shape[0] // 4)
        model = DailyShapeLSTM(width)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x_mean = np.asarray(checkpoint["mean"], dtype="float32")
        x_scale = np.asarray(checkpoint["std"], dtype="float32")
    else:
        model, x_mean, x_scale = _load_cnn(model_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), x_mean, x_scale, device


def _annual_predictions(model_root: Path, static_input: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict Grid with gradient boosting and Heating with random forest."""
    annual_dir = model_root / "Annual"
    grid_path = annual_dir / "grid_gradient_boosting_seed_001.joblib"
    heating_path = annual_dir / "heating_random_forest_seed_001.joblib"
    values = static_input.T.to_numpy(dtype=float)
    if grid_path.exists() and heating_path.exists():
        grid_model = _load_joblib(grid_path)[0]
        heating_model = _load_joblib(heating_path)[1]
        grid = np.maximum(np.expm1(np.clip(grid_model.predict(values), 0.0, 16.1)), 0.0)
        heating = np.maximum(np.expm1(np.clip(heating_model.predict(values), 0.0, 16.1)), 0.0)
        return grid, heating

    annual_model = _load_joblib(annual_dir / "gradient_boosting.joblib")
    annual_log = annual_model.predict(values)
    annual_raw = np.maximum(np.expm1(np.clip(annual_log, 0.0, 16.1)), 0.0)
    if annual_raw.shape[1] < 2:
        raise ValueError("Annual model must output Grid and Heating in its first two columns")
    return annual_raw[:, 0], annual_raw[:, 1]


def demand_surrogate(
    inputs: tuple[pd.DataFrame, pd.DataFrame],
    model_root: str | Path = MODEL_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict four annual and four hourly demand columns.

    ``inputs`` is the tuple returned by :func:`demand_input`.  Grid is copied
    to Electricity and Cooling is masked to zero.  Heating and Grid hourly
    values use the deployable cascade:

        annual prediction x predicted daily-peak pattern x LSTM daily shape
    """
    static_input, timeseries_input = inputs
    model_root = Path(model_root)
    buildings = list(static_input.columns)
    if not buildings:
        raise ValueError("static_input contains no buildings")

    try:
        annual_grid, annual_heating = _annual_predictions(model_root, static_input)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the annual model. This bundle was verified with the "
            "HEME environment (scikit-learn 1.8, NumPy 2.x)."
        ) from exc
    static_output = pd.DataFrame(
        {
            "GridConsumption": annual_grid,
            "ElectricityConsumption": annual_grid.copy(),
            "HeatingConsumption": annual_heating,
            "CoolingConsumption": np.zeros(len(buildings)),
        },
        index=pd.Index(buildings, name="building_IRI"),
    )

    selected_pattern_path = model_root / "Annual pattern" / "annual_shape_gradient_boosting_seed_42.joblib"
    selected_pattern = _load_pattern(selected_pattern_path) if selected_pattern_path.exists() else None
    if selected_pattern is None:
        grid_bundle = _load_pattern(model_root / "Annual pattern" / "grid_pattern_gradient_boosting.joblib")
        heating_bundle = _load_pattern(model_root / "Annual pattern" / "heating_pattern_gradient_boosting.joblib")
        grid_pattern = _pattern(grid_bundle, static_input)
        heating_pattern = _pattern(heating_bundle, static_input)

    shape_model, x_mean, x_scale, shape_device = _load_daily_shape(model_root)
    hourly_by_building: list[np.ndarray] = []
    for building in buildings:
        block = timeseries_input.loc[building]
        if block.shape != (52, 8760):
            raise ValueError(f"Time-series input for {building!r} has shape {block.shape}, expected (52, 8760)")
        hourly_by_building.append(
            block.to_numpy(dtype="float32").T.reshape(365, 24, 52)
        )
    hourly_batch = np.stack(hourly_by_building, axis=0)
    if selected_pattern is not None:
        pattern_batch = _annual_shape_batch(selected_pattern, hourly_batch)
        heating_daily_patterns = pattern_batch[:, :, 0]
        grid_daily_patterns = pattern_batch[:, :, 1]
    else:
        heating_daily_patterns = heating_pattern
        grid_daily_patterns = grid_pattern

    scaled = (hourly_batch - x_mean[None, None, :, :]) / x_scale[None, None, :, :]
    with torch.no_grad():
        raw_shape_batch = (
            shape_model(
                torch.from_numpy(scaled.reshape(-1, 24, 52)).to(shape_device)
            )[:, :48]
            .clamp(0.0, 1.0)
            .cpu().numpy()
            .reshape(len(buildings), 365, 48)
        )

    output_blocks: list[pd.DataFrame] = []
    for building_index, building in enumerate(buildings):
        block = timeseries_input.loc[building]
        heating_daily_pattern = heating_daily_patterns[building_index]
        grid_daily_pattern = grid_daily_patterns[building_index]
        raw_shape = raw_shape_batch[building_index]

        heating = _annual_conserving_profile(
            annual_heating[building_index], heating_daily_pattern, raw_shape[:, :24]
        ).reshape(-1)
        grid = _annual_conserving_profile(
            annual_grid[building_index], grid_daily_pattern, raw_shape[:, 24:48]
        ).reshape(-1)
        predicted = pd.DataFrame(
            {
                "GridConsumption": grid,
                "ElectricityConsumption": grid.copy(),
                "HeatingConsumption": heating,
                "CoolingConsumption": np.zeros(8760),
            },
            index=pd.DatetimeIndex(block.columns, name="time"),
        )
        if len(buildings) > 1:
            predicted.columns = pd.MultiIndex.from_product(
                [[building], DEMAND_COLUMNS], names=["building_IRI", "demand"]
            )
        output_blocks.append(predicted)

    timeseries_output = pd.concat(output_blocks, axis=1)
    if len(buildings) == 1:
        timeseries_output = timeseries_output[DEMAND_COLUMNS]
    if not np.array_equal(
        timeseries_output.xs("GridConsumption", axis=1, level="demand").to_numpy()
        if isinstance(timeseries_output.columns, pd.MultiIndex)
        else timeseries_output[["GridConsumption"]].to_numpy(),
        timeseries_output.xs("ElectricityConsumption", axis=1, level="demand").to_numpy()
        if isinstance(timeseries_output.columns, pd.MultiIndex)
        else timeseries_output[["ElectricityConsumption"]].to_numpy(),
    ):
        raise RuntimeError("Grid-to-Electricity copy mask failed")
    return static_output[DEMAND_COLUMNS], timeseries_output


__all__ = ["demand_input", "demand_surrogate", "DEMAND_COLUMNS"]
