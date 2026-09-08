"""Final_bundle GUI for annual and hourly four-demand prediction."""

from __future__ import annotations

import calendar
import re
import sys
from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "Scripts"
DATA = ROOT / "data"
sys.path.insert(0, str(SCRIPTS))

from demand_input_agent import GEOMETRY_FEATURES, SURROUNDING_FEATURES, USAGE_FEATURES, demand_input
from demand_pipline import DEMAND_COLUMNS, demand_surrogate
from weather_mapper import mapweather


GEOMETRY_CSV = DATA / "building_geometry_full.csv"
USAGE_CSV = DATA / "building_usage.csv"
TARGET_MAPPING_CSV = DATA / "target_mapping.csv"
ADMINER_DIR = Path(
    r"C:\Users\ROG\OneDrive - University of Cambridge\Study\THESIS"
    r"\City data\Kaiserslautern\adminer"
)
RESULT_DEMANDS = ["ElectricityConsumption", "HeatingConsumption"]
DYNAMIC_FEATURES = [
    "w_AirTemperature", "w_AtmosphericPressure", "w_CloudCover", "w_DewPoint",
    "w_DiffuseHorizontalIrradiance", "w_DirectNormalIrradiance", "w_Rainfall",
    "w_RelativeHumidity", "w_Snowfall", "w_WindDirection", "w_WindSpeed",
    "h_sin", "h_cos", "d_sin", "d_cos", "s_heating", "s_cooling",
]


class WeatherTable(ttk.Frame):
    """Scrollable, read-only 17-by-hour input preview."""

    label_width, cell_width, row_height, header_height = 225, 105, 28, 34

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.times = pd.DatetimeIndex([])
        self.values = np.empty((17, 0))
        self.offset = 0
        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.scroll)
        self.canvas.pack(fill="both", expand=True)
        self.hbar.pack(fill="x")
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def set_data(self, times: pd.DatetimeIndex, values: np.ndarray) -> None:
        self.times = pd.DatetimeIndex(times)
        self.values = np.asarray(values, float)
        self.offset = 0
        self._sync_scroll()
        self.redraw()

    def _visible_count(self) -> int:
        return max(1, int((self.canvas.winfo_width() - self.label_width) / self.cell_width) + 2)

    def _sync_scroll(self) -> None:
        count, visible = len(self.times), self._visible_count()
        self.hbar.set(self.offset / max(count, 1), min(1.0, (self.offset + visible) / max(count, 1)))

    def scroll(self, *args: str) -> None:
        count = len(self.times)
        if not count:
            return
        if args[0] == "moveto":
            self.offset = int(float(args[1]) * count)
        else:
            step = 1 if args[2] == "units" else self._visible_count()
            self.offset += int(args[1]) * step
        self.offset = max(0, min(count - 1, self.offset))
        self._sync_scroll()
        self.redraw()

    def redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        end = min(len(self.times), self.offset + self._visible_count())
        canvas.create_rectangle(0, 0, self.label_width, self.header_height, fill="#e8edf3", outline="#aab2bd")
        canvas.create_text(8, self.header_height / 2, text="feature / timestamp", anchor="w", font=("TkDefaultFont", 9, "bold"))
        for display_column, source_column in enumerate(range(self.offset, end)):
            x = self.label_width + display_column * self.cell_width
            canvas.create_rectangle(x, 0, x + self.cell_width, self.header_height, fill="#e8edf3", outline="#aab2bd")
            canvas.create_text(x + self.cell_width / 2, self.header_height / 2, text=self.times[source_column].strftime("%m-%d\n%H:00"), justify="center", font=("TkDefaultFont", 8))
        for row, name in enumerate(DYNAMIC_FEATURES):
            y = self.header_height + row * self.row_height
            canvas.create_rectangle(0, y, self.label_width, y + self.row_height, fill="#f5f7fa", outline="#c8ced7")
            canvas.create_text(8, y + self.row_height / 2, text=name, anchor="w")
            for display_column, source_column in enumerate(range(self.offset, end)):
                x = self.label_width + display_column * self.cell_width
                canvas.create_rectangle(x, y, x + self.cell_width, y + self.row_height, fill="white", outline="#d5d9df")
                canvas.create_text(x + self.cell_width / 2, y + self.row_height / 2, text=f"{self.values[row, source_column]:.4g}")
        canvas.configure(scrollregion=(0, 0, self.label_width + self._visible_count() * self.cell_width, self.header_height + 17 * self.row_height))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Final Bundle Demand Surrogate")
        self.geometry("1160x780")
        self.minsize(900, 650)
        self.iri = tk.StringVar(value="Select an Adminer building output CSV")
        self.adminer_path: Path | None = None
        self.station = tk.StringVar(value="Not loaded")
        self.start_month, self.start_day = tk.IntVar(value=1), tk.IntVar(value=1)
        self.end_month, self.end_day = tk.IntVar(value=1), tk.IntVar(value=2)
        self.static_vars = {name: tk.StringVar(value="0") for name in GEOMETRY_FEATURES + SURROUNDING_FEATURES}
        self.usage = tk.StringVar(value="Not loaded")
        self.inputs = self.annual_output = self.hourly_output = None
        self.target_hourly = None
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Building Adminer output:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.iri, state="readonly").grid(row=0, column=1, columnspan=4, sticky="ew", padx=6)
        ttk.Button(top, text="Load from file", command=self.load_from_file).grid(row=0, column=5)
        ttk.Label(top, text="Weather station").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Label(top, textvariable=self.station).grid(row=1, column=1, columnspan=5, sticky="w", padx=6)
        ttk.Label(top, text="Display period (2024, Feb 29 omitted)").grid(row=2, column=0, sticky="w")
        period = ttk.Frame(top)
        period.grid(row=2, column=1, columnspan=4, sticky="w")
        ttk.Label(period, text="Start").pack(side="left")
        self.start_month_box = ttk.Combobox(period, textvariable=self.start_month, values=list(range(1, 13)), state="readonly", width=4)
        self.start_month_box.pack(side="left", padx=(4, 2))
        self.start_day_box = ttk.Combobox(period, textvariable=self.start_day, state="readonly", width=4)
        self.start_day_box.pack(side="left", padx=(0, 14))
        ttk.Label(period, text="End").pack(side="left")
        self.end_month_box = ttk.Combobox(period, textvariable=self.end_month, values=list(range(1, 13)), state="readonly", width=4)
        self.end_month_box.pack(side="left", padx=(4, 2))
        self.end_day_box = ttk.Combobox(period, textvariable=self.end_day, state="readonly", width=4)
        self.end_day_box.pack(side="left")
        self.start_month_box.bind("<<ComboboxSelected>>", lambda _event: self._update_days())
        self.end_month_box.bind("<<ComboboxSelected>>", lambda _event: self._update_days())
        self._update_days()
        ttk.Button(top, text="Load period", command=self.load_period).grid(row=2, column=5)
        top.columnconfigure(1, weight=1)

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10)
        for title, names in (("Geometry (5)", GEOMETRY_FEATURES), ("Surrounding (10)", SURROUNDING_FEATURES)):
            frame = ttk.Frame(tabs, padding=12)
            tabs.add(frame, text=title)
            for index, name in enumerate(names):
                row, column = divmod(index, 2)
                ttk.Label(frame, text=name).grid(row=row, column=column * 2, sticky="w", padx=5, pady=4)
                ttk.Entry(frame, textvariable=self.static_vars[name], state="readonly").grid(row=row, column=column * 2 + 1, sticky="ew", padx=5)
            frame.columnconfigure(1, weight=1)
            frame.columnconfigure(3, weight=1)
        usage_frame = ttk.Frame(tabs, padding=20)
        tabs.add(usage_frame, text="Usage (20 shares)")
        ttk.Label(usage_frame, text="Dominant usage:").pack(side="left")
        ttk.Label(usage_frame, textvariable=self.usage).pack(side="left", padx=8)
        weather_frame = ttk.Frame(tabs, padding=6)
        tabs.add(weather_frame, text="Weather + temporal (17 x hours)")
        self.weather_table = WeatherTable(weather_frame)
        self.weather_table.pack(fill="both", expand=True)

        bar = ttk.Frame(self, padding=10)
        bar.pack(fill="x")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status).pack(side="left")
        ttk.Button(bar, text="Run annual prediction", command=self.run_annual).pack(side="right")
        ttk.Button(bar, text="Run time-series prediction", command=self.run_timeseries).pack(side="right", padx=8)

    def _update_days(self) -> None:
        for month, day, box in ((self.start_month, self.start_day, self.start_day_box), (self.end_month, self.end_day, self.end_day_box)):
            maximum = calendar.monthrange(2024, int(month.get()))[1]
            box.configure(values=list(range(1, maximum + 1)))
            if int(day.get()) > maximum:
                day.set(maximum)

    def _period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(date(2024, self.start_month.get(), self.start_day.get()), tz="UTC")
        end = pd.Timestamp(date(2024, self.end_month.get(), self.end_day.get()), tz="UTC") + pd.Timedelta(hours=23)
        if end < start:
            raise ValueError("End date must not be before start date")
        return start, end

    def load_from_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select building Adminer output",
            initialdir=str(ADMINER_DIR if ADMINER_DIR.is_dir() else Path.home()),
            filetypes=(("Building CSV", "building_*.csv"), ("CSV files", "*.csv")),
        )
        if not selected:
            return
        self.load_building(Path(selected))

    def load_building(self, adminer_path: Path) -> None:
        try:
            self.status.set("Mapping Adminer output to building IRI...")
            self.update_idletasks()
            adminer_path = adminer_path.resolve()
            table_uuid, building_iri = self._iri_from_adminer_file(adminer_path)
            self.adminer_path = adminer_path
            self.iri.set(building_iri)
            self.target_hourly = self._load_targets_from_file(adminer_path)

            station = mapweather(building_iri, GEOMETRY_CSV)
            weather_uuid = self._building_uuid(station)
            weather_path = DATA / f"weather_{weather_uuid}.csv"
            if not weather_path.is_file():
                raise FileNotFoundError(
                    f"Nearest-station weather file is not bundled: {weather_path}"
                )
            self.inputs = demand_input(
                [building_iri], station, str(GEOMETRY_CSV), str(USAGE_CSV)
            )
            self.annual_output = self.hourly_output = None
            static, _hourly = self.inputs
            for name in self.static_vars:
                self.static_vars[name].set(f"{float(static.loc[name].iloc[0]):.8g}")
            usage_values = static.loc[USAGE_FEATURES].iloc[:, 0].astype(float)
            self.usage.set(usage_values.idxmax().removeprefix("usage_"))
            self.station.set(station)
            self.load_period()
            self.status.set(
                f"Loaded table {table_uuid}; target {len(self.target_hourly)} rows; "
                f"weather {weather_path.name}"
            )
            messagebox.showinfo(
                "Input ready",
                "Adminer target, building IRI, nearest weather, and Final_bundle "
                "model inputs loaded successfully.",
            )
        except Exception as exc:
            self.inputs = self.annual_output = self.hourly_output = None
            self.target_hourly = None
            self.status.set("Input failed")
            messagebox.showerror("Input error", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _building_uuid(value: str) -> str:
        match = re.search(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", str(value)
        )
        if not match:
            raise ValueError(f"Building identifier has no UUID: {value!r}")
        return match.group(0).lower()

    def _iri_from_adminer_file(self, target_path: Path) -> tuple[str, str]:
        """Map a selected Adminer table UUID to its canonical building IRI."""
        if not target_path.is_file():
            raise FileNotFoundError(f"Adminer output not found: {target_path}")
        match = re.fullmatch(
            r"building_([0-9a-fA-F-]{36})\.csv", target_path.name, flags=re.IGNORECASE
        )
        if not match:
            raise ValueError(
                "Expected an Adminer filename like building_<timeseries-table-UUID>.csv"
            )
        table_uuid = self._building_uuid(match.group(1))
        if not TARGET_MAPPING_CSV.is_file():
            raise FileNotFoundError(f"Target mapping not found: {TARGET_MAPPING_CSV}")
        mapping = pd.read_csv(
            TARGET_MAPPING_CSV,
            usecols=["timeseries_table", "building_uuid"],
            dtype=str,
        )
        mapping["timeseries_table"] = mapping["timeseries_table"].str.lower()
        hit = mapping[mapping["timeseries_table"] == table_uuid]
        if hit.empty:
            raise KeyError(
                f"Timeseries table {table_uuid} is not present in {TARGET_MAPPING_CSV.name}"
            )
        building_uuids = hit["building_uuid"].dropna().str.lower().unique()
        if len(building_uuids) != 1:
            raise ValueError(
                f"Timeseries table {table_uuid} maps to {len(building_uuids)} buildings"
            )
        building_uuid = self._building_uuid(building_uuids[0])
        return table_uuid, f"https://theworldavatar.io/kg/Building/{building_uuid}"

    @staticmethod
    def _load_targets_from_file(target_path: Path) -> pd.DataFrame:
        """Read Electricity/Heating targets directly from the selected Adminer CSV."""
        required = ["time", "column2", "column3"]
        raw = pd.read_csv(target_path, usecols=required)
        raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["time"]).drop_duplicates("time", keep="last").sort_values("time")
        if raw.empty:
            raise ValueError(f"Adminer output has no valid timestamp rows: {target_path}")
        result = raw.set_index("time").rename(
            columns={"column2": "ElectricityConsumption", "column3": "HeatingConsumption"}
        ).astype(float)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError(f"Adminer target contains non-finite demand values: {target_path}")
        return result

    def load_period(self) -> None:
        if self.inputs is None:
            return
        start, end = self._period()
        building = self.inputs[0].columns[0]
        hourly = self.inputs[1].loc[building]
        selected = hourly.loc[:, (hourly.columns >= start) & (hourly.columns <= end)]
        self.weather_table.set_data(selected.columns, selected.loc[DYNAMIC_FEATURES].to_numpy(float))
        self.status.set(f"Displaying {selected.shape[1]} model-input hours")

    def _predict(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Load a building Adminer output file first")
        if self.annual_output is None or self.hourly_output is None:
            self.status.set("Running selected annual + shape cascade...")
            self.update_idletasks()
            self.annual_output, self.hourly_output = demand_surrogate(self.inputs)

    def run_annual(self) -> None:
        try:
            self._predict()
            prediction = self.annual_output.iloc[0]
            target = self.target_hourly[RESULT_DEMANDS].sum() if self.target_hourly is not None else None
            window = tk.Toplevel(self)
            window.title("Annual demand prediction")
            window.geometry("820x260")
            frame = ttk.Frame(window, padding=18)
            frame.pack(fill="both", expand=True)
            ttk.Label(frame, text="Annual demand: prediction vs target", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 10))
            table = ttk.Treeview(frame, columns=("demand", "prediction", "target", "error", "ape"), show="headings", height=2)
            headings = {"demand": "Demand", "prediction": "Prediction (kWh)", "target": "Target (kWh)", "error": "Error (kWh)", "ape": "Absolute % error"}
            widths = {"demand": 190, "prediction": 135, "target": 135, "error": 125, "ape": 125}
            for column in table["columns"]:
                table.heading(column, text=headings[column])
                table.column(column, width=widths[column], anchor="e" if column != "demand" else "w")
            for name in RESULT_DEMANDS:
                predicted = float(prediction[name])
                true = float(target[name]) if target is not None else None
                error = predicted - true if true is not None else None
                ape = abs(error) / abs(true) * 100 if error is not None and true != 0 else None
                table.insert("", "end", values=(name, f"{predicted:,.3f}", f"{true:,.3f}" if true is not None else "—", f"{error:,.3f}" if error is not None else "—", f"{ape:.2f}%" if ape is not None else "—"))
            table.pack(fill="x")
            ttk.Label(frame, text="Grid and Cooling are intentionally omitted from this result view.", foreground="#555").pack(anchor="w", pady=(12, 0))
            self.status.set("Annual prediction complete")
        except Exception as exc:
            messagebox.showerror("Annual prediction error", f"{type(exc).__name__}: {exc}")

    def run_timeseries(self) -> None:
        try:
            self._predict()
            start, end = self._period()
            predicted = self.hourly_output.loc[(self.hourly_output.index >= start) & (self.hourly_output.index <= end)]
            if predicted.empty:
                raise ValueError("Selected period contains no model timestamps")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure

            window = tk.Toplevel(self)
            window.title("Four-demand time-series prediction")
            window.geometry("1120x720")
            figure = Figure(figsize=(11, 7), dpi=100)
            axes = figure.subplots(2, 1, sharex=True)
            notes = []
            for axis, name, title in zip(axes, RESULT_DEMANDS, ("Electricity consumption", "Heating consumption")):
                axis.plot(predicted.index, predicted[name], label="Prediction", lw=1.2)
                if self.target_hourly is not None:
                    target = self.target_hourly[name].reindex(predicted.index)
                    axis.plot(target.index, target, label="Target", lw=1.3)
                    valid = target.notna() & np.isfinite(predicted[name])
                    if valid.any():
                        true_values = target[valid].to_numpy(float)
                        pred_values = predicted.loc[valid, name].to_numpy(float)
                        error = pred_values - true_values
                        mae = float(np.mean(np.abs(error)))
                        rmse = float(np.sqrt(np.mean(error**2)))
                        nonzero = np.abs(true_values) > 1e-8
                        mape = float(np.mean(np.abs(error[nonzero] / true_values[nonzero])) * 100) if nonzero.any() else float("nan")
                        notes.append(f"{title}: MAE={mae:.3f} kWh, RMSE={rmse:.3f} kWh, MAPE={mape:.2f}%, valid={int(valid.sum())}/{len(valid)}")
                axis.set_title(title)
                axis.set_ylabel("kWh")
                axis.grid(alpha=0.25)
                axis.legend()
            axes[1].set_xlabel("Time (UTC)")
            figure.tight_layout(rect=(0, 0.075, 1, 1))
            figure.text(0.01, 0.01, "\n".join(notes) if notes else "Target is not bundled for this building; prediction only.", fontsize=9)
            canvas = FigureCanvasTkAgg(figure, master=window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(canvas, window).update()
            self.status.set(f"Predicted and displayed {len(predicted)} hourly rows")
        except Exception as exc:
            messagebox.showerror("Time-series prediction error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    App().mainloop()
