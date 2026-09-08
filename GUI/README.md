# Final Bundle Demand GUI

Double-click `launch_gui.bat`. The GUI uses only sibling Final_bundle resources:

- `data/building_geometry_full.csv`, `data/building_usage.csv`, and `data/weather_{UUID}.csv`
- `Scripts/weather_mapper.py`
- `Scripts/demand_input_agent.py`
- `Scripts/demand_pipline.py`
- models under `Models/Annual`, `Models/Annual pattern`, and `Models/Daily shape`

Enter a building IRI or UUID and click **Load from Final_bundle**. Use **Run annual prediction** for the four annual demand columns, or **Run time-series prediction** for the selected part of the 8,760-hour cascade output.

The result views show Electricity and Heating only. Annual results include target, signed error, and absolute percentage error. Time-series results overlay target and prediction and report MAE, RMSE, and MAPE. Target curves are loaded from `data/targets` through the verified `data/target_mapping.csv` table-UUID to building-UUID mapping.

The internal output mask remains fixed: `ElectricityConsumption` copies `GridConsumption`, while `CoolingConsumption` is zero; Grid and Cooling are omitted from the GUI result views.
