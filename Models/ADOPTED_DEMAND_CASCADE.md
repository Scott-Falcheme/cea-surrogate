# Adopted demand cascade

The deployed configuration follows the thesis subsection **Adopted surrogate model**:

- annual Grid: Gradient Boosting, seed 1;
- annual Heating: Random Forest, seed 1;
- annual daily-peak shape: Gradient Boosting, seed 42;
- daily 24-hour shape: LSTM, seed 42;
- final 8,760-hour output: normalised so its sum equals the predicted annual total.

`Scripts.demand_pipline.demand_surrogate` keeps the existing GUI/API input and output contract. The former weights remain in place as fallbacks, while the selected files listed in `deployment_manifest.json` take priority.

The reproducible strict held-out evaluation and paste-ready table are in `End-to-end evaluation`. It uses the 202 buildings held out by both the annual seed-1 split and the annual/daily-shape seed-42 split. The annual features are the current 35-feature mapping used by the GUI, and 29 February is removed to form 365 days.

The thesis conclusion still calls the daily model a CNN. That sentence conflicts with the later explicit adopted-model subsection, which selects the LSTM; deployment follows the explicit adopted-model subsection.
