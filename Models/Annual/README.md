# Annual demand model

## Definition

`gradient_boosting.joblib` predicts seven annual outputs from 35 static building and surrounding features. The first two outputs used by the demand cascade are:

1. annual Grid consumption in `log1p(kWh)`;
2. annual Heating consumption in `log1p(kWh)`.

The physical annual predictions are:

```python
annual_raw = np.maximum(np.expm1(np.clip(model.predict(x), 0.0, 16.1)), 0.0)
annual_grid_kwh = annual_raw[:, 0]
annual_heating_kwh = annual_raw[:, 1]
```

The exact input order is stored in `metrics.json` under `feature_columns`.

## Interface

```python
import json
from pathlib import Path

import joblib
import numpy as np

model_dir = Path("Models/Annual")
features = json.loads((model_dir / "metrics.json").read_text())["feature_columns"]
model = joblib.load(model_dir / "gradient_boosting.joblib")
x = static_frame[features].to_numpy(float)
annual_kwh = np.maximum(np.expm1(np.clip(model.predict(x), 0.0, 16.1)), 0.0)
```

For normal use, call `Scripts.demand_pipline.demand_surrogate`; it connects this model to the annual-pattern models and `Daily shape/cnn.pt`.

## Annual-model error

The deployed model's stored independent test metrics are:

| Series | MAE | RMSE | NMAE | MAPE | R2 |
|---|---:|---:|---:|---:|---:|
| Heating | 3,458.34 kWh | 8,499.40 kWh | 12.61% | 17.48% | 0.9506 |
| Grid | 227.53 kWh | 1,998.05 kWh | 5.32% | 4.29% | 0.9454 |

## Complete three-level cascade error

The complete cascade converts the two predicted temporal levels into annual-conserving hourly weights:

```text
hourly = annual_prediction * (daily_peak_ratio * CNN_shape)
         / sum_over_365x24(daily_peak_ratio * CNN_shape)
```

It was evaluated on 1,891 cache-matched buildings from the same deterministic UUID holdout used for the fixed-pattern comparison, with `cnn.pt` running on an RTX 4070 GPU:

| Series | Hourly MAE | Hourly RMSE | Hourly NMAE |
|---|---:|---:|---:|
| Heating | 0.7803 kWh | 6.6897 kWh | 22.99% |
| Grid | 0.0544 kWh | 0.9784 kWh | 9.40% |

Before annual conservation, NMAE was 24.24% for Heating and 16.10% for Grid. The maximum post-correction difference between each predicted annual value and its summed hourly output was below `7e-10 kWh`. These end-to-end errors include all three model levels. The evaluation cache covers 363 source days, so the reported metrics are not a literal full-365-day test.
