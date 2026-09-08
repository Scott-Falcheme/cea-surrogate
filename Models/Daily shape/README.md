# Daily-shape models

The production cascade now prioritises `lstm.pt`, following the thesis's final
adopted-model subsection. `cnn.pt` remains only as a backwards-compatible
fallback when the selected LSTM checkpoint is absent.

## Definition

`cnn.pt` maps one building-day input of shape `(24, 52)` to 50 outputs:

- outputs `0:24`: Heating shape normalised by that day's Heating maximum;
- outputs `24:48`: Grid shape normalised by that day's Grid maximum;
- outputs `48:50`: legacy direct daily-peak predictions.

The deployed three-level cascade uses only the first 48 outputs. It obtains daily physical scale from:

```text
annual model prediction x annual-pattern model daily-peak ratio
```

The input is standardised with `input_and_scale_scaler.npz`, and shape predictions are clipped to `[0, 1]`.

## Interface

```python
from pathlib import Path

import numpy as np
import torch

from Scripts.demand_pipline import _load_cnn

model, x_mean, x_scale = _load_cnn(Path("Models"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# hourly_features has shape (building_days, 24, 52).
scaled = (hourly_features - x_mean[None]) / x_scale[None]
with torch.no_grad():
    daily_shape = model(torch.from_numpy(scaled).to(device))[:, :48].clamp(0, 1)
```

For normal use, call `Scripts.demand_pipline.demand_surrogate`. The production pipeline is configured to use `cnn.pt` directly and selects CUDA automatically when available.

## CNN checkpoint error

The validation metrics stored inside `cnn.pt` are:

| Series | Normalised-shape MAE | Normalised-shape RMSE | Normalised-shape MAPE |
|---|---:|---:|---:|
| Heating | 0.12835 | 0.18329 | 50.36% |
| Grid | 0.01452 | 0.04740 | 5.71% |

The checkpoint also contains absolute errors based on its legacy direct daily-peak outputs: Heating MAE/RMSE `2.2006/7.9713 kWh`, and Grid MAE/RMSE `0.4204/5.7044 kWh`. Those values are not the deployed three-level cascade error.

## Complete three-level cascade error

Using the specified Annual model, both Annual-pattern models, and this `cnn.pt`, the raw temporal weights are normalised over the year before multiplication by the annual prediction:

```text
hourly = annual_prediction * (daily_peak_ratio * CNN_shape)
         / sum_over_365x24(daily_peak_ratio * CNN_shape)
```

On 1,891 cache-matched buildings from the deterministic UUID holdout:

| Series | Hourly MAE | Hourly RMSE | Hourly NMAE |
|---|---:|---:|---:|
| Heating | 0.7803 kWh | 6.6897 kWh | 22.99% |
| Grid | 0.0544 kWh | 0.9784 kWh | 9.40% |

Before annual conservation, NMAE was 24.24% for Heating and 16.10% for Grid. Annual conservation therefore reduced Heating NMAE by 1.25 percentage points and Grid NMAE by 6.70 percentage points. The cache used for this comparison covers 363 source days.

## RTX 4070 CUDA environment

The HEME environment was verified with PyTorch `2.11.0+cu128`, CUDA `12.8`, and an NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9):

```powershell
F:\ANACONDA\envs\HEME\python.exe -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation with:

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```
