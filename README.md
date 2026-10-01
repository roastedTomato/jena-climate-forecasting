# Jena Climate Forecasting

Multivariate temperature forecasting with TensorFlow and Keras, combining Conv1D feature extraction with an LSTM. The model uses 144 past observations to predict the next 7 temperature observations.

## Pipeline

- Download the Jena climate 2009–2016 dataset through Keras.
- Use numeric weather variables as inputs and `T (degC)` as the target.
- Split observations chronologically: 50% training, 25% validation, 25% test.
- Normalise inputs using training statistics only; keep targets in degrees Celsius.
- Build windows entirely within each split; shuffle only training windows.
- Compare the model against a persistence baseline that repeats the last temperature.
- Display learning curves and an example forecast.

## Model

| Setting | Value |
| --- | --- |
| Lookback | 144 observations |
| Forecast horizon | 7 observations |
| Conv1D | 32 filters, kernel size 5, ReLU |
| MaxPooling1D | Pool size 2 |
| LSTM | 32 units |
| Dense hidden layer | 32 units, ReLU |
| Output | 7 temperature values |
| Optimiser / loss | RMSprop / MAE |
| Batch size | 256 |
| Maximum epochs | 20 |
| Early stopping | Validation MAE, patience 3, restore best weights |

## Historical results

| Method | Test MAE |
| --- | --- |
| Persistence baseline | 0.506°C |
| Conv1D + LSTM | 0.452°C |

These rounded values come from the original saved coursework outputs. They represent about a 10.7% reduction in MAE relative to the baseline. Training stopped after seven epochs. Full training has not been repeated for this repository; no trained weights are included. Provenance is recorded in `results/original_run.json`.

## Run locally

Python 3.13 is the version in the inspected local environment. Dependencies are pinned to that environment, not asserted to be the versions used in the historical run.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab jena_climate_forecasting.ipynb
```

On Windows, use `.venv\Scripts\activate` to activate the environment. Run the notebook in order. The first run requires internet access to download the dataset. Full training takes time and memory; the dataset itself is not committed.

## Files

- `jena_climate_forecasting.ipynb`: data preparation, baseline, model, evaluation and plots.
- `requirements.txt`: inspected environment versions.
- `results/original_run.json`: historical result provenance.

## Limitations

The forecast horizon is **seven observations, not seven days**. Historical accuracy may vary on rerun; the original experiment did not set a global random seed. Overlapping windows produce correlated examples. Aggregate MAE does not establish performance for each forecast step or season. The notebook does not include a deployment service or long-term climate prediction.

## Provenance and AI assistance

Originally developed for DATAX504 Assignment 5. The assignment disclosed Codex GPT-5.5 assistance with code suggestions and explanations, with author review and execution. Codex assisted in preparing this standalone version, fixing the extraction path for the inspected Keras version, and adding documentation and visualisations. The model and evaluation protocol are preserved. The unrelated news-classification route and submission instructions are excluded. Notebook outputs are cleared for a consistent fresh run.
