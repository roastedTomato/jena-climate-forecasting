"""Shared training/evaluation functions and command-line entry point."""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from data import load_climate, prepare_data, build_datasets
from model import build_model


def evaluate_naive(dataset, data, horizon=7):
    """Persistence MAE in Celsius, weighted over all forecast values."""
    total_error, total_values = 0.0, 0
    for x, y in dataset:
        last_temp = x[:, -1, data.temp_col] * data.std[data.temp_col] + data.mean[data.temp_col]
        predicted = tf.repeat(last_temp[:, None], repeats=horizon, axis=1)
        total_error += float(tf.reduce_sum(tf.abs(y - predicted)).numpy())
        total_values += int(np.prod(y.shape))
    if not total_values:
        raise ValueError("Cannot evaluate an empty dataset")
    return total_error / total_values


def train_model(model, train_ds, val_ds, epochs=20):
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=3, restore_best_weights=True)
    ])


def evaluate_model(model, test_ds):
    return {key: float(value) for key, value in model.evaluate(test_ds, return_dict=True).items()}


def main():
    parser = argparse.ArgumentParser(description="Train a seven-step Jena temperature forecaster")
    parser.add_argument("--csv", type=Path, help="Local CSV; otherwise download the dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    data = prepare_data(load_climate(args.csv))
    train_ds, val_ds, test_ds = build_datasets(data, batch_size=args.batch_size)
    baseline = evaluate_naive(test_ds, data)
    model = build_model(data.features.shape[1])
    history = train_model(model, train_ds, val_ds, args.epochs)
    metrics = evaluate_model(model, test_ds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model_test_metrics": metrics, "persistence_test_mae_celsius": baseline,
              "lookback": 144, "horizon": 7, "batch_size": args.batch_size,
              "epochs_completed": len(history.epoch), "history": history.history}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Model test MAE: {metrics['mae']:.3f} °C; persistence: {baseline:.3f} °C")


if __name__ == "__main__":
    main()
