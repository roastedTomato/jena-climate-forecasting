"""Download climate observations and build windows without crossing split boundaries."""
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass
class ClimateData:
    features: np.ndarray
    temperature: np.ndarray
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    temp_col: int
    num_train: int
    num_val: int


def load_climate(csv_path=None):
    """Read a local CSV or download the original Jena dataset via Keras."""
    if csv_path is None:
        directory = Path(keras.utils.get_file(
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            fname="jena_climate_2009_2016.csv.zip", extract=True,
        ))
        files = sorted(directory.rglob("*.csv"))
        if not files:
            raise FileNotFoundError("No climate CSV found in the extracted dataset")
        csv_path = files[0]
    return pd.read_csv(csv_path)


def prepare_data(frame):
    """Split 50/25/25 chronologically; normalise inputs using training rows only."""
    feature_frame = frame.drop(columns=["Date Time"])
    names = feature_frame.columns.tolist()
    features = feature_frame.astype("float32").to_numpy()
    temperature = frame["T (degC)"].astype("float32").to_numpy()
    if not np.isfinite(features).all():
        raise ValueError("Weather features must be finite")
    num_train = int(0.5 * len(features))
    num_val = int(0.25 * len(features))
    if num_train < 2 or num_val < 1:
        raise ValueError("Too few observations for chronological splits")
    mean = features[:num_train].mean(axis=0)
    std = features[:num_train].std(axis=0)
    if np.any(std == 0):
        raise ValueError("A training feature has zero variance")
    return ClimateData((features - mean) / std, temperature, names, mean, std,
                       names.index("T (degC)"), num_train, num_val)


def make_dataset(data, start, end, lookback=144, horizon=7, batch_size=256, shuffle=False):
    """Use only input and target observations entirely within [start, end)."""
    if min(lookback, horizon, batch_size) < 1:
        raise ValueError("Window and batch sizes must be positive")
    if not 0 <= start < end <= len(data.features):
        raise ValueError("Invalid split boundaries")
    stop = end - lookback - horizon + 1
    if stop <= start:
        raise ValueError("Split too short for the requested input and forecast windows")
    features = tf.convert_to_tensor(data.features, dtype=tf.float32)
    targets = tf.convert_to_tensor(data.temperature, dtype=tf.float32)
    ds = tf.data.Dataset.range(start, stop)
    if shuffle:
        ds = ds.shuffle(10000, seed=42)

    def window(i):
        return features[i:i + lookback], targets[i + lookback:i + lookback + horizon]

    return ds.map(window, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_datasets(data, lookback=144, horizon=7, batch_size=256):
    boundaries = [(0, data.num_train), (data.num_train, data.num_train + data.num_val),
                  (data.num_train + data.num_val, len(data.features))]
    return tuple(make_dataset(data, start, end, lookback, horizon, batch_size, shuffle=(i == 0))
                 for i, (start, end) in enumerate(boundaries))
