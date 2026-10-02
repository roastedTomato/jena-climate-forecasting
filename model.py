"""Conv1D + LSTM architecture from the original temperature experiment."""
import keras
from keras import layers


def build_model(num_features, lookback=144, horizon=7):
    model = keras.Sequential([
        layers.Input(shape=(lookback, num_features)),
        layers.Conv1D(32, 5, activation="relu"),
        layers.MaxPooling1D(2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(horizon),
    ])
    model.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])
    return model
