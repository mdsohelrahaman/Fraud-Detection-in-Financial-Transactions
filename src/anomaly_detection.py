"""Isolation Forest and Autoencoder utilities."""
from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow import keras
from tensorflow.keras import layers


def fit_isolation_forest(X: np.ndarray, contamination: float = 0.01) -> IsolationForest:
    """Fit Isolation Forest on prepared features."""
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def isolation_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return higher-is-more-anomalous scores normalized to [0, 1]."""
    raw = -model.score_samples(X)
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo + 1e-9)


def build_autoencoder(input_dim: int) -> keras.Model:
    """Build a compact dense autoencoder for anomaly detection."""
    bottleneck = max(4, min(32, input_dim // 2))
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(max(32, input_dim // 2), activation="relu")(inputs)
    x = layers.Dense(bottleneck, activation="relu")(x)
    x = layers.Dense(max(32, input_dim // 2), activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_scores(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute normalized reconstruction errors."""
    reconstructed = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1)
    lo, hi = errors.min(), errors.max()
    return (errors - lo) / (hi - lo + 1e-9)


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
) -> keras.Model:
    """Train an autoencoder, using early stopping on reconstruction loss."""
    model = build_autoencoder(X_train.shape[1])
    callback = keras.callbacks.EarlyStopping(
        monitor="loss", patience=4, restore_best_weights=True
    )
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[callback],
        verbose=1,
    )
    return model
