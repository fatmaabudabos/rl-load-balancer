import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.prepare_data import get_data


def train_and_evaluate():
    """
    Trains a Linear Regression model on the Google Cluster Data.

    Linear Regression finds the best straight-line relationship between
    the 40 input features (10 timesteps x 4 features, flattened) and
    the 4 output values (next cpu, memory, disk_io, max_cpu).

    It does not understand sequences or time — it just sees 40 numbers
    and finds the best linear combination to predict the next 4.
    """

    print("=" * 55)
    print("MODEL — Linear Regression")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    # We use the flat version (samples, 40) — sklearn cannot handle 3D input
    X_train, X_test, y_train, y_test, X_train_flat, X_test_flat = get_data()

    print(f"\nTraining on : {len(X_train_flat):,} samples")
    print(f"Testing on  : {len(X_test_flat):,} samples")
    print(f"Input shape : {X_train_flat.shape}")
    print(f"Target shape: {y_train.shape}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining...")
    model = LinearRegression()
    model.fit(X_train_flat, y_train)
    print("Done.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_flat)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n  MSE : {mse:.6f}  (lower is better)")
    print(f"  MAE : {mae:.6f}  (lower is better)")

    # ── Per-feature breakdown ─────────────────────────────────────────────────
    # Shows how well the model predicts each of the 4 features individually
    feature_names = ["cpu", "memory", "disk_io", "max_cpu"]
    print("\n  Per-feature MAE:")
    for i, name in enumerate(feature_names):
        feature_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        print(f"    {name:<10}: {feature_mae:.6f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/linear_regression.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {save_path}")

    print("\n" + "=" * 55)

    return {"model": "Linear Regression", "mse": mse, "mae": mae}


if __name__ == "__main__":
    train_and_evaluate()
