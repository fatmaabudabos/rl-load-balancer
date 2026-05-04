import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.prepare_data import get_data

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 200_000   # same as Random Forest — keeps training time reasonable
N_ESTIMATORS = 100       # number of boosting rounds (trees)
LEARNING_RATE = 0.1      # how much each tree corrects the previous one
MAX_DEPTH    = 6         # how deep each tree can grow


def train_and_evaluate():
    """
    Trains an XGBoost model on the Google Cluster Data.

    XGBoost builds trees sequentially — each tree corrects the mistakes
    of the previous one. This makes it more accurate than Random Forest
    in most cases because it actively targets errors rather than
    averaging independent trees.

    Like Linear Regression and Random Forest, it uses the flat version
    of the data (samples, 40) — it does not understand sequences.

    XGBRegressor only supports single output by default.
    We wrap it in MultiOutputRegressor to predict all 4 features
    (cpu, memory, disk_io, max_cpu) at once.
    """

    print("=" * 55)
    print("MODEL — XGBoost")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, X_train_flat, X_test_flat = get_data()

    # ── Sample for training ───────────────────────────────────────────────────
    # XGBoost is fast but 1.9M samples is still too much for a laptop CPU.
    # 200k gives a good result in reasonable time.
    rng          = np.random.default_rng(42)
    idx          = rng.choice(len(X_train_flat), size=SAMPLE_SIZE, replace=False)
    X_train_flat = X_train_flat[idx]
    y_train      = y_train[idx]

    print(f"\nTraining on : {len(X_train_flat):,} samples (sampled from full set)")
    print(f"Testing on  : {len(X_test_flat):,} samples (full test set)")
    print(f"Input shape : {X_train_flat.shape}")
    print(f"Target shape: {y_train.shape}")

    # ── Train ─────────────────────────────────────────────────────────────────
    # MultiOutputRegressor trains one XGBoost model per output feature.
    # So it trains 4 models internally: one for cpu, one for memory,
    # one for disk_io, one for max_cpu.
    print("\nTraining...")
    base_model = XGBRegressor(
        n_estimators  = N_ESTIMATORS,
        learning_rate = LEARNING_RATE,
        max_depth     = MAX_DEPTH,
        n_jobs        = -1,          # use all CPU cores
        random_state  = 42,
        verbosity     = 0,           # suppress XGBoost internal logs
    )
    model = MultiOutputRegressor(base_model, n_jobs=-1)
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
    feature_names = ["cpu", "memory", "disk_io", "max_cpu"]
    print("\n  Per-feature MAE:")
    for i, name in enumerate(feature_names):
        feature_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        print(f"    {name:<10}: {feature_mae:.6f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/xgboost.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {save_path}")

    print("\n" + "=" * 55)

    return {"model": "XGBoost", "mse": mse, "mae": mae}


if __name__ == "__main__":
    train_and_evaluate()
