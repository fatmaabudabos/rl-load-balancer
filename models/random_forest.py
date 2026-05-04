import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.prepare_data import get_data


def train_and_evaluate():
    """
    Trains a Random Forest model on the Google Cluster Data.

    A Random Forest builds many decision trees, each trained on a
    random subset of the data. Each tree makes its own prediction
    and the forest averages them all together.

    This makes it more accurate and robust than a single decision tree,
    and more reliable than Linear Regression on complex non-linear data.

    Like Linear Regression, it uses the flat version of the data
    (samples, 40) — it does not understand sequences.

    n_estimators = number of trees in the forest
    n_jobs       = use all CPU cores to train in parallel (faster)
    random_state = fixes randomness so results are reproducible
    """

    print("=" * 55)
    print("MODEL — Random Forest")
    print("=" * 55)

    # ── Load data ─────────────────────────────────────────────────────────────
    # We use the flat version (samples, 40) — same as Linear Regression
    X_train, X_test, y_train, y_test, X_train_flat, X_test_flat = get_data()

    # ── Sample for training ───────────────────────────────────────────────────
    # Random Forest does not benefit much from 1.9M samples — it gets
    # diminishing returns past a few hundred thousand. We sample 200,000
    # randomly to keep training time reasonable. The test set stays full
    # so evaluation is still honest.
    SAMPLE_SIZE = 200_000
    rng         = np.random.default_rng(42)
    idx         = rng.choice(len(X_train_flat), size=SAMPLE_SIZE, replace=False)
    X_train_flat = X_train_flat[idx]
    y_train      = y_train[idx]

    print(f"\nTraining on : {len(X_train_flat):,} samples (sampled from full set)")
    print(f"Testing on  : {len(X_test_flat):,} samples (full test set)")
    print(f"Input shape : {X_train_flat.shape}")
    print(f"Target shape: {y_train.shape}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining...")
    model = RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,          # use all available CPU cores
        random_state=42,
    )
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

    # ── Feature importance ────────────────────────────────────────────────────
    # Random Forest can tell us which of the 40 input features it relied
    # on most when making predictions. We group them back into timesteps.
    print("\n  Feature importance by input feature (grouped):")
    importance    = model.feature_importances_   # shape (40,)
    feature_names_input = ["cpu", "memory", "disk_io", "max_cpu"]
    grouped = np.zeros(4)
    for i in range(4):
        # sum importance of this feature across all 10 timesteps
        grouped[i] = importance[i::4].sum()
    for name, imp in zip(feature_names_input, grouped):
        bar = "#" * int(imp * 100)
        print(f"    {name:<10}: {imp:.4f}  {bar}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/random_forest.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {save_path}")

    print("\n" + "=" * 55)

    return {"model": "Random Forest", "mse": mse, "mae": mae}


if __name__ == "__main__":
    train_and_evaluate()
