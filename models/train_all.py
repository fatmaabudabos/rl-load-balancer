import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.prepare_data import get_data
from models.lstm import LSTMPredictor

# ── Output folder ─────────────────────────────────────────────────────────────
os.makedirs("results/plots", exist_ok=True)


def load_models():
    """
    Loads all 3 trained models from models/saved/.
    No retraining — just loads what was already saved.
    """
    print("Loading saved models...")

    # Linear Regression
    with open("models/saved/linear_regression.pkl", "rb") as f:
        lr_model = pickle.load(f)
    print("  Linear Regression loaded")

    # Random Forest
    with open("models/saved/random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)
    print("  Random Forest loaded")

    # LSTM
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMPredictor()
    lstm_model.load_state_dict(torch.load("models/saved/lstm.pt", map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    print("  LSTM loaded")

    return lr_model, rf_model, lstm_model, device


def evaluate_all(lr_model, rf_model, lstm_model, device):
    """
    Runs all 3 models on the same test set and collects MSE and MAE.
    """
    print("\nLoading test data...")
    X_train, X_test, y_train, y_test, X_train_flat, X_test_flat = get_data()

    results = {}

    # ── Linear Regression ─────────────────────────────────────────────────────
    print("\nEvaluating Linear Regression...")
    y_pred_lr = lr_model.predict(X_test_flat)
    results["Linear Regression"] = {
        "mse": mean_squared_error(y_test, y_pred_lr),
        "mae": mean_absolute_error(y_test, y_pred_lr),
        "per_feature_mae": [
            mean_absolute_error(y_test[:, i], y_pred_lr[:, i])
            for i in range(4)
        ]
    }

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("Evaluating Random Forest...")
    y_pred_rf = rf_model.predict(X_test_flat)
    results["Random Forest"] = {
        "mse": mean_squared_error(y_test, y_pred_rf),
        "mae": mean_absolute_error(y_test, y_pred_rf),
        "per_feature_mae": [
            mean_absolute_error(y_test[:, i], y_pred_rf[:, i])
            for i in range(4)
        ]
    }

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("Evaluating LSTM...")
    X_test_t = torch.tensor(X_test).to(device)
    with torch.no_grad():
        y_pred_lstm_t = lstm_model(X_test_t)
    y_pred_lstm = y_pred_lstm_t.cpu().numpy()
    results["LSTM"] = {
        "mse": mean_squared_error(y_test, y_pred_lstm),
        "mae": mean_absolute_error(y_test, y_pred_lstm),
        "per_feature_mae": [
            mean_absolute_error(y_test[:, i], y_pred_lstm[:, i])
            for i in range(4)
        ]
    }

    return results


def print_comparison(results):
    """
    Prints a clean comparison table in the terminal.
    """
    print("\n")
    print("=" * 55)
    print("MODEL COMPARISON RESULTS")
    print("=" * 55)
    print(f"  {'Model':<22} {'MSE':>10}  {'MAE':>10}")
    print("-" * 55)

    for name, metrics in results.items():
        print(f"  {name:<22} {metrics['mse']:>10.6f}  {metrics['mae']:>10.6f}")

    print("-" * 55)

    # find winner by lowest MAE
    winner = min(results, key=lambda m: results[m]["mae"])
    print(f"\n  Winner: {winner} (lowest MAE)")
    print("=" * 55)

    # per-feature breakdown
    feature_names = ["cpu", "memory", "disk_io", "max_cpu"]
    print("\n  Per-feature MAE breakdown:")
    print(f"  {'Feature':<12}", end="")
    for name in results:
        print(f"  {name:>18}", end="")
    print()
    print("  " + "-" * 65)
    for i, feat in enumerate(feature_names):
        print(f"  {feat:<12}", end="")
        for name in results:
            print(f"  {results[name]['per_feature_mae'][i]:>18.6f}", end="")
        print()
    print()


def plot_comparison(results):
    """
    Generates two plots:
    1. Bar chart comparing MSE across all 3 models
    2. Bar chart comparing MAE across all 3 models
    Saved as a single figure with two subplots.
    """
    model_names = list(results.keys())
    mse_values  = [results[m]["mse"] for m in model_names]
    mae_values  = [results[m]["mae"] for m in model_names]
    colors      = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── MSE plot ──────────────────────────────────────────────────────────────
    bars1 = ax1.bar(model_names, mse_values, color=colors, alpha=0.85, width=0.5)
    ax1.set_title("Mean Squared Error (MSE)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("MSE (lower is better)")
    ax1.set_ylim(0, max(mse_values) * 1.3)
    for bar, val in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.00005,
                 f"{val:.6f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── MAE plot ──────────────────────────────────────────────────────────────
    bars2 = ax2.bar(model_names, mae_values, color=colors, alpha=0.85, width=0.5)
    ax2.set_title("Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("MAE (lower is better)")
    ax2.set_ylim(0, max(mae_values) * 1.3)
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0002,
                 f"{val:.6f}", ha="center", va="bottom", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("ML Model Comparison — Google Cluster Data 2011",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = "results/plots/model_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("TRAIN ALL — Model Comparison")
    print("=" * 55)

    lr_model, rf_model, lstm_model, device = load_models()
    results = evaluate_all(lr_model, rf_model, lstm_model, device)
    print_comparison(results)
    plot_comparison(results)

    print("Done. Check results/plots/model_comparison.png")
