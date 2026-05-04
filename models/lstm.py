import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.prepare_data import get_data

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_SIZE   = 128     # increased from 64 — more capacity to learn patterns
NUM_LAYERS    = 2       # how many LSTM layers stacked
EPOCHS        = 30      # increased from 10 — more learning passes
BATCH_SIZE    = 512     # how many samples processed at once
LEARNING_RATE = 0.001   # starting learning rate
SAMPLE_SIZE   = 800_000 # increased from 500k — more training data


# ── Model definition ──────────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    """
    A two-layer LSTM that takes in a sequence of resource usage readings
    and predicts the next timestep.

    Architecture:
        Input       : (batch, sequence_length=10, features=4)
        LSTM layer 1: 128 hidden units — learns low-level temporal patterns
        LSTM layer 2: 128 hidden units — learns higher-level patterns
        Linear layer: maps 128 → 4 (one output per feature)
        Output      : (batch, 4)  — predicted cpu, memory, disk_io, max_cpu

    Changes from v1:
        - Hidden size doubled: 64 → 128
        - Learning rate scheduler added (reduces LR when loss plateaus)
        - Sample size increased: 500k → 800k
        - Epochs increased: 10 → 30
    """

    def __init__(self, input_size=4, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=4):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,    # input shape: (batch, seq, features)
            dropout     = 0.2,     # randomly zero 20% of connections between layers
        )

        # fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, sequence_length, hidden_size)

        # take only the last timestep's output for prediction
        last_out = lstm_out[:, -1, :]
        # last_out shape: (batch, hidden_size)

        out = self.fc(last_out)
        # out shape: (batch, output_size=4)
        return out


def train_and_evaluate():
    """
    Trains the improved LSTM model and evaluates it on the test set.

    Key improvements over v1:
        1. Larger hidden size (128 vs 64) — more learning capacity
        2. More epochs (30 vs 10) — more time to learn
        3. More training data (800k vs 500k) — more patterns to learn from
        4. Learning rate scheduler — automatically reduces LR when loss
           stops improving, helping the model fine-tune rather than overshoot
    """

    print("=" * 55)
    print("MODEL — LSTM (improved)")
    print("=" * 55)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _, _ = get_data()

    # ── Sample for training ───────────────────────────────────────────────────
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"\nTraining on : {len(X_train):,} samples (sampled from full set)")
    print(f"Testing on  : {len(X_test):,} samples (full test set)")
    print(f"Input shape : {X_train.shape}")
    print(f"Target shape: {y_train.shape}")

    # ── Convert to PyTorch tensors ────────────────────────────────────────────
    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t  = torch.tensor(X_test).to(device)
    y_test_t  = torch.tensor(y_test).to(device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── Model, loss, optimizer, scheduler ─────────────────────────────────────
    model     = LSTMPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ReduceLROnPlateau: if the loss doesn't improve for 3 epochs in a row,
    # reduce the learning rate by half. Helps the model fine-tune carefully
    # once it's close to a good solution.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Batches/epoch : {len(train_loader)}\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # step the scheduler — it checks if loss improved
        scheduler.step(avg_loss)

        print(f"  Epoch {epoch+1:>2}/{EPOCHS}  |  Loss: {avg_loss:.6f}  |  LR: {current_lr:.6f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)

    y_pred    = y_pred_t.cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()

    mse = mean_squared_error(y_test_np, y_pred)
    mae = mean_absolute_error(y_test_np, y_pred)

    print(f"\n  MSE : {mse:.6f}  (lower is better)")
    print(f"  MAE : {mae:.6f}  (lower is better)")

    feature_names = ["cpu", "memory", "disk_io", "max_cpu"]
    print("\n  Per-feature MAE:")
    for i, name in enumerate(feature_names):
        feature_mae = mean_absolute_error(y_test_np[:, i], y_pred[:, i])
        print(f"    {name:<10}: {feature_mae:.6f}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models/saved", exist_ok=True)
    save_path = "models/saved/lstm.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")

    print("\n" + "=" * 55)

    return {"model": "LSTM", "mse": mse, "mae": mae}


if __name__ == "__main__":
    train_and_evaluate()
