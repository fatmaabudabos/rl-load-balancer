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
HIDDEN_SIZE  = 64      # number of LSTM units per layer
NUM_LAYERS   = 2       # how many LSTM layers stacked
EPOCHS       = 10      # how many full passes through the training data
BATCH_SIZE   = 512     # how many samples processed at once
LEARNING_RATE = 0.001  # how fast the model updates its weights
SAMPLE_SIZE  = 500_000 # train on 500k samples instead of 1.9M (CPU speedup)


# ── Model definition ──────────────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    """
    A two-layer LSTM that takes in a sequence of resource usage readings
    and predicts the next timestep.

    Architecture:
        Input       : (batch, sequence_length=10, features=4)
        LSTM layer 1: 64 hidden units — learns low-level temporal patterns
        LSTM layer 2: 64 hidden units — learns higher-level patterns
        Linear layer: maps 64 → 4 (one output per feature)
        Output      : (batch, 4)  — predicted cpu, memory, disk_io, max_cpu
    """

    def __init__(self, input_size=4, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=4):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,      # input shape: (batch, seq, features)
            dropout     = 0.2,       # randomly zero 20% of connections between layers
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
    Trains the LSTM model and evaluates it on the test set.
    Uses PyTorch DataLoader to feed data in batches during training.
    """

    print("=" * 55)
    print("MODEL — LSTM")
    print("=" * 55)

    # ── Device ────────────────────────────────────────────────────────────────
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    # LSTM uses the sequence version (samples, 10, 4) — NOT the flat version
    X_train, X_test, y_train, y_test, _, _ = get_data()

    # ── Sample for training ───────────────────────────────────────────────────
    # Running on CPU so we limit training to 500k samples to keep time
    # reasonable. Test set stays full so evaluation is still honest.
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(X_train), size=SAMPLE_SIZE, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"\nTraining on : {len(X_train):,} samples (sampled from full set)")
    print(f"Testing on  : {len(X_test):,} samples")
    print(f"Input shape : {X_train.shape}")
    print(f"Target shape: {y_train.shape}")

    # ── Convert to PyTorch tensors ────────────────────────────────────────────
    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t  = torch.tensor(X_test).to(device)
    y_test_t  = torch.tensor(y_test).to(device)

    # ── DataLoader ────────────────────────────────────────────────────────────
    # Feeds data to the model in batches of BATCH_SIZE during training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── Model, loss function, optimizer ───────────────────────────────────────
    model     = LSTMPredictor().to(device)
    criterion = nn.MSELoss()                          # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Batches/epoch: {len(train_loader)}\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()           # clear previous gradients
            y_pred = model(X_batch)         # forward pass
            loss   = criterion(y_pred, y_batch)  # compute loss
            loss.backward()                 # backward pass (BPTT)
            optimizer.step()                # update weights
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1:>2}/{EPOCHS}  |  Loss: {avg_loss:.6f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)

    y_pred = y_pred_t.cpu().numpy()
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
