import numpy as np
import pandas as pd

# ── File path ─────────────────────────────────────────────────────────────────
DATA_PATH       = r"C:\Users\AUB\Desktop\299A\part-00000-of-00500.csv"

# ── Parameters ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 10      # how many past timesteps the model looks at
TRAIN_RATIO     = 0.8     # 80% train, 20% test
MIN_ROWS        = 50      # drop machines with fewer rows than this


def load_and_clean():
    """
    Step 1 — Load
        Read the full CSV. Pull only the 6 columns we need.
        Name them properly.

    Step 2 — Clean
        - Clip max_cpu to 1.0  (it can spike above 1.0 in the raw data)
        - Drop machines with fewer than MIN_ROWS timesteps
          (too short to build meaningful sequences from)

    Returns a clean pandas DataFrame.
    """

    print("=" * 55)
    print("PREPARE DATA")
    print("=" * 55)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("\n[1] Loading full CSV...")
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        usecols=[0, 4, 5, 6, 11, 13],
        names=["timestamp", "machine_id", "cpu", "memory", "disk_io", "max_cpu"],
    )
    print(f"    Rows loaded      : {len(df):,}")
    print(f"    Unique machines  : {df['machine_id'].nunique():,}")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print("\n[2] Cleaning...")

    # clip max_cpu — anything above 1.0 is an outlier burst, cap it at 1.0
    before = (df["max_cpu"] > 1.0).sum()
    df["max_cpu"] = df["max_cpu"].clip(upper=1.0)
    print(f"    max_cpu values clipped to 1.0  : {before:,}")

    # drop machines with too few rows
    counts        = df.groupby("machine_id").size()
    valid_machines = counts[counts >= MIN_ROWS].index
    before_machines = df["machine_id"].nunique()
    df = df[df["machine_id"].isin(valid_machines)]
    after_machines  = df["machine_id"].nunique()
    print(f"    Machines dropped (< {MIN_ROWS} rows) : {before_machines - after_machines:,}")
    print(f"    Machines kept               : {after_machines:,}")
    print(f"    Rows after cleaning         : {len(df):,}")

    return df


def build_sequences(df):
    """
    Step 3 — Build sliding window sequences.

    For each machine:
        Sort by timestamp.
        Slide a window of SEQUENCE_LENGTH across the time series.
        Input  X : the window  (SEQUENCE_LENGTH timesteps, 4 features)
        Target y : the very next timestep (4 features)

    Returns:
        X : shape (total_samples, SEQUENCE_LENGTH, 4)
        y : shape (total_samples, 4)
    """

    print("\n[3] Building sequences...")

    X_all = []
    y_all = []

    machines = df["machine_id"].unique()

    for machine_id in machines:
        machine_df = df[df["machine_id"] == machine_id].copy()
        machine_df.sort_values("timestamp", inplace=True)

        # extract the 4 feature columns as a numpy array
        series = machine_df[["cpu", "memory", "disk_io", "max_cpu"]].values
        # series shape: (num_timesteps, 4)

        # slide the window across the full time series
        for t in range(len(series) - SEQUENCE_LENGTH):
            x = series[t : t + SEQUENCE_LENGTH]   # (SEQUENCE_LENGTH, 4)
            y = series[t + SEQUENCE_LENGTH]        # (4,)
            X_all.append(x)
            y_all.append(y)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    print(f"    Sequence length  : {SEQUENCE_LENGTH} timesteps")
    print(f"    Total samples    : {len(X):,}")
    print(f"    X shape          : {X.shape}  (samples, timesteps, features)")
    print(f"    y shape          : {y.shape}  (samples, features)")

    return X, y


def split(X, y):
    """
    Step 4 — Train / test split.
    80% of samples for training, 20% for testing.
    The split is done in time order — we do not shuffle.
    (Shuffling would leak future data into training, which is cheating.)

    Returns:
        X_train, X_test, y_train, y_test  — shape (SEQUENCE_LENGTH, 4) version
    """

    print("\n[4] Splitting 80/20...")

    split_idx = int(len(X) * TRAIN_RATIO)

    X_train = X[:split_idx]
    X_test  = X[split_idx:]
    y_train = y[:split_idx]
    y_test  = y[split_idx:]

    print(f"    Training samples : {len(X_train):,}")
    print(f"    Test samples     : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def flatten(X_train, X_test):
    """
    Step 5 — Flatten sequences for sklearn models.

    Linear Regression and Random Forest cannot handle 3D input.
    They need a flat 2D array.

    (samples, 10, 4)  →  (samples, 40)

    Returns:
        X_train_flat, X_test_flat
    """

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)

    return X_train_flat, X_test_flat


def get_data():
    """
    Main function — runs all steps and returns everything each model needs.

    Returns:
        X_train        : (samples, 10, 4)  — for LSTM
        X_test         : (samples, 10, 4)  — for LSTM
        y_train        : (samples, 4)
        y_test         : (samples, 4)
        X_train_flat   : (samples, 40)     — for Linear Regression + Random Forest
        X_test_flat    : (samples, 40)     — for Linear Regression + Random Forest
    """

    df                       = load_and_clean()
    X, y                     = build_sequences(df)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train_flat, X_test_flat = flatten(X_train, X_test)

    print("\n" + "=" * 55)
    print("DATA READY")
    print(f"  LSTM input shape    : {X_train.shape}")
    print(f"  Sklearn input shape : {X_train_flat.shape}")
    print(f"  Target shape        : {y_train.shape}")
    print("=" * 55 + "\n")

    return X_train, X_test, y_train, y_test, X_train_flat, X_test_flat


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_flat, X_test_flat = get_data()

    print("Sample input (X_train[0]) — 10 timesteps x 4 features:")
    print("  [cpu,      memory,    disk_io,   max_cpu ]")
    for row in X_train[0]:
        print(f"  {row}")

    print(f"\nTarget (y_train[0]) — what the model should predict:")
    print(f"  {y_train[0]}")
    print(f"  [cpu, memory, disk_io, max_cpu]")
