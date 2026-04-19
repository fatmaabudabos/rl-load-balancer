import numpy as np


def build_sequences(data, sequence_length=10):
    """
    Converts raw time-series data into input/output pairs for LSTM training.

    The idea:
        Given the last `sequence_length` timesteps of CPU and memory usage,
        predict the NEXT timestep.

    Example with sequence_length=3:
        Input:  [t1, t2, t3]  → Output: [t4]
        Input:  [t2, t3, t4]  → Output: [t5]
        Input:  [t3, t4, t5]  → Output: [t6]
        ...and so on

    Parameters:
        data            : numpy array of shape (num_vms, num_timesteps, 2)
                          the output of loader.load_cluster_data()
        sequence_length : how many past timesteps the LSTM looks at
                          to predict the next one

    Returns:
        X : numpy array shape (total_samples, sequence_length, 2)
            input sequences (past CPU + memory readings)
        y : numpy array shape (total_samples, 2)
            target values (next CPU + memory reading to predict)
    """

    X_all = []
    y_all = []

    num_vms, num_timesteps, num_features = data.shape

    for vm_idx in range(num_vms):
        vm_series = data[vm_idx]   # shape (num_timesteps, 2)

        # slide a window of size sequence_length across the time series
        for t in range(num_timesteps - sequence_length):
            # input: sequence_length timesteps starting at t
            x = vm_series[t : t + sequence_length]         # shape (sequence_length, 2)
            # output: the very next timestep
            y = vm_series[t + sequence_length]              # shape (2,)

            X_all.append(x)
            y_all.append(y)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    return X, y


def split_dataset(X, y, train_ratio=0.8):
    """
    Splits data into training and test sets.
    80% of samples used for training, 20% for testing.

    Parameters:
        X           : input sequences
        y           : target values
        train_ratio : fraction of data used for training

    Returns:
        X_train, X_test, y_train, y_test
    """
    split     = int(len(X) * train_ratio)
    X_train   = X[:split]
    X_test    = X[split:]
    y_train   = y[:split]
    y_test    = y[split:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # quick test — run with: python data/dataset.py
    from data.loader import load_cluster_data

    filepath = r"C:\Users\AUB\Desktop\299A\part-00000-of-00500.csv"
    data, _  = load_cluster_data(filepath, num_vms=4, num_timesteps=100)

    X, y = build_sequences(data, sequence_length=10)

    print("=" * 45)
    print("DATASET BUILD — Results")
    print("=" * 45)
    print(f"  Raw data shape      : {data.shape}")
    print(f"  Sequence length     : 10 timesteps")
    print(f"  Total samples       : {len(X)}")
    print(f"  X shape             : {X.shape}  (samples, timesteps, features)")
    print(f"  y shape             : {y.shape}  (samples, features)")

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"\n  Training samples    : {len(X_train)}")
    print(f"  Test samples        : {len(X_test)}")
    print(f"\n  Example input (X[0]):")
    print(f"  [cpu,      memory  ]")
    for row in X[0]:
        print(f"  {row}")
    print(f"\n  Target output (y[0]):")
    print(f"  {y[0]}  ← this is what LSTM will learn to predict")
    print("=" * 45)
