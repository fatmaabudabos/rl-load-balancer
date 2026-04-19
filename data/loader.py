import pandas as pd
import numpy as np


# ── Column indices from Google Cluster Data 2011 task_usage ───────────────────
# Full schema: https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md
COL_TIMESTAMP  = 0   # time in microseconds
COL_JOB_ID     = 2   # job identifier
COL_TASK_ID    = 3   # task index within job
COL_MACHINE_ID = 4   # physical machine hosting the task
COL_CPU        = 5   # CPU usage (normalized 0.0 - 1.0)
COL_MEMORY     = 6   # memory usage (normalized 0.0 - 1.0)


def load_cluster_data(filepath, num_vms=4, num_timesteps=100):
    """
    Reads a Google Cluster Data task_usage CSV file and returns
    a clean time-series of CPU and memory usage per VM.

    Steps:
    1. Load the CSV and extract only the columns we need
    2. Pick the top num_vms most active machines (most data points)
    3. For each machine, sort by timestamp and sample num_timesteps readings
    4. Return a numpy array of shape (num_vms, num_timesteps, 2)
       where the last dimension is [cpu_usage, memory_usage]

    Parameters:
        filepath    : path to the .csv file
        num_vms     : how many machines to use (we use 4 to match the paper)
        num_timesteps: how many time steps per VM to keep

    Returns:
        data        : numpy array shape (num_vms, num_timesteps, 2)
        machine_ids : list of machine ids selected
    """

    print(f"Loading data from {filepath}...")

    # ── Load only the columns we need ────────────────────────────────────────
    df = pd.read_csv(
        filepath,
        header=None,
        usecols=[COL_TIMESTAMP, COL_MACHINE_ID, COL_CPU, COL_MEMORY],
        names=["timestamp", "machine_id", "cpu", "memory"],
    )

    print(f"  Total rows loaded : {len(df)}")
    print(f"  Unique machines   : {df['machine_id'].nunique()}")

    # ── Drop rows with missing values ─────────────────────────────────────────
    df.dropna(inplace=True)

    # ── Pick the top num_vms machines with the most data points ───────────────
    # More data points = more reliable time-series for LSTM training
    top_machines = (
        df.groupby("machine_id")
        .size()
        .nlargest(num_vms)
        .index
        .tolist()
    )

    print(f"  Selected machines : {top_machines}")

    # ── Build time-series per machine ─────────────────────────────────────────
    data = []

    for machine_id in top_machines:
        machine_df = df[df["machine_id"] == machine_id].copy()
        machine_df.sort_values("timestamp", inplace=True)

        cpu    = machine_df["cpu"].values
        memory = machine_df["memory"].values

        # sample exactly num_timesteps evenly spaced points
        indices = np.linspace(0, len(cpu) - 1, num_timesteps, dtype=int)
        cpu     = cpu[indices]
        memory  = memory[indices]

        # stack into shape (num_timesteps, 2)
        time_series = np.stack([cpu, memory], axis=1)
        data.append(time_series)

    # final shape: (num_vms, num_timesteps, 2)
    data = np.array(data, dtype=np.float32)

    print(f"  Output shape      : {data.shape}")
    print(f"  CPU range         : {data[:,:,0].min():.4f} - {data[:,:,0].max():.4f}")
    print(f"  Memory range      : {data[:,:,1].min():.4f} - {data[:,:,1].max():.4f}")

    return data, top_machines


if __name__ == "__main__":
    # quick test — run with: python data/loader.py
    filepath = r"C:\Users\AUB\Desktop\299A\part-00000-of-00500.csv"
    data, machines = load_cluster_data(filepath, num_vms=4, num_timesteps=100)

    print("\nFirst VM — first 5 timesteps:")
    print("  [cpu,    memory]")
    for row in data[0][:5]:
        print(f"  {row}")
