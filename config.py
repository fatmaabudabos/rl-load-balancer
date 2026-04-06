# ── Environment ──────────────────────────────────────────────────────────────
NUM_VMS = 4
NUM_JOBS = 1000
ARRIVAL_RATES = [10, 15, 20, 25, 30]   # jobs per second, swept in main.py

# ── SLA & Overload ────────────────────────────────────────────────────────────
SLA_THRESHOLD = 200        # ms — jobs exceeding this count as SLA violations
OVERLOAD_THRESHOLD = 0.80  # 80% CPU — paper's migration trigger point

# ── Reward function weights (R = α(1 - Umax) + β(1 - mean(Ui))) ──────────────
ALPHA = 0.5   # weight for penalising the most overloaded VM
BETA  = 0.5   # weight for penalising average utilisation across all VMs

# ── RL Hyperparameters ────────────────────────────────────────────────────────
LEARNING_RATE      = 0.001
GAMMA              = 0.99    # discount factor — how much future rewards matter
EPSILON_START      = 1.0     # start fully exploring
EPSILON_END        = 0.01    # end mostly exploiting
EPSILON_DECAY      = 0.995   # multiply epsilon by this after every job
REPLAY_BUFFER_SIZE = 10000   # max experiences stored
BATCH_SIZE         = 64      # experiences sampled per training step
TARGET_UPDATE_FREQ = 100     # steps between syncing target network
