# RL-Based VM Load Balancer

An offline simulation comparing AI-based (DQN) load balancing against Round Robin and Least Connection baselines for virtual machine scheduling in cloud environments.

Inspired by: *Enhancing Machine Learning Performance with AI-Based Virtual Machine Load Balancing*

---

## Project Structure

```
rl-load-balancer/
├── algorithms/
│   ├── round_robin/scheduler.py      # Round Robin baseline
│   ├── least_connection/scheduler.py # Least Connection baseline
│   └── rl/
│       ├── agent.py                  # DQN agent (epsilon-greedy, replay buffer)
│       └── network.py                # Neural network (Q-value estimator)
├── environment/
│   ├── job.py                        # Job class
│   ├── vm.py                         # VM class
│   └── cloud_environment.py          # Simulation engine
├── metrics/
│   └── collector.py                  # Computes all evaluation metrics
├── results/
│   └── plots/                        # Output charts saved here
├── config.py                         # All hyperparameters in one place
├── utils.py                          # Helpers: create env, run scheduler
├── main.py                           # Experiment runner
└── requirements.txt
```

---

## How to Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the experiment
python main.py
```

Results and plots will be saved to `results/plots/`.

---

## Metrics Evaluated

- Average Response Time (ms)
- CPU Utilization per VM (%)
- Energy Consumption (kWh)
- Task Completion Rate (%)
- SLA Compliance (%)

---

## Algorithms Compared

| Algorithm | Type | Description |
|---|---|---|
| Round Robin | Baseline | Assigns jobs in circular order |
| Least Connection | Baseline | Assigns to VM with fewest active jobs |
| DQN (RL Agent) | Proposed | Learns optimal assignment via Q-learning |
