# RL-Based VM Load Balancer

An offline simulation comparing AI-based load balancing against Round Robin and Least Connection baselines for virtual machine scheduling in cloud environments.

Inspired by: *Enhancing Machine Learning Performance with AI-Based Virtual Machine Load Balancing*

---

## Project Structure

```
rl-load-balancer/
├── data/
│   ├── loader.py        # Loads Google Cluster Data, extracts CPU and memory per VM
│   └── dataset.py       # Builds sliding window sequences for LSTM training
├── algorithms/
│   ├── round_robin/
│   │   └── scheduler.py # Round Robin baseline
│   └── least_connection/
│       └── scheduler.py # Least Connection baseline
├── environment/
│   ├── job.py           # Job class
│   ├── vm.py            # VM class
│   └── cloud_environment.py  # Simulation engine
├── results/
│   └── plots/           # Output charts saved here
├── config.py            # All parameters in one place
├── main.py              # Experiment runner
└── requirements.txt
```

---

## Progress

### Done
- **Environment** — 4 VMs with varied capacities, 1000 jobs with random CPU/memory demands
- **Round Robin baseline** — assigns jobs in circular order
- **Least Connection baseline** — assigns to VM with least backlog
- **Metrics** — task completion rate and CPU utilization per VM
- **Plots** — task completion vs arrival rate, CPU utilization per VM
- **Data pipeline** — loads Google Cluster Data 2011, builds (360, 10, 2) dataset for LSTM training

### In Progress
- LSTM load prediction model
- RL agent for VM assignment decisions

### Coming Later
- VM migration
- Dynamic resource allocation
- Auto-tuning
- Full evaluation with Google Cluster Data

---

## How to Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the experiment
python main.py
```

Results and plots will be saved to `results/plots/`.

---

## Metrics

- Task Completion Rate (%)
- CPU Utilisation per VM (%)

---

## Algorithms Compared

| Algorithm | Type | Description |
|---|---|---|
| Round Robin | Baseline | Assigns jobs in circular order |
| Least Connection | Baseline | Assigns to VM with least backlog |
| RL Agent | Proposed (coming) | Learns optimal assignment via Q-learning |
