import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import create_environment, run_scheduler, run_rl_agent
from algorithms.round_robin.scheduler import RoundRobinScheduler
from algorithms.least_connection.scheduler import LeastConnectionScheduler
from metrics.collector import print_metrics
from config import ARRIVAL_RATES, NUM_VMS

# ── Fix all random seeds for reproducible results ──────────────────────────────
# Without this, RL training gives different results every run because
# epsilon-greedy exploration and replay buffer sampling are random.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SEED     = 42
N_EPOCHS = 8   # number of training passes for the RL agent per arrival rate

# ── Results storage ────────────────────────────────────────────────────────────
# results[scheduler_name][metric_name] = list of values, one per arrival rate
results = {
    "Round Robin"      : {"avg_response_time": [], "cpu_utilization": [],
                          "total_energy": [], "task_completion": [], "sla_compliance": []},
    "Least Connection" : {"avg_response_time": [], "cpu_utilization": [],
                          "total_energy": [], "task_completion": [], "sla_compliance": []},
    "RL Agent"         : {"avg_response_time": [], "cpu_utilization": [],
                          "total_energy": [], "task_completion": [], "sla_compliance": []},
}


# ── Run experiments ────────────────────────────────────────────────────────────
print("=" * 60)
print("MAIN EXPERIMENT — Sweeping arrival rates")
print("=" * 60)

for rate in ARRIVAL_RATES:
    print(f"\nArrival rate: {rate} jobs/sec")
    print("-" * 40)

    env = create_environment(arrival_rate=rate, seed=SEED)

    # Round Robin
    rr      = RoundRobinScheduler(num_vms=NUM_VMS)
    rr_m    = run_scheduler(env, rr)
    print(f"  RR  → response={rr_m['avg_response_time']:.1f}ms  "
          f"SLA={rr_m['sla_compliance']:.1f}%")

    # Least Connection
    lc      = LeastConnectionScheduler(vms=env.vms)
    lc_m    = run_scheduler(env, lc)
    print(f"  LC  → response={lc_m['avg_response_time']:.1f}ms  "
          f"SLA={lc_m['sla_compliance']:.1f}%")

    # RL Agent
    print(f"  RL  → training for {N_EPOCHS} epochs...")
    rl_m    = run_rl_agent(env, n_epochs=N_EPOCHS)
    print(f"  RL  → response={rl_m['avg_response_time']:.1f}ms  "
          f"SLA={rl_m['sla_compliance']:.1f}%")

    # store results
    for name, m in [("Round Robin", rr_m),
                    ("Least Connection", lc_m),
                    ("RL Agent", rl_m)]:
        results[name]["avg_response_time"].append(m["avg_response_time"])
        results[name]["cpu_utilization"].append(m["cpu_utilization"])
        results[name]["total_energy"].append(m["total_energy"])
        results[name]["task_completion"].append(m["task_completion"])
        results[name]["sla_compliance"].append(m["sla_compliance"])


# ── Plotting ───────────────────────────────────────────────────────────────────
os.makedirs("results/plots", exist_ok=True)

colors = {
    "Round Robin"      : "#e74c3c",
    "Least Connection" : "#f39c12",
    "RL Agent"         : "#2ecc71",
}
markers = {
    "Round Robin"      : "o",
    "Least Connection" : "s",
    "RL Agent"         : "^",
}

# ── Plot 1: Average response time vs arrival rate ─────────────────────────────
plt.figure(figsize=(8, 5))
for name in results:
    plt.plot(ARRIVAL_RATES, results[name]["avg_response_time"],
             label=name, color=colors[name],
             marker=markers[name], linewidth=2)
plt.xlabel("Arrival Rate (jobs/sec)")
plt.ylabel("Avg Response Time (ms)")
plt.title("Response Time vs Arrival Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/response_time.png", dpi=150)
plt.close()
print("\nSaved: results/plots/response_time.png")

# ── Plot 2: CPU utilisation per VM (bar chart at middle arrival rate) ─────────
mid_rate_idx = len(ARRIVAL_RATES) // 2   # use the middle arrival rate
x      = np.arange(NUM_VMS)
width  = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, name in enumerate(results):
    utils = results[name]["cpu_utilization"][mid_rate_idx]
    ax.bar(x + i * width, utils, width,
           label=name, color=colors[name], alpha=0.85)

ax.set_xlabel("VM")
ax.set_ylabel("CPU Utilisation")
ax.set_title(f"CPU Utilisation per VM "
             f"(arrival rate = {ARRIVAL_RATES[mid_rate_idx]} jobs/sec)")
ax.set_xticks(x + width)
ax.set_xticklabels([f"VM{i}" for i in range(NUM_VMS)])
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("results/plots/cpu_utilization.png", dpi=150)
plt.close()
print("Saved: results/plots/cpu_utilization.png")

# ── Plot 3: Energy consumption vs arrival rate ────────────────────────────────
plt.figure(figsize=(8, 5))
for name in results:
    plt.plot(ARRIVAL_RATES, results[name]["total_energy"],
             label=name, color=colors[name],
             marker=markers[name], linewidth=2)
plt.xlabel("Arrival Rate (jobs/sec)")
plt.ylabel("Total Energy")
plt.title("Energy Consumption vs Arrival Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/energy.png", dpi=150)
plt.close()
print("Saved: results/plots/energy.png")

# ── Plot 4: Task completion and SLA compliance vs arrival rate ────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name in results:
    ax1.plot(ARRIVAL_RATES, results[name]["task_completion"],
             label=name, color=colors[name],
             marker=markers[name], linewidth=2)
    ax2.plot(ARRIVAL_RATES, results[name]["sla_compliance"],
             label=name, color=colors[name],
             marker=markers[name], linewidth=2)

ax1.set_xlabel("Arrival Rate (jobs/sec)")
ax1.set_ylabel("Task Completion Rate (%)")
ax1.set_title("Task Completion vs Arrival Rate")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Arrival Rate (jobs/sec)")
ax2.set_ylabel("SLA Compliance (%)")
ax2.set_title("SLA Compliance vs Arrival Rate")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/plots/completion_sla.png", dpi=150)
plt.close()
print("Saved: results/plots/completion_sla.png")

print("\n" + "=" * 60)
print("Experiment complete. Plots saved to results/plots/")
print("=" * 60)
