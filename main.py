import os
import numpy as np
import matplotlib.pyplot as plt

from environment.cloud_environment import CloudEnvironment
from algorithms.round_robin.scheduler import RoundRobinScheduler
from algorithms.least_connection.scheduler import LeastConnectionScheduler
from config import ARRIVAL_RATES, NUM_VMS

SEED = 42

# ── Results storage ────────────────────────────────────────────────────────────
results = {
    "Round Robin"      : {"cpu_utilization": [], "task_completion": []},
    "Least Connection" : {"cpu_utilization": [], "task_completion": []},
}


def run_scheduler(env, scheduler):
    """Runs a scheduler through all jobs and returns metrics."""
    env.reset()
    if hasattr(scheduler, 'reset'):
        scheduler.reset()

    state = env.get_state()
    done  = False

    while not done:
        vm_index    = scheduler.select_vm(state)
        state, done = env.step(vm_index)

    return collect_metrics(env)


def collect_metrics(env):
    """Computes the 2 metrics we care about from a finished simulation."""
    all_jobs        = [job for vm in env.vms for job in vm.completed_jobs]
    sim_end         = max(vm.available_time for vm in env.vms)
    total           = len(all_jobs)

    cpu_utils       = [vm.get_avg_utilization(sim_end) for vm in env.vms]
    task_completion = sum(1 for j in all_jobs if j.success) / total * 100

    return {
        "cpu_utilization" : cpu_utils,
        "task_completion" : task_completion,
    }


# ── Run experiments ────────────────────────────────────────────────────────────
print("=" * 55)
print("EXPERIMENT — Round Robin vs Least Connection")
print("=" * 55)

for rate in ARRIVAL_RATES:
    print(f"\nArrival rate: {rate} jobs/sec")
    print("-" * 35)

    env = CloudEnvironment(arrival_rate=rate, seed=SEED)

    rr   = RoundRobinScheduler(num_vms=NUM_VMS)
    rr_m = run_scheduler(env, rr)
    print(f"  RR → task completion={rr_m['task_completion']:.1f}%")

    lc   = LeastConnectionScheduler(vms=env.vms)
    lc_m = run_scheduler(env, lc)
    print(f"  LC → task completion={lc_m['task_completion']:.1f}%")

    for name, m in [("Round Robin", rr_m), ("Least Connection", lc_m)]:
        results[name]["cpu_utilization"].append(m["cpu_utilization"])
        results[name]["task_completion"].append(m["task_completion"])


# ── Plotting ───────────────────────────────────────────────────────────────────
os.makedirs("results/plots", exist_ok=True)

colors  = {"Round Robin": "#e74c3c", "Least Connection": "#f39c12"}
markers = {"Round Robin": "o",       "Least Connection": "s"}

# ── Plot 1: Task completion rate vs arrival rate ───────────────────────────────
plt.figure(figsize=(8, 5))
for name in results:
    plt.plot(ARRIVAL_RATES, results[name]["task_completion"],
             label=name, color=colors[name], marker=markers[name], linewidth=2)
plt.xlabel("Arrival Rate (jobs/sec)")
plt.ylabel("Task Completion Rate (%)")
plt.title("Task Completion vs Arrival Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/task_completion.png", dpi=150)
plt.close()
print("\nSaved: results/plots/task_completion.png")

# ── Plot 2: CPU utilisation per VM at middle arrival rate ──────────────────────
mid = len(ARRIVAL_RATES) // 2
x   = np.arange(NUM_VMS)
w   = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
for i, name in enumerate(results):
    ax.bar(x + i * w, results[name]["cpu_utilization"][mid],
           w, label=name, color=colors[name], alpha=0.85)
ax.set_xlabel("VM")
ax.set_ylabel("CPU Utilisation")
ax.set_title(f"CPU Utilisation per VM (arrival rate = {ARRIVAL_RATES[mid]} jobs/sec)")
ax.set_xticks(x + w / 2)
ax.set_xticklabels([f"VM{i}" for i in range(NUM_VMS)])
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("results/plots/cpu_utilization.png", dpi=150)
plt.close()
print("Saved: results/plots/cpu_utilization.png")

print("\n" + "=" * 55)
print("Done. Plots saved to results/plots/")
print("=" * 55)
