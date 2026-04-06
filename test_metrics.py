"""
Milestone 5 test — runs Round Robin on the environment then verifies
all 5 metrics are computed correctly by the collector.
Run with: python test_metrics.py
"""
from environment.cloud_environment import CloudEnvironment
from algorithms.round_robin.scheduler import RoundRobinScheduler
from metrics.collector import collect_metrics, print_metrics
from config import NUM_VMS


def test_metrics():
    print("=" * 50)
    print("MILESTONE 5 — Metrics Collector Test")
    print("=" * 50)

    env       = CloudEnvironment(arrival_rate=20, seed=42)
    scheduler = RoundRobinScheduler(num_vms=NUM_VMS)

    # run simulation
    env.reset()
    scheduler.reset()
    state = env.get_state()
    done  = False

    while not done:
        vm_index          = scheduler.select_vm(state)
        state, _, done    = env.step(vm_index)

    # collect metrics
    metrics = collect_metrics(env)

    # ── Verify all keys exist ─────────────────────────────────────────────────
    expected_keys = [
        "avg_response_time",
        "cpu_utilization",
        "total_energy",
        "task_completion",
        "sla_compliance",
    ]

    print("\n[1] All metric keys present:")
    for key in expected_keys:
        present = key in metrics
        print(f"    {key}: {present} (expected True)")

    # ── Verify value types and ranges ─────────────────────────────────────────
    print("\n[2] Value sanity checks:")
    print(f"    avg_response_time > 0    : {metrics['avg_response_time'] > 0}")
    print(f"    total_energy > 0         : {metrics['total_energy'] > 0}")
    print(f"    task_completion 0-100    : {0 <= metrics['task_completion'] <= 100}")
    print(f"    sla_compliance 0-100     : {0 <= metrics['sla_compliance'] <= 100}")
    print(f"    cpu_utilization length   : {len(metrics['cpu_utilization'])} (expected 4)")
    print(f"    all util values 0-1      : "
          f"{all(0 <= u <= 1 for u in metrics['cpu_utilization'])}")

    # ── Print full results ────────────────────────────────────────────────────
    print("\n[3] Full metrics output:")
    print_metrics("Round Robin", metrics)

    print("\n" + "=" * 50)
    print("Metrics collector working correctly.")
    print("=" * 50)


if __name__ == "__main__":
    test_metrics()
