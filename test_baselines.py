"""
Milestone 3 test — runs Round Robin and Least Connection end-to-end
and prints basic metrics to verify both schedulers work correctly.
Run with: python test_baselines.py
"""
from environment.cloud_environment import CloudEnvironment
from algorithms.round_robin.scheduler import RoundRobinScheduler
from algorithms.least_connection.scheduler import LeastConnectionScheduler
from config import NUM_VMS


def run_scheduler(scheduler, env):
    """Runs a scheduler through all jobs in the environment."""
    env.reset()
    if hasattr(scheduler, 'reset'):
        scheduler.reset()

    done = False
    state = env.get_state()

    while not done:
        vm_index = scheduler.select_vm(state)
        state, reward, done = env.step(vm_index)

    return env


def print_results(name, env):
    all_jobs = [job for vm in env.vms for job in vm.completed_jobs]
    simulation_end_time = max(vm.available_time for vm in env.vms)

    avg_response = sum(j.response_time for j in all_jobs) / len(all_jobs)
    completion   = sum(1 for j in all_jobs if j.success) / len(all_jobs) * 100
    sla          = sum(1 for j in all_jobs if j.success) / len(all_jobs) * 100
    total_energy = sum(vm.total_energy for vm in env.vms)

    print(f"\n  {name}")
    print(f"    Avg response time : {avg_response:.2f} ms")
    print(f"    Task completion   : {completion:.1f}%")
    print(f"    SLA compliance    : {sla:.1f}%")
    print(f"    Total energy      : {total_energy:.2f}")
    print(f"    CPU utilisation per VM:")
    for vm in env.vms:
        print(f"      VM{vm.vm_id}: {vm.get_avg_utilization(simulation_end_time):.2f}")


def test_baselines():
    print("=" * 50)
    print("MILESTONE 3 — Baseline Schedulers Test")
    print("=" * 50)

    env = CloudEnvironment(arrival_rate=20, seed=42)

    rr = RoundRobinScheduler(num_vms=NUM_VMS)
    lc = LeastConnectionScheduler(vms=env.vms)

    print("\nRunning Round Robin...")
    run_scheduler(rr, env)
    print_results("Round Robin", env)

    print("\nRunning Least Connection...")
    run_scheduler(lc, env)
    print_results("Least Connection", env)

    print("\n" + "=" * 50)
    print("Both schedulers completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    test_baselines()
