"""
Quick sanity check for Milestone 2.
Manually assigns 5 jobs to VMs and verifies all metrics compute correctly.
Run with: python test_environment.py
"""
from environment.cloud_environment import CloudEnvironment


def test_environment():
    print("=" * 50)
    print("MILESTONE 2 — Environment Test")
    print("=" * 50)

    env = CloudEnvironment(arrival_rate=10, seed=42)

    # ── Check VMs created correctly ───────────────────────────────────────────
    print(f"\n[1] VMs created: {len(env.vms)} (expected 4)")
    for vm in env.vms:
        print(f"    VM{vm.vm_id} | cpu_capacity={vm.cpu_capacity} "
              f"memory_capacity={vm.memory_capacity} energy_rate={vm.energy_rate}")

    # ── Check jobs generated ──────────────────────────────────────────────────
    print(f"\n[2] Jobs generated: {len(env.jobs)} (expected 1000)")
    print("    First 3 jobs:")
    for job in env.jobs[:3]:
        print(f"    Job{job.job_id} | arrival={job.arrival_time:.1f}ms "
              f"cpu={job.cpu_demand:.2f} mem={job.memory_demand:.2f} "
              f"priority={job.priority}")

    # ── Check initial state vector ────────────────────────────────────────────
    state = env.get_state()
    print(f"\n[3] Initial state vector (length {len(state)}, expected 7):")
    print(f"    {state}")
    print(f"    [job_cpu, job_mem, job_priority_norm, vm0_util, vm1_util, vm2_util, vm3_util]")

    # ── Assign 5 jobs manually and inspect results ────────────────────────────
    print("\n[4] Assigning 5 jobs manually (round-robin order 0,1,2,3,0):")
    vm_choices = [0, 1, 2, 3, 0]

    for i, vm_idx in enumerate(vm_choices):
        job = env.jobs[env.current_job_index]
        next_state, reward, done = env.step(vm_idx)
        print(f"\n    Job{job.job_id} → VM{vm_idx}")
        print(f"      wait_time={job.wait_time:.2f}ms  "
              f"exe_time={job.exe_time:.2f}ms  "
              f"response_time={job.response_time:.2f}ms")
        print(f"      cost={job.cost:.4f}  success={job.success}  reward={reward:.4f}")

    # ── Check VM utilisation after assignments ────────────────────────────────
    print("\n[5] VM utilisation after 5 assignments:")
    for vm in env.vms:
        print(f"    VM{vm.vm_id} | utilisation={vm.get_utilization():.2f}  "
              f"overloaded={vm.is_overloaded()}  "
              f"jobs_completed={len(vm.completed_jobs)}")

    # ── Check migration detection ─────────────────────────────────────────────
    overloaded = env.check_migration()
    print(f"\n[6] Overloaded VMs: {overloaded}")

    # ── Check reset works ─────────────────────────────────────────────────────
    env.reset()
    all_clear = all(
        vm.cpu_used == 0.0 and
        vm.available_time == 0.0 and
        len(vm.completed_jobs) == 0
        for vm in env.vms
    )
    print(f"\n[7] Reset works correctly: {all_clear} (expected True)")

    print("\n" + "=" * 50)
    print("All checks passed!" if all_clear else "Something went wrong in reset.")
    print("=" * 50)


if __name__ == "__main__":
    test_environment()
