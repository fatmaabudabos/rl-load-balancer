from config import SLA_THRESHOLD


def collect_metrics(env):
    """
    Takes a finished simulation environment and computes all 5 metrics
    from the paper:

      1. Average response time   (ms)
      2. CPU utilisation per VM  (fraction 0-1)
      3. Total energy consumed   (kWh)
      4. Task completion rate    (%)
      5. SLA compliance          (%)

    Returns a dictionary so main.py can store and compare results
    across all three schedulers.
    """

    all_jobs         = [job for vm in env.vms for job in vm.completed_jobs]
    simulation_end   = max(vm.available_time for vm in env.vms)
    total_jobs       = len(all_jobs)

    # guard against empty simulation
    if total_jobs == 0:
        return {
            "avg_response_time"  : 0.0,
            "cpu_utilization"    : [0.0] * len(env.vms),
            "total_energy"       : 0.0,
            "task_completion"    : 0.0,
            "sla_compliance"     : 0.0,
        }

    # ── 1. Average response time (ms) ─────────────────────────────────────────
    # Total time from job arrival to job completion, averaged across all jobs.
    # Paper target: AI ≈ 160ms, LC ≈ 210ms, RR ≈ 250ms
    avg_response_time = sum(j.response_time for j in all_jobs) / total_jobs

    # ── 2. CPU utilisation per VM ─────────────────────────────────────────────
    # Average fraction of CPU time each VM was busy over the full simulation.
    # Paper target: AI ≈ 68-75% balanced, RR ≈ 20-85% unbalanced
    cpu_utilization = [
        vm.get_avg_utilization(simulation_end) for vm in env.vms
    ]

    # ── 3. Total energy consumed ──────────────────────────────────────────────
    # Sum of energy cost across all VMs.
    # Paper target: AI ≈ 440 kWh, LC ≈ 520 kWh, RR ≈ 550 kWh
    total_energy = sum(vm.total_energy for vm in env.vms)

    # ── 4. Task completion rate (%) ───────────────────────────────────────────
    # Fraction of jobs that completed successfully (response_time <= SLA_THRESHOLD).
    # Paper target: AI ≈ 95%, LC ≈ 88%, RR ≈ 85%
    task_completion = (sum(1 for j in all_jobs if j.success) / total_jobs) * 100

    # ── 5. SLA compliance (%) ─────────────────────────────────────────────────
    # Same calculation as task completion in our simulation — a job either
    # meets the SLA threshold or it doesn't.
    # Paper target: AI ≈ 98%, LC ≈ 92%, RR ≈ 90%
    sla_compliance = task_completion

    return {
        "avg_response_time" : avg_response_time,
        "cpu_utilization"   : cpu_utilization,
        "total_energy"      : total_energy,
        "task_completion"   : task_completion,
        "sla_compliance"    : sla_compliance,
    }


def print_metrics(scheduler_name, metrics):
    """Prints a formatted summary of collected metrics."""
    print(f"\n  {scheduler_name}")
    print(f"    Avg response time  : {metrics['avg_response_time']:.2f} ms")
    print(f"    Total energy       : {metrics['total_energy']:.2f}")
    print(f"    Task completion    : {metrics['task_completion']:.1f}%")
    print(f"    SLA compliance     : {metrics['sla_compliance']:.1f}%")
    print(f"    CPU utilisation per VM:")
    for i, util in enumerate(metrics['cpu_utilization']):
        print(f"      VM{i}: {util:.2f}")
