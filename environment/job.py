class Job:
    def __init__(self, job_id, arrival_time, cpu_demand, memory_demand, priority):
        # ── Identity & arrival ─────────────────────────────────────────────────
        self.job_id       = job_id
        self.arrival_time = arrival_time   # simulation time when job arrived (ms)

        # ── Resource demands ───────────────────────────────────────────────────
        self.cpu_demand    = cpu_demand       # fraction of VM CPU needed (0.0 - 1.0)
        self.memory_demand = memory_demand    # fraction of VM memory needed (0.0 - 1.0)
        self.priority      = priority         # 1 = low, 2 = medium, 3 = high

        # ── Timing (filled in after assignment) ────────────────────────────────
        self.wait_time     = 0.0   # time spent waiting in VM queue (ms)
        self.exe_time      = 0.0   # time spent executing on VM (ms)
        self.response_time = 0.0   # wait_time + exe_time (ms)

        # ── Outcome (filled in after execution) ────────────────────────────────
        self.success = False   # True if response_time <= SLA_THRESHOLD
