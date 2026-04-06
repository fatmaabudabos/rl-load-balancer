class Job:
    def __init__(self, job_id, arrival_time, cpu_demand, memory_demand, priority):
        # ── Identity & arrival ────────────────────────────────────────────────
        self.job_id       = job_id
        self.arrival_time = arrival_time   # simulation time when job arrived

        # ── Resource demands ──────────────────────────────────────────────────
        self.cpu_demand    = cpu_demand       # fraction of one VM's CPU needed (0.0 - 1.0)
        self.memory_demand = memory_demand    # fraction of one VM's memory needed (0.0 - 1.0)
        self.priority      = priority         # 1 = low, 2 = medium, 3 = high

        # ── Timing (filled in after assignment) ───────────────────────────────
        self.wait_time     = 0.0   # time spent waiting in VM queue (ms)
        self.exe_time      = 0.0   # time spent executing on VM (ms)
        self.response_time = 0.0   # total = wait_time + exe_time (ms)

        # ── Outcome (filled in after execution) ───────────────────────────────
        self.cost    = 0.0    # energy cost charged for this job
        self.success = False  # True if job completed within SLA threshold
