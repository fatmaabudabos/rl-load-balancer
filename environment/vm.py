from config import OVERLOAD_THRESHOLD


class VM:
    def __init__(self, vm_id, cpu_capacity, memory_capacity, energy_rate):
        # ── Identity & capacity ───────────────────────────────────────────────
        self.vm_id           = vm_id
        self.cpu_capacity    = cpu_capacity      # total CPU units available
        self.memory_capacity = memory_capacity   # total memory units available
        self.energy_rate     = energy_rate       # energy consumed per ms of execution (kWh)

        # ── Runtime state ─────────────────────────────────────────────────────
        self.queue          = []    # jobs waiting to be executed
        self.cpu_used       = 0.0   # CPU units currently in use
        self.available_time = 0.0   # simulation clock time when VM is free

        # ── History (for metrics) ─────────────────────────────────────────────
        self.total_energy   = 0.0   # total energy consumed across all jobs
        self.completed_jobs = []    # all jobs this VM has finished

    # ── Utilisation ───────────────────────────────────────────────────────────
    def get_utilization(self):
        """Returns CPU utilisation as a fraction between 0.0 and 1.0."""
        return min(self.cpu_used / self.cpu_capacity, 1.0)

    def is_overloaded(self):
        """Returns True if utilisation exceeds the overload threshold."""
        return self.get_utilization() >= OVERLOAD_THRESHOLD

    # ── Job assignment ────────────────────────────────────────────────────────
    def assign_job(self, job, current_time):
        """
        Assigns a job to this VM, calculates wait/exe/response times,
        updates VM state, and marks the job outcome.
        """
        # wait time = how long until VM is free from current time
        wait_time = max(0.0, self.available_time - current_time)

        # execution time = job's CPU demand scaled by VM capacity (ms)
        exe_time = (job.cpu_demand / self.cpu_capacity) * 1000

        # priority scaling — higher priority jobs execute faster
        priority_factor = 1.0 / job.priority
        exe_time *= priority_factor

        # fill in job timing fields
        job.wait_time     = wait_time
        job.exe_time      = exe_time
        job.response_time = wait_time + exe_time

        # energy cost for this job
        job.cost = exe_time * self.energy_rate

        # update VM state
        self.cpu_used        = min(self.cpu_used + job.cpu_demand, self.cpu_capacity)
        self.available_time  = current_time + wait_time + exe_time
        self.total_energy   += job.cost
        self.completed_jobs.append(job)
        self.queue.append(job)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        """Resets VM to initial state for a fresh simulation run."""
        self.queue          = []
        self.cpu_used       = 0.0
        self.available_time = 0.0
        self.total_energy   = 0.0
        self.completed_jobs = []
