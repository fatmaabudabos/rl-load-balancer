from config import OVERLOAD_THRESHOLD


class VM:
    def __init__(self, vm_id, cpu_capacity, memory_capacity):
        # ── Identity & capacity ────────────────────────────────────────────────
        self.vm_id           = vm_id
        self.cpu_capacity    = cpu_capacity      # total CPU units available
        self.memory_capacity = memory_capacity   # total memory units available

        # ── Runtime state ──────────────────────────────────────────────────────
        self.queue          = []    # jobs assigned to this VM
        self.available_time = 0.0   # when this VM finishes all queued work (ms)

        # ── History (for metrics) ──────────────────────────────────────────────
        self.completed_jobs = []    # all jobs this VM has finished

    # ── Utilisation ───────────────────────────────────────────────────────────
    def get_utilization(self, current_time=0.0):
        """
        Real-time utilisation (0.0 - 1.0).
        Based on how much backlog exists relative to a 500ms reference window.
        Returns 0 if the VM is currently free.
        """
        if self.available_time <= current_time:
            return 0.0
        return min((self.available_time - current_time) / 500.0, 1.0)

    def get_avg_utilization(self, simulation_end_time):
        """
        Average CPU utilisation over the full simulation.
        = total time spent executing / total simulation time.
        Used for the CPU utilisation plot.
        """
        if simulation_end_time <= 0:
            return 0.0
        total_cpu_time = sum(job.exe_time for job in self.completed_jobs)
        return min(total_cpu_time / simulation_end_time, 1.0)

    def is_overloaded(self, current_time=0.0):
        """Returns True if utilisation exceeds the overload threshold."""
        return self.get_utilization(current_time) >= OVERLOAD_THRESHOLD

    # ── Job assignment ─────────────────────────────────────────────────────────
    def assign_job(self, job, current_time):
        """
        Assigns a job to this VM and calculates its timing fields.
        wait_time  = how long the job waits before it starts running
        exe_time   = how long the job takes to run on this VM
        response_time = wait + exe (this is what we measure against SLA)
        """
        wait_time = max(0.0, self.available_time - current_time)
        exe_time  = (job.cpu_demand / self.cpu_capacity) * 1000
        exe_time *= (1.0 / job.priority)   # higher priority = faster execution

        job.wait_time     = wait_time
        job.exe_time      = exe_time
        job.response_time = wait_time + exe_time

        self.available_time = current_time + wait_time + exe_time
        self.completed_jobs.append(job)
        self.queue.append(job)

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self):
        """Resets VM to a clean state for a fresh simulation run."""
        self.queue          = []
        self.available_time = 0.0
        self.completed_jobs = []
