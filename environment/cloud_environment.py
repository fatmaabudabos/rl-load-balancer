import numpy as np
from environment.vm import VM
from environment.job import Job
from config import NUM_VMS, NUM_JOBS, SLA_THRESHOLD, OVERLOAD_THRESHOLD


class CloudEnvironment:
    def __init__(self, arrival_rate, seed=42):
        self.arrival_rate = arrival_rate   # jobs per second
        self.seed         = seed
        self.rng          = np.random.default_rng(seed)

        self.vms  = self._create_vms()
        self.jobs = self._generate_jobs()

        self.current_job_index = 0   # pointer to the next job to be scheduled
        self.current_time      = 0.0 # simulation clock (ms)

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _create_vms(self):
        """
        Creates 4 VMs with varied capacities.
        VM0 and VM2 are strong (cpu=4.0), VM1 and VM3 are weak (cpu=2.0).
        This makes scheduling non-trivial — a dumb scheduler will overload the weak VMs.
        """
        configs = [
            # (vm_id, cpu_capacity, memory_capacity)
            (0, 4.0, 8.0),
            (1, 2.0, 4.0),
            (2, 4.0, 8.0),
            (3, 2.0, 4.0),
        ]
        return [VM(vid, cpu, mem) for vid, cpu, mem in configs]

    def _generate_jobs(self):
        """
        Generates NUM_JOBS jobs with random demands and Poisson inter-arrivals.
        Higher arrival_rate = jobs arrive closer together = more system pressure.
        """
        jobs         = []
        current_time = 0.0

        for i in range(NUM_JOBS):
            inter_arrival = self.rng.exponential(1000.0 / self.arrival_rate)
            current_time += inter_arrival

            jobs.append(Job(
                job_id        = i,
                arrival_time  = current_time,
                cpu_demand    = self.rng.uniform(0.1, 1.0),
                memory_demand = self.rng.uniform(0.1, 1.0),
                priority      = int(self.rng.integers(1, 4)),
            ))

        return jobs

    # ── State ──────────────────────────────────────────────────────────────────
    def get_state(self):
        """
        Returns the current state as a numpy array:
        [job_cpu, job_mem, job_priority, vm0_util, vm1_util, vm2_util, vm3_util]
        Used by the RL agent (added in a later phase).
        """
        job          = self.jobs[self.current_job_index]
        job_features = [job.cpu_demand, job.memory_demand, job.priority / 3.0]
        vm_utils     = [vm.get_utilization(self.current_time) for vm in self.vms]
        return np.array(job_features + vm_utils, dtype=np.float32)

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, vm_index):
        """
        Assigns the current job to the chosen VM and advances the simulation.
        Returns (next_state, done).
        """
        job               = self.jobs[self.current_job_index]
        self.current_time = job.arrival_time

        self.vms[vm_index].assign_job(job, self.current_time)
        job.success = job.response_time <= SLA_THRESHOLD

        self.current_job_index += 1
        done       = self.current_job_index >= len(self.jobs)
        next_state = self.get_state() if not done else None

        return next_state, done

    # ── Migration check ────────────────────────────────────────────────────────
    def check_migration(self):
        """Returns VM ids currently overloaded. Used in a later phase."""
        return [vm.vm_id for vm in self.vms if vm.is_overloaded(self.current_time)]

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self):
        """Resets all VMs for a fresh simulation run."""
        for vm in self.vms:
            vm.reset()
        self.current_job_index = 0
        self.current_time      = 0.0
