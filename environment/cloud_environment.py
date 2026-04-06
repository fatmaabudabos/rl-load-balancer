import numpy as np
from environment.vm import VM
from environment.job import Job
from config import (
    NUM_VMS, NUM_JOBS, ALPHA, BETA, SLA_THRESHOLD, OVERLOAD_THRESHOLD, SLA_PENALTY
)


class CloudEnvironment:
    def __init__(self, arrival_rate, seed=42):
        self.arrival_rate = arrival_rate   # jobs per second
        self.seed         = seed
        self.rng          = np.random.default_rng(seed)

        self.vms  = self._create_vms()
        self.jobs = self._generate_jobs()

        self.current_job_index = 0   # pointer to the next job to be scheduled
        self.current_time      = 0.0 # simulation clock (ms)

    # ── Setup ─────────────────────────────────────────────────────────────────
    def _create_vms(self):
        """Creates 4 VMs with varied capacities and energy rates."""
        configs = [
            # (vm_id, cpu_capacity, memory_capacity, energy_rate)
            (0, 4.0, 8.0,  0.05),
            (1, 2.0, 4.0,  0.03),
            (2, 4.0, 8.0,  0.05),
            (3, 2.0, 4.0,  0.03),
        ]
        return [VM(vid, cpu, mem, er) for vid, cpu, mem, er in configs]

    def _generate_jobs(self):
        """
        Generates NUM_JOBS jobs with random demands and inter-arrival times
        based on arrival_rate (Poisson process).
        """
        jobs = []
        current_time = 0.0

        for i in range(NUM_JOBS):
            # inter-arrival time drawn from exponential distribution (ms)
            inter_arrival = self.rng.exponential(1000.0 / self.arrival_rate)
            current_time += inter_arrival

            jobs.append(Job(
                job_id        = i,
                arrival_time  = current_time,
                cpu_demand    = self.rng.uniform(0.1, 1.0),
                memory_demand = self.rng.uniform(0.1, 1.0),
                priority      = int(self.rng.integers(1, 4)),  # 1, 2, or 3
            ))

        return jobs

    # ── State ─────────────────────────────────────────────────────────────────
    def get_state(self):
        """
        Returns the current state as a flat numpy array of size 7:
        [job_cpu, job_mem, job_priority,
         vm0_util, vm1_util, vm2_util, vm3_util]

        VM utilisation is computed as the fraction of a 500ms reference
        window that the VM's backlog occupies (clamped to 0-1).
        This gives the agent a clear signal about which VMs are free
        vs loaded at normal operating arrival rates.
        """
        job = self.jobs[self.current_job_index]

        job_features = [
            job.cpu_demand,
            job.memory_demand,
            job.priority / 3.0,
        ]

        vm_utils = [vm.get_utilization(self.current_time) for vm in self.vms]

        return np.array(job_features + vm_utils, dtype=np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────
    def calculate_reward(self, job):
        """
        R = α(1 - Umax) + β(1 - mean(Ui)) - SLA_PENALTY (if SLA violated)

        Base formula taken directly from the paper:
        - Umax      = utilisation of the most loaded VM (penalises overload)
        - mean(Ui)  = average utilisation across all VMs (penalises imbalance)

        SLA penalty added to give the agent a direct signal when a job
        exceeds the response time threshold — without this the agent can
        balance utilisation but still produce terrible response times.
        """
        utilizations = [vm.get_utilization(self.current_time) for vm in self.vms]
        u_max  = max(utilizations)
        u_mean = sum(utilizations) / len(utilizations)

        reward = ALPHA * (1 - u_max) + BETA * (1 - u_mean)

        # deduct penalty if this job violated the SLA
        if not job.success:
            reward -= SLA_PENALTY

        return reward

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, vm_index):
        """
        Assigns the current job to the chosen VM, advances the simulation,
        and returns (next_state, reward, done).

        Called once per job by any scheduler or the RL agent.
        """
        job = self.jobs[self.current_job_index]
        self.current_time = job.arrival_time

        # assign job to chosen VM
        self.vms[vm_index].assign_job(job, self.current_time)

        # mark job success based on SLA threshold
        job.success = job.response_time <= SLA_THRESHOLD

        # compute reward after assignment (job fields now populated)
        reward = self.calculate_reward(job)

        # advance to next job
        self.current_job_index += 1
        done = self.current_job_index >= len(self.jobs)

        next_state = self.get_state() if not done else None

        return next_state, reward, done

    # ── Migration check ───────────────────────────────────────────────────────
    def check_migration(self):
        """
        Returns a list of VM ids that are currently overloaded (util >= 80%).
        Migration logic will be added in a later milestone.
        """
        return [vm.vm_id for vm in self.vms if vm.is_overloaded(self.current_time)]

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        """Resets the environment for a fresh run with the same jobs and VMs."""
        for vm in self.vms:
            vm.reset()
        self.current_job_index = 0
        self.current_time      = 0.0
