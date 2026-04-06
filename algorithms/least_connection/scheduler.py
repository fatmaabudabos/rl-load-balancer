class LeastConnectionScheduler:
    def __init__(self, vms):
        self.vms = vms

    def select_vm(self, state):
        """
        Returns the index of the VM that will be free soonest.
        Using available_time as the selection criterion gives a meaningful
        difference from Round Robin — it routes jobs to the least loaded VM
        rather than cycling blindly.
        """
        return min(range(len(self.vms)), key=lambda i: self.vms[i].available_time)

    def reset(self):
        pass   # stateless — nothing to reset
