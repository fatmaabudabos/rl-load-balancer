class RoundRobinScheduler:
    def __init__(self, num_vms):
        self.num_vms = num_vms
        self.current_index = 0   # tracks whose turn it is

    def select_vm(self, state):
        """Returns the next VM index in circular order. Ignores state entirely."""
        vm_index = self.current_index
        self.current_index = (self.current_index + 1) % self.num_vms
        return vm_index

    def reset(self):
        self.current_index = 0
