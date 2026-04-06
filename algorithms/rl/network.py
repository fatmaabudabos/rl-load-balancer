import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Feedforward neural network that approximates Q-values.

    Maps the current state to one Q-value per VM.
    The agent picks the VM with the highest Q-value.

    Architecture:
        Input  (7)  → [job_cpu, job_memory, job_priority, vm0_util, vm1_util, vm2_util, vm3_util]
        Hidden (64) → ReLU
        Hidden (64) → ReLU
        Output (4)  → one Q-value per VM
    """

    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),   # input layer → hidden layer 1
            nn.ReLU(),
            nn.Linear(64, 64),          # hidden layer 1 → hidden layer 2
            nn.ReLU(),
            nn.Linear(64, output_dim),  # hidden layer 2 → output (one Q-value per VM)
        )

    def forward(self, x):
        """
        Forward pass.
        x: state tensor of shape (batch_size, input_dim)
        returns: Q-values tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
