import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from algorithms.rl.network import QNetwork
from config import (
    LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END,
    EPSILON_DECAY, REPLAY_BUFFER_SIZE, BATCH_SIZE,
    TARGET_UPDATE_FREQ, NUM_VMS
)

# state vector size: 3 job features + NUM_VMS utilizations
STATE_DIM  = 3 + NUM_VMS
ACTION_DIM = NUM_VMS


class RLAgent:
    """
    DQN-based RL agent for VM load balancing.

    Implements the RL component described in the paper:
    - Observes the current state (job features + VM utilizations)
    - Selects a VM using epsilon-greedy policy
    - Stores experiences in a replay buffer
    - Trains the online network using sampled batches
    - Periodically syncs the target network for stable training
    """

    def __init__(self):
        # ── Networks ──────────────────────────────────────────────────────────
        # Online network: updated every training step
        self.online_net = QNetwork(STATE_DIM, ACTION_DIM)
        # Target network: frozen copy, synced every TARGET_UPDATE_FREQ steps
        # Provides stable Q-value targets so the agent isn't chasing a moving target
        self.target_net = QNetwork(STATE_DIM, ACTION_DIM)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        # ── Optimiser & loss ──────────────────────────────────────────────────
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn   = nn.MSELoss()

        # ── Replay buffer ─────────────────────────────────────────────────────
        # Stores (state, action, reward, next_state, done) tuples
        # deque automatically drops oldest experience when full
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # ── Exploration parameters ────────────────────────────────────────────
        # Starts at 1.0 (fully random) and decays toward 0.01 (mostly learned)
        self.epsilon = EPSILON_START

        # ── Step counter (used to trigger target network sync) ────────────────
        self.step_count = 0

    # ── Action selection ──────────────────────────────────────────────────────
    def select_vm(self, state):
        """
        Epsilon-greedy VM selection.
        - With probability epsilon: pick a random VM (exploration)
        - Otherwise: pick the VM with the highest Q-value (exploitation)

        This matches the paper's RL decision-making:
        the agent observes state and outputs the best action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape (1, STATE_DIM)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)          # shape (1, ACTION_DIM)
        return q_values.argmax().item()                        # index of highest Q-value

    # ── Memory ────────────────────────────────────────────────────────────────
    def store_experience(self, state, action, reward, next_state, done):
        """Saves one (s, a, r, s', done) tuple to the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self):
        """
        Samples a random batch from the replay buffer and performs one
        gradient update on the online network.

        Uses the Bellman equation:
            Q_target = r + γ * max(Q_target_net(s'))

        Only runs when the buffer has enough experiences to fill a batch.
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return None  # not enough data yet

        # sample a random batch
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert to tensors
        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array([s if s is not None
                                                  else np.zeros(STATE_DIM)
                                                  for s in next_states]))
        dones       = torch.FloatTensor(dones)

        # current Q-values from online network for the actions that were taken
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q-values from target network (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        # compute loss and backpropagate
        loss = self.loss_fn(current_q, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # decay epsilon after every training step
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # sync target network every TARGET_UPDATE_FREQ steps
        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        """Resets exploration rate but keeps learned weights across runs."""
        self.epsilon = EPSILON_START
        self.step_count = 0
