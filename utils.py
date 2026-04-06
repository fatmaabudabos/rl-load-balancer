from environment.cloud_environment import CloudEnvironment
from algorithms.round_robin.scheduler import RoundRobinScheduler
from algorithms.least_connection.scheduler import LeastConnectionScheduler
from algorithms.rl.agent import RLAgent
from metrics.collector import collect_metrics
from config import NUM_VMS, ARRIVAL_RATES


def create_environment(arrival_rate, seed=42):
    """Creates a fresh CloudEnvironment with the given arrival rate and seed."""
    return CloudEnvironment(arrival_rate=arrival_rate, seed=seed)


def run_scheduler(env, scheduler):
    """
    Runs a traditional scheduler (RR or LC) through all jobs in the environment.
    Returns the collected metrics dictionary.
    """
    env.reset()

    if hasattr(scheduler, 'reset'):
        scheduler.reset()

    state = env.get_state()
    done  = False

    while not done:
        vm_index       = scheduler.select_vm(state)
        state, _, done = env.step(vm_index)

    return collect_metrics(env)


def run_rl_agent(env, n_epochs=5):
    """
    Trains and runs the RL agent across multiple epochs on the same environment.

    Each epoch runs through all jobs once.
    The agent carries its learned weights across epochs — it gets smarter
    each pass. Epsilon starts high (exploring) and decays toward exploitation.

    Returns the metrics from the final epoch only (when the agent is most learned).
    """
    agent = RLAgent()

    for epoch in range(n_epochs):
        env.reset()
        state = env.get_state()
        done  = False

        while not done:
            action                   = agent.select_vm(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state if next_state is not None else state

        if epoch < n_epochs - 1:
            print(f"    Epoch {epoch + 1}/{n_epochs} done | "
                  f"epsilon={agent.epsilon:.4f}")

    print(f"    Epoch {n_epochs}/{n_epochs} done | "
          f"epsilon={agent.epsilon:.4f} (final)")

    # collect metrics from the last epoch (most trained)
    return collect_metrics(env)
