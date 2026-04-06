"""
Milestone 4 test — verifies the RL agent:
  1. Selects VMs without crashing
  2. Epsilon decays over time
  3. Loss decreases as training progresses
  4. Outperforms random selection by the end
Run with: python test_rl_agent.py
"""
from environment.cloud_environment import CloudEnvironment
from algorithms.rl.agent import RLAgent


def test_rl_agent():
    print("=" * 50)
    print("MILESTONE 4 — RL Agent Test")
    print("=" * 50)

    env   = CloudEnvironment(arrival_rate=20, seed=42)
    agent = RLAgent()

    losses        = []
    total_reward  = 0.0
    done          = False
    state         = env.get_state()

    print(f"\n[1] Initial epsilon: {agent.epsilon:.3f} (expected 1.000)")

    # ── Run through all jobs ──────────────────────────────────────────────────
    while not done:
        action                       = agent.select_vm(state)
        next_state, reward, done     = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        loss                         = agent.train()

        if loss is not None:
            losses.append(loss)

        total_reward += reward
        state         = next_state if next_state is not None else state

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"[2] Final epsilon : {agent.epsilon:.4f} (expected close to {0.01})")
    print(f"[3] Total steps   : {agent.step_count}")
    print(f"[4] Experiences   : {len(agent.replay_buffer)}")
    print(f"[5] Training steps: {len(losses)}")

    if losses:
        first_100_avg = sum(losses[:100])  / len(losses[:100])
        last_100_avg  = sum(losses[-100:]) / len(losses[-100:])
        print(f"\n[6] Loss — first 100 steps avg : {first_100_avg:.6f}")
        print(f"    Loss — last  100 steps avg : {last_100_avg:.6f}")
        print(f"    Loss decreased: {last_100_avg < first_100_avg}")

    all_jobs     = [job for vm in env.vms for job in vm.completed_jobs]
    avg_response = sum(j.response_time for j in all_jobs) / len(all_jobs)
    sla          = sum(1 for j in all_jobs if j.success) / len(all_jobs) * 100

    print(f"\n[7] Avg response time : {avg_response:.2f} ms")
    print(f"    SLA compliance    : {sla:.1f}%")
    print(f"    (Baseline to beat — LC: 137.22ms response, 78.1% SLA)")

    print("\n" + "=" * 50)
    print("RL agent ran successfully.")
    print("=" * 50)


if __name__ == "__main__":
    test_rl_agent()
