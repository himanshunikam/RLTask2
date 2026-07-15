from asteroid import AsteroidStatic
from collections import deque
from action_space import CallCounter
import numpy as np
import torch
import torch.nn as nn
from actor_critic import Actor, Critic, Buffer, get_n_step_transition
from DDPG import OUNoise, ddpg_act        # OU exploration is identical to DDPG


def td3_learn(batch, steps, actor, critic1, critic2,
              actor_target, critic1_target, critic2_target,
              actor_opt, critic_opt, gamma, n_step, tau,
              policy_noise, noise_clip, policy_delay):
    states, actions, rewards, next_states, dones = batch

    # --- critic update: twin critics + min, with target-policy smoothing ---
    with torch.no_grad():
        noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
        next_actions = (actor_target(next_states) + noise).clamp(-1.0, 1.0)
        target_q = torch.min(critic1_target(next_states, next_actions),
                             critic2_target(next_states, next_actions))
        y = rewards + (gamma ** n_step) * (1 - dones) * target_q
    critic_loss = nn.MSELoss()(critic1(states, actions), y) \
                + nn.MSELoss()(critic2(states, actions), y)
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(critic1.parameters()) + list(critic2.parameters()), 10.0)
    critic_opt.step()

    # --- delayed actor + target updates ---
    if steps % policy_delay == 0:
        actor_loss = -critic1(states, actor(states)).mean()     # actor uses critic1 only
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        for net, tgt in [(actor, actor_target), (critic1, critic1_target), (critic2, critic2_target)]:
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)


def train(seed, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, n_step=3,
          tau=0.005, ou_sigma=0.2, policy_noise=0.2, noise_clip=0.5,
          policy_delay=2, batch=128, hidden=256, warmup=2000,
          crash_floor=-0.5, max_episodes=5000, report_every=25,
          report_step_offset=0, trial=None, verbose=False):
    """One fresh TD3 training run. Returns (metric1, best_min_dist, best_success_len)."""
    env = AsteroidStatic(seed=seed)
    counter = CallCounter(env=env)

    actor          = Actor(hidden=hidden)
    critic1        = Critic(hidden=hidden)
    critic2        = Critic(hidden=hidden)
    actor_target   = Actor(hidden=hidden)
    critic1_target = Critic(hidden=hidden)
    critic2_target = Critic(hidden=hidden)
    actor_target.load_state_dict(actor.state_dict())        # start targets identical
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt  = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=critic_lr)

    buffer = Buffer()
    n_step_buffer = deque(maxlen=n_step)
    ou = OUNoise(dim=2, sigma=ou_sigma)
    steps_done = 0

    metric1_calls_to_first_solve = None
    best_success_len             = None
    metric2_calls                = None
    best_min_dist                = 1e9
    success_count                = 0

    for episode in range(max_episodes):
        s = env.reset()
        ou.reset()
        n_step_buffer.clear()
        episode_reward = 0.0
        ep_len = 0
        info = {}

        for t in range(1000):
            a = ddpg_act(s, actor, buffer, ou, warmup)
            s2, r, done, info = env.step(a)
            best_min_dist = min(best_min_dist, info["dist_to_goal"])
            n_step_buffer.append((s, a, max(r, crash_floor), s2, done))
            if len(n_step_buffer) == n_step:
                buffer.add(*get_n_step_transition(n_step_buffer, gamma))
            steps_done += 1
            ep_len += 1
            if len(buffer.buffer) > warmup:
                td3_learn(buffer.sample(batch), steps_done, actor, critic1, critic2,
                          actor_target, critic1_target, critic2_target,
                          actor_opt, critic_opt, gamma, n_step, tau,
                          policy_noise, noise_clip, policy_delay)
            s = s2
            episode_reward += r
            if done:
                while len(n_step_buffer) > 0:            # flush the tail
                    buffer.add(*get_n_step_transition(n_step_buffer, gamma))
                    n_step_buffer.popleft()
                break

        reached = bool(info.get("reached_goal", False))
        if reached:
            success_count += 1
            if metric1_calls_to_first_solve is None:
                metric1_calls_to_first_solve = counter.n
            if best_success_len is None or ep_len < best_success_len:
                best_success_len = ep_len
                metric2_calls = counter.n

        if verbose:
            success_rate = success_count / (episode + 1)
            print(f"ep {episode:4d} | len {ep_len:3d} | reward {episode_reward:8.2f} | "
                  f"reached {reached} | minDist {best_min_dist:5.2f} | "
                  f"calls {counter.n} | succ% {success_rate:.2f}")

        if trial is not None and episode % report_every == 0:
            trial.report(best_min_dist, step=report_step_offset + episode)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    if verbose:
        print("\n=== graded metrics ===")
        print("Metric 1 (calls to first solve):", metric1_calls_to_first_solve)
        print("Metric 2 (shortest success len):", best_success_len, "at calls:", metric2_calls)

    return metric1_calls_to_first_solve, best_min_dist, best_success_len


if __name__ == "__main__":
    train(seed=7, verbose=True)
