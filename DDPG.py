from asteroid import AsteroidStatic
from action_space import CallCounter
import numpy as np
import torch
import torch.nn as nn
from actor_critic import Actor, Critic, Buffer


class OUNoise:
    def __init__(self, dim=2, theta=0.15, sigma=0.2, dt=1.0):
        self.theta, self.sigma, self.dt, self.dim = theta, sigma, dt, dim
        self.reset()
    def reset(self):
        self.x = np.zeros(self.dim)
    def sample(self):
        self.x = self.x + self.theta * (-self.x) * self.dt \
                 + self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)
        return self.x


def ddpg_act(s, actor, buffer, ou, warmup):
    if len(buffer.buffer) < warmup:
        return np.random.uniform(-1, 1, 2)                    # warmup: random fill
    with torch.no_grad():
        a = actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return np.clip(a + ou.sample(), -1.0, 1.0)                # correlated exploration push


def ddpg_learn(batch, actor, critic, actor_target, critic_target,
               actor_opt, critic_opt, gamma, tau):
    states, actions, rewards, next_states, dones = batch

    # --- critic update: 1-step TD target ---
    with torch.no_grad():
        target_q = critic_target(next_states, actor_target(next_states))
        y = rewards + gamma * (1 - dones) * target_q
    critic_loss = nn.MSELoss()(critic(states, actions), y)
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_opt.step()

    # --- actor update (deterministic policy gradient) ---
    actor_loss = -critic(states, actor(states)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # --- soft (Polyak) target updates ---
    for net, tgt in [(actor, actor_target), (critic, critic_target)]:
        for p, tp in zip(net.parameters(), tgt.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def train(seed, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
          tau=0.005, ou_sigma=0.2, batch=128, hidden=256, warmup=2000,
          crash_floor=-0.5, max_episodes=5000, report_every=25,
          report_step_offset=0, trial=None, verbose=False):
    """One fresh DDPG training run. Returns (metric1, best_min_dist, best_success_len)."""
    env = AsteroidStatic(seed=seed)
    counter = CallCounter(env=env)

    actor         = Actor(hidden=hidden)
    critic        = Critic(hidden=hidden)
    actor_target  = Actor(hidden=hidden)
    critic_target = Critic(hidden=hidden)
    actor_target.load_state_dict(actor.state_dict())     # start targets identical
    critic_target.load_state_dict(critic.state_dict())

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    buffer = Buffer()
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
        episode_reward = 0.0
        ep_len = 0
        info = {}

        for t in range(1000):
            a = ddpg_act(s, actor, buffer, ou, warmup)
            s2, r, done, info = env.step(a)
            best_min_dist = min(best_min_dist, info["dist_to_goal"])
            buffer.add(s, a, max(r, crash_floor), s2, done)   # plain 1-step transition
            steps_done += 1
            ep_len += 1
            if len(buffer.buffer) > warmup:
                ddpg_learn(buffer.sample(batch), actor, critic, actor_target,
                           critic_target, actor_opt, critic_opt, gamma, tau)
            s = s2
            episode_reward += r
            if done:
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
