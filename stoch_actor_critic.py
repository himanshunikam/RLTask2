from asteroid import AsteroidStatic
from collections import deque
import random
from action_space import CallCounter
import numpy as np
import torch
import torch.nn as nn


# --- input normalization (env defaults: world_bounds=(-0.5,4.5,-0.5,4.5), v_max=4.0) ---
NORM_CENTER = torch.tensor([2.0, 2.0, 0.0, 0.0])
NORM_SCALE  = torch.tensor([2.5, 2.5, 4.0, 4.0])

def normalize(s):
    return (s - NORM_CENTER) / NORM_SCALE


import torch.distributions as D
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
class Actor(nn.Module):
    def __init__(self, s_dim=4, a_dim=2, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mean    = nn.Linear(hidden, a_dim)
        self.log_std = nn.Parameter(torch.zeros(a_dim))   # learnable, state-independent std

    

    def forward(self, s):
        h = self.body(normalize(s))
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return D.Normal(self.mean(h), log_std.exp())


class Critic(nn.Module):
    def __init__(self, s_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, s):
        return self.net(normalize(s)).squeeze(-1)      # note: no action input





def train(seed, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, hidden=256, beta= 0.01,
          crash_floor=-0.5, max_episodes=5000, report_every=25,
          report_step_offset=0, trial=None, verbose=False):
    """One fresh training run. Returns (metric1_calls_to_first_solve, best_min_dist, best_success_len)."""
    env = AsteroidStatic(seed=seed)
    counter = CallCounter(env=env)
    metric1_calls_to_first_solve = None   # calls when goal first reached
    best_success_len             = None   # shortest successful episode
    metric2_calls                = None   # calls when that shortest length was first hit
    best_min_dist                = 1e9    # closest to goal across the whole run
    success_count                = 0
    epsilon = 1.0
    steps_done = 0

    actor  = Actor(hidden=hidden)
    critic = Critic(hidden=hidden)
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    for episode in range(max_episodes):
        s = env.reset()
        I = 1.0         # the slide's discount accumulator
        episode_reward = 0.0
        ep_len = 0
                         
        for t in range(1000):
            s_t = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
            dist = actor(s_t)
            a    = dist.sample()                 # exploration is HERE (sampling)
            logp = dist.log_prob(a).sum(-1)      # keep grad
            a_env = a.clamp(-1, 1).squeeze(0).detach().numpy()

            s2, r, done, info = env.step(a_env)
            ep_len += 1
            episode_reward += r
            best_min_dist = min(best_min_dist, info["dist_to_goal"])

            v_s  = critic(s_t)
            with torch.no_grad():
                v_s2 = critic(torch.as_tensor(s2, dtype=torch.float32).unsqueeze(0))
                r_c = max(r, crash_floor)
                target = r_c + gamma * (1.0 - done) * v_s2      # V̂(S')=0 if terminal
            delta = target - v_s                                  # TD error δ

            critic_loss = delta.pow(2).mean()
            actor_loss  = -(I * logp * delta.detach()).mean() - beta * dist.entropy().sum(-1).mean()

            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            actor_opt.zero_grad();  actor_loss.backward();  actor_opt.step()

            I *= gamma                            # I ← γI  (the slide's factor)
            s = s2
            if done:
                break


        reached = bool(info.get("reached_goal", False))
        if reached:
            success_count += 1
            if metric1_calls_to_first_solve is None:              # Metric 1: first solve
                metric1_calls_to_first_solve = counter.n
            if best_success_len is None or ep_len < best_success_len:   # Metric 2: fastest path
                best_success_len = ep_len
                metric2_calls = counter.n


        if verbose:
            
            success_rate = success_count / (episode + 1)
            print(f"ep {episode:4d} | len {ep_len:3d} | reward {episode_reward:8.2f} | "
                  f"eps {epsilon:.3f} | reached {reached} | minDist {best_min_dist:5.2f} | "
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
    return metric1_calls_to_first_solve, best_min_dist, best_success_len, metric2_calls


if __name__ == "__main__":
    train(seed=2)
