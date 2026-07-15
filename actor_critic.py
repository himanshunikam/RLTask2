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


class Actor(nn.Module):
    def __init__(self, s_dim=4, a_dim=2, hidden=256, a_max=1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, a_dim)
        )
        self.a_max = a_max

    def forward(self, s):
        return self.a_max * torch.tanh(self.net(normalize(s)))


class Critic(nn.Module):
    def __init__(self, sdim=4, adim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim + adim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([normalize(s), a], dim=1)).squeeze(1)


class Buffer():
    def __init__(self) -> None:
        self.buffer = deque(maxlen=100000)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch=128):
        self.batch_array = random.sample(self.buffer, batch)
        states, actions, rewards, next_states, dones = zip(*self.batch_array)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32))


def get_n_step_transition(n_buf, gamma):
    s0, a0 = n_buf[0][0], n_buf[0][1]
    R = 0.0
    next_s, done = n_buf[-1][3], n_buf[-1][4]
    for k, (_, _, r, s2, d) in enumerate(n_buf):
        R += (gamma ** k) * r
        if d:
            next_s, done = s2, True
            break
    return s0, a0, R, next_s, done


def act(s, actor, buffer, epsilon, noise_std, warmup):
    if len(buffer.buffer) < warmup:
        return np.random.choice([-1.0, 1.0], size=2)          # warmup: bang-bang random
    if np.random.random() < epsilon:
        return np.random.choice([-1.0, 1.0], size=2)          # epsilon injection: full-range random
    with torch.no_grad():
        a = actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    a = a + np.random.normal(0.0, noise_std, size=2)
    return np.clip(a, -1.0, 1.0)


def learn(batch, steps, actor, critic, actor_target, critic_target,
          actor_opt, critic_opt, gamma, n_step, target_update_freq):
    states, actions, rewards, next_states, dones = batch

    # --- critic update: regress Q(s,a) toward n-step return + gamma^n * Q'(s', mu'(s')) ---
    with torch.no_grad():
        next_actions = actor_target(next_states)
        target_q     = critic_target(next_states, next_actions)
        y = rewards + (gamma ** n_step) * (1 - dones) * target_q
    q = critic(states, actions)
    critic_loss = nn.MSELoss()(q, y)
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_opt.step()

    # --- actor update: deterministic policy gradient, maximize Q(s, mu(s)) ---
    actor_loss = -critic(states, actor(states)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # --- refresh targets: hard copy every C steps ---
    if steps % target_update_freq == 0:
        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())


def train(seed, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, n_step=3,
          noise_std=0.2, epsilon_min=0.2, epsilon_decay=0.99,
          target_update_freq=500, batch=128, hidden=256, warmup=2000,
          crash_floor=-0.5, max_episodes=5000, report_every=25,
          report_step_offset=0, trial=None, verbose=False):
    """One fresh training run. Returns (metric1_calls_to_first_solve, best_min_dist, best_success_len)."""
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
    n_step_buffer = deque(maxlen=n_step)
    epsilon = 1.0
    steps_done = 0

    # --- graded metrics (guide Part 5) ---
    metric1_calls_to_first_solve = None   # calls when goal first reached
    best_success_len             = None   # shortest successful episode
    metric2_calls                = None   # calls when that shortest length was first hit
    best_min_dist                = 1e9    # closest to goal across the whole run
    success_count                = 0

    for episode in range(max_episodes):
        s = env.reset()
        n_step_buffer.clear()
        episode_reward = 0.0
        ep_len = 0
        info = {}

        for t in range(1000):
            a = act(s, actor, buffer, epsilon, noise_std, warmup)
            s2, r, done, info = env.step(a)
            best_min_dist = min(best_min_dist, info["dist_to_goal"])
            n_step_buffer.append((s, a, max(r, crash_floor), s2, done))
            if len(n_step_buffer) == n_step:
                buffer.add(*get_n_step_transition(n_step_buffer, gamma))
            steps_done += 1
            ep_len += 1
            if len(buffer.buffer) > warmup:
                learn(buffer.sample(batch), steps_done, actor, critic,
                      actor_target, critic_target, actor_opt, critic_opt,
                      gamma, n_step, target_update_freq)
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
            if metric1_calls_to_first_solve is None:              # Metric 1: first solve
                metric1_calls_to_first_solve = counter.n
            if best_success_len is None or ep_len < best_success_len:   # Metric 2: fastest path
                best_success_len = ep_len
                metric2_calls = counter.n

        epsilon = max(epsilon_min, epsilon * epsilon_decay)   # decay exploration once per episode

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

    return metric1_calls_to_first_solve, best_min_dist, best_success_len


if __name__ == "__main__":
    train(seed=0, verbose=True)
