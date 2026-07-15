from asteroid import AsteroidStatic
from action_space import ACTIONS, NUM_ACTIONS, get_action, CallCounter
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
from network import DQN
from replay import Replay

env = AsteroidStatic(seed=0)
counter = CallCounter(env=env)
alpha = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.2              # keep 20% exploration alive: goal is reached so rarely we must keep sampling
epsilon_decay = 0.9998
crash_floor = -0.5            # how bad a crash is (clip floor). Less negative -> bolder                             # ventures toward the goal instead of hovering. Sweep this: -1.0, -0.5, -0.25.
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 10000

replay = Replay()
n_step = 3
n_step_buffer = deque(maxlen=n_step)

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

# Input normalization (from env defaults: world_bounds=(-0.5,4.5,-0.5,4.5), v_max=4.0).
# Positions centered at 2.0 / half-range 2.5 -> ~[-1,1]; velocities scaled by v_max.
# Broadcasts over [4], [1,4] and [batch,4].
NORM_CENTER = torch.tensor([2.0, 2.0, 0.0, 0.0], dtype=torch.float32)
NORM_SCALE  = torch.tensor([2.5, 2.5, 4.0, 4.0], dtype=torch.float32)
def normalize(states):
    return (states - NORM_CENTER) / NORM_SCALE

def get_n_step_transition(n_buf, gamma):
    """(s0, a0, R, s_n, done) with R = sum_k gamma^k * r_k, truncated at any terminal."""
    s0, a0 = n_buf[0][0], n_buf[0][1]
    R = 0.0
    next_s, done = n_buf[-1][3], n_buf[-1][4]
    for k, (_, _, r, s2, d) in enumerate(n_buf):
        R += (gamma ** k) * r
        if d:
            next_s, done = s2, True
            break
    return s0, a0, R, next_s, done


def select_action(state, epsilon, policy_net : torch.nn.Module, in_features):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS-1)
    else :
        state_tensor = normalize(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        q_values = policy_net(state_tensor)
        return torch.argmax(q_values).item()
    
def optimize_model(steps):
    if len(replay.buffer) < 1000: 
        return
    
    states, actions, rewards, next_states, dones = replay.sample()
    states = normalize(states)
    next_states = normalize(next_states)
    q_vlaues = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        a_star = policy_net(next_states).argmax(dim=1, keepdim=True)          # [batch, 1]
        max_next_q_values = target_net(next_states).gather(1, a_star).squeeze(1)  # [batch]
        y = rewards + (gamma ** n_step) * max_next_q_values*(1- dones)


    loss = nn.SmoothL1Loss()(q_vlaues, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    if steps%target_update_freq ==0:
        target_net.load_state_dict(policy_net.state_dict())


rewards_per_episode = []
episode_lengths = []
steps_done = 0
# --- graded metrics (guide Part 5) ---
metric1_calls_to_first_solve = None   # calls when goal first reached
best_success_len             = None   # shortest successful episode
metric2_calls                = None   # calls when that shortest length was first hit
success_count                = 0

for episode in range(episodes):
    state = env.reset()
    n_step_buffer.clear()
    episode_reward = 0.0
    ep_len = 0
    ep_min_dist = 1e9          # closest the agent got to the goal this episode
    info = {}

    for t in range(1000):
        action = select_action(state, epsilon, policy_net, 9)
        next_state, reward, done, info = env.step(get_action(action))
        ep_min_dist = min(ep_min_dist, info["dist_to_goal"])
        # Asymmetric reward clip: floor the -100 crash at -1 so it can't collapse
        # every Q-value to ~-100 (it would otherwise propagate through the Bellman
        # backup and drown the +0.5*progress shaping). We do NOT cap the positive
        # side, so the +10 goal bonus stays large and salient -- the goal is reached
        # so rarely that we want that event to strongly imprint. (Note in the report.)
        clipped_reward = float(max(reward, crash_floor))
        n_step_buffer.append((state, action, clipped_reward, next_state, done))
        if len(n_step_buffer) == n_step:
            replay.add(*get_n_step_transition(n_step_buffer, gamma))

        steps_done += 1
        ep_len += 1
        optimize_model(steps_done)
        state = next_state
        episode_reward += reward
        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        if done:
            while len(n_step_buffer) > 0:
                replay.add(*get_n_step_transition(n_step_buffer, gamma))
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

    rewards_per_episode.append(episode_reward)
    episode_lengths.append(ep_len)

    success_rate = success_count / (episode + 1)
    print(f"ep {episode:4d} | len {ep_len:3d} | reward {episode_reward:8.2f} | "
          f"eps {epsilon:.3f} | reached {reached} | minDist {ep_min_dist:5.2f} | "
          f"calls {counter.n} | succ% {success_rate:.2f}")

print("\n=== graded metrics ===")
print("Metric 1 (calls to first solve):", metric1_calls_to_first_solve)
print("Metric 2 (shortest success len):", best_success_len, "at calls:", metric2_calls)


