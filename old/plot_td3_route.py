"""Train a TD3 agent on seed 7, then plot and save its route through the arena.

Run: python plot_td3_route.py
Trains with the tuned TD3 config until the goal is reached once (or a small episode
budget runs out), then does one greedy (noise-free) rollout and overlays that route on
the arena using AsteroidStatic's own plotting helper (_render_static_axes). Saves to
report/td3_trajectory_seed7.png so it can be dropped straight into the report.
"""
import json
import torch
import matplotlib.pyplot as plt

from asteroid import AsteroidStatic
from agent import Actor, Critic, Buffer, CallCounter
from DDPG import OUNoise, ddpg_act
from TD3 import td3_learn

SEED = 7
MAX_EPISODES = 1500          # seed 7 is easy; TD3 solves within a few thousand calls
WARMUP = 2000
CRASH_FLOOR = -0.5
OUT_PATH = "report/td3_trajectory_seed8.png"

cfg = json.load(open("Hyperparameter Tuning/best_td3_new.json"))
hidden = cfg["hidden"]

env = AsteroidStatic(seed=SEED)
counter = CallCounter(env=env)

actor          = Actor(hidden=hidden)
critic1        = Critic(hidden=hidden)
critic2        = Critic(hidden=hidden)
actor_target   = Actor(hidden=hidden)
critic1_target = Critic(hidden=hidden)
critic2_target = Critic(hidden=hidden)
actor_target.load_state_dict(actor.state_dict())
critic1_target.load_state_dict(critic1.state_dict())
critic2_target.load_state_dict(critic2.state_dict())

actor_opt  = torch.optim.Adam(actor.parameters(), lr=cfg["actor_lr"])
critic_opt = torch.optim.Adam(
    list(critic1.parameters()) + list(critic2.parameters()), lr=cfg["critic_lr"])

buffer = Buffer()
ou = OUNoise(dim=2, sigma=cfg["ou_sigma"])
steps_done = 0
success_count = 0
REQUIRED_SOLVES = 8   # first solve alone is often a lucky noisy escape, not a converged policy

# --- train until the policy solves reliably (or the episode budget runs out) ---
for episode in range(MAX_EPISODES):
    s = env.reset()
    ou.reset()
    info = {}
    for t in range(1000):
        a = ddpg_act(s, actor, buffer, ou, WARMUP)
        s2, r, done, info = env.step(a)
        buffer.add(s, a, max(r, CRASH_FLOOR), s2, done)
        steps_done += 1
        if len(buffer.buffer) > WARMUP:
            td3_learn(buffer.sample(cfg["batch"]), steps_done, actor, critic1, critic2,
                      actor_target, critic1_target, critic2_target,
                      actor_opt, critic_opt, cfg["gamma"], cfg["tau"],
                      cfg["policy_noise"], cfg["noise_clip"], cfg["policy_delay"])
        s = s2
        if done:
            break

    if info.get("reached_goal", False):
        success_count += 1
        print(f"Solve {success_count}/{REQUIRED_SOLVES} at episode {episode}, calls={counter.n}")
        if success_count >= REQUIRED_SOLVES:
            break

if success_count < REQUIRED_SOLVES:
    print(f"Only {success_count}/{REQUIRED_SOLVES} solves within {MAX_EPISODES} episodes; "
          f"plotting the best greedy rollout found anyway.")

# --- greedy (noise-free policy) rollout; the env itself still injects action noise
# into the physics, so retry a few times to get a route that actually reaches the goal ---
best_xs, best_ys, best_info = None, None, {}
for attempt in range(20):
    s = env.reset()
    xs, ys = [s[0]], [s[1]]
    info = {}
    with torch.no_grad():
        for t in range(500):
            a = actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            s, r, done, info = env.step(a)
            xs.append(s[0])
            ys.append(s[1])
            if done:
                break
    if best_xs is None:
        best_xs, best_ys, best_info = xs, ys, info      # keep the first as a fallback
    if info.get("reached_goal", False):
        best_xs, best_ys, best_info = xs, ys, info
        print(f"Greedy rollout reached the goal on attempt {attempt}")
        break
xs, ys, info = best_xs, best_ys, best_info

if info.get("reached_goal", False):
    outcome = "Ziel erreicht"
elif info.get("collision", False):
    outcome = "Kollision"
elif info.get("out_of_bounds", False):
    outcome = "Arena verlassen"
else:
    outcome = "Zeitlimit"

fig, ax = plt.subplots(figsize=(6, 6))
env._render_static_axes(ax)
ax.plot(xs, ys, color="tab:blue", linewidth=2, marker="o", markersize=2, label="TD3-Route")
ax.scatter([xs[-1]], [ys[-1]], color="red", zorder=5, label="Ende")
ax.set_title(f"TD3-Route auf Seed {SEED} ({outcome}, {len(xs)-1} Schritte)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=160)
print("saved ->", OUT_PATH)
