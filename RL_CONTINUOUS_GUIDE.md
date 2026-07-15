# Solving `AsteroidStatic` with Continuous-Control Deep RL — Beginner Step-by-Step Guide

> A hands-on guide to solve the Task 2 environment using **Actor-Critic → DDPG → TD3**
> in **PyTorch** — the *continuous-control* family, which acts directly on `[ax, ay]` with **no
> discretization**. Each part ends with a **checkpoint** — get it working before moving on.
>
> This is a companion to `RL_IMPLEMENTATION_GUIDE.md` (the DQN/DDQN/Dueling route). It is a
> **separate track**: create new files, and leave the existing DQN code untouched.

---

## 0. The big picture (read first)

**What the environment is.** `AsteroidStatic` (in `asteroid.py`) is a point-mass spaceship in a 2D
arena full of circular asteroids. You push it with acceleration `[ax, ay]`; it drifts (inertia). You
win by getting within `0.3` of the goal. You lose by hitting an asteroid, leaving the arena, or running
out of time (500 steps).

- **What you observe each step:** `obs = [x, y, vx, vy]` (position + velocity). You do NOT see the goal
  or asteroids — you learn where to go from the reward.
- **What you control:** `action = [ax, ay]`, each a *real number* in `[-1, +1]`. **This is already
  continuous** — `get_action_dim()` is `2`, `get_action_limits()` is `(-1, 1)`.
- **Reward (already shaped):** `+0.5 ×` progress toward goal per step, `+10` for the goal, `−100` for a
  crash. The progress term gently points learning toward the goal.

**Why continuous control (and why *no* discretization).** DQN-family methods need a *finite* action
list, so the DQN track discretizes into 9 bang-bang actions. Actor-Critic / DDPG / TD3 instead learn a
policy that **outputs the continuous vector directly**. Two payoffs here:
1. **Finer control.** The agent can output *gentle* accelerations and true *braking* (accelerate
   opposite to velocity by a small amount) — impossible to do smoothly with 9 full-throttle actions.
   In a dense asteroid field this can mean the difference between threading a gap and overshooting into
   a rock.
2. **Directed exploration.** Instead of ε-greedy random jumps, you explore by adding *noise around the
   current policy* — smoother, more sample-efficient probing of nearby behaviors.

**Your two goals (from Task2.pdf):**
1. Reach the goal using the **fewest `_calc_next_state` calls** (sample-efficient learning).
2. Reach the goal in the **fewest time steps per episode** (fastest path once trained).

**The rules you must respect (Task2.pdf):**
- ❌ Don't modify `asteroid.py` or any provided file.
- ❌ Don't "analyze the environment algorithmically" (no path-planner, don't read asteroid positions).
  Learn only from `obs`, `reward`, `info`.
- ✅ Use default environment settings. Only change the **seed** to get different arenas.
- ✅ You may use any *public* function of the environment.
- ✅ Tune hyperparameters on seeds **disjoint** from your eval seeds (e.g. tune on 100–119, report on 0–9).

**The three methods (each builds on the last):**
| Method | One-line idea | Key change vs. previous |
|---|---|---|
| **Actor-Critic** | An **actor** `μ(s)→a` proposes actions; a **critic** `Q(s,a)` scores them and teaches the actor | the base deterministic actor-critic loop |
| **DDPG** | Professionalize it: replay + **soft** target nets + exploration noise | stability machinery around the AC core |
| **TD3** | Fix DDPG's overestimation & brittleness | **twin critics + min**, **target-policy smoothing**, **delayed** actor updates |

**Tools to install:** `python 3`, `numpy`, `torch` (CPU is fine — nets are tiny), `matplotlib`,
`optuna`, `imageio`.

**Files you'll create (all NEW — do not touch the DQN files):**
```
networks_ac.py    # Actor and Critic networks
replay_ac.py      # replay buffer storing CONTINUOUS actions
agent_ac.py       # one agent, switchable Actor-Critic / DDPG / TD3 via flags
trainer_ac.py     # training loop + metric counting
tune_ac.py        # Optuna hyperparameter search
main_ac.py        # final run over 10 arenas -> the 3 report numbers + plots
readme_ac.txt     # how to run
```
> You may reuse the `CallCounter` class from the existing `action_space.py` by **importing** it
> (importing is not modifying). Everything else is new.

---

## Part 1 — Wrap the environment (no learning yet)

**Goal of this part:** run episodes with continuous actions and *count* calls.

**Step 1.1 — Look around.** `env = AsteroidStatic(seed=0)`; print `env.get_observation_dim()` (=4),
`env.get_action_dim()` (=2), `env.get_action_limits()` (=(-1,1)). `obs = env.reset()`; try a few
`env.step(np.array([0.3, -0.2]))` — note that *any* real vector in `[-1,1]²` is valid (the env clips).

**Step 1.2 — Action helper.** There is nothing to discretize. Your actor will output a vector in
`[-1, 1]²` (via `tanh`), and you pass it straight to `env.step`. Just keep a clip for safety:
```python
import numpy as np
A_MAX = 1.0
def clip_action(a): return np.clip(a, -A_MAX, A_MAX)
```

**Step 1.3 — Count `_calc_next_state` without editing the file.** Reuse the `CallCounter` wrapper (same
as the DQN track). Because you only take *real* steps, **calls == steps**.

**Step 1.4 — Random agent sanity loop.** Reset, then repeatedly sample `a = np.random.uniform(-1, 1, 2)`,
step until `done`. Print episode length, total reward, `info["reached_goal"]`. Expect most episodes to
crash quickly — this is a hard, obstacle-dense arena (25 asteroids, goal ≥3 units away). That motivates
learning *and* good exploration.

> ✅ **Checkpoint 1:** You can run full episodes with continuous actions and your counter matches the
> step count exactly.

---

## Part 2 — Actor-Critic core (deterministic)

**Idea.** Keep two networks:
- an **actor** `μ(s)` that outputs an action (a *deterministic* policy), and
- a **critic** `Q(s, a)` that estimates the value of taking action `a` in state `s`.

The critic learns like DQN (regress toward `r + γ Q(s', μ(s'))`). The actor learns to **output actions
the critic rates highly** — you literally push the actor to *maximize* `Q(s, μ(s))`. That gradient
flowing from critic into actor is the "deterministic policy gradient."

**Step 2.1 — The networks** (`networks_ac.py`). *Always normalize the input first* — we learned in the
DQN track that this is essential for stability. Use the env defaults: positions centered at `2.0` and
divided by `2.5` (world is `[-0.5, 4.5]`), velocities divided by `4.0` (`v_max`).
```python
import torch, torch.nn as nn

NORM_CENTER = torch.tensor([2.0, 2.0, 0.0, 0.0])
NORM_SCALE  = torch.tensor([2.5, 2.5, 4.0, 4.0])
def normalize(s): return (s - NORM_CENTER) / NORM_SCALE

class Actor(nn.Module):
    def __init__(self, s_dim=4, a_dim=2, hidden=256, a_max=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, a_dim))
        self.a_max = a_max
    def forward(self, s):
        return self.a_max * torch.tanh(self.net(normalize(s)))   # squashes output into [-a_max, a_max]

class Critic(nn.Module):
    def __init__(self, s_dim=4, a_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, s, a):
        return self.net(torch.cat([normalize(s), a], dim=1)).squeeze(1)   # shape [batch]
```
*Note:* the `tanh` guarantees the actor's action is always in range — no clipping needed on the *policy*
output (you still clip after adding exploration noise).

**Step 2.2 — Replay buffer** (`replay_ac.py`). Same as the DQN buffer, except **`action` is a
2-D float vector**, not an integer index. Store `(s, a, r, s2, done)`; `sample(batch=128)` returns
tensors (`actions` as `float32`, shape `[batch, 2]`). `collections.deque(maxlen=100000)`.

**Step 2.3 — Exploration by noise** (in `agent_ac.py`). Deterministic policies don't explore on their
own. To act: `a = μ(s) + N(0, σ)`, then clip to `[-1, 1]`. Start `σ` around `0.2` and optionally decay
toward `~0.05`. (This replaces ε-greedy.)
```python
def act(self, s, noise_std):
    with torch.no_grad():
        a = self.actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    a = a + np.random.normal(0.0, noise_std, size=2)
    return np.clip(a, -1.0, 1.0)
```

**Step 2.4 — The learning step.** Keep target copies `actor_target`, `critic_target`. For a batch:
```
# --- critic update (like DQN) ---
with no_grad:
    a2 = actor_target(s2)
    y  = r + gamma * (1 - done) * critic_target(s2, a2)
critic_loss = MSE(critic(s, a), y)
optimize critic

# --- actor update (deterministic policy gradient) ---
actor_loss = -critic(s, actor(s)).mean()     # push actor toward high-Q actions
optimize actor

# --- refresh targets (for the base AC, a periodic hard copy is fine) ---
every C steps: copy actor->actor_target, critic->critic_target
```
Use `Adam` (`actor_lr ≈ 1e-4`, `critic_lr ≈ 1e-3`). This base version already learns on easy arenas.

> ✅ **Checkpoint 2:** On an easy seed (try `seed=7`), over a few hundred episodes the critic loss
> settles, episode length trends **down**, and `reached_goal` starts becoming `True`.

---

## Part 3 — DDPG (professionalize the actor-critic)

DDPG *is* the deterministic actor-critic above, plus two stabilizers that make it reliably trainable.

**Change 1 — Soft ("Polyak") target updates.** Instead of a periodic hard copy, nudge the targets a
little **every** step. This is smoother and is the single biggest stability win:
```python
tau = 0.005
for p, tp in zip(net.parameters(), target.parameters()):
    tp.data.mul_(1 - tau).add_(tau * p.data)
```
Do this for both actor and critic targets, every update.

**Change 2 — Proper exploration noise.** Gaussian noise (Step 2.3) works well; the original DDPG paper
used temporally-correlated **Ornstein-Uhlenbeck** noise, which produces smoother "sustained pushes" —
worth trying on this inertial env. Add a `warmup` of pure-random actions (e.g. first 1–5k steps) to
fill the buffer before learning.

**Full DDPG loop** (`trainer_ac.py`):
```
for episode in range(max_episodes):
    s = env.reset()
    for t in range(max_steps):
        a = agent.act(s, noise_std)                 # random during warmup
        s2, r, done, info = env.step(a)             # a is the continuous vector
        buffer.add(s, a, r, s2, done)
        if len(buffer) > warmup: agent.learn(buffer.sample(batch))   # soft-updates every learn
        s = s2
        if done: break
```
**Starter hyperparameters:** `hidden=256`, `gamma=0.99`, `actor_lr=1e-4`, `critic_lr=1e-3`,
`batch=128`, `buffer=100000`, `warmup=2000`, `tau=0.005`, `noise_std=0.1–0.2`.

> ✅ **Checkpoint 3:** DDPG learns at least as reliably as the base AC (usually much steadier). Keep both
> for the report.

---

## Part 4 — TD3 (three tricks over DDPG)

DDPG has one notorious weakness: the critic **over-estimates** `Q` (like vanilla DQN), and the actor
then exploits those errors. TD3 = DDPG + three fixes. Add flags so one agent covers all three methods.

**Trick 1 — Twin critics + min (clipped double-Q).** Keep **two** critics `Q1, Q2` (and two targets).
Use the **smaller** of the two target values to form the learning target — this curbs overestimation:
```
with no_grad:
    a2 = clip(actor_target(s2) + target_noise, -1, 1)      # see Trick 2
    y  = r + gamma * (1 - done) * min(critic1_target(s2, a2), critic2_target(s2, a2))
critic_loss = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
```

**Trick 2 — Target-policy smoothing.** Add small *clipped* noise to the target action so the critic
can't chase a sharp peak (regularizes the value estimate):
```python
noise = (torch.randn_like(a2) * policy_noise).clamp(-noise_clip, noise_clip)   # e.g. 0.2, 0.5
```

**Trick 3 — Delayed policy updates.** Update the **actor and the targets only every `d` critic
updates** (`d=2`). The critic gets more accurate before the actor trusts it:
```
if step % d == 0:
    actor_loss = -critic1(s, actor(s)).mean()   # actor uses Q1 only
    optimize actor
    soft-update all four targets
```
**TD3 starter additions:** `policy_noise=0.2`, `noise_clip=0.5`, `policy_delay=2`.

> ✅ **Checkpoint 4:** One `AgentAC(twin, policy_delay, policy_noise)` class produces **Actor-Critic**
> (twin=False, delay=1, hard/soft target), **DDPG** (twin=False, delay=1, soft target + noise), and
> **TD3** (twin=True, delay=2, smoothing). This mirrors the DQN track's `(double, dueling)` flags.

---

## Part 5 — Measure the two graded numbers

Reuse the `CallCounter` from Step 1.3 and count exactly as in the DQN track:

- **Metric 1 — calls to first solve:** `counter.n` at the end of the **first** episode where
  `info["reached_goal"]` is `True`.
- **Metric 2 — fewest time steps:** the **shortest** successful episode length seen, and the `counter.n`
  at the moment that shortest length was first achieved (the ungraded "associated calls").

Run for **seeds 0–9** and **average**. Those averages are your three report numbers.

> ✅ **Checkpoint 5:** `evaluate(agent_cfg, seeds)` returns the three averaged numbers. (Refactor
> training into a `train(seed, cfg)` function so nothing leaks between seeds.)

---

## Part 6 — Tune hyperparameters with Optuna

**Critical rule (Task2.pdf):** tune on **different seeds** than your final eval — e.g. tune on
`100–119`, report on `0–9`. The knobs that matter most for continuous control:
```python
import optuna
def objective(trial):
    actor_lr  = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True)
    tau       = trial.suggest_float("tau", 1e-3, 2e-2, log=True)
    noise_std = trial.suggest_float("noise_std", 0.05, 0.4)
    policy_delay = trial.suggest_int("policy_delay", 1, 4)   # 1 ~ DDPG-ish, 2 ~ TD3
    return mean_calls_to_first_solve(cfg(...), seeds=range(100, 120))
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)
```
Run one study aimed at **Metric 1** and (optionally) one aimed at **Metric 2**.

> ✅ **Checkpoint 6:** A tuned config (or two), found on non-eval seeds.

---

## Part 7 — Final run, plots, and the report

**`main_ac.py`:** load the tuned config, run over seeds `0–9`, print the three averaged numbers, and save:
- a **learning curve** (episode length & success vs. number of calls), and
- a **trajectory plot** of the trained policy (roll out the actor *without* exploration noise and overlay
  the path on `env.save_frame(...)`).

**`readme_ac.txt`:** one paragraph on how to run (`python main_ac.py`), deps, expected outputs.

**Report (≤5 pages):** describe the continuous-control approach and **why you didn't discretize**; give
an **Actor-Critic vs DDPG vs TD3** comparison table on both metrics; your best result + trajectory plot;
and a short note on **SAC** (Soft Actor-Critic — stochastic policy + entropy bonus, strong exploration,
a natural next step you considered) and **PPO** (on-policy, robust, no replay). Show what *didn't* work.

> ✅ **Checkpoint 7 (done):** `python main_ac.py` prints the three numbers and saves both plots.

---

## Debugging — if it won't learn (lessons carried over from the DQN track)

This env is a genuinely **hard-exploration** task (25 obstacles, goal ≥3 units away, fast noisy
dynamics). Empirically, undirected exploration reaches the goal on only ~0–0.3% of episodes and **never**
on some seeds. Expect the same difficulty here. Practical fixes:

- **Always normalize inputs** (positions ÷2.5 centered, velocities ÷4). Skipping this destabilizes both
  actor and critic. This was the biggest single stability lever in the DQN track.
- **Critic diverges / Q explodes:** lower `critic_lr`, use gradient clipping (`clip_grad_norm_` at ~10),
  make sure `tau` isn't too large. TD3's twin-min directly attacks this — if DDPG is unstable, go TD3.
- **Actor collapses to a corner (always full throttle):** too much exploration noise or `actor_lr` too
  high; the `tanh` saturates. Lower `actor_lr` and `noise_std`.
- **Never reaches the goal / stuck hovering:** the shaped reward's straight-line "progress" term lures
  the agent into the obstacle wall in front of the goal, so it learns to *flee to safety*. Same trap as
  the DQN track. Levers: keep exploration alive longer (higher `noise_std` for longer), and consider the
  reward-shaping tricks that helped there — **asymmetric reward clipping** (floor the −100 crash at a
  small negative like −0.5 so the agent is bold enough to venture toward the goal, while keeping the +10
  goal bonus large and salient). Note any reward reshaping in the report.
- **Overestimation makes it look good then collapse:** that's exactly what TD3's clipped-double-Q and
  delayed updates fix — the main reason to prefer TD3 over DDPG here.
- **Counter ≠ steps:** you're stepping somewhere extra (e.g. a stray eval rollout) — count every step.

## Suggested order to build (milestones)
Part 1 → random agent runs & counts → Part 2 base Actor-Critic solves an easy seed → Part 3 DDPG →
Part 4 TD3 → Part 5 metrics over 10 seeds → Part 6 Optuna → Part 7 final numbers + plots + report.

## Algorithm cheat-sheet (for the report)

| Method | One-line idea | Key change vs. previous |
|---|---|---|
| **Actor-Critic (deterministic)** | Actor `μ(s)` proposes, critic `Q(s,a)` scores and teaches it | base deterministic policy gradient |
| **DDPG** | Off-policy deterministic AC done right | + **replay** + **soft target nets (τ)** + exploration noise |
| **TD3** | Reduce DDPG overestimation & brittleness | + **twin critics (min)** + **target-policy smoothing** + **delayed actor updates** |
| **SAC** | Stochastic policy with entropy bonus | maximizes reward **and** exploration (entropy); very sample-efficient |
| **PPO** | On-policy, clipped policy-gradient | robust, simple, **no replay** (less sample-efficient here) |
