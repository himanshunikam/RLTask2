# Solving `AsteroidStatic` with Deep RL — Beginner Step-by-Step Guide

> A hands-on guide to solve the Task 2 environment yourself using **DQN → DDQN → Dueling DDQN**
> in **PyTorch**. Each part ends with a **checkpoint** — get it working before moving on.

---

## 0. The big picture (read first)

**What the environment is.** `AsteroidStatic` (in `asteroid.py`) is a little spaceship (a point with
mass) in a 2D arena full of circular asteroids. You push it with an acceleration `[ax, ay]`; it drifts
because it has velocity (inertia). You win by getting within `0.3` of the goal. You lose by hitting an
asteroid, leaving the arena, or running out of time (500 steps).

- **What you observe each step:** `obs = [x, y, vx, vy]` (position + velocity). *That's it* — you do NOT
  see where the goal or asteroids are. You must *learn* where to go from the reward.
- **What you control:** `action = [ax, ay]`, each between `-1` and `+1`.
- **Reward (already shaped for you):** `+0.5 ×` (progress toward goal) every step, `+10` for reaching
  the goal, `−100` for crashing. The progress term is your friend — it gently points learning toward the goal.

**Your two goals (from Task2.pdf):**
1. Reach the goal using the **fewest `_calc_next_state` calls** (= fewest environment steps spent
   *learning*). This rewards *sample-efficient* learning.
2. Reach the goal in the **fewest time steps per episode** (= the *fastest* path once trained).

**The rules you must respect (Task2.pdf):**
- ❌ Don't modify `asteroid.py` or any provided file.
- ❌ Don't "analyze the environment algorithmically" (no path-planner, don't read the asteroid
  positions). Learn only from `obs`, `reward`, `info`.
- ✅ Use default environment settings. Only change the **seed** to get different arenas.
- ✅ You may use any *public* function of the environment.

**Why DQN-family + why we discretize.** DQN, DDQN, Dueling DDQN learn a value `Q(s, a)` for a *finite*
list of actions. But our action is continuous. So we pick **9 fixed "bang-bang" actions** —
every combination of `{-1, 0, +1}` for `ax` and `ay`. Full-throttle pushes (including diagonals) are
also the *fastest* way to move a mass, which helps Goal 2.

**Tools to install:** `python 3`, `numpy`, `torch` (PyTorch, CPU is fine — the network is tiny),
`matplotlib`, `optuna`, `imageio` (for the env's GIF/plot saving).

**Files you'll create (all new, next to `asteroid.py`):**
```
action_space.py   # the 9 discrete actions + index<->vector helpers
network.py        # the neural network(s)
replay.py         # the memory buffer
agent.py          # DQN/DDQN/Dueling agent (one class, switches)
trainer.py        # the training loop + metric counting
tune.py           # Optuna hyperparameter search
main.py           # final run over 10 arenas -> the 3 report numbers + plots
readme.txt        # how to run
```

---

## Part 1 — Wrap the environment (no learning yet)

**Goal of this part:** be able to run episodes, take the 9 discrete actions, and *count* calls.

**Step 1.1 — Look around.** In a scratch script: `env = AsteroidStatic(seed=0)`, print
`env.get_observation_dim()` (=4), `env.get_action_dim()` (=2), `env.get_action_limits()` (=(-1,1)).
Call `obs = env.reset()` and a few `env.step([0.0, 0.0])` to see the `obs, reward, done, info` shape.

**Step 1.2 — Define the 9 actions** (`action_space.py`):
```python
import itertools, numpy as np
ACTIONS = [np.array(a, dtype=float) for a in itertools.product((-1.,0.,1.), (-1.,0.,1.))]  # 9 of them
N_ACTIONS = len(ACTIONS)          # 9
def to_continuous(i): return ACTIONS[i]   # agent picks index 0..8 -> real [ax,ay]
```

**Step 1.3 — Count `_calc_next_state` *without editing the file*.** Wrap the method on the instance:
```python
class CallCounter:
    def __init__(self, env):
        self.n = 0
        self._orig = env._calc_next_state
        env._calc_next_state = self._wrap
    def _wrap(self, a):
        self.n += 1
        return self._orig(a)
```
Because you only ever take *real* steps (no simulated look-ahead), **calls == steps**. Verify this:
after an episode, `counter.n` should equal the number of `env.step(...)` calls.

**Step 1.4 — Random agent sanity loop.** Reset, then repeatedly pick a random action index, step, until
`done`. Print episode length, total reward, and `info["reached_goal"]`. You'll see most random episodes
crash quickly — that's expected and motivates learning.

> ✅ **Checkpoint 1:** You can run full episodes with the 9 actions and your counter matches the step
> count exactly.

---

## Part 2 — Build a vanilla DQN

**Idea:** a neural net guesses, for the current state, the value of each of the 9 actions. Act greedily
(usually) on the highest value; occasionally act randomly to explore (ε-greedy). Learn by nudging the
predicted value toward `reward + γ × (best value of the next state)`.

**Step 2.1 — The network** (`network.py`): input 4 numbers, output 9 values.
```python
import torch, torch.nn as nn
class QNet(nn.Module):
    def __init__(self, n_in=4, n_out=9, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,hidden), nn.ReLU(),
                                 nn.Linear(hidden,hidden), nn.ReLU(),
                                 nn.Linear(hidden,n_out))
    def forward(self, x): return self.net(x)   # shape [batch, 9]
```
*Tip:* normalize the input first (e.g. divide x,y by ~5 using the known default `world_bounds`, and
vx,vy by `v_max=4`). This only scales inputs — it is not "analyzing" the env — and it makes training stable.

**Step 2.2 — Replay buffer** (`replay.py`): a fixed-size memory of past transitions
`(state, action, reward, next_state, done)`. Store every step; sample random minibatches to learn from.
This reuses data (good for Goal 1) and decorrelates samples. Use a `collections.deque(maxlen=50000)`;
`sample(batch=64)` returns numpy arrays.

**Step 2.3 — ε-greedy action selection** (in `agent.py`): with probability ε pick a random action index,
else pick `argmax` of `QNet(state)`. Start ε high (1.0) and **decay** it toward ~0.05 over training so
you explore early and exploit later.

**Step 2.4 — The learning step.** Keep TWO copies of the network: `online` (trained every step) and
`target` (a frozen copy, refreshed every `C` steps). For a sampled batch:
```
q_pred   = online(states).gather(action)              # value the net gave the action you took
with no_grad:
    q_next = target(next_states).max(dim=1)           # best value in the next state (vanilla DQN)
    y = rewards + gamma * q_next * (1 - dones)         # 'done' kills the future term
loss = SmoothL1(q_pred, y)                             # Huber loss
optimizer.zero_grad(); loss.backward(); optimizer.step()
every C steps: target.load_state_dict(online.state_dict())
```
Use `Adam(lr≈1e-3)`. The `target` net stops the learning target from chasing itself (stability).

**Step 2.5 — Training loop** (`trainer.py`):
```
for episode in range(max_episodes):
    s = env.reset()
    for t in range(max_steps):
        a = agent.act(s, epsilon)
        s2, r, done, info = env.step(to_continuous(a))
        buffer.add(s, a, r, s2, done)
        if len(buffer) > warmup: agent.learn(buffer.sample(batch))
        s = s2; decay(epsilon)
        if done: break
```

**Starter hyperparameters** (tune later): `hidden=128`, `gamma=0.99`, `lr=1e-3`, `batch=64`,
`buffer=50000`, `warmup=1000`, target sync `C=500` steps, ε `1.0 → 0.05` over ~20k steps.

> ✅ **Checkpoint 2:** On `seed=0`, over a few hundred episodes the episode length trends **down** and
> `reached_goal` starts becoming `True`. If it never reaches the goal, see **Debugging** below.

---

## Part 3 — Upgrade to DDQN (one tiny change)

Vanilla DQN's `max` makes it *over-estimate* values. Fix: let the **online** net *choose* the best next
action, but let the **target** net *score* it:
```
with no_grad:
    a_star = online(next_states).argmax(dim=1)
    q_next = target(next_states).gather(a_star)
    y = rewards + gamma * q_next * (1 - dones)
```
That's the only change. Add a flag `double=True/False` to your agent so you can compare DQN vs DDQN.

> ✅ **Checkpoint 3:** DDQN learns at least as reliably as DQN (usually steadier). Keep both for the report.

---

## Part 4 — Upgrade to Dueling DDQN (network change)

Split the network's last part into two heads — a single **state value** `V(s)` and per-action
**advantages** `A(s, a)` — then recombine:
```python
# after a shared trunk producing features h:
V = self.value(h)                       # shape [batch, 1]
A = self.adv(h)                         # shape [batch, 9]
Q = V + (A - A.mean(dim=1, keepdim=True))
```
This helps the net learn "is this a good *place* to be" separately from "which *action* is best", which
generalizes value better. Add a `dueling=True/False` flag. Combine with `double=True` → **Dueling DDQN**
(your strongest variant for Goal 2).

> ✅ **Checkpoint 4:** One `DQNAgent(double, dueling)` class produces DQN `(F,F)`, DDQN `(T,F)`,
> Dueling DDQN `(T,T)`.

---

## Part 5 — Measure the two graded numbers

While training (one continuous run per arena, **using the call counter from Step 1.3**):

- **Metric 1 — calls to first solve:** record `counter.n` at the end of the **first** episode where
  `info["reached_goal"]` is `True`. (Definition of "solved" = first goal reach; mention this choice in
  your report.)
- **Metric 2 — fewest time steps:** track the **shortest** successful episode length seen so far, and
  remember `counter.n` at the moment that shortest length was first achieved (the ungraded "associated
  calls").

Run this for **seeds 0–9** (ten different arenas) and **average** the numbers. Those averages are your
three report numbers: avg Metric 1, avg Metric 2, avg associated calls.

> ✅ **Checkpoint 5:** A function `evaluate(agent_cfg, seeds)` returns the three averaged numbers.

---

## Part 6 — Tune hyperparameters with Optuna

Let Optuna search good values instead of guessing. **Critical rule (Task2.pdf):** tune on **different
seeds** than your final eval — e.g. tune on seeds `100–119`, report on seeds `0–9`.
```python
import optuna
def objective(trial):
    lr   = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gamma= trial.suggest_float("gamma", 0.95, 0.999)
    eps_decay = trial.suggest_int("eps_decay_steps", 5000, 50000)
    C    = trial.suggest_int("target_sync", 100, 2000)
    # train on tuning seeds, return the metric you care about (lower = better)
    return mean_calls_to_first_solve(cfg(lr,gamma,eps_decay,C), seeds=range(100,120))
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40)
```
Run one study aimed at **Metric 1** and (optionally) one aimed at **Metric 2** — they may prefer
different settings (fast exploration vs. polished policy).

> ✅ **Checkpoint 6:** You have a tuned config (or two) found on non-eval seeds.

---

## Part 7 — Final run, plots, and the report

**`main.py`:** load the tuned config, run over seeds `0–9`, print the three averaged numbers, and save:
- a **learning curve** (episode length & success vs. number of calls), and
- a **trajectory plot** of the trained policy (roll out greedily and overlay the path on
  `env.save_frame(...)`) — this is the "plot that it can be solved" the report asks for.

**`readme.txt`:** one paragraph on how to run (`python main.py`), dependencies, expected outputs.

**Report (≤5 pages, from Task2.pdf):** describe the approaches you tried (incl. *why* you discretized,
and a short note on **Deep SARSA** — on-policy, risk-aware near the −100 crash penalty — and **Rainbow**
— DQN + Double + Dueling + Prioritized Replay + n-step + Distributional + Noisy Nets — as methods you
considered), a DQN-vs-DDQN-vs-Dueling comparison table on both metrics, your final best result, and the
trajectory plot. You're encouraged to show what *didn't* work.

> ✅ **Checkpoint 7 (done):** `python main.py` prints the three numbers and saves both plots; all 10
> arenas are solved.

---

## Debugging — if it won't learn

- **Never reaches the goal:** explore longer (slower ε decay), raise `warmup`, confirm the reward sign
  (progress should be positive when moving toward goal). Check you pass `to_continuous(a)` (a vector),
  not the integer index, to `env.step`.
- **Learns then collapses:** lower `lr`, sync the target net less often (larger `C`), use Huber loss.
- **Crashes into asteroids constantly:** that's the −100 penalty doing its job; give it more episodes,
  and note that **Deep SARSA** (on-policy) is naturally more cautious here — a good "things I tried" item.
- **Counter ≠ steps:** you're calling `step` somewhere extra (e.g. a stray evaluation rollout) — make
  sure *every* step you take is one you intend to count.

## Suggested order to build (milestones)
Part 1 → random agent runs & counts → Part 2 DQN solves seed 0 → Part 3 DDQN → Part 4 Dueling →
Part 5 metrics over 10 seeds → Part 6 Optuna → Part 7 final numbers + plots + report.

## Algorithm cheat-sheet (for the report)

| Method | One-line idea | Key change vs. previous |
|---|---|---|
| **Deep SARSA** | On-policy TD; target uses the action ε-greedy *actually takes* next | risk-aware (cautious near −100 crashes); usually no replay |
| **DQN** | Off-policy Q-learning + neural net | adds **experience replay** + **target network** |
| **DDQN** | Reduce `max` overestimation | online net *selects* next action, target net *scores* it |
| **Dueling DDQN** | Separate "how good is this state" from "which action" | network split into **V(s)** + **A(s,a)** heads |
| **Rainbow** | Combine the best DQN upgrades | + Prioritized Replay + n-step + Distributional (C51) + Noisy Nets |
