"""Evaluate the tuned Actor-Critic on the 10 eval seeds (0-9).

Run:  python eval_actor_critic.py
Loads best_actor_critic.json, trains one fresh agent per seed, and reports the three
graded numbers averaged over the seeds that solved. Results persist to eval_actor_critic.json
after every seed, so partial progress survives interruption.
"""
import json
import numpy as np
from stoch_actor_critic import train

METHOD       = "Actor-Critic"
CFG_PATH     = "new_best_actor_critic.json"
OUT_PATH     = "new_eval_actor_critic_results.json"
EVAL_SEEDS   = range(10)
MAX_EPISODES = 5000

cfg = json.load(open(CFG_PATH))
print(f"[{METHOD}] config: {cfg}\n", flush=True)

rows = []
for seed in EVAL_SEEDS:
    m1, bmd, m2_len, m2_calls = train(seed=seed, max_episodes=MAX_EPISODES, **cfg)
    rows.append({"seed": seed,
                 "metric1_calls_to_solve": m1,
                 "metric2_shortest_len": m2_len,
                 "metric2_assoc_calls": m2_calls,
                 "best_min_dist": bmd})
    print(f"[{METHOD}] seed {seed}: metric1={m1} | metric2_len={m2_len} | "
          f"metric2_calls={m2_calls} | minDist={bmd:.2f}", flush=True)
    json.dump(rows, open(OUT_PATH, "w"), indent=2)      # persist after each seed

# ---- aggregate over the seeds that actually solved ----
m1s = [r["metric1_calls_to_solve"] for r in rows if r["metric1_calls_to_solve"] is not None]
m2s = [r["metric2_shortest_len"]  for r in rows if r["metric2_shortest_len"]  is not None]
m2c = [r["metric2_assoc_calls"]   for r in rows if r["metric2_assoc_calls"]   is not None]

summary = {
    "method": METHOD,
    "solved": f"{len(m1s)}/10",
    "avg_metric1_calls_to_solve": float(np.mean(m1s)) if m1s else None,
    "avg_metric2_shortest_len":   float(np.mean(m2s)) if m2s else None,
    "avg_metric2_assoc_calls":    float(np.mean(m2c)) if m2c else None,
}
print(f"\n=== {METHOD} SUMMARY (averages over solved seeds) ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
json.dump({"per_seed": rows, "summary": summary}, open(OUT_PATH, "w"), indent=2)
print(f"\nsaved -> {OUT_PATH}")
