import json
import os
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from actor_critic import train as train_ac
from DDPG import train as train_ddpg
from TD3 import train as train_td3

CFG_DIR      = "Hyperparameter Tuning"
EVAL_SEEDS   = range(10)
MAX_EPISODES = 5000


METHODS = [
    ("Actor-Critic", train_ac,   "new_best_actor_critic.json"),
    ("DDPG",         train_ddpg, "best_ddpg_new.json"),
    ("TD3",          train_td3,  "best_td3_new.json"),
]


def evaluate(name, train_fn, cfg_file):
    cfg = json.load(open(os.path.join(CFG_DIR, cfg_file)))
    print(f"\n{'='*64}\n{name}  |  config: {cfg}\n{'='*64}", flush=True)

    rows = []
    for seed in EVAL_SEEDS:
        m1, bmd, m2_len, m2_calls = train_fn(seed=seed, max_episodes=MAX_EPISODES, **cfg)
        rows.append({"seed": seed,
                     "metric1_calls_to_solve": m1,
                     "metric2_shortest_len": m2_len,
                     "metric2_assoc_calls": m2_calls,
                     "best_min_dist": bmd})
        print(f"[{name}] seed {seed}: metric1={m1} | metric2_len={m2_len} | "
              f"metric2_calls={m2_calls} | minDist={bmd:.2f}", flush=True)

    m1s = [r["metric1_calls_to_solve"] for r in rows if r["metric1_calls_to_solve"] is not None]
    m2s = [r["metric2_shortest_len"]  for r in rows if r["metric2_shortest_len"]  is not None]
    m2c = [r["metric2_assoc_calls"]   for r in rows if r["metric2_assoc_calls"]   is not None]
    summary = {
        "method": name,
        "solved": f"{len(m1s)}/{len(EVAL_SEEDS)}",
        "avg_metric1_calls_to_solve": float(np.mean(m1s)) if m1s else None,
        "avg_metric2_shortest_len":   float(np.mean(m2s)) if m2s else None,
        "avg_metric2_assoc_calls":    float(np.mean(m2c)) if m2c else None,
    }
    return {"per_seed": rows, "summary": summary}


if __name__ == "__main__":

    results = {}
    for name, fn, cfg_file in METHODS:
        results[name] = evaluate(name, fn, cfg_file)
    print("\n=== SUMMARY (averages over solved seeds) ===")
    for name in results:
        print(name, results[name]["summary"])

