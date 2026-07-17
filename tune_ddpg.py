"""Optuna hyperparameter search for DDPG.

Run:  python tune_ddpg.py

Tunes on TUNE_SEEDS (disjoint from the 0-9 eval seeds -- Task2 rule). Objective =
calls-to-first-solve (lower is better) with a distance fallback when a config never
solves within the budget. Best params printed and saved to best_ddpg.json.

Reuses the real train() from DDPG.py (no duplicated algorithm here).
"""
import json
import numpy as np
import optuna
from DDPG import train

# ---- search / budget knobs (raise for a better search, lower for speed) ----------
TUNE_SEEDS   = [115, 119]     # easiest of 100-119 (random occasionally reaches goal)
MAX_EPISODES = 5000
N_TRIALS     = 30


def score(metric1, best_min_dist):
    if metric1 is not None:
        return float(metric1)
    return 1e7 + best_min_dist * 1e6


def objective(trial):
    params = dict(
        actor_lr  = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        critic_lr = trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True),
        tau       = trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        gamma     = trial.suggest_float("gamma", 0.95, 0.999),
        ou_sigma  = trial.suggest_float("ou_sigma", 0.1, 0.4),
        batch     = trial.suggest_categorical("batch", [64, 128, 256]),
        hidden    = trial.suggest_categorical("hidden", [128, 256]),
    )
    scores = []
    for i, seed in enumerate(TUNE_SEEDS):
        m1, bmd, m2_len = train(seed=seed, max_episodes=MAX_EPISODES,
                                report_step_offset=i * MAX_EPISODES, trial=trial, **params)
        scores.append(score(m1, bmd))
        trial.set_user_attr(f"metric2_len_seed{seed}", m2_len)   # shortest successful episode (or None)
    solved = [v for k, v in trial.user_attrs.items()
              if k.startswith("metric2_len_seed") and v is not None]
    trial.set_user_attr("metric2_len_min", int(min(solved)) if solved else None)
    return float(np.mean(scores))


if __name__ == "__main__":
    print("new code running")
    study = optuna.create_study(
        study_name="ddpg_tune_session2",
        storage="sqlite:///tune_ddpg_s2.db",     # <-- persists every completed trial to disk
        direction="minimize",
        load_if_exists=True,  
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("\n=== best trial ===")
    print("score :", study.best_value)
    print("params:", study.best_params)
    print("metric2 (best trial):", study.best_trial.user_attrs)
    with open("best_ddpg.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("saved -> best_ddpg.json")
