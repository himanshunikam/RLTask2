"""Optuna hyperparameter search for TD3.

Run:  python tune_td3.py

Tunes on TUNE_SEEDS (disjoint from the 0-9 eval seeds -- Task2 rule). Objective =
calls-to-first-solve (lower is better) with a distance fallback when a config never
solves within the budget. Best params printed and saved to best_td3.json.

Reuses the real train() from TD3.py (no duplicated algorithm here).
"""
import json
import numpy as np
import optuna
from TD3 import train

# ---- search / budget knobs (raise for a better search, lower for speed) ----------
TUNE_SEEDS   = [100, 101]
MAX_EPISODES = 2500
N_TRIALS     = 30


def score(metric1, best_min_dist):
    if metric1 is not None:
        return float(metric1)
    return 1e7 + best_min_dist * 1e6


def objective(trial):
    params = dict(
        actor_lr     = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        critic_lr    = trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True),
        tau          = trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        gamma        = trial.suggest_float("gamma", 0.95, 0.999),
        n_step       = trial.suggest_int("n_step", 1, 5),
        ou_sigma     = trial.suggest_float("ou_sigma", 0.1, 0.4),
        policy_noise = trial.suggest_float("policy_noise", 0.1, 0.4),
        noise_clip   = trial.suggest_float("noise_clip", 0.3, 0.6),
        policy_delay = trial.suggest_int("policy_delay", 1, 4),
        batch        = trial.suggest_categorical("batch", [64, 128, 256]),
        hidden       = trial.suggest_categorical("hidden", [128, 256]),
    )
    scores = []
    for i, seed in enumerate(TUNE_SEEDS):
        m1, bmd, _ = train(seed=seed, max_episodes=MAX_EPISODES,
                           report_step_offset=i * MAX_EPISODES, trial=trial, **params)
        scores.append(score(m1, bmd))
    return float(np.mean(scores))


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="td3_new",
        storage="sqlite:///tune_td3.db",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("\n=== best trial ===")
    print("score :", study.best_value)
    print("params:", study.best_params)
    with open("best_td3_new.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("saved -> best_td3_new.json")
