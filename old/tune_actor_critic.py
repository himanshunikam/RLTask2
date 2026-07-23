"""Optuna hyperparameter search for the base Actor-Critic.

Run:  python tune_actor_critic.py

Tunes on TUNE_SEEDS (disjoint from the 0-9 eval seeds -- Task2 rule). Objective =
calls-to-first-solve (lower is better); if a config never solves within the episode
budget it falls back to a distance penalty so the search still prefers configs that
get closer to the goal. Best params printed and saved to best_actor_critic.json.

Reuses the real train() from actor_critic.py (no duplicated algorithm here).
"""
import json
import numpy as np
import optuna
from old.old_actor_critic import train

# ---- search / budget knobs (raise for a better search, lower for speed) ----------
TUNE_SEEDS   = [115, 119]     # 107 111
MAX_EPISODES = 5000           # per-seed episode budget per trial
N_TRIALS     = 50


def score(metric1, best_min_dist):
    if metric1 is not None:
        return float(metric1)                    # solved -> minimize calls-to-first-solve
    return 1e7 + best_min_dist * 1e6             # never solved -> prefer getting closer


def objective(trial):
    params = dict(
        actor_lr           = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        critic_lr          = trial.suggest_float("critic_lr", 1e-4, 3e-3, log=True),
        gamma              = trial.suggest_float("gamma", 0.95, 0.999),
        noise_std          = trial.suggest_float("noise_std", 0.05, 0.4),
        epsilon_min        = trial.suggest_float("epsilon_min", 0.05, 0.3),
        epsilon_decay      = trial.suggest_float("epsilon_decay", 0.99, 0.9999),
        target_update_freq = trial.suggest_int("target_update_freq", 100, 1000),
        batch              = trial.suggest_categorical("batch", [64, 128, 256]),
        hidden             = trial.suggest_categorical("hidden", [128, 256]),
    )
    scores = []
    for i, seed in enumerate(TUNE_SEEDS):
        m1, bmd, m2_len, _ = train(seed=seed, max_episodes=MAX_EPISODES,
                                   report_step_offset=i * MAX_EPISODES, trial=trial, **params)
        scores.append(score(m1, bmd))
        trial.set_user_attr(f"metric2_len_seed{seed}", m2_len)   # shortest successful episode (or None)
    solved = [v for k, v in trial.user_attrs.items()
              if k.startswith("metric2_len_seed") and v is not None]
    trial.set_user_attr("metric2_len_min", int(min(solved)) if solved else None)
    return float(np.mean(scores))


if __name__ == "__main__":
    print("new code running!")
    study = optuna.create_study(
        study_name="ac_session2",
        storage="sqlite:///tune_ac_s3.db",  
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1000),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("\n=== best trial ===")
    print("score :", study.best_value)
    print("params:", study.best_params)
    print("metric2 (best trial):", study.best_trial.user_attrs)
    with open("best_actor_critic.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("saved -> best_actor_critic.json")
