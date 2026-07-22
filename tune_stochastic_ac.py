import json
import numpy as np
import optuna
from stoch_actor_critic import train

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
        gamma              = trial.suggest_float("gamma", 0.9, 0.999),
        beta = trial.suggest_float("beta", 1e-3, 0.1, log=True),
        hidden             = trial.suggest_categorical("hidden", [128, 256, 512]),
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
        study_name="new_ac",
        storage="sqlite:///new_ac_s4.db",  
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(ValueError, FloatingPointError))
    print("\n=== best trial ===")
    print("score :", study.best_value)
    print("params:", study.best_params)
    print("metric2 (best trial):", study.best_trial.user_attrs)
    with open("new_best_actor_critic.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("saved -> new_best_actor_critic.json")
