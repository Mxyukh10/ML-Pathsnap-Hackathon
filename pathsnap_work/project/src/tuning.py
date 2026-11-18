
import json
import joblib
import optuna
import pandas as pd

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from src.pipeline import load_partition, FEATURES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

print("Loading training data...")
df, _ = load_partition("train", 8)

# Reproduce the original pipeline's deterministic 180k cap
if len(df) > 180000:
    pos = df[df.y == 1]
    neg = df[df.y == 0].sample(
        n=180000 - len(pos), random_state=42
    )
    df = pd.concat([pos, neg]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

X = df[FEATURES].replace([float("inf"), float("-inf")], 0).fillna(0)
y = df["y"].astype(int)
groups = df["trip_id"]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)
train_idx, val_idx = next(splitter.split(X, y, groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

def score(model, X_data, y_data):
    pred = model.predict(X_data)
    return {
        "accuracy": accuracy_score(y_data, pred),
        "precision": precision_score(y_data, pred, zero_division=0),
        "recall": recall_score(y_data, pred, zero_division=0),
        "f1": f1_score(y_data, pred, zero_division=0)
    }

# Original baseline
baseline = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=20,
    learning_rate=0.08,
    l2_regularization=1.0,
    random_state=42
)

print("Training baseline...")
baseline.fit(X_train, y_train)
baseline_metrics = score(baseline, X_val, y_val)

# Bayesian optimisation using Optuna TPE
def objective(trial):
    params = {
        "max_iter": trial.suggest_int("max_iter", 80, 250),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 40),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.2, log=True
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 0.01, 10.0, log=True
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 10, 50
        ),
        "random_state": 42
    }

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return f1_score(
        y_val, model.predict(X_val), zero_division=0
    )

print("Starting Bayesian optimisation...")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=15)

# Save trial history and best parameters
study.trials_dataframe().to_csv(
    RESULTS / "bayesian_trials.csv", index=False
)

best_params = study.best_params
best_params["random_state"] = 42

with open(RESULTS / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best parameters:", best_params)

# Compare baseline and tuned model on same validation split
tuned = HistGradientBoostingClassifier(**best_params)
tuned.fit(X_train, y_train)
tuned_metrics = score(tuned, X_val, y_val)

comparison = pd.DataFrame([
    {"model": "Baseline HistGradientBoosting", **baseline_metrics},
    {"model": "Tuned HistGradientBoosting", **tuned_metrics}
])
comparison.to_csv(
    RESULTS / "tuning_comparison.csv", index=False
)

# Refit tuned model on all permitted training rows
print("Retraining tuned model on all training rows...")
final_model = HistGradientBoostingClassifier(**best_params)
final_model.fit(X, y)

joblib.dump(final_model, MODELS / "tuned_model.joblib")

print("\nTUNING COMPLETE")
print(comparison.to_string(index=False))
print("\nSaved:")
print(RESULTS / "bayesian_trials.csv")
print(RESULTS / "best_params.json")
print(RESULTS / "tuning_comparison.csv")
print(MODELS / "tuned_model.joblib")
