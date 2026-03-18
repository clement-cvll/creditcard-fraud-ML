"""Credit-card fraud detection — LightGBM with Optuna hyperparameter tuning."""

import os
import warnings

import kagglehub
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 100
N_TRIALS = 100


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_data():
    """Download and load the credit-card fraud dataset."""
    kagglehub.dataset_download("mlg-ulb/creditcardfraud", output_dir="data")
    df = pl.read_csv(os.path.join("data", "creditcard.csv"), ignore_errors=True)
    X, y = df.drop("Class").to_numpy(), df["Class"].to_numpy()
    return X, y


def split_data(X, y):
    """Stratified 60/20/20 train/val/test split."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE
    )
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5s}: {len(ys):>7,}  (fraud: {ys.sum()})")
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_fixed_params(fraud_weight):
    """Return the fixed (non-tuned) LightGBM parameters."""
    return {
        "objective": "binary",
        "metric": "custom",
        "boosting_type": "gbdt",
        "class_weight": {0: 1, 1: fraud_weight},
        "n_jobs": -1,
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
    }


def sample_params(trial):
    """Sample hyperparameters for a single Optuna trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }


def evaluate(model, X_val, y_val, X_test, y_test):
    """Tune threshold on val set for F1, then compute metrics on test set."""
    # Find best threshold on validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

    # Compute F1 for each threshold (handle div by zero)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Apply threshold to test set
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= best_thresh).astype(int)

    return {
        "mcc": matthews_corrcoef(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_test_proba),
        "f1": f1_score(y_test, y_pred),
        "y_proba": y_test_proba,
        "best_iter": model.best_iteration_,
        "threshold": best_thresh,
    }


def eval_pr_auc(y_true, y_pred):
    """Custom LightGBM metric for PR-AUC."""
    is_higher_better = True
    return "pr_auc", average_precision_score(y_true, y_pred), is_higher_better


def plot_pr_curve(
    y_test, y_proba, pr_auc, path=os.path.join("visuals", "pr_curve.png")
):
    """Save a Precision-Recall curve to disk."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline = y_test.sum() / len(y_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        color="steelblue",
        lw=2,
        label=f"LightGBM (PR-AUC = {pr_auc:.4f})",
    )
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({baseline:.4f})",
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve (LightGBM)",
        xlim=[0, 1],
        ylim=[0, 1.05],
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    print(f"PR curve saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Data
    print("Loading data...")
    X, y = load_data()
    n_neg, n_pos = np.bincount(y)
    fraud_weight = n_neg / n_pos
    print(
        f"Class distribution: {n_neg:,} non-fraud / {n_pos:,} fraud ({fraud_weight:.0f}:1)\n"
    )

    print("Splitting data:")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    fixed_params = make_fixed_params(fraud_weight)

    # ── Optuna optimization with pruning ─────────────────────────────────
    print(f"\nOptuna optimization ({N_TRIALS} trials with pruning)...")

    def objective(trial):
        params = {**fixed_params, **sample_params(trial)}
        pruning_cb = optuna.integration.LightGBMPruningCallback(
            trial, "pr_auc", valid_name="valid_0"
        )

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_pr_auc,
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
                pruning_cb,
            ],
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_proba)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\nBest trial: #{study.best_trial.number} (PR-AUC = {study.best_value:.4f})")
    print(f"Best params: {study.best_params}")

    # Retrain with best params
    best_params = {**fixed_params, **study.best_params}
    print("\nTraining final model...")

    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_pr_auc,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    # Evaluate
    results = evaluate(model, X_val, y_val, X_test, y_test)
    print(
        f"\nTest results | MCC: {results['mcc']:.4f}"
        f" | PR-AUC: {results['pr_auc']:.4f}"
        f" | F1: {results['f1']:.4f}"
        f" | best_iter: {results['best_iter']}"
        f" | threshold: {results['threshold']:.4f}"
    )

    # Plot
    plot_pr_curve(y_test, results["y_proba"], results["pr_auc"])


if __name__ == "__main__":
    main()
