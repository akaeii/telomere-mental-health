import os
import shutil
import optuna
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

import pandas as pd

SEED = 42
N_FOLDS = 10
SPW = 1.56


def objective(trial):
    data = pd.read_csv("data/hamd_data.csv")
    y = data.pop("hamd_response")
    X = data

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": SEED,
    }

    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    mcc_scores = []
    best_n_estimators_list = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        clf = xgb.XGBClassifier(
            **param,
            n_estimators=10000,
            early_stopping_rounds=100,
            seed=SEED,
            scale_pos_weight=SPW,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        preds = clf.predict(X_valid)
        mcc = matthews_corrcoef(y_valid, preds)
        mcc_scores.append(mcc)
        best_n_estimators_list.append(clf.best_iteration)

    trial.set_user_attr("n_estimators", int(np.max(best_n_estimators_list)))
    return np.mean(mcc_scores)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=1800)

    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (MCC): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"  Avg. number of estimators: {trial.user_attrs['n_estimators']}")
