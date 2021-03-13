import pandas as pd
import numpy as np
import optuna
import logging
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import loss
import trainers

def cross_validation(train_df, y, test_df, params, categorical_features=[]):

    fit_params = {
        'num_boost_round': 10000,
        'early_stopping_rounds': 200,
        'verbose_eval': 200,
    }

    print(fit_params)
    feat_name = [*train_df.columns]

    # 交差検証
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    y_oof = np.empty([len(train_df),])
    y_test = []
    feature_importances = pd.DataFrame()
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, y.astype(int))):
        print('Fold {}'.format(fold + 1))

        x_train, y_train = train_df.iloc[train_idx][feat_name], y.iloc[train_idx]
        x_val, y_val = train_df.iloc[valid_idx][feat_name], y.iloc[valid_idx]
        x_test = test_df[feat_name]

        y_pred_valid, y_pred_test, valid_loss, importances, best_iter = trainers.train_lgbm(
            x_train, y_train, x_val, y_val, x_test, params, fit_params, 
            categorical_features=categorical_features,
            feature_name=feat_name,
            fold_id=fold,
            loss_func=loss.calc_loss,
            calc_importances=True
        )

        y_oof[valid_idx] = y_pred_valid
        score = loss.calc_loss(y[valid_idx], y_pred_valid)
        y_test.append(y_pred_test)
        feature_importances = pd.concat([feature_importances, importances], axis=0, sort=False)

    # feature_importances.to_csv(output_path.joinpath("feature_importances.csv"), index=False)
    # feature_importances.groupby("feature", as_index=False).mean().to_csv(output_path.joinpath("feature_importances_cvmean.csv"), index=False)

    # validのスコア計算
    score = loss.calc_loss(y, y_oof)
    print(f"valid score: {score}")

    # submission用(CVの平均を結果とする)
    y_test = np.mean(y_test,axis=0)

    return score, y_test

def run_optuna(train_df, y, test_df, categorical_features=[]):

    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('./optuna.log'))

    def objective(trial, train_df, y, test_df, categorical_features):

        fixed_params = {
            'objective': 'regression',
            'metric':"rmse",
            "verbosity": -1,
            'learning_rate': 0.05,
            'max_depth': -1,
            'num_threads': 0,
        }

        params = {
            "boosting"        : trial.suggest_categorical("boosting", ['gbdt']),
            'num_leaves'      : trial.suggest_int        ("num_leaves"      ,     8,    512, step=8  ),
            'max_bin'         : trial.suggest_int        ("max_bin"         ,     8,    512, step=8  ),
            'min_data_in_bin' : trial.suggest_int        ("min_data_in_bin" ,     8,    512, step=8  ),
            'feature_fraction': trial.suggest_float      ("feature_fraction",   0.1,    1.0, step=0.1),
            "bagging_freq"    : trial.suggest_int        ("bagging_freq"    ,     1,    100, step=5  ),
            "bagging_fraction": trial.suggest_float      ("feature_fraction",   0.1,    1.0, step=0.1),
            'lambda_l1'       : trial.suggest_loguniform ("lambda_l1"       , 0.001, 1000.0),
            'lambda_l2'       : trial.suggest_loguniform ("lambda_l2"       , 0.001, 1000.0),
        }

        params.update(fixed_params)

        score, y_test = cross_validation(train_df, y, test_df, params, categorical_features)

        return score

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, train_df, y, test_df, categorical_features), n_trials=200, n_jobs=1)

    best_params = study.best_params

    # hist_df = study.trials_dataframe(multi_index=True)
    # hist_df.to_csv("result.csv")

    best_params.update({
        'objective': 'regression',
        'metric':"rmse",
        "verbosity": -1,
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_threads': 0,
    })

    return best_params