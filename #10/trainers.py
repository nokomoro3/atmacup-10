import pandas as pd
import lightgbm as lgb
from catboost import Pool

def train_lgbm(X_train, y_train, X_valid, y_valid, X_test, params, fit_params, categorical_features, feature_name, fold_id, loss_func, calc_importances=True):

    train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)
    if X_valid is not None:
        valid = lgb.Dataset(X_valid, y_valid,
                            categorical_feature=categorical_features,
                            feature_name=feature_name)

    if X_valid is not None:
        model = lgb.train(
            params,
            train,
            valid_sets=[train,valid],
            **fit_params
        )
    else:
        model = lgb.train(
            params,
            train,
            **fit_params
        )

    # train score
    if X_valid is not None:
        y_pred_valid = model.predict(X_valid)
        valid_loss = loss_func(y_valid, y_pred_valid)
    else:
        y_pred_valid = None
        valid_loss = None

    #test
    if X_test is not None:
        y_pred_test = model.predict(X_test)
    else:
        y_pred_test = None

    if calc_importances:
        importances = pd.DataFrame()
        importances['feature'] = feature_name
        importances['gain'] = model.feature_importance(importance_type='gain')
        importances['split'] = model.feature_importance(importance_type='split')
        importances['fold'] = fold_id
    else:
        importances = None

    return y_pred_valid, y_pred_test, valid_loss, importances, model.best_iteration

def train_catboost(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name, fold_id, loss_func, calc_importances=True):

    # lgb_params = {'num_leaves': 32,
    #             'min_data_in_leaf': 64,
    #             'objective': 'regression',
    #             'max_depth': -1,
    #             'learning_rate': 0.05,
    #             "boosting": "gbdt",
    #             "bagging_freq": 1,
    #             "bagging_fraction": 0.8,
    #             "bagging_seed": 0,
    #             "verbosity": -1,
    #             'reg_alpha': 0.1,
    #             'reg_lambda': 0.3,
    #             'colsample_bytree': 0.7,
    #             'metric':"rmse",
    #             'num_threads':6,
    #         }

    fit_params = {
        'early_stopping_rounds': 200,
        'verbose_eval': 200,
    }

    # データセットの作成。Poolで説明変数、目的変数、
    # カラムのデータ型を指定できる
    train = Pool(X_train, y_train, cat_features=categorical_features, feature_names=feature_name)
    if X_valid is not None:
        valid = Pool(X_valid, y_valid, cat_features=categorical_features, feature_names=feature_name)

    # 分類用のインスタンスを作成
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(random_seed=42)
    if X_valid is not None:
        model.fit(train, eval_set=valid, use_best_model=True, **fit_params)
    else:
        model.fit(train, use_best_model=True, **fit_params)

    # train score
    if X_valid is not None:
        y_pred_valid = model.predict(X_valid)
        valid_loss = loss_func(y_valid, y_pred_valid)
    else:
        y_pred_valid = None
        valid_loss = None

    #test
    if X_test is not None:
        y_pred_test = model.predict(X_test)
    else:
        y_pred_test = None

    # if calc_importances:
    #     importances = pd.DataFrame()
    #     importances['feature'] = feature_name
    #     importances['gain'] = model.feature_importance(importance_type='gain')
    #     importances['split'] = model.feature_importance(importance_type='split')
    #     importances['fold'] = fold_id
    # else:
    #     importances = None
    importances = None

    return y_pred_valid, y_pred_test, valid_loss, importances, None
